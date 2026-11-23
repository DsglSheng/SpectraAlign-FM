from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from scipy import io as sio

try:
    from torchvision import transforms as tv_transforms
except ImportError:
    tv_transforms = None

from aware_open.downstream import AwareFeatureExtractor, build_downstream_metadata, resolve_unmixing_scene_config
from aware_open.spectral_prototypes import PrototypeBank, load_prototype_bank


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPERSIGMA_ROOT = REPO_ROOT.parent / "HyperSIGMA-main"


# The data loading, train patch sampling, whole-image tiling, and
# SAD/MSE evaluation functions below are copied from the official
# HyperSIGMA `HyperspectralUnmixing/func.py`, then path- and typing-adjusted
# so the whole downstream unmixing flow stays in this single script.


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def load_unmixing_data(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = sio.loadmat(str(file_path))
    cube = dataset["img_3d"].transpose([1, 2, 0]).astype(np.float32)
    endmembers = dataset["endmember"].transpose([1, 0]).astype(np.float32)
    abundance = dataset["abundance"]
    abundance = np.reshape(abundance, [abundance.shape[0], cube.shape[0], cube.shape[1]]).astype(np.float32)
    init_em = dataset["init_em"].astype(np.float32)
    print("data.shape:", cube.shape)
    print("init endmember.shape:", init_em.shape)
    print("endmember.shape:", endmembers.shape)
    print("abundance.shape:", abundance.shape)
    return cube, endmembers, abundance, init_em


def sad_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = F.normalize(y_true, dim=1, p=2)
    y_pred = F.normalize(y_pred, dim=1, p=2)
    cosine = torch.sum(y_true * y_pred, dim=1).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    return torch.mean(torch.acos(cosine))


def numpy_sad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if denom <= 0:
        return float("nan")
    cosine = np.clip(float(y_pred.dot(y_true) / denom), -1.0, 1.0)
    return float(np.arccos(cosine))


def order_endmembers(endmembers: np.ndarray, endmembers_gt: np.ndarray) -> tuple[dict[int, int], np.ndarray, float]:
    num_endmembers = endmembers.shape[0]
    mapping: dict[int, int] = {}
    sad_values: list[float] = []
    sad_mat = np.ones((num_endmembers, num_endmembers), dtype=np.float32)
    ordered = endmembers.copy()
    ordered_gt = endmembers_gt.copy()
    for i in range(num_endmembers):
        ordered[i, :] = ordered[i, :] / max(float(ordered[i, :].max()), 1.0e-12)
        ordered_gt[i, :] = ordered_gt[i, :] / max(float(ordered_gt[i, :].max()), 1.0e-12)
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_sad(ordered[i, :], ordered_gt[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = float(sad_mat.min())
        index_arr = np.where(sad_mat == minimum)
        if minimum == 100:
            break
        if len(index_arr) < 2:
            break
        index = (int(index_arr[0][0]), int(index_arr[1][0]))
        mapping[index[1]] = index[0]
        sad_values.append(minimum)
        sad_mat[index[0], index[1]] = 100
        rows += 1
        sad_mat[index[0], :] = 100
        sad_mat[:, index[1]] = 100
    sad_arr = np.asarray(sad_values, dtype=np.float32)
    average_sam = float(np.sum(sad_arr) / max(len(sad_arr), 1))
    return mapping, sad_arr, average_sam


def match_endmembers_and_abundances(
    endmembers: np.ndarray,
    endmembers_gt: np.ndarray,
    abundances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("predict_em:", endmembers.shape)
    print("GT_em:", endmembers_gt.shape)
    num_endmembers = endmembers.shape[0]
    mapping, _, average_sam = order_endmembers(endmembers, endmembers_gt)
    ordered_endmembers = []
    ordered_abundances = []
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / max(float(endmembers[i, :].max()), 1.0e-12)
        endmembers_gt[i, :] = endmembers_gt[i, :] / max(float(endmembers_gt[i, :].max()), 1.0e-12)
        ordered_endmembers.append(endmembers[mapping[i]])
        ordered_abundances.append(abundances[mapping[i], :, :])
    ordered_endmembers = np.asarray(ordered_endmembers)
    ordered_abundances = np.asarray(ordered_abundances)
    sad_ordered = []
    for i in range(num_endmembers):
        sad_ordered.append(numpy_sad(ordered_endmembers[i], endmembers_gt[i, :]))
    sad_ordered.append(average_sam)
    return np.asarray(sad_ordered, dtype=np.float32), ordered_endmembers, ordered_abundances


def alter_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    num_em = y_true.shape[0]
    y_true = np.reshape(y_true, [num_em, -1])
    y_pred = np.reshape(y_pred, [num_em, -1])
    residual = y_pred - y_true
    mse = np.mean(residual * residual, axis=1)
    average_mse = float(np.sum(mse) / len(mse))
    return np.insert(mse, num_em, average_mse, axis=0)


def whole2_train_and_test_data(img_size: int, img: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width, _ = img.shape
    if height < img_size:
        gap = img_size - height
        img = np.concatenate([img, img[(height - gap) : height, :, :]], axis=0)
    if width < img_size:
        gap = img_size - width
        img = np.concatenate([img, img[:, (width - gap) : width, :]], axis=1)
    height, width, _ = img.shape
    num_h = height // img_size
    num_w = width // img_size
    if height % img_size != 0:
        gap = (num_h + 1) * img_size - height
        img = np.concatenate([img, img[(height - gap) : height, :, :]], axis=0)
    if width % img_size != 0:
        gap = (num_w + 1) * img_size - width
        img = np.concatenate([img, img[:, (width - gap) : width, :]], axis=1)
    height, width, _ = img.shape
    print("padding img:", img.shape)
    num_h = height // img_size
    num_w = width // img_size
    sub_imgs = []
    for i in range(num_h):
        for j in range(num_w):
            sub_imgs.append(img[i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size, :])
    return np.asarray(sub_imgs, dtype=np.float32), num_h, num_w


def whole_get_train_and_test_data(
    img_size: int,
    img: np.ndarray,
    *,
    batch_size: int,
) -> tuple[data.DataLoader, data.DataLoader, int, int, int]:
    print("---------func: whole_get_train_and_test_data--------------")
    x_train, num_h, num_w = whole2_train_and_test_data(img_size, img)
    x_test, num_h, num_w = whole2_train_and_test_data(img_size, img)
    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    train_loader = data.DataLoader(data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(data.TensorDataset(x_test), batch_size=num_h, shuffle=False)
    return train_loader, test_loader, num_h, num_w, img_size


def mirror_hsi(height: int, width: int, band: int, input_normalize: np.ndarray, patch: int = 5) -> np.ndarray:
    padding = patch // 2
    mirror = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    mirror[padding : (padding + height), padding : (padding + width), :] = input_normalize
    for i in range(padding):
        mirror[padding : (height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    for i in range(padding):
        mirror[padding : (height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    for i in range(padding):
        mirror[i, :, :] = mirror[padding * 2 - i - 1, :, :]
    for i in range(padding):
        mirror[height + padding + i, :, :] = mirror[height + padding - 1 - i, :, :]
    return mirror


def gain_neighborhood_pixel(mirror_image: np.ndarray, points: np.ndarray, index: int, patch: int = 5) -> np.ndarray:
    x = int(points[index, 0])
    y = int(points[index, 1])
    return mirror_image[x : (x + patch), y : (y + patch), :]


def get_train_data(
    mirror_image: np.ndarray,
    band: int,
    train_points: np.ndarray,
    *,
    patch: int = 5,
) -> np.ndarray:
    x_train = np.zeros((train_points.shape[0], patch, patch, band), dtype=float)
    for i in range(train_points.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_points, i, patch)
    return x_train.squeeze()


def get_train_patch_seed_i(
    img_size: int,
    img: np.ndarray,
    train_number: int,
    batch_size: int,
    seed_i: int,
) -> data.DataLoader:
    np.random.seed(seed_i)
    random.seed(seed_i)
    height, width, band = img.shape
    data_label = np.ones([height, width])
    position = np.array(np.where(data_label == 1)).transpose(1, 0)
    selected = np.random.choice(position.shape[0], int(train_number), replace=False)
    selected = position[selected]
    train_mask = np.zeros(data_label.shape)
    for i in range(int(train_number)):
        train_mask[selected[i][0], selected[i][1]] = 1
    total_pos_train = np.argwhere(train_mask == 1)
    mirror_img = mirror_hsi(height, width, band, img, patch=img_size)
    img_train = get_train_data(mirror_img, band, total_pos_train, patch=img_size)
    img_train = torch.from_numpy(img_train.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    return data.DataLoader(data.TensorDataset(img_train), batch_size=batch_size, shuffle=True)


class SumToOne(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.scale * x, dim=1)


class NonnegativeWeightConstraint:
    def __call__(self, module: nn.Module) -> None:
        if hasattr(module, "weight"):
            module.weight.data = torch.clamp_min(module.weight.data, 0)


class EndmemberDecoder(nn.Module):
    def __init__(self, *, channels: int, num_em: int, kernel: int) -> None:
        super().__init__()
        padding = kernel // 2
        self.decoder = nn.Conv2d(num_em, channels, kernel_size=kernel, stride=1, padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        return self.relu(self.decoder(code))

    def getEndmembers(self) -> torch.Tensor:
        self.decoder.apply(NonnegativeWeightConstraint())
        return self.decoder.weight.data


class SigmaStyleAwareUnmix(nn.Module):
    # This abundance head follows the official HyperSIGMA unmixing fusion head layout,
    # while swapping only the encoder source from Sigma ViT to the Aware feature extractor.
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        channels: int,
        num_em: int,
        kernel: int,
        scale: float,
        num_tokens: int,
        pixel_branch_channels: int = 0,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.num_em = int(num_em)
        self.num_tokens = int(num_tokens)
        self.pixel_branch_channels = int(pixel_branch_channels)

        model_dim = int(extractor.model_dim)
        self.conv_features1 = nn.Conv2d(model_dim, self.num_tokens, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(model_dim, self.num_tokens, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(model_dim, self.num_tokens, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(model_dim, self.num_tokens, kernel_size=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.num_tokens),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.num_tokens),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.num_tokens),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.num_tokens),
            nn.Dropout(0.2),
        )
        self.smooth = nn.Conv2d(self.num_tokens * 4, self.num_tokens, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.num_tokens, self.num_em, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.num_em),
            nn.Dropout(0.2),
        )
        self.pixel_branch = None
        if self.pixel_branch_channels > 0:
            self.pixel_branch = nn.Sequential(
                nn.Conv2d(channels, self.pixel_branch_channels, kernel_size=1, bias=False),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(self.pixel_branch_channels),
                nn.Conv2d(self.pixel_branch_channels, self.pixel_branch_channels, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(self.pixel_branch_channels),
                nn.Conv2d(self.pixel_branch_channels, self.num_em, kernel_size=1),
            )
        self.sumtoone = SumToOne(scale)
        self.decoder = EndmemberDecoder(channels=channels, num_em=num_em, kernel=kernel)

    @staticmethod
    def _token_map(tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        grid_h, grid_w = grid_size
        if grid_h * grid_w != num_tokens:
            raise ValueError(f"grid_size {grid_size} is inconsistent with token count {num_tokens}.")
        return tokens.transpose(1, 2).reshape(batch_size, dim, grid_h, grid_w)

    def _multi_level_features(self, features: dict[str, torch.Tensor | tuple[int, int]]) -> list[torch.Tensor]:
        intermediate = features["intermediate_tokens"]
        grid_size = features["grid_size"]
        if not isinstance(intermediate, list) or len(intermediate) != 4:
            raise RuntimeError("Aware extractor must return exactly four intermediate token tensors for Sigma-style fusion.")
        if not isinstance(grid_size, tuple):
            raise RuntimeError("Aware extractor did not return a valid grid_size tuple.")
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        projected: list[torch.Tensor] = []
        for op, token_tensor in zip(ops, intermediate):
            if not isinstance(token_tensor, torch.Tensor):
                raise RuntimeError("Intermediate token entries must be torch.Tensor values.")
            projected.append(op(self._token_map(token_tensor, grid_size)))
        return projected

    def getAbundances(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | tuple[int, int]]]:
        height, width = int(x.shape[2]), int(x.shape[3])
        features = self.extractor.forward_features(x)
        multi_scale = self._multi_level_features(features)
        p4 = self.conv1(multi_scale[3])
        p3 = self.conv2(multi_scale[2])
        p2 = self.conv3(multi_scale[1])
        p1 = self.conv4(multi_scale[0])
        fused = torch.cat([p1, p2, p3, p4], dim=1)
        fused = F.interpolate(fused, size=(height, width), mode="bilinear", align_corners=True)
        fused = self.smooth(fused)
        logits = self.conv5(fused)
        if self.pixel_branch is not None:
            # Aware tokens are 8x8 for 64x64 inputs; this branch restores pixel-level detail
            # while keeping the HyperSIGMA reconstruction losses unchanged.
            logits = logits + self.pixel_branch(x)
        abunds = self.sumtoone(logits)
        if return_features:
            return abunds, features
        return abunds

    def forward(
        self,
        patch: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | tuple[int, int]]]:
        abunds_output = self.getAbundances(patch, return_features=return_features)
        if return_features:
            abunds, features = abunds_output
            return abunds, self.decoder(abunds), features
        abunds = abunds_output
        return abunds, self.decoder(abunds)

    def getEndmembers(self) -> torch.Tensor:
        endmembers = self.decoder.getEndmembers()
        if endmembers.shape[2] > 1:
            return torch.squeeze(endmembers).mean(dim=2).mean(dim=2)
        return torch.squeeze(endmembers)


class OfficialFlipAugmentation:
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            batch = torch.flip(batch, dims=(-1,))
        if torch.rand(()) < 0.5:
            batch = torch.flip(batch, dims=(-2,))
        return batch


def abundance_spatial_tv(abunds: torch.Tensor) -> torch.Tensor:
    horizontal = torch.abs(abunds[:, :, :, 1:] - abunds[:, :, :, :-1]).mean()
    vertical = torch.abs(abunds[:, :, 1:, :] - abunds[:, :, :-1, :]).mean()
    return horizontal + vertical


def load_compatible_model_state(model: nn.Module, checkpoint_path: Path) -> dict[str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = payload.get("model", payload)
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target_state.update(compatible)
    model.load_state_dict(target_state)
    return {
        "loaded": len(compatible),
        "source": len(source_state),
        "target": len(target_state),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Aware pretrained encoder for Urban4 unmixing.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--prototype-bank-downstream", type=Path, default=None)
    parser.add_argument("--proto-bank-target", choices=("eo1", "gf5"), default="eo1")
    parser.add_argument("--hypersigma-root", type=Path, default=DEFAULT_HYPERSIGMA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "finetune_unmixing")
    parser.add_argument("--dataset", type=str, default="Urban4")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-patches", type=int, default=400)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--w-ab", type=float, default=0.35)
    parser.add_argument("--w-tv", type=float, default=0.1)
    parser.add_argument("--w-abundance-tv", type=float, default=0.0)
    parser.add_argument("--w-mse", type=float, default=0.0, help="Optional pixel-wise reconstruction MSE weight for unmixing fine-tuning.")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unmix-tokens", type=int, default=64)
    parser.add_argument("--pixel-branch-channels", type=int, default=0)
    parser.add_argument("--proto-kl-weight", type=float, default=0.05)
    parser.add_argument("--proto-balance-weight", type=float, default=0.005)
    parser.add_argument("--proto-warmup-epochs", type=int, default=20)
    parser.add_argument("--proto-center-window", type=int, default=3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()
    if args.freeze_encoder:
        print("[Aware] --freeze-encoder is disabled for fair full-parameter fine-tuning; using freeze_encoder=False.")
        args.freeze_encoder = False
    return args


def ramp_weight(base_weight: float, epoch: int, warmup_epochs: int) -> float:
    if base_weight <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return float(base_weight)
    progress = min(max(epoch, 0) / float(warmup_epochs), 1.0)
    return float(base_weight) * progress


def build_model(
    args: argparse.Namespace,
    *,
    channels: int,
    num_em: int,
    init_em: np.ndarray,
    scene_config,
    device: torch.device,
) -> SigmaStyleAwareUnmix:
    metadata = build_downstream_metadata(
        num_bands=channels,
        sensor=scene_config.sensor,
        level=scene_config.level,
        gsd=scene_config.gsd_m,
        wavelengths_nm=scene_config.wavelengths,
        fwhm_nm=scene_config.fwhm,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )
    model = SigmaStyleAwareUnmix(
        extractor,
        channels=channels,
        num_em=num_em,
        kernel=args.kernel,
        scale=args.scale,
        num_tokens=args.unmix_tokens,
        pixel_branch_channels=args.pixel_branch_channels,
    ).to(device)
    with torch.no_grad():
        model.decoder.decoder.weight[:, :, args.kernel // 2, args.kernel // 2] = torch.from_numpy(init_em).float().to(device)

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = vars(args).copy()
        payload["scene_metadata"] = {
            **asdict(scene_config),
            "wavelengths": scene_config.wavelengths.tolist(),
            "fwhm": scene_config.fwhm.tolist(),
        }
        payload["downstream_metadata"] = {
            "sensor": metadata.sensor,
            "level": metadata.level,
            "gsd": metadata.gsd,
            "wavelengths_nm": metadata.wavelengths_nm.tolist(),
            "fwhm_nm": metadata.fwhm_nm.tolist(),
        }
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
    return model


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.img_size % 8 != 0:
        raise ValueError("img-size must be divisible by the pretrained Aware local patch size 8.")

    scene_config = resolve_unmixing_scene_config(args.dataset)
    data_file = args.hypersigma_root / "HyperspectralUnmixing" / "data" / "Urban_188_em4_init.mat"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing Urban_188_em4_init.mat under {data_file.parent}.")

    img_3d, endmember_gt, abundance_gt, init_em = load_unmixing_data(data_file)
    if img_3d.shape[-1] != scene_config.wavelengths.numel():
        raise ValueError(
            f"Scene metadata expects {scene_config.wavelengths.numel()} bands but Urban4 data has {img_3d.shape[-1]}."
        )

    loader = get_train_patch_seed_i(args.img_size, img_3d, args.num_patches, args.batch_size, args.seed)
    if tv_transforms is not None:
        train_transform = tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(p=0.5),
                tv_transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        train_transform = OfficialFlipAugmentation()

    channels = int(img_3d.shape[-1])
    num_em = int(init_em.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args,
        channels=channels,
        num_em=num_em,
        init_em=init_em,
        scene_config=scene_config,
        device=device,
    )
    init_model_stats = None
    if args.init_model is not None:
        init_model_stats = load_compatible_model_state(model, args.init_model)
        print(f"[Aware] warm-started unmixing model from {args.init_model}: {init_model_stats}", flush=True)
        config_path = args.output_dir / "config.json"
        try:
            payload = json.loads(config_path.read_text())
        except Exception:
            payload = vars(args).copy()
        payload["init_model"] = str(args.init_model)
        payload["init_model_stats"] = init_model_stats
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)

    downstream_bank: PrototypeBank | None = None
    if args.prototype_bank_downstream is not None:
        downstream_bank = load_prototype_bank(args.prototype_bank_downstream)

    if args.encoder_lr_scale != 1.0 and not args.freeze_encoder:
        encoder_param_ids = {id(parameter) for parameter in model.extractor.encoder.parameters()}
        encoder_params = []
        head_params = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in encoder_param_ids:
                encoder_params.append(parameter)
            else:
                head_params.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.lr},
                {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"
    best_loss = float("inf")
    best_path = args.output_dir / "best_unmixing.pt"
    log_path = args.output_dir / "unmixing_log.jsonl"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        current_proto_kl_weight = ramp_weight(args.proto_kl_weight, epoch, args.proto_warmup_epochs)
        current_proto_balance_weight = ramp_weight(args.proto_balance_weight, epoch, args.proto_warmup_epochs)
        totals = {"loss": 0.0, "sad": 0.0, "mse": 0.0, "ab": 0.0, "tv": 0.0, "abundance_tv": 0.0, "proto_kl": 0.0, "proto_balance": 0.0}
        batches = 0
        epoch_start = time.time()
        for (data,) in loader:
            data = train_transform(data).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                abunds, output, features = model(data, return_features=True)
                loss_sad = sad_loss(data, output)
                loss_mse = args.w_mse * F.mse_loss(output, data)
                loss_ab = args.w_ab * torch.sqrt(abunds.clamp_min(1.0e-8)).mean()
                endmembers = model.getEndmembers()
                loss_tv = args.w_tv * torch.abs(endmembers[:, 1:] - endmembers[:, :-1]).sum()
                loss_abundance_tv = args.w_abundance_tv * abundance_spatial_tv(abunds)
                proto_kl = output.new_zeros(())
                proto_balance = output.new_zeros(())
                total_loss = loss_sad + loss_mse + loss_ab + loss_tv + loss_abundance_tv
                if current_proto_kl_weight > 0.0 or current_proto_balance_weight > 0.0:
                    target_bank = downstream_bank if downstream_bank is not None else args.proto_bank_target
                    proto_losses = model.extractor.prototype_auxiliary_loss(
                        data,
                        features,
                        target_bank=target_bank,
                        center_window=args.proto_center_window,
                    )
                    proto_kl = proto_losses["proto_kl"]
                    proto_balance = proto_losses["proto_balance"]
                    total_loss = total_loss + current_proto_kl_weight * proto_kl + current_proto_balance_weight * proto_balance
            total_loss.backward()
            optimizer.step()

            totals["loss"] += float(total_loss.item())
            totals["sad"] += float(loss_sad.item())
            totals["mse"] += float(loss_mse.item())
            totals["ab"] += float(loss_ab.item())
            totals["tv"] += float(loss_tv.item())
            totals["abundance_tv"] += float(loss_abundance_tv.item())
            totals["proto_kl"] += float(proto_kl.item())
            totals["proto_balance"] += float(proto_balance.item())
            batches += 1
        scheduler.step()

        record = {
            "epoch": epoch,
            "loss": totals["loss"] / max(batches, 1),
            "loss_sad": totals["sad"] / max(batches, 1),
            "loss_mse": totals["mse"] / max(batches, 1),
            "loss_ab": totals["ab"] / max(batches, 1),
            "loss_tv": totals["tv"] / max(batches, 1),
            "loss_abundance_tv": totals["abundance_tv"] / max(batches, 1),
            "loss_proto_kl": totals["proto_kl"] / max(batches, 1),
            "loss_proto_balance": totals["proto_balance"] / max(batches, 1),
            "proto_kl_weight": current_proto_kl_weight,
            "proto_balance_weight": current_proto_balance_weight,
            "prototype_bank": downstream_bank.sensor if downstream_bank is not None else args.proto_bank_target,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - start,
        }
        print(json.dumps(record, ensure_ascii=True), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "train": record, "args": vars(args)}, best_path)

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    _, label_test_loader, num_h, num_w, img_size_test = whole_get_train_and_test_data(
        args.img_size,
        img_3d,
        batch_size=args.batch_size,
    )
    endmembers = model.getEndmembers().detach().cpu().numpy()
    pred = torch.zeros([num_w, num_h, num_em, img_size_test, img_size_test])
    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(label_test_loader):
            pred[batch_idx] = model.getAbundances(batch_data.to(device)).detach().cpu()
    pred = torch.permute(pred, [2, 0, 3, 1, 4])
    abundances = np.reshape(pred.numpy(), [num_em, num_h * img_size_test, num_w * img_size_test])
    abundances = np.asarray(abundances[:, : img_3d.shape[0], : img_3d.shape[1]])
    sad_ordered, endmember_sordered, abundance_sordered = match_endmembers_and_abundances(
        endmembers.T,
        endmember_gt.copy(),
        abundances,
    )
    mse = alter_mse(abundance_gt, abundance_sordered)
    result = {
        "best_epoch": int(payload["epoch"]),
        "prototype_bank": downstream_bank.sensor if downstream_bank is not None else args.proto_bank_target,
        "sad": np.asarray(sad_ordered).tolist(),
        "mse": np.asarray(mse).tolist(),
        "avg_sad": float(sad_ordered[-1]),
        "avg_mse": float(mse[-1]),
    }
    sio.savemat(
        args.output_dir / "unmixing_result.mat",
        {
            "decoder_weight": endmembers,
            "endmembers": endmember_sordered,
            "abundances": abundance_sordered,
            "SAD": sad_ordered,
            "MSE": mse,
        },
    )
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
