from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from scipy import io as sio

try:
    from torchvision import transforms as tv_transforms
except ImportError:
    tv_transforms = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPERSIGMA_ROOT = REPO_ROOT.parent / "HyperSIGMA-main"
torch.set_num_threads(2)


# The dataset sampling, patch tiling, and evaluation utilities below are copied
# from the official HyperSIGMA `HyperspectralUnmixing/func.py`, with only path
# handling and typing cleaned up.


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


def write_sad_mse_summary(
    output_path: Path,
    *,
    dataset: str,
    seed: int,
    w_tv: float,
    w_ab: float,
    kernel: int,
    sad_ordered: np.ndarray,
    mse: np.ndarray,
) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(f"----------{dataset}----------\n")
        handle.write(f"seed: {seed}\n")
        handle.write(f"w_tv: {w_tv}    w_ab: {w_ab}    kernel: {kernel}\n")
        handle.write("1. endmember-SAD:\n")
        for value in sad_ordered:
            handle.write(f"{float(value):.6f}\n")
        handle.write("2. abundance-MSE:\n")
        for value in mse:
            handle.write(f"{float(value):.6f}\n")


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


class OfficialFlipAugmentation:
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            batch = torch.flip(batch, dims=(-1,))
        if torch.rand(()) < 0.5:
            batch = torch.flip(batch, dims=(-2,))
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official HyperSIGMA Urban unmixing reference runner.")
    parser.add_argument("--hypersigma-root", type=Path, default=DEFAULT_HYPERSIGMA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="Urban4")
    parser.add_argument("--mode", choices=("hypersigma", "spatsigma"), default="hypersigma")
    parser.add_argument("--spatial-init-source", choices=("sigma", "aware"), default="sigma")
    parser.add_argument("--spectral-init-source", choices=("sigma", "random"), default="sigma")
    parser.add_argument("--aware-checkpoint", type=Path, default=None)
    parser.add_argument("--spat-checkpoint", type=Path, default=None)
    parser.add_argument("--spec-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-patches", type=int, default=400)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--seg-patches", type=int, default=2)
    parser.add_argument("--kernel", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--w-ab", type=float, default=0.35)
    parser.add_argument("--w-tv", type=float, default=0.1)
    parser.add_argument("--aware-sp-reg-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    return parser.parse_args()


def ensure_hypersigma_imports(hypersigma_root: Path):
    unmix_root = hypersigma_root / "HyperspectralUnmixing"
    if not unmix_root.exists():
        raise FileNotFoundError(f"Missing HyperSIGMA unmixing directory: {unmix_root}")
    if str(unmix_root) not in sys.path:
        sys.path.insert(0, str(unmix_root))
    from mmengine.optim import build_optim_wrapper
    from mmcv__custom import custom_layer_decay_optimizer_constructor  # noqa: F401
    from mmcv__custom import layer_decay_optimizer_constructor_vit  # noqa: F401
    from models.model import HyperSIGMA_Unmix, SpatSIGMA_Unmix

    return build_optim_wrapper, HyperSIGMA_Unmix, SpatSIGMA_Unmix


def build_model_args(
    *,
    channels: int,
    num_em: int,
    img_size: int,
    seg_patches: int,
    kernel: int,
    scale: float,
    num_patches: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        patch_size=img_size,
        channels=channels,
        seg_patches=seg_patches,
        embed_dim=768,
        NUM_TOKENS=64,
        num_em=num_em,
        scale=scale,
        kernel=kernel,
        num_patches=num_patches,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a top-level 'model' state dict.")
    return checkpoint["model"]


def load_aware_spatial_weights(
    model_state: dict[str, torch.Tensor],
    *,
    aware_checkpoint: Path,
    target_prefix: str,
) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(aware_checkpoint, map_location="cpu", weights_only=False)
    aware_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(aware_state, dict):
        raise ValueError(f"Aware checkpoint {aware_checkpoint} does not contain a state dict.")

    loaded: dict[str, torch.Tensor] = {}
    for key, value in aware_state.items():
        if not key.startswith("backbone.transformer."):
            continue
        stripped = key.removeprefix("backbone.transformer.")
        if stripped.startswith("patch_embed.proj."):
            continue
        if stripped.startswith("rope_embed."):
            continue
        if stripped == "storage_tokens":
            continue
        if stripped.endswith("bias_mask"):
            continue

        mapped_key = f"{target_prefix}{stripped}"
        if mapped_key not in model_state:
            continue
        if model_state[mapped_key].shape != value.shape:
            continue
        loaded[mapped_key] = value
    if not loaded:
        raise RuntimeError(f"No compatible Aware spatial weights were found in {aware_checkpoint}.")
    print(f"Loaded {len(loaded)} compatible Aware spatial parameters from {aware_checkpoint}.")
    return loaded


def load_initial_weights(
    model: torch.nn.Module,
    *,
    mode: str,
    spatial_init_source: str,
    spectral_init_source: str,
    aware_checkpoint: Path | None,
    spat_checkpoint: Path | None,
    spec_checkpoint: Path | None,
) -> dict[str, torch.Tensor]:
    model_params = model.state_dict()
    spatial_prefix = "encoder." if mode == "spatsigma" else "spat_encoder."
    aware_loaded_params: dict[str, torch.Tensor] = {}

    if spatial_init_source == "aware":
        if aware_checkpoint is None:
            raise ValueError("--aware-checkpoint is required when --spatial-init-source aware.")
        aware_loaded_params = load_aware_spatial_weights(
            model_params,
            aware_checkpoint=aware_checkpoint,
            target_prefix=spatial_prefix,
        )
        model_params.update(aware_loaded_params)
        model.load_state_dict(model_params)
    else:
        if spat_checkpoint is None:
            raise ValueError("--spat-checkpoint is required when --spatial-init-source sigma.")
        pretrained = load_checkpoint(spat_checkpoint)
        for key in list(pretrained.keys()):
            if "patch_embed.proj" in key:
                del pretrained[key]
        for key in list(pretrained.keys()):
            pretrained[f"{spatial_prefix}{key}"] = pretrained.pop(key)
        same_params = {key: value for key, value in pretrained.items() if key in model_params}
        print(f"Loaded {len(same_params)} compatible Sigma spatial parameters from {spat_checkpoint}.")
        model_params.update(same_params)
        model.load_state_dict(model_params)

    if mode == "spatsigma":
        return aware_loaded_params

    if spectral_init_source == "random":
        print("Using random initialization for the HyperSIGMA spectral encoder.")
        return aware_loaded_params

    if spec_checkpoint is None:
        raise ValueError("--spec-checkpoint is required when --spectral-init-source sigma in --mode hypersigma.")

    spec_pretrained = load_checkpoint(spec_checkpoint)
    for key in list(spec_pretrained.keys()):
        if "spec" in key or "spat" in key:
            del spec_pretrained[key]
    for key in list(spec_pretrained.keys()):
        spec_pretrained[f"spec_encoder.{key}"] = spec_pretrained.pop(key)

    same_params = {key: value for key, value in spec_pretrained.items() if key in model_params}
    print(f"Loaded {len(same_params)} compatible Sigma spectral parameters from {spec_checkpoint}.")
    model_params.update(same_params)
    model.load_state_dict(model_params)
    return aware_loaded_params


def build_sp_anchors(
    model: torch.nn.Module,
    reference_state: dict[str, torch.Tensor],
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    if not reference_state:
        return []
    named_parameters = dict(model.named_parameters())
    anchors: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    for name, reference in reference_state.items():
        parameter = named_parameters.get(name)
        if parameter is None:
            continue
        anchors.append((name, parameter, reference.detach().clone().to(device=parameter.device, dtype=parameter.dtype)))
    return anchors


def sp_regularization(anchors: list[tuple[str, torch.nn.Parameter, torch.Tensor]]) -> torch.Tensor | None:
    if not anchors:
        return None
    total: torch.Tensor | None = None
    for _, parameter, reference in anchors:
        if not parameter.requires_grad:
            continue
        term = F.mse_loss(parameter.float(), reference.float(), reduction="mean")
        total = term if total is None else total + term
    if total is None:
        return None
    return total


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if args.dataset.lower() != "urban4":
        raise ValueError("This reference runner currently supports only Urban4.")

    build_optim_wrapper, HyperSIGMA_Unmix, SpatSIGMA_Unmix = ensure_hypersigma_imports(args.hypersigma_root)
    data_file = args.hypersigma_root / "HyperspectralUnmixing" / "data" / "Urban_188_em4_init.mat"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing Urban_188_em4_init.mat under {data_file.parent}.")

    img_3d, endmember_gt, abundance_gt, init_em = load_unmixing_data(data_file)
    loader = get_train_patch_seed_i(args.img_size, img_3d, args.num_patches, args.batch_size, args.seed)
    channels = int(img_3d.shape[-1])
    num_em = int(init_em.shape[1])
    model_args = build_model_args(
        channels=channels,
        num_em=num_em,
        img_size=args.img_size,
        seg_patches=args.seg_patches,
        kernel=args.kernel,
        scale=args.scale,
        num_patches=args.num_patches,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if tv_transforms is not None:
        train_transform = tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(p=0.5),
                tv_transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        train_transform = OfficialFlipAugmentation()

    if args.mode == "spatsigma":
        model = SpatSIGMA_Unmix(model_args)
    else:
        model = HyperSIGMA_Unmix(model_args)

    aware_reference_state = load_initial_weights(
        model,
        mode=args.mode,
        spatial_init_source=args.spatial_init_source,
        spectral_init_source=args.spectral_init_source,
        aware_checkpoint=args.aware_checkpoint,
        spat_checkpoint=args.spat_checkpoint,
        spec_checkpoint=args.spec_checkpoint,
    )

    with torch.no_grad():
        model_dict = model.state_dict()
        model_dict["decoder.decoder.weight"][:, :, args.kernel // 2, args.kernel // 2] = torch.from_numpy(init_em).float()
        model.load_state_dict(model_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sp_anchors = build_sp_anchors(model, aware_reference_state)
    if args.aware_sp_reg_weight > 0 and not sp_anchors:
        raise ValueError("--aware-sp-reg-weight > 0 requires an Aware-initialized run with compatible loaded parameters.")
    optim_wrapper_cfg = dict(
        optimizer=dict(type="AdamW", lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay),
        constructor="LayerDecayOptimizerConstructor_ViT",
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    )
    optimizer_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    optimizer = optimizer_wrapper.optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"
    log_path = args.output_dir / "unmixing_log.jsonl"
    model_path = args.output_dir / "model.pkl"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        model.train()
        epoch_start = time.time()
        totals = {"loss": 0.0, "sad": 0.0, "ab": 0.0, "tv": 0.0, "sp": 0.0}
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            batch = train_transform(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                abunds, output = model(batch)
                loss_sad = sad_loss(batch, output)
                loss_ab = args.w_ab * torch.sqrt(abunds.clamp_min(1.0e-8)).mean()
                endmembers = model.getEndmembers()
                if isinstance(endmembers, np.ndarray):
                    endmembers_tensor = torch.from_numpy(endmembers).to(device=device, dtype=torch.float32)
                else:
                    endmembers_tensor = endmembers.to(device=device, dtype=torch.float32)
                loss_tv = args.w_tv * torch.abs(endmembers_tensor[:, 1:] - endmembers_tensor[:, :-1]).sum()
                loss_sp = loss_tv.new_zeros(())
                if args.aware_sp_reg_weight > 0:
                    sp_term = sp_regularization(sp_anchors)
                    if sp_term is not None:
                        loss_sp = args.aware_sp_reg_weight * sp_term
                loss = loss_sad + loss_ab + loss_tv
                loss = loss + loss_sp
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["sad"] += float(loss_sad.item())
            totals["ab"] += float(loss_ab.item())
            totals["tv"] += float(loss_tv.item())
            totals["sp"] += float(loss_sp.item())
            batches += 1
        record = {
            "epoch": epoch,
            "mode": args.mode,
            "loss": totals["loss"] / max(batches, 1),
            "loss_sad": totals["sad"] / max(batches, 1),
            "loss_ab": totals["ab"] / max(batches, 1),
            "loss_tv": totals["tv"] / max(batches, 1),
            "loss_sp": totals["sp"] / max(batches, 1),
            "aware_sp_reg_weight": args.aware_sp_reg_weight,
            "aware_sp_anchor_count": len(sp_anchors),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - start,
        }
        print(json.dumps(record, ensure_ascii=True), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.output_dir / "final.pt")

    model.eval()
    _, label_test_loader, num_h, num_w, img_size_test = whole_get_train_and_test_data(
        args.img_size,
        img_3d,
        batch_size=args.batch_size,
    )
    if isinstance(model.getEndmembers(), np.ndarray):
        endmembers = model.getEndmembers()
    else:
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
        "mode": args.mode,
        "dataset": args.dataset,
        "sad": np.asarray(sad_ordered).tolist(),
        "mse": np.asarray(mse).tolist(),
        "avg_sad": float(sad_ordered[-1]),
        "avg_mse": float(mse[-1]),
    }
    write_sad_mse_summary(
        args.output_dir / "summary.txt",
        dataset=args.dataset,
        seed=args.seed,
        w_tv=args.w_tv,
        w_ab=args.w_ab,
        kernel=args.kernel,
        sad_ordered=np.asarray(sad_ordered),
        mse=np.asarray(mse),
    )
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
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=True, default=str)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
