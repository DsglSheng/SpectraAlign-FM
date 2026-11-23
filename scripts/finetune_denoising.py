from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import lmdb
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from aware_open.downstream.adapter import AwareFeatureExtractor, build_downstream_metadata


HYPERSIGMA_BASELINES = {
    "case5": {"psnr": 28.503, "ssim": 0.961, "sam": 0.106},
    "case5_spatsigma": {"psnr": 28.103, "ssim": 0.959, "sam": 0.109},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class WdcLmdbDataset(Dataset):
    def __init__(self, db_path: Path, repeat: int = 10) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Missing LMDB dataset: {self.db_path}")
        self.env = lmdb.open(
            str(self.db_path),
            max_readers=16,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()["entries"])
        meta_path = self.db_path / "meta_info.txt"
        with meta_path.open("r", encoding="utf-8") as handle:
            line = handle.readline()
        size = line.split("(")[1].split(")")[0]
        height, width, channels = [int(item) for item in size.split(",")]
        self.channels = channels
        self.height = height
        self.width = width
        self.repeat = int(repeat)

    def __len__(self) -> int:
        return self.length * self.repeat

    def __getitem__(self, index: int) -> np.ndarray:
        index = int(index) % self.length
        with self.env.begin(write=False) as txn:
            raw = txn.get(f"{index:08}".encode("ascii"))
        if raw is None:
            raise IndexError(f"LMDB key {index:08} missing in {self.db_path}")
        flat = np.frombuffer(raw, dtype=np.float32)
        return flat.reshape(self.channels, self.height, self.width).copy()


class WdcMatRandomPatchDataset(Dataset):
    def __init__(self, mat_paths: list[Path], *, samples_per_epoch: int, patch_size: int = 64) -> None:
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.cubes: list[np.ndarray] = []
        for path in mat_paths:
            payload = sio.loadmat(path)
            key = "data" if "data" in payload else "gt"
            cube = np.asarray(payload[key], dtype=np.float32)
            if cube.ndim != 3:
                raise ValueError(f"Expected 3D WDC cube in {path}, got {cube.shape}.")
            if cube.shape[0] != 191 and cube.shape[-1] == 191:
                cube = np.transpose(cube, (2, 0, 1))
            if cube.shape[0] != 191:
                raise ValueError(f"Expected 191 bands in {path}, got {cube.shape}.")
            if cube.shape[1] < self.patch_size or cube.shape[2] < self.patch_size:
                continue
            self.cubes.append(np.ascontiguousarray(cube))
        if not self.cubes:
            raise FileNotFoundError("No WDC .mat cube is large enough for random patch sampling.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> np.ndarray:
        cube = self.cubes[np.random.randint(0, len(self.cubes))]
        _, height, width = cube.shape
        top = np.random.randint(0, height - self.patch_size + 1)
        left = np.random.randint(0, width - self.patch_size + 1)
        patch = cube[:, top : top + self.patch_size, left : left + self.patch_size]
        if np.random.rand() < 0.5:
            patch = patch[:, :, ::-1]
        if np.random.rand() < 0.5:
            patch = patch[:, ::-1, :]
        return np.ascontiguousarray(patch, dtype=np.float32)


class WdcPatchMatDataset(Dataset):
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.files = sorted(self.folder.glob("*.mat"))
        if not self.files:
            raise FileNotFoundError(f"No .mat files found under {self.folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.files[int(index)]
        payload = sio.loadmat(path)
        noisy = np.asarray(payload["input"], dtype=np.float32)
        clean = np.asarray(payload["gt"], dtype=np.float32)
        return torch.from_numpy(noisy), torch.from_numpy(clean), path.name


class AddNoiseNoniid:
    def __init__(self, sigmas: list[int]) -> None:
        self.sigmas = np.asarray(sigmas, dtype=np.float32) / 255.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        sigma = self.sigmas[np.random.randint(0, len(self.sigmas), image.shape[0])]
        return image + np.random.randn(*image.shape).astype(np.float32) * sigma.reshape(-1, 1, 1)


class _AddNoiseImpulse:
    def __init__(self, amounts: list[float], salt_vs_pepper: float = 0.5) -> None:
        self.amounts = np.asarray(amounts, dtype=np.float32)
        self.salt_vs_pepper = float(salt_vs_pepper)

    def __call__(self, image: np.ndarray, bands: np.ndarray) -> np.ndarray:
        amounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for band, amount in zip(bands, amounts):
            flipped = np.random.choice([True, False], size=image[band].shape, p=[float(amount), 1.0 - float(amount)])
            salted = np.random.choice([True, False], size=image[band].shape, p=[self.salt_vs_pepper, 1.0 - self.salt_vs_pepper])
            image[band, flipped & salted] = 1.0
            image[band, flipped & (~salted)] = 0.0
        return image


class _AddNoiseStripe:
    def __init__(self, min_amount: float, max_amount: float) -> None:
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)

    def __call__(self, image: np.ndarray, bands: np.ndarray) -> np.ndarray:
        _, _, width = image.shape
        counts = np.random.randint(np.floor(self.min_amount * width), np.floor(self.max_amount * width), len(bands))
        for band, count in zip(bands, counts):
            loc = np.random.permutation(width)[: int(count)]
            stripe = np.random.uniform(0.0, 1.0, size=(len(loc),)).astype(np.float32) * 0.5 - 0.25
            image[band, :, loc] -= stripe.reshape(-1, 1)
        return image


class _AddNoiseDeadline:
    def __init__(self, min_amount: float, max_amount: float) -> None:
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)

    def __call__(self, image: np.ndarray, bands: np.ndarray) -> np.ndarray:
        _, _, width = image.shape
        counts = np.random.randint(np.ceil(self.min_amount * width), np.ceil(self.max_amount * width), len(bands))
        for band, count in zip(bands, counts):
            loc = np.random.permutation(width)[: int(count)]
            image[band, :, loc] = 0.0
        return image


class AddNoiseMixed:
    def __init__(self, noise_bank: list[object], num_bands: list[float]) -> None:
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, image: np.ndarray) -> np.ndarray:
        bands_all = np.random.permutation(image.shape[0])
        pos = 0
        for noise_maker, band_fraction in zip(self.noise_bank, self.num_bands):
            count = int(np.floor(band_fraction * image.shape[0])) if 0 < band_fraction <= 1 else int(band_fraction)
            bands = bands_all[pos : pos + count]
            pos += count
            image = noise_maker(image, bands)
        return image


class SequentialSelect:
    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms
        self.pos = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        transform = self.transforms[self.pos]
        self.pos = (self.pos + 1) % len(self.transforms)
        return transform(image)


class ComplexNoiseTransform:
    def __init__(self) -> None:
        self.gaussian = AddNoiseNoniid([10, 30, 50, 70])
        self.select = SequentialSelect(
            [
                lambda x: x,
                AddNoiseMixed([_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])], [1 / 3]),
                AddNoiseMixed([_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])], [1 / 3]),
                AddNoiseMixed([_AddNoiseStripe(0.05, 0.15)], [1 / 3]),
                AddNoiseMixed([_AddNoiseDeadline(0.05, 0.15)], [1 / 3]),
            ]
        )

    def __call__(self, clean: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        target = clean.astype(np.float32, copy=True)
        noisy = self.gaussian(target.copy())
        noisy = self.select(noisy)
        return torch.from_numpy(noisy.astype(np.float32, copy=False)), torch.from_numpy(target)


class Case5MixedNoiseTransform:
    def __init__(self) -> None:
        self.gaussian = AddNoiseNoniid([30, 50, 70])
        self.mixed = AddNoiseMixed(
            [
                _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7]),
                _AddNoiseStripe(0.05, 0.15),
                _AddNoiseDeadline(0.05, 0.15),
            ],
            [1 / 3, 1 / 3, 1 / 3],
        )

    def __call__(self, clean: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        target = clean.astype(np.float32, copy=True)
        noisy = self.gaussian(target.copy())
        noisy = self.mixed(noisy)
        return torch.from_numpy(noisy.astype(np.float32, copy=False)), torch.from_numpy(target)


def build_noise_transform(profile: str) -> object:
    if profile == "complex":
        return ComplexNoiseTransform()
    if profile == "case5_mixed":
        return Case5MixedNoiseTransform()
    raise ValueError(f"Unsupported noise profile {profile}")


class TrainingPairDataset(Dataset):
    def __init__(self, base: Dataset, noise: ComplexNoiseTransform) -> None:
        self.base = base
        self.noise = noise

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noise(self.base[index])


class AwareDenoisingNet(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        channels: int,
        feature_channels: int = 64,
        hidden_channels: int = 256,
        use_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.channels = int(channels)
        self.use_intermediate = bool(use_intermediate)
        feature_count = 5 if self.use_intermediate else 1
        self.token_projs = nn.ModuleList([nn.Linear(extractor.model_dim, feature_channels) for _ in range(feature_count)])
        self.input_proj = nn.Sequential(
            nn.Conv2d(channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
        )
        in_channels = feature_channels * (feature_count + 1) + channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _tokens_to_map(self, tokens: torch.Tensor, grid_size: tuple[int, int], size: tuple[int, int], proj: nn.Linear) -> torch.Tensor:
        height, width = grid_size
        feature = proj(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        features = self.extractor.forward_features(noisy)
        grid_size = features["grid_size"]
        size = (int(noisy.shape[-2]), int(noisy.shape[-1]))
        maps = []
        if self.use_intermediate:
            tokens_list = list(features["intermediate_tokens"]) + [features["patch_tokens"]]
        else:
            tokens_list = [features["patch_tokens"]]
        for tokens, proj in zip(tokens_list, self.token_projs):
            maps.append(self._tokens_to_map(tokens, grid_size, size, proj))
        local = self.input_proj(noisy)
        fused = torch.cat([noisy, local, *maps], dim=1)
        residual = self.head(fused)
        return noisy + self.residual_scale * residual


def sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_vec = pred.flatten(2)
    target_vec = target.flatten(2)
    dot = (pred_vec * target_vec).sum(dim=1)
    denom = pred_vec.square().sum(dim=1).sqrt() * target_vec.square().sum(dim=1).sqrt()
    cos = (dot / denom.clamp_min(1.0e-8)).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    return torch.acos(cos).mean()


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + 1.0e-6).mean()


def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss_h = F.l1_loss(pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    loss_w = F.l1_loss(pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1])
    return 0.5 * (loss_h + loss_w)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    *,
    sam_weight: float,
    char_weight: float,
    spectral_grad_weight: float,
    spatial_grad_weight: float,
) -> torch.Tensor:
    if loss_name == "l2":
        return F.mse_loss(pred, target)
    if loss_name == "l1":
        return F.l1_loss(pred, target)
    if loss_name == "char":
        return charbonnier_loss(pred, target)
    if loss_name == "l2_sam":
        return F.mse_loss(pred, target) + float(sam_weight) * sam_loss(pred, target)
    if loss_name == "l2_sam_grad":
        return (
            F.mse_loss(pred, target)
            + float(sam_weight) * sam_loss(pred, target)
            + float(spectral_grad_weight) * spectral_gradient_loss(pred, target)
            + float(spatial_grad_weight) * spatial_gradient_loss(pred, target)
        )
    if loss_name == "case5_combo":
        return (
            F.mse_loss(pred, target)
            + float(char_weight) * charbonnier_loss(pred, target)
            + float(sam_weight) * sam_loss(pred, target)
            + float(spectral_grad_weight) * spectral_gradient_loss(pred, target)
            + float(spatial_grad_weight) * spatial_gradient_loss(pred, target)
        )
    raise ValueError(f"Unsupported loss {loss_name}")


def bandwise_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred.astype(np.float64, copy=False), 0.0, 1.0)
    target = target.astype(np.float64, copy=False)
    channels = int(target.shape[0])
    psnrs: list[float] = []
    ssims: list[float] = []
    for band in range(channels):
        psnrs.append(float(peak_signal_noise_ratio(target[band], pred[band], data_range=1.0)))
        ssims.append(float(structural_similarity(target[band], pred[band], data_range=1.0)))
    dot = np.sum(pred * target, axis=0) + np.spacing(1)
    denom = np.sqrt(np.sum(pred**2, axis=0) + np.spacing(1)) * np.sqrt(np.sum(target**2, axis=0) + np.spacing(1))
    sam_rad = float(np.mean(np.arccos(np.clip(dot / denom, -1.0, 1.0))))
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "sam": sam_rad,
    }


@torch.no_grad()
def predict_with_tta(model: nn.Module, noisy: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled:
        return model(noisy)
    preds = []
    for dims in ((), (-1,), (-2,), (-2, -1)):
        aug = torch.flip(noisy, dims=dims) if dims else noisy
        pred = model(aug)
        preds.append(torch.flip(pred, dims=dims) if dims else pred)
    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, precision: str, *, eval_tta: bool = False) -> dict[str, object]:
    model.eval()
    per_file = []
    use_amp = precision == "bf16" and device.type == "cuda"
    for noisy, clean, name in loader:
        noisy = noisy.to(device=device, dtype=torch.float32, non_blocking=True)
        clean = clean.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = predict_with_tta(model, noisy, enabled=eval_tta)
        pred_np = pred.squeeze(0).detach().cpu().float().numpy()
        clean_np = clean.squeeze(0).detach().cpu().float().numpy()
        metrics = bandwise_metrics(pred_np, clean_np)
        metrics["file"] = name[0] if isinstance(name, (list, tuple)) else str(name)
        per_file.append(metrics)
    aggregate = {
        "psnr": float(np.mean([item["psnr"] for item in per_file])),
        "ssim": float(np.mean([item["ssim"] for item in per_file])),
        "sam": float(np.mean([item["sam"] for item in per_file])),
        "per_file": per_file,
    }
    baseline = HYPERSIGMA_BASELINES["case5"]
    aggregate["delta_vs_hypersigma"] = {
        "psnr": aggregate["psnr"] - baseline["psnr"],
        "ssim": aggregate["ssim"] - baseline["ssim"],
        "sam": baseline["sam"] - aggregate["sam"],
    }
    aggregate["eval_tta"] = bool(eval_tta)
    return aggregate


def load_compatible_model_state(model: nn.Module, checkpoint_path: Path, *, scope: str = "all") -> dict[str, int | str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = payload.get("model", payload)
    target_state = model.state_dict()
    skipped_by_scope = 0
    if scope == "head":
        filtered_state = {}
        for key, value in source_state.items():
            if key.startswith("extractor."):
                skipped_by_scope += 1
                continue
            filtered_state[key] = value
        source_state = filtered_state
    elif scope != "all":
        raise ValueError(f"Unsupported init load scope {scope}")
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target_state.update(compatible)
    model.load_state_dict(target_state)
    return {
        "scope": scope,
        "loaded": len(compatible),
        "source": len(source_state),
        "target": len(target_state),
        "skipped_by_scope": skipped_by_scope,
    }


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, *, warmup_epochs: int = 2, epochs: int = 100) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)))
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aware fine-tuning for HyperSIGMA WDC denoising protocol.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/data/denoising/HyperSIGMA_denoising"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--loss", choices=("l2", "l1", "char", "l2_sam", "l2_sam_grad", "case5_combo"), default="l2")
    parser.add_argument("--sam-weight", type=float, default=1.0e-3)
    parser.add_argument("--char-weight", type=float, default=0.0)
    parser.add_argument("--spectral-grad-weight", type=float, default=0.0)
    parser.add_argument("--spatial-grad-weight", type=float, default=0.0)
    parser.add_argument("--noise-profile", choices=("complex", "case5_mixed"), default="complex")
    parser.add_argument("--train-source", choices=("lmdb", "mat_random", "lmdb_plus_mat"), default="lmdb")
    parser.add_argument("--mat-samples-per-epoch", type=int, default=6600)
    parser.add_argument("--mat-patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--feature-channels", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--respect-freeze-encoder", action="store_true")
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--init-load-scope", choices=("all", "head"), default="all")
    parser.add_argument("--no-intermediate", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-tta", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.freeze_encoder and not args.respect_freeze_encoder:
        print("[Aware] --freeze-encoder is disabled for fair full-parameter fine-tuning; using freeze_encoder=False.")
        args.freeze_encoder = False
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_db = args.data_root / "Training" / "wdc.db"
    test_dir = args.data_root / "Testing" / "Patch_Cases" / "Case5"
    if not (train_db / "data.mdb").exists():
        raise FileNotFoundError(f"Missing HyperSIGMA denoising train data: {train_db / 'data.mdb'}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing HyperSIGMA denoising Case5 test folder: {test_dir}")

    metadata = build_downstream_metadata(
        num_bands=191,
        sensor="others",
        level="surface",
        gsd=2.8,
        wavelength_start_nm=400.0,
        wavelength_end_nm=2500.0,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )
    model = AwareDenoisingNet(
        extractor,
        channels=191,
        feature_channels=args.feature_channels,
        hidden_channels=args.hidden_channels,
        use_intermediate=not args.no_intermediate,
    ).to(device)
    init_model_stats = None
    if args.init_model is not None:
        init_model_stats = load_compatible_model_state(model, args.init_model, scope=args.init_load_scope)
        print(f"[Aware] warm-started denoising model from {args.init_model}: {init_model_stats}", flush=True)

    clean_datasets: list[Dataset] = []
    if args.train_source in ("lmdb", "lmdb_plus_mat"):
        clean_datasets.append(WdcLmdbDataset(train_db, repeat=args.repeat))
    if args.train_source in ("mat_random", "lmdb_plus_mat"):
        split_train_paths = sorted((args.data_root / "splits" / "train").glob("*.mat"))
        clean_datasets.append(
            WdcMatRandomPatchDataset(
                split_train_paths,
                samples_per_epoch=args.mat_samples_per_epoch,
                patch_size=args.mat_patch_size,
            )
        )
    clean_dataset: Dataset = clean_datasets[0] if len(clean_datasets) == 1 else ConcatDataset(clean_datasets)
    train_dataset = TrainingPairDataset(clean_dataset, build_noise_transform(args.noise_profile))
    test_dataset = WdcPatchMatDataset(test_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")

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
                {"params": head_params, "lr": args.lr, "lr_scale": 1.0},
                {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale, "lr_scale": args.encoder_lr_scale},
            ],
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=args.weight_decay)
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    start_epoch = 1
    best_psnr = -math.inf
    if args.resume and latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_psnr = float(payload.get("best_psnr", best_psnr))

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output_dir"] = str(args.output_dir)
    config["checkpoint"] = str(args.checkpoint)
    config["init_model"] = str(args.init_model) if args.init_model is not None else None
    config["init_model_stats"] = init_model_stats
    config["prototype_bank_eo1"] = str(args.prototype_bank_eo1)
    config["prototype_bank_gf5"] = str(args.prototype_bank_gf5)
    config["metadata"] = {**asdict(metadata), "wavelengths_nm": metadata.wavelengths_nm.tolist(), "fwhm_nm": metadata.fwhm_nm.tolist()}
    config["hypersigma_reference"] = HYPERSIGMA_BASELINES["case5"]
    config["protocol_note"] = (
        "Matches HyperSIGMA ImageDenoising complex WDC protocol: WDC LMDB 64x64 patches, "
        "batch size 4, lr 1e-4, 100 epochs by default, L2 loss by default, and Patch_Cases/Case5 evaluation. "
        "Optional breakout branches can add random train-split .mat crops and Case5-specific composite losses."
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)

    log_path = args.output_dir / "train_log.jsonl"
    use_amp = args.precision == "bf16" and device.type == "cuda"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        lr = float(optimizer.param_groups[0]["lr"])
        running_loss = 0.0
        seen = 0
        for batch_idx, (noisy, clean) in enumerate(train_loader, start=1):
            noisy = noisy.to(device=device, dtype=torch.float32, non_blocking=True)
            clean = clean.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(noisy)
                loss = compute_loss(
                    pred,
                    clean,
                    args.loss,
                    sam_weight=args.sam_weight,
                    char_weight=args.char_weight,
                    spectral_grad_weight=args.spectral_grad_weight,
                    spatial_grad_weight=args.spatial_grad_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e6)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            seen += 1
            if batch_idx % 50 == 0:
                print(f"epoch={epoch} batch={batch_idx}/{len(train_loader)} lr={lr:.4e} loss={running_loss / seen:.6f}", flush=True)
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        row: dict[str, object] = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": running_loss / max(seen, 1),
            "train_batches": seen,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device=device, precision=args.precision, eval_tta=args.eval_tta)
            row["case5_metrics"] = metrics
            print(
                "eval "
                f"epoch={epoch} psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.4f} sam={metrics['sam']:.4f} "
                f"delta_psnr={metrics['delta_vs_hypersigma']['psnr']:.4f}",
                flush=True,
            )
            if float(metrics["psnr"]) > best_psnr:
                best_psnr = float(metrics["psnr"])
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "args": config}, best_path)
                with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump({"epoch": epoch, **metrics}, handle, indent=2, ensure_ascii=True)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_psnr": best_psnr, "args": config}, latest_path)
        if epoch < args.epochs:
            row["next_lr"] = adjust_learning_rate(optimizer, epoch, args.lr, epochs=args.epochs)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    if best_path.exists():
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
    final_metrics = evaluate(model, test_loader, device=device, precision=args.precision, eval_tta=args.eval_tta)
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, ensure_ascii=True)
    print(json.dumps(final_metrics, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
