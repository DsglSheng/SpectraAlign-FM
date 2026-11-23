from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_adapters import AwareBackbone


DEFAULTS = {
    "aware_repo": Path("/mnt/Data/SpectralData/Aware_open"),
    "aware_checkpoint": Path(
        "/mnt/Data/SpectralData/Aware_open/runs/pretrain_ours_v3wa_kl003_full450k_20260505/"
        "branches/ours_v3wa_kl003_full450k/checkpoints/best_total.pt"
    ),
    "prototype_bank_eo1": Path(
        "/mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_20260503/prototype_bank_eo1.pt"
    ),
    "prototype_bank_gf5": Path(
        "/mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_20260503/prototype_bank_gf5.pt"
    ),
}


class RegressionDataset(Dataset):
    def __init__(
        self,
        spectra: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        *,
        spectrum_mean: np.ndarray,
        spectrum_std: np.ndarray,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.spectra = spectra[indices].astype(np.float32, copy=False)
        self.targets = targets[indices].astype(np.float32, copy=False)
        self.spectrum_mean = spectrum_mean.astype(np.float32, copy=False)
        self.spectrum_std = spectrum_std.astype(np.float32, copy=False)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spectrum = (self.spectra[index] - self.spectrum_mean) / self.spectrum_std
        target = (float(self.targets[index]) - self.target_mean) / self.target_std
        return {
            "spectrum": torch.from_numpy(spectrum),
            "target": torch.tensor(target, dtype=torch.float32),
            "raw_target": torch.tensor(self.targets[index], dtype=torch.float32),
        }


class ConvBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SpectralResidualBranch(nn.Module):
    def __init__(self, bands: int, *, width: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        if width % 8:
            raise ValueError("Residual width must be divisible by 8 for GroupNorm.")
        self.stem = nn.Sequential(
            nn.Conv1d(3, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.scales = nn.ModuleList(ConvBlock(width, kernel) for kernel in (3, 7, 15))
        self.mix = nn.Sequential(
            nn.Conv1d(width * 3, feature_dim, 1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.raw_projection = nn.Sequential(
            nn.LayerNorm(bands),
            nn.Linear(bands, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = feature_dim * 3 + 4

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        first = F.pad(spectrum[:, 1:] - spectrum[:, :-1], (1, 0))
        second = F.pad(first[:, 1:] - first[:, :-1], (1, 0))
        features = self.stem(torch.stack((spectrum, first, second), dim=1))
        mixed = self.mix(torch.cat([branch(features) for branch in self.scales], dim=1))
        pooled = torch.cat((mixed.mean(dim=-1), mixed.amax(dim=-1)), dim=1)
        raw = self.raw_projection(spectrum)
        stats = torch.stack(
            (
                spectrum.mean(dim=1),
                spectrum.std(dim=1),
                spectrum.amin(dim=1),
                spectrum.amax(dim=1),
            ),
            dim=1,
        )
        return torch.cat((pooled, raw, stats), dim=1)


class AwareSpectralFusionRegressor(nn.Module):
    def __init__(
        self,
        backbone: AwareBackbone,
        *,
        bands: int,
        residual_width: int,
        residual_dim: int,
        fusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.spectral_branch = SpectralResidualBranch(
            bands,
            width=residual_width,
            feature_dim=residual_dim,
            dropout=dropout,
        )
        self.aware_projection = nn.Sequential(
            nn.LayerNorm(backbone.feature_dim),
            nn.Linear(backbone.feature_dim, fusion_dim),
            nn.GELU(),
        )
        self.spectral_projection = nn.Sequential(
            nn.LayerNorm(self.spectral_branch.output_dim),
            nn.Linear(self.spectral_branch.output_dim, fusion_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        aware = self.aware_projection(self.backbone(spectrum))
        spectral = self.spectral_projection(self.spectral_branch(spectrum))
        gate = self.gate(torch.cat((aware, spectral), dim=1))
        fused = aware + gate * spectral
        return self.head(fused).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aware-HSI spectral-residual fusion regression.")
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--encoder-lr", type=float, default=5.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--residual-width", type=int, default=48)
    parser.add_argument("--residual-dim", type=int, default=128)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-noise-std", type=float, default=0.005)
    parser.add_argument("--band-dropout", type=float, default=0.02)
    parser.add_argument("--loss", choices=("mse", "smoothl1"), default="mse")
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=value)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def fwhm_from_wavelengths(wavelengths_nm: np.ndarray) -> np.ndarray:
    if wavelengths_nm.size == 1:
        return np.ones_like(wavelengths_nm)
    differences = np.abs(np.diff(wavelengths_nm))
    fwhm = np.empty_like(wavelengths_nm)
    fwhm[0] = differences[0]
    fwhm[-1] = differences[-1]
    if wavelengths_nm.size > 2:
        fwhm[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return np.maximum(fwhm, 1.0e-3)


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    huber_beta: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    return F.smooth_l1_loss(prediction, target, beta=huber_beta)


def augment_spectrum(spectrum: torch.Tensor, *, noise_std: float, band_dropout: float) -> torch.Tensor:
    if noise_std > 0.0:
        spectrum = spectrum + torch.randn_like(spectrum) * noise_std
    if band_dropout > 0.0:
        keep = torch.rand_like(spectrum).ge(band_dropout)
        spectrum = spectrum * keep.to(dtype=spectrum.dtype)
    return spectrum


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    target_mean: float,
    target_std: float,
    loss_name: str,
    huber_beta: float,
    include_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, precision):
            prediction = model(spectrum)
            loss = regression_loss(prediction, target, loss_name=loss_name, huber_beta=huber_beta)
        losses.append(float(loss.item()))
        predictions.append(prediction.float().cpu().numpy() * target_std + target_mean)
        targets.append(batch["raw_target"].numpy())
    y_pred = np.concatenate(predictions).astype(np.float32)
    y_true = np.concatenate(targets).astype(np.float32)
    result: dict[str, Any] = {
        "loss": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": r2_score(y_true, y_pred),
    }
    if include_predictions:
        result["predictions"] = y_pred
        result["targets"] = y_true
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    loss_name: str,
    huber_beta: float,
    noise_std: float,
    band_dropout: float,
    grad_clip: float,
) -> float:
    model.train()
    backbone = getattr(model, "backbone", None)
    if backbone is not None and not any(parameter.requires_grad for parameter in backbone.parameters()):
        backbone.eval()
    losses: list[float] = []
    for batch in loader:
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        spectrum = augment_spectrum(spectrum, noise_std=noise_std, band_dropout=band_dropout)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            prediction = model(spectrum)
            loss = regression_loss(prediction, target, loss_name=loss_name, huber_beta=huber_beta)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics_path = args.output_dir / "test_metrics.json"
    if final_metrics_path.exists():
        print(f"[skip] completed result exists: {final_metrics_path}", flush=True)
        return 0

    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with np.load(args.dataset_npz, allow_pickle=True) as data:
        spectra = np.asarray(data["spectra"], dtype=np.float32)
        targets = np.asarray(data["targets"], dtype=np.float32)
        wavelengths_nm = np.asarray(data["wavelengths_nm"], dtype=np.float32)
        train_idx = np.asarray(data["train_idx"], dtype=np.int64)
        val_idx = np.asarray(data["val_idx"], dtype=np.int64)
        test_idx = np.asarray(data["test_idx"], dtype=np.int64)
        dataset_name = str(np.asarray(data["dataset"]).item())
        target_name = str(np.asarray(data["target_name"]).item())
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))

    train_spectra = spectra[train_idx]
    spectrum_mean = train_spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    spectrum_std = np.maximum(
        train_spectra.std(axis=0, dtype=np.float64).astype(np.float32),
        1.0e-6,
    )
    target_mean = float(targets[train_idx].mean())
    target_std = float(max(targets[train_idx].std(), 1.0e-6))
    datasets = {
        split: RegressionDataset(
            spectra,
            targets,
            indices,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        for split, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx))
    }
    common_loader = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    loaders = {
        split: DataLoader(
            dataset,
            shuffle=split == "train",
            drop_last=False,
            **common_loader,
        )
        for split, dataset in datasets.items()
    }

    backbone = AwareBackbone(
        aware_repo=args.aware_repo,
        checkpoint=args.aware_checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        wavelengths_nm=torch.from_numpy(wavelengths_nm),
        sensor=str(metadata.get("sensor", "field")),
        level=str(metadata.get("level", "field")),
        gsd=float(metadata.get("gsd", 1.0)),
        device=device,
    )
    model = AwareSpectralFusionRegressor(
        backbone,
        bands=int(spectra.shape[1]),
        residual_width=int(args.residual_width),
        residual_dim=int(args.residual_dim),
        fusion_dim=int(args.fusion_dim),
        dropout=float(args.dropout),
    ).to(device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    task_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": task_parameters, "lr": float(args.head_lr), "name": "spectral_fusion"}
    ]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": float(args.encoder_lr), "name": "aware_backbone"}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))

    def learning_rate_multiplier(epoch_index: int) -> float:
        warmup = max(0, int(args.warmup_epochs))
        if warmup and epoch_index < warmup:
            return float(epoch_index + 1) / float(warmup)
        progress = (epoch_index - warmup) / max(1, int(args.epochs) - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best_val_rmse.pt"
    best_rmse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    if args.auto_resume and latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_rmse = float(payload["best_val_rmse"])
        best_epoch = int(payload["best_epoch"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        print(f"[resume] epoch={start_epoch} best_val_rmse={best_rmse:.6f}", flush=True)

    config = {
        "args": vars(args),
        "dataset": dataset_name,
        "target": target_name,
        "metadata": metadata,
        "samples": {name: len(dataset) for name, dataset in datasets.items()},
        "bands": int(spectra.shape[1]),
        "wavelength_range_nm": [float(wavelengths_nm.min()), float(wavelengths_nm.max())],
        "target_train_mean": target_mean,
        "target_train_std": target_std,
        "architecture": {
            "name": "aware_spectral_residual_fusion_v1",
            "aware_adapter": backbone.load_info,
            "spectral_inputs": ["normalized reflectance", "first difference", "second difference"],
            "spectral_kernels": [3, 7, 15],
            "fusion": "validation-trained elementwise sigmoid gate with aware residual",
        },
        "parameters": {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        },
        "device": str(device),
    }
    save_json(args.output_dir / "config.json", config)

    log_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer=optimizer,
            device=device,
            precision=args.precision,
            loss_name=args.loss,
            huber_beta=float(args.huber_beta),
            noise_std=float(args.input_noise_std),
            band_dropout=float(args.band_dropout),
            grad_clip=float(args.grad_clip),
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            device=device,
            precision=args.precision,
            target_mean=target_mean,
            target_std=target_std,
            loss_name=args.loss,
            huber_beta=float(args.huber_beta),
        )
        scheduler.step()
        improved = float(val_metrics["rmse"]) < best_rmse
        if improved:
            best_rmse = float(val_metrics["rmse"])
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_rmse": best_rmse,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "config": config,
        }
        torch.save(checkpoint, latest_path)
        if improved:
            torch.save(checkpoint, best_path)
        row = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val": val_metrics,
            "best_val_rmse": best_rmse,
            "best_epoch": best_epoch,
            "lr": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
            "seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=True) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.6f} "
            f"best={best_rmse:.6f}@{best_epoch} stale={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= int(args.patience):
            print(f"[early-stop] no validation RMSE improvement for {stale_epochs} epochs", flush=True)
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    test_metrics = evaluate(
        model,
        loaders["test"],
        device=device,
        precision=args.precision,
        target_mean=target_mean,
        target_std=target_std,
        loss_name=args.loss,
        huber_beta=float(args.huber_beta),
        include_predictions=True,
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        targets=test_metrics.pop("targets"),
        predictions=test_metrics.pop("predictions"),
        test_idx=test_idx,
    )
    result = {
        "model": "aware_spectral_residual_fusion_v1",
        "dataset": dataset_name,
        "target": target_name,
        "best_epoch": int(best_payload["epoch"]),
        "best_val_rmse": float(best_payload["best_val_rmse"]),
        "test": test_metrics,
        "protocol": (
            "shared split and 256-band input; train-only normalization; validation-RMSE selection; "
            "Aware-HSI plus a disclosed raw-spectrum residual branch"
        ),
    }
    save_json(final_metrics_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
