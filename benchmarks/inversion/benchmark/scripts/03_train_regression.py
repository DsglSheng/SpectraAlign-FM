from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_adapters import build_model


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
    "hypersigma_repo": Path("/mnt/Data/SpectralData/HyperSIGMA-main"),
    "hypersigma_checkpoint": Path(
        "/mnt/Data/SpectralData/HyperSIGMA-main/Pretrain/spec-vit-base-ultra-checkpoint-1599.pth"
    ),
    "spectralgpt_repo": Path(
        "/mnt/Data/SpectralData/SpectralGPT_Aware_Comparison_20260516/official_code/IEEE_TPAMI_SpectralGPT"
    ),
    "spectralgpt_checkpoint": Path(
        "/mnt/Data/SpectralData/SpectralGPT_Aware_Comparison_20260516/pretrained/SpectralGPT+.pth"
    ),
    "spectralearth_repo": Path(
        "/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/official_code/spectral_earth"
    ),
    "spectralearth_checkpoint": Path(
        "/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/pretrained/spec_vit_small/mae.pth"
    ),
    "specaware_repo": Path(
        "/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/official_code/SpecAware"
    ),
    "specaware_checkpoint": Path(
        "/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/pretrained/SpecAware_Base_model.pth"
    ),
    "specaware_sentence_model": Path(
        "/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/pretrained/"
        "sentence-transformers_all-MiniLM-L6-v2"
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
        normalized_target = (float(self.targets[index]) - self.target_mean) / self.target_std
        return {
            "spectrum": torch.from_numpy(spectrum),
            "target": torch.tensor(normalized_target, dtype=torch.float32),
            "raw_target": torch.tensor(self.targets[index], dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified hyperspectral foundation-model regression benchmark.")
    parser.add_argument("--model", choices=("aware", "hypersigma", "spectralgpt", "spectralearth", "specaware"), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=value)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def fwhm_from_wavelengths(wavelengths_nm: np.ndarray) -> np.ndarray:
    if wavelengths_nm.size == 1:
        return np.ones_like(wavelengths_nm)
    edges = np.empty(wavelengths_nm.size + 1, dtype=np.float32)
    edges[1:-1] = 0.5 * (wavelengths_nm[:-1] + wavelengths_nm[1:])
    edges[0] = wavelengths_nm[0] - (edges[1] - wavelengths_nm[0])
    edges[-1] = wavelengths_nm[-1] + (wavelengths_nm[-1] - edges[-2])
    return np.maximum(np.diff(edges), 1.0e-3).astype(np.float32)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    target_mean: float,
    target_std: float,
    include_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    truth: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, precision):
            prediction = model(spectrum)
            loss = F.mse_loss(prediction, target)
        pred_raw = prediction.float().cpu().numpy() * target_std + target_mean
        losses.append(float(loss.item()))
        truth.append(batch["raw_target"].numpy())
        predictions.append(pred_raw)
    y_true = np.concatenate(truth).astype(np.float64)
    y_pred = np.concatenate(predictions).astype(np.float64)
    result: dict[str, Any] = {
        "loss": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": r2_score(y_true, y_pred),
        "samples": int(y_true.size),
    }
    if include_predictions:
        result["targets"] = y_true.astype(np.float32)
        result["predictions"] = y_pred.astype(np.float32)
    return result


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            prediction = model(spectrum)
            loss = F.mse_loss(prediction, target)
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
    spectrum_std = train_spectra.std(axis=0, dtype=np.float64).astype(np.float32)
    spectrum_std = np.maximum(spectrum_std, 1.0e-6)
    target_mean = float(targets[train_idx].mean())
    target_std = float(max(targets[train_idx].std(), 1.0e-6))

    datasets = {
        "train": RegressionDataset(
            spectra,
            targets,
            train_idx,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
        ),
        "val": RegressionDataset(
            spectra,
            targets,
            val_idx,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
        ),
        "test": RegressionDataset(
            spectra,
            targets,
            test_idx,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
        ),
    }
    loader_common = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=False, **loader_common),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=False, **loader_common),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **loader_common),
    }

    paths = {name: Path(getattr(args, name)) for name in DEFAULTS}
    wavelengths_tensor = torch.from_numpy(wavelengths_nm)
    fwhm_tensor = torch.from_numpy(fwhm_from_wavelengths(wavelengths_nm))
    model, model_info = build_model(
        args.model,
        wavelengths_nm=wavelengths_tensor,
        fwhm_nm=fwhm_tensor,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        device=device,
        paths=paths,
        sensor=str(metadata.get("sensor", "field")),
        level=str(metadata.get("level", "L2")),
        gsd=float(metadata.get("gsd", 1.0)),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=max(1.0e-7, float(args.lr) * 0.01),
    )
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
        "model_info": model_info,
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
            grad_clip=float(args.grad_clip),
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            device=device,
            precision=args.precision,
            target_mean=target_mean,
            target_std=target_std,
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
            "lr": float(optimizer.param_groups[0]["lr"]),
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
        include_predictions=True,
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        targets=test_metrics.pop("targets"),
        predictions=test_metrics.pop("predictions"),
        test_idx=test_idx,
    )
    result = {
        "model": args.model,
        "dataset": dataset_name,
        "target": target_name,
        "best_epoch": int(best_payload["epoch"]),
        "best_val_rmse": float(best_payload["best_val_rmse"]),
        "test": test_metrics,
        "protocol": "shared split, 256-band interpolation, standardized raw-target MSE, validation-RMSE selection",
    }
    save_json(final_metrics_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
