from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aware_open.downstream.adapter import AwareFeatureExtractor, build_downstream_metadata


class SpectralRegressionDataset(Dataset):
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
        target_transform: str = "raw",
        target_weight_power: float = 0.0,
        target_weight_offset: float = 1.0,
        target_weight_norm: float = 1.0,
        target_weight_max: float = 1.0,
    ) -> None:
        self.spectra = spectra[indices].astype(np.float32)
        self.targets = targets[indices].astype(np.float32)
        self.spectrum_mean = spectrum_mean.astype(np.float32)
        self.spectrum_std = spectrum_std.astype(np.float32)
        self.target_mean = float(target_mean)
        self.target_std = float(max(target_std, 1.0e-6))
        self.target_transform = target_transform
        self.target_weight_power = float(target_weight_power)
        self.target_weight_offset = float(target_weight_offset)
        self.target_weight_norm = float(max(target_weight_norm, 1.0e-6))
        self.target_weight_max = float(max(target_weight_max, 1.0))

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = (self.spectra[idx] - self.spectrum_mean) / self.spectrum_std
        target_value = transform_targets_np(self.targets[idx], self.target_transform)
        y = (target_value - self.target_mean) / self.target_std
        weight = target_weight_np(
            self.targets[idx],
            power=self.target_weight_power,
            offset=self.target_weight_offset,
            norm=self.target_weight_norm,
            max_weight=self.target_weight_max,
        )
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "raw_y": torch.tensor(self.targets[idx], dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


class AwarePixelRegressor(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        patch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.patch_size = int(patch_size or extractor.local_patch_size)
        self.head = nn.Sequential(
            nn.LayerNorm(extractor.model_dim),
            nn.Linear(extractor.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        if spectra.dim() != 2:
            raise ValueError(f"Expected spectra shaped (B, C), got {tuple(spectra.shape)}.")
        patch = spectra[:, :, None, None].expand(-1, -1, self.patch_size, self.patch_size).contiguous()
        features = self.extractor.forward_features(patch)
        pooled = features["pooled_embedding"]
        return self.head(pooled).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def transform_targets_np(values: np.ndarray | np.float32 | float, transform: str) -> np.ndarray | np.float32:
    arr = np.asarray(values, dtype=np.float32)
    if transform == "raw":
        out = arr
    elif transform == "log1p":
        out = np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown target transform: {transform}")
    if np.isscalar(values):
        return np.float32(out)
    return out.astype(np.float32)


def inverse_targets_np(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    raise ValueError(f"Unknown target transform: {transform}")


def target_weight_np(
    value: np.float32 | float,
    *,
    power: float,
    offset: float,
    norm: float,
    max_weight: float,
) -> np.float32:
    if power <= 0.0:
        return np.float32(1.0)
    base = (float(value) + offset) / (float(norm) + offset)
    weight = max(base, 1.0e-6) ** power
    return np.float32(min(max(weight, 1.0e-3), max_weight))


def regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    huber_beta: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_name == "mse":
        per_item = F.mse_loss(pred, target, reduction="none")
    elif loss_name == "l1":
        per_item = F.l1_loss(pred, target, reduction="none")
    elif loss_name == "smoothl1":
        per_item = F.smooth_l1_loss(pred, target, beta=huber_beta, reduction="none")
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    if weight is None:
        return per_item.mean()
    weight = weight.to(device=per_item.device, dtype=per_item.dtype)
    return (per_item * weight).sum() / weight.sum().clamp_min(1.0e-6)


def augment_spectra(x: torch.Tensor, *, noise_std: float, band_dropout: float) -> torch.Tensor:
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std
    if band_dropout > 0.0:
        keep = torch.rand_like(x) > band_dropout
        x = x * keep.to(dtype=x.dtype)
    return x


@torch.no_grad()
def evaluate(
    model: AwarePixelRegressor,
    loader: DataLoader,
    *,
    device: torch.device,
    target_mean: float,
    target_std: float,
    target_transform: str,
    loss_name: str,
    huber_beta: float,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        pred_norm = model(x)
        loss = regression_loss(pred_norm, y, loss_name, huber_beta)
        pred_transformed = pred_norm.detach().float().cpu().numpy() * target_std + target_mean
        pred = inverse_targets_np(pred_transformed, target_transform)
        raw_y = batch["raw_y"].detach().float().cpu().numpy()
        preds.append(pred)
        truth.append(raw_y)
        losses.append(float(loss.item()))
    pred_all = np.concatenate(preds)
    y_all = np.concatenate(truth)
    rmse = float(np.sqrt(np.mean((pred_all - y_all) ** 2)))
    return {
        "loss": float(np.mean(losses)),
        "rmse": rmse,
        "r2": r2_score(y_all, pred_all),
        "samples": int(y_all.size),
    }


def configure_trainability(model: AwarePixelRegressor, mode: str) -> dict[str, int]:
    if mode == "linear_probe":
        for param in model.extractor.parameters():
            param.requires_grad = False
    elif mode == "freeze_aware_embeddings":
        freeze_keywords = (
            "tokenizer",
            "band_metadata",
            "sample_metadata",
            "metadata",
            "routing",
            "band_hyper",
            "hypernet",
        )
        for name, param in model.extractor.named_parameters():
            if any(key in name.lower() for key in freeze_keywords):
                param.requires_grad = False
    elif mode == "full_finetune":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Aware-HSI for pixel spectroscopy regression.")
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--sensor", default="field")
    parser.add_argument("--level", default="field")
    parser.add_argument("--gsd", type=float, default=1.0)
    parser.add_argument("--mode", choices=("linear_probe", "freeze_aware_embeddings", "full_finetune"), default="full_finetune")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--target-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--loss", choices=("mse", "smoothl1", "l1"), default="mse")
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--band-dropout", type=float, default=0.0)
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--target-weight-power", type=float, default=0.0)
    parser.add_argument("--target-weight-offset", type=float, default=1.0)
    parser.add_argument("--target-weight-max", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.dataset_npz)
    spectra = np.asarray(data["spectra"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    wavelengths = np.asarray(data["wavelengths_nm"], dtype=np.float32)
    train_idx = np.asarray(data["train_idx"], dtype=np.int64)
    val_idx = np.asarray(data["val_idx"], dtype=np.int64)
    test_idx = np.asarray(data["test_idx"], dtype=np.int64)

    train_spectra = spectra[train_idx]
    spectrum_mean = np.nanmean(train_spectra, axis=0).astype(np.float32)
    spectrum_std = np.nanstd(train_spectra, axis=0).astype(np.float32)
    spectrum_std = np.maximum(spectrum_std, 1.0e-6)
    transformed_train_targets = transform_targets_np(targets[train_idx], args.target_transform)
    target_mean = float(np.mean(transformed_train_targets))
    target_std = float(np.std(transformed_train_targets))
    target_weight_norm = float(np.median(targets[train_idx]))

    datasets = {
        "train": SpectralRegressionDataset(
            spectra,
            targets,
            train_idx,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=args.target_transform,
            target_weight_power=args.target_weight_power,
            target_weight_offset=args.target_weight_offset,
            target_weight_norm=target_weight_norm,
            target_weight_max=args.target_weight_max,
        ),
        "val": SpectralRegressionDataset(
            spectra,
            targets,
            val_idx,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=args.target_transform,
            target_weight_power=0.0,
            target_weight_offset=args.target_weight_offset,
            target_weight_norm=target_weight_norm,
            target_weight_max=args.target_weight_max,
        ),
        "test": SpectralRegressionDataset(
            spectra,
            targets,
            test_idx,
            spectrum_mean=spectrum_mean,
            spectrum_std=spectrum_std,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=args.target_transform,
            target_weight_power=0.0,
            target_weight_offset=args.target_weight_offset,
            target_weight_norm=target_weight_norm,
            target_weight_max=args.target_weight_max,
        ),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }

    metadata = build_downstream_metadata(
        num_bands=int(spectra.shape[1]),
        sensor=args.sensor,
        level=args.level,
        gsd=args.gsd,
        wavelengths_nm=wavelengths,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=False,
        device=device,
    )
    model = AwarePixelRegressor(extractor, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best_val_rmse.pt"
    resume_payload = None
    resume_path = None
    if args.resume:
        for candidate_path in (latest_path, best_path):
            if not candidate_path.exists():
                continue
            try:
                resume_payload = torch.load(candidate_path, map_location=device, weights_only=False)
            except Exception as exc:  # noqa: BLE001 - checkpoint corruption should fall back cleanly.
                print(f"[resume] failed to load {candidate_path}: {exc}", flush=True)
                resume_payload = None
                continue
            resume_path = candidate_path
            missing, unexpected = model.load_state_dict(resume_payload["model"], strict=False)
            print(
                f"[resume] loaded {candidate_path}; epoch={resume_payload.get('epoch')} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
            break
    if resume_payload is None:
        if args.init_model is not None:
            if args.resume:
                print("[resume] no usable checkpoint found; falling back to --init-model", flush=True)
            payload = torch.load(args.init_model, map_location=device, weights_only=False)
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(
                f"[init] loaded {args.init_model}; missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        elif args.resume:
            print(f"[resume] requested but {latest_path} does not exist; starting from checkpoint", flush=True)
    param_info = configure_trainability(model, args.mode)
    head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("extractor.")]
    extractor_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("extractor.")]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr, "name": "head"})
    if extractor_params:
        param_groups.append({"params": extractor_params, "lr": args.lr * args.encoder_lr_scale, "name": "extractor"})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if resume_payload is not None and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
    scaler_enabled = args.precision == "bf16" and device.type == "cuda"

    saved_args = vars(args).copy()
    saved_args.update(
        {
            "dataset_npz": str(args.dataset_npz),
            "output_dir": str(args.output_dir),
            "checkpoint": str(args.checkpoint),
            "prototype_bank_eo1": str(args.prototype_bank_eo1),
            "prototype_bank_gf5": str(args.prototype_bank_gf5),
        }
    )
    config = {
        "args": saved_args,
        "samples": {k: len(v) for k, v in datasets.items()},
        "bands": int(spectra.shape[1]),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_transform": args.target_transform,
        "target_weight_norm": target_weight_norm,
        **param_info,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    metrics_path = args.output_dir / "metrics.jsonl"
    best_val_rmse = math.inf
    best_epoch = -1
    epochs_since_best = 0
    start_epoch = 1
    if resume_payload is not None:
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        last_metric_epoch = int(resume_payload.get("epoch", 0))
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        print("[resume] skipped corrupt metrics.jsonl line", flush=True)
                        continue
                    last_metric_epoch = int(row.get("epoch", last_metric_epoch))
                    val_rmse = float(row["val"]["rmse"])
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_epoch = int(row["epoch"])
        if best_epoch >= 0:
            if resume_path == latest_path:
                epochs_since_best = max(0, last_metric_epoch - best_epoch)
            else:
                epochs_since_best = max(0, int(resume_payload.get("epoch", best_epoch)) - best_epoch)
        print(
            f"[resume] start_epoch={start_epoch} best_epoch={best_epoch} "
            f"best_val_rmse={best_val_rmse:.6f} epochs_since_best={epochs_since_best}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []
        for batch in loaders["train"]:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            weight = batch["weight"].to(device=device, dtype=torch.float32)
            x = augment_spectra(x, noise_std=args.input_noise_std, band_dropout=args.band_dropout)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=scaler_enabled):
                pred = model(x)
                loss = regression_loss(pred, y, args.loss, args.huber_beta, weight=weight)
            loss.backward()
            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            running.append(float(loss.item()))

        train_metrics = evaluate(
            model,
            loaders["train"],
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=args.target_transform,
            loss_name=args.loss,
            huber_beta=args.huber_beta,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=args.target_transform,
            loss_name=args.loss,
            huber_beta=args.huber_beta,
        )
        row = {
            "epoch": epoch,
            "train_loss_loop": float(np.mean(running)),
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "normalization": {
                "spectrum_mean": spectrum_mean,
                "spectrum_std": spectrum_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "target_transform": args.target_transform,
            },
        }
        torch.save(payload, latest_path)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(payload, best_path)
        else:
            epochs_since_best += 1
        if epochs_since_best >= args.patience:
            print(f"[early-stop] epoch={epoch} best_epoch={best_epoch} best_val_rmse={best_val_rmse:.6f}", flush=True)
            break

    best_payload_path = best_path if best_path.exists() else latest_path
    best_payload = torch.load(best_payload_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    test_metrics = evaluate(
        model,
        loaders["test"],
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        target_transform=args.target_transform,
        loss_name=args.loss,
        huber_beta=args.huber_beta,
    )
    final = {
        "best_epoch": int(best_payload["epoch"]),
        "best_val_rmse": best_val_rmse,
        "test": test_metrics,
    }
    (args.output_dir / "test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
