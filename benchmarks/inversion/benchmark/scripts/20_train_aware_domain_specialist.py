from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Candidate:
    name: str
    loss: str
    target_transform: str
    use_spectral_cnn: bool
    learning_rate: float


REGISTRY = {
    "green": (
        Candidate("veg_log_mse", "mse", "log1p", True, 4.0e-4),
        Candidate("veg_raw_mse", "mse", "raw", True, 4.0e-4),
        Candidate("veg_log_huber", "smoothl1", "log1p", True, 4.0e-4),
        Candidate("veg_indices_only", "mse", "log1p", False, 5.0e-4),
    ),
    "qxrd": (
        Candidate("mineral_raw_mse", "mse", "raw", True, 4.0e-4),
        Candidate("mineral_raw_huber", "smoothl1", "raw", True, 4.0e-4),
        Candidate("mineral_sqrt_mse", "mse", "sqrt", True, 4.0e-4),
        Candidate("mineral_depth_only", "mse", "raw", False, 5.0e-4),
    ),
    "karly": (
        Candidate("soil_native_mse", "mse", "raw", True, 4.0e-4),
        Candidate("soil_native_huber", "smoothl1", "raw", True, 4.0e-4),
        Candidate("soil_indices_only", "mse", "raw", False, 5.0e-4),
    ),
    "gloria": (
        Candidate("water_log_mse", "mse", "log1p", True, 4.0e-4),
        Candidate("water_log_huber", "smoothl1", "log1p", True, 4.0e-4),
        Candidate("water_tail_mse", "tail_mse", "raw", True, 4.0e-4),
        Candidate("water_indices_only", "mse", "log1p", False, 5.0e-4),
    ),
}


class DomainDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        domain_features: np.ndarray,
        spectral_view: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        *,
        target_mean: float,
        target_std: float,
        target_transform: str,
    ) -> None:
        self.tokens = tokens
        self.domain_features = domain_features
        self.spectral_view = spectral_view
        self.targets = targets
        self.indices = indices
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_transform = target_transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        raw = float(self.targets[index])
        transformed = transform_scalar(raw, self.target_transform)
        return {
            "tokens": torch.from_numpy(self.tokens[index].astype(np.float32)),
            "domain": torch.from_numpy(self.domain_features[index]),
            "spectrum": torch.from_numpy(self.spectral_view[index]),
            "target": torch.tensor((transformed - self.target_mean) / self.target_std, dtype=torch.float32),
            "raw_target": torch.tensor(raw, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.int64),
        }


class SpecialistRegressor(nn.Module):
    def __init__(self, token_dim: int, domain_dim: int, use_spectral_cnn: bool, dropout: float) -> None:
        super().__init__()
        hidden = 256
        self.token_project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU())
        self.token_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.domain_encoder = nn.Sequential(
            nn.LayerNorm(domain_dim),
            nn.Linear(domain_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.use_spectral_cnn = use_spectral_cnn
        if use_spectral_cnn:
            self.spectral_encoder = nn.Sequential(
                nn.Conv1d(3, 64, 7, padding=3, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv1d(64, 128, 11, padding=5, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv1d(128, hidden, 15, padding=7, bias=False),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
        branch_count = 3 if use_spectral_cnn else 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * branch_count),
            nn.Linear(hidden * branch_count, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, tokens: torch.Tensor, domain: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        token_features = self.token_project(tokens)
        token_weights = torch.softmax(self.token_score(token_features).squeeze(-1), dim=1)
        token_pooled = torch.sum(token_features * token_weights.unsqueeze(-1), dim=1)
        domain_encoded = self.domain_encoder(domain)
        branches = [token_pooled, domain_encoded]
        if self.use_spectral_cnn:
            first = F.pad(spectrum[:, 1:] - spectrum[:, :-1], (1, 0))
            second = F.pad(first[:, 1:] - first[:, :-1], (1, 0))
            spectral = self.spectral_encoder(torch.stack((spectrum, first, second), dim=1))
            branches.append(spectral.mean(dim=-1) + spectral.amax(dim=-1))
        return self.head(self.fusion(torch.cat(branches, dim=1))).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train validation-selected dataset-specific Aware inversion heads.")
    parser.add_argument("--branch", choices=tuple(REGISTRY), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--aware-features-npz", type=Path, required=True)
    parser.add_argument("--domain-features-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260628, 20260629, 20260630))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_scalar(value: float, transform: str) -> float:
    if transform == "raw":
        return value
    if transform == "log1p":
        return math.log1p(max(value, 0.0))
    if transform == "sqrt":
        return math.sqrt(max(value, 0.0))
    raise ValueError(transform)


def transform_array(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values.astype(np.float32)
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0.0)).astype(np.float32)
    raise ValueError(transform)


def inverse_array(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    if transform == "sqrt":
        return np.maximum(values, 0.0) ** 2
    raise ValueError(transform)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float("nan") if denominator <= 0 else 1.0 - float(np.sum(error**2)) / denominator,
    }


def training_loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "smoothl1":
        return F.smooth_l1_loss(prediction, target, beta=0.5)
    if loss_name == "tail_mse":
        weight = 1.0 + target.detach().clamp_min(0.0).pow(2).clamp(max=8.0) * 0.35
        return torch.sum(weight * (prediction - target) ** 2) / weight.sum().clamp_min(1.0e-6)
    raise ValueError(loss_name)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    target_mean: float,
    target_std: float,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truth, predictions, indices = [], [], []
    for batch in loader:
        tokens = batch["tokens"].to(device=device, dtype=torch.float32, non_blocking=True)
        domain = batch["domain"].to(device=device, dtype=torch.float32, non_blocking=True)
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            prediction = model(tokens, domain, spectrum)
        transformed = prediction.float().cpu().numpy() * target_std + target_mean
        predictions.append(inverse_array(transformed, transform))
        truth.append(batch["raw_target"].numpy())
        indices.append(batch["index"].numpy())
    return np.concatenate(truth), np.concatenate(predictions), np.concatenate(indices)


def run_seed(
    candidate: Candidate,
    seed: int,
    *,
    tokens: np.ndarray,
    domain_features: np.ndarray,
    spectral_view: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = output_dir / candidate.name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "val_metrics.json"
    checkpoint_path = run_dir / "best.pt"
    predictions_path = run_dir / "val_predictions.npz"
    if result_path.exists() and checkpoint_path.exists() and predictions_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    seed_everything(seed)
    transformed = transform_array(targets[train_idx], candidate.target_transform)
    target_mean = float(transformed.mean())
    target_std = float(max(transformed.std(), 1.0e-6))
    datasets = {
        name: DomainDataset(
            tokens,
            domain_features,
            spectral_view,
            targets,
            indices,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=candidate.target_transform,
        )
        for name, indices in (("train", train_idx), ("val", val_idx))
    }
    common = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
    }
    model = SpecialistRegressor(
        token_dim=int(tokens.shape[-1]),
        domain_dim=int(domain_features.shape[1]),
        use_spectral_cnn=candidate.use_spectral_cnn,
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=candidate.learning_rate * 0.01
    )
    best = float("inf")
    stale = 0
    max_epochs = 3 if args.smoke else args.epochs
    patience = 3 if args.smoke else args.patience
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loaders["train"]:
            tokens_batch = batch["tokens"].to(device=device, dtype=torch.float32, non_blocking=True)
            domain = batch["domain"].to(device=device, dtype=torch.float32, non_blocking=True)
            spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bf16",
            ):
                prediction = model(tokens_batch, domain, spectrum)
                loss = training_loss(prediction, target, candidate.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        y_true, y_pred, _ = predict(
            model,
            loaders["val"],
            device=device,
            precision=args.precision,
            target_mean=target_mean,
            target_std=target_std,
            transform=candidate.target_transform,
        )
        val = metrics(y_true, y_pred)
        if val["rmse"] < best:
            best = val["rmse"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "candidate": candidate.__dict__,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "epoch": epoch,
                    "best_val_rmse": best,
                },
                checkpoint_path,
            )
            print(f"[{candidate.name} seed={seed}] epoch={epoch} val_rmse={best:.8g}", flush=True)
        else:
            stale += 1
        if stale >= patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_true, y_pred, indices = predict(
        model,
        loaders["val"],
        device=device,
        precision=args.precision,
        target_mean=checkpoint["target_mean"],
        target_std=checkpoint["target_std"],
        transform=candidate.target_transform,
    )
    np.savez_compressed(predictions_path, targets=y_true, predictions=y_pred, indices=indices)
    result = {
        "candidate": candidate.__dict__,
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "best_val_rmse": checkpoint["best_val_rmse"],
        "val": metrics(y_true, y_pred),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "test_metrics.json"
    if final_path.exists() and not args.smoke:
        print(f"[skip] completed: {final_path}")
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        targets = np.asarray(payload["targets"], dtype=np.float32)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        dataset_name = str(np.asarray(payload["dataset"]).item())
        target_name = str(np.asarray(payload["target_name"]).item())
    with np.load(args.aware_features_npz, allow_pickle=True) as payload:
        tokens = np.asarray(payload["multiview_tokens"], dtype=np.float16)
    with np.load(args.domain_features_npz, allow_pickle=True) as payload:
        domain_features = np.asarray(payload["domain_features"], dtype=np.float32)
        spectral_view = np.asarray(payload["spectral_view"], dtype=np.float32)
    domain_mean = domain_features[train_idx].mean(axis=0)
    domain_std = np.maximum(domain_features[train_idx].std(axis=0), 1.0e-6)
    domain_features = ((domain_features - domain_mean) / domain_std).astype(np.float32)
    spectral_mean = spectral_view[train_idx].mean(axis=0)
    spectral_std = np.maximum(spectral_view[train_idx].std(axis=0), 1.0e-6)
    spectral_view = ((spectral_view - spectral_mean) / spectral_std).astype(np.float32)
    candidates = REGISTRY[args.branch][:2] if args.smoke else REGISTRY[args.branch]
    seeds = tuple(args.seeds[:1]) if args.smoke else tuple(args.seeds)
    search = []
    for candidate in candidates:
        seed_results = [
            run_seed(
                candidate,
                seed,
                tokens=tokens,
                domain_features=domain_features,
                spectral_view=spectral_view,
                targets=targets,
                train_idx=train_idx,
                val_idx=val_idx,
                output_dir=args.output_dir,
                args=args,
                device=device,
            )
            for seed in seeds
        ]
        val_predictions = []
        val_targets = val_indices = None
        for seed in seeds:
            payload = np.load(args.output_dir / candidate.name / f"seed{seed}" / "val_predictions.npz")
            val_predictions.append(payload["predictions"])
            val_targets, val_indices = payload["targets"], payload["indices"]
        ensemble = np.mean(val_predictions, axis=0)
        row = {
            "candidate": candidate.__dict__,
            "seed_results": seed_results,
            "ensemble_val": metrics(val_targets, ensemble),
        }
        search.append(row)
        print(f"[candidate] {candidate.name} val_rmse={row['ensemble_val']['rmse']:.8g}", flush=True)
    selected = min(search, key=lambda row: row["ensemble_val"]["rmse"])
    candidate = next(item for item in candidates if item.name == selected["candidate"]["name"])
    transformed = transform_array(targets[train_idx], candidate.target_transform)
    target_mean = float(transformed.mean())
    target_std = float(max(transformed.std(), 1.0e-6))
    test_dataset = DomainDataset(
        tokens,
        domain_features,
        spectral_view,
        targets,
        test_idx,
        target_mean=target_mean,
        target_std=target_std,
        target_transform=candidate.target_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_predictions = []
    test_targets = test_indices = None
    for seed in seeds:
        checkpoint = torch.load(
            args.output_dir / candidate.name / f"seed{seed}" / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model = SpecialistRegressor(
            token_dim=int(tokens.shape[-1]),
            domain_dim=int(domain_features.shape[1]),
            use_spectral_cnn=candidate.use_spectral_cnn,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        test_targets, prediction, test_indices = predict(
            model,
            test_loader,
            device=device,
            precision=args.precision,
            target_mean=checkpoint["target_mean"],
            target_std=checkpoint["target_std"],
            transform=candidate.target_transform,
        )
        test_predictions.append(prediction)
    ensemble_test = np.mean(test_predictions, axis=0)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        targets=test_targets,
        predictions=ensemble_test,
        test_idx=test_indices,
    )
    (args.output_dir / "validation_search.json").write_text(
        json.dumps(sorted(search, key=lambda row: row["ensemble_val"]["rmse"]), indent=2),
        encoding="utf-8",
    )
    result = {
        "model": f"aware_{args.branch}_domain_specialist_v4",
        "dataset": dataset_name,
        "target": target_name,
        "best_val_rmse": selected["ensemble_val"]["rmse"],
        "selection": selected,
        "test": metrics(test_targets, ensemble_test),
        "protocol": (
            "strict-loaded frozen Aware-HSI multiview tokens plus dataset-specific physical spectral "
            "features; three-seed ensemble validation selection; test evaluated only after selection"
        ),
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
