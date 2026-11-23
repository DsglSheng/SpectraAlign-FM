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
    profile: str
    spectral_branch: str
    loss: str
    target_transform: str
    learning_rate: float


CANDIDATES = (
    Candidate("single_mlp_mse", "single", "mlp", "mse", "raw", 5.0e-4),
    Candidate("mv4_mlp_mse", "multiview", "mlp", "mse", "raw", 5.0e-4),
    Candidate("mv4_cnn_mse", "multiview", "cnn", "mse", "raw", 4.0e-4),
    Candidate("mv4_cnn_weighted", "multiview", "cnn", "weighted_mse", "raw", 4.0e-4),
    Candidate("mv4_cnn_huber", "multiview", "cnn", "smoothl1", "raw", 4.0e-4),
    Candidate("mv4_cnn_log", "multiview", "cnn", "mse", "log1p", 4.0e-4),
)


class FixedFeatureDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        spectra: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        *,
        target_mean: float,
        target_std: float,
        target_transform: str,
    ) -> None:
        self.tokens = tokens
        self.spectra = spectra
        self.targets = targets
        self.indices = indices
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.target_transform = target_transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[position])
        raw_target = float(self.targets[index])
        transformed = math.log1p(max(raw_target, 0.0)) if self.target_transform == "log1p" else raw_target
        target = (transformed - self.target_mean) / self.target_std
        return {
            "tokens": torch.from_numpy(self.tokens[index].astype(np.float32)),
            "spectrum": torch.from_numpy(self.spectra[index]),
            "target": torch.tensor(target, dtype=torch.float32),
            "raw_target": torch.tensor(raw_target, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.int64),
        }


class TokenPool(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.project(tokens))
        weights = torch.softmax(self.score(encoded).squeeze(-1), dim=1)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=1)


class SpectralMLP(nn.Module):
    def __init__(self, bands: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(bands),
            nn.Linear(bands, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.network(spectrum)


class SpectralCNN(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(3, 64, 7, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 96, 9, padding=4, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv1d(96, hidden_dim, 11, padding=5, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        first = F.pad(spectrum[:, 1:] - spectrum[:, :-1], (1, 0))
        second = F.pad(first[:, 1:] - first[:, :-1], (1, 0))
        features = self.network(torch.stack((spectrum, first, second), dim=1))
        return features.mean(dim=-1) + features.amax(dim=-1)


class AwareMultiviewRegressor(nn.Module):
    def __init__(
        self,
        *,
        bands: int,
        spectral_branch: str,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.token_pool = TokenPool(hidden_dim=hidden_dim, dropout=dropout)
        if spectral_branch == "mlp":
            self.spectral = SpectralMLP(bands, hidden_dim, dropout)
        elif spectral_branch == "cnn":
            self.spectral = SpectralCNN(hidden_dim, dropout)
        else:
            raise ValueError(spectral_branch)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        aware = self.token_pool(tokens)
        spectral = self.spectral(spectrum)
        gate = self.gate(torch.cat((aware, spectral), dim=1))
        return self.head(aware + gate * spectral).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-select and seed-ensemble fixed Aware-HSI multiview inversion heads."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--features-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--input-noise-std", type=float, default=0.005)
    parser.add_argument("--band-dropout", type=float, default=0.02)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260625, 20260626, 20260627))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_targets(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values.astype(np.float32)
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def inverse_targets(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    return np.maximum(np.expm1(values), 0.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": r2_score(y_true, y_pred),
    }


def loss_function(prediction: torch.Tensor, target: torch.Tensor, name: str, beta: float) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "smoothl1":
        return F.smooth_l1_loss(prediction, target, beta=beta)
    if name == "weighted_mse":
        weight = 1.0 + 0.5 * target.detach().abs().clamp(max=4.0)
        return torch.sum(weight * (prediction - target) ** 2) / weight.sum().clamp_min(1.0e-6)
    raise ValueError(name)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    target_mean: float,
    target_std: float,
    target_transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        tokens = batch["tokens"].to(device=device, dtype=torch.float32, non_blocking=True)
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            prediction = model(tokens, spectrum)
        transformed = prediction.float().cpu().numpy() * target_std + target_mean
        predictions.append(inverse_targets(transformed, target_transform))
        targets.append(batch["raw_target"].numpy())
        indices.append(batch["index"].numpy())
    return (
        np.concatenate(targets).astype(np.float32),
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(indices).astype(np.int64),
    )


def train_seed(
    *,
    candidate: Candidate,
    seed: int,
    tokens: np.ndarray,
    spectra: np.ndarray,
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
    checkpoint_path = run_dir / "best_val_rmse.pt"
    predictions_path = run_dir / "val_predictions.npz"
    if result_path.exists() and checkpoint_path.exists() and predictions_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    seed_everything(seed)
    transformed_train = transform_targets(targets[train_idx], candidate.target_transform)
    target_mean = float(transformed_train.mean())
    target_std = float(max(transformed_train.std(), 1.0e-6))
    datasets = {
        split: FixedFeatureDataset(
            tokens,
            spectra,
            targets,
            indices,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=candidate.target_transform,
        )
        for split, indices in (("train", train_idx), ("val", val_idx))
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
    model = AwareMultiviewRegressor(
        bands=int(spectra.shape[1]),
        spectral_branch=candidate.spectral_branch,
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(candidate.learning_rate) * 0.01,
    )
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    log_path = run_dir / "metrics.jsonl"
    max_epochs = 3 if args.smoke else int(args.epochs)
    patience = 3 if args.smoke else int(args.patience)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            tokens_batch = batch["tokens"].to(device=device, dtype=torch.float32, non_blocking=True)
            spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            if args.input_noise_std > 0.0:
                spectrum = spectrum + torch.randn_like(spectrum) * float(args.input_noise_std)
            if args.band_dropout > 0.0:
                keep = torch.rand_like(spectrum).ge(float(args.band_dropout))
                spectrum = spectrum * keep.to(dtype=spectrum.dtype)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bf16",
            ):
                prediction = model(tokens_batch, spectrum)
                loss = loss_function(prediction, target, candidate.loss, float(args.huber_beta))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        y_true, y_pred, _ = predict(
            model,
            loaders["val"],
            device=device,
            precision=args.precision,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=candidate.target_transform,
        )
        val_metrics = metrics(y_true, y_pred)
        improved = val_metrics["rmse"] < best_val
        if improved:
            best_val = val_metrics["rmse"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "candidate": candidate.__dict__,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "epoch": epoch,
                    "best_val_rmse": best_val,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": val_metrics,
            "best_val_rmse": best_val,
            "best_epoch": best_epoch,
            "stale": stale,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if epoch == 1 or epoch % 20 == 0 or improved:
            print(
                f"[{candidate.name} seed={seed}] epoch={epoch} "
                f"val_rmse={val_metrics['rmse']:.8g} best={best_val:.8g}@{best_epoch}",
                flush=True,
            )
        if stale >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    y_true, y_pred, indices = predict(
        model,
        loaders["val"],
        device=device,
        precision=args.precision,
        target_mean=target_mean,
        target_std=target_std,
        target_transform=candidate.target_transform,
    )
    np.savez_compressed(predictions_path, targets=y_true, predictions=y_pred, indices=indices)
    result = {
        "candidate": candidate.__dict__,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_rmse": float(checkpoint["best_val_rmse"]),
        "val": metrics(y_true, y_pred),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "test_metrics.json"
    if final_path.exists() and not args.smoke:
        print(f"[skip] completed: {final_path}", flush=True)
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        raw_spectra = np.asarray(payload["spectra"], dtype=np.float32)
        targets = np.asarray(payload["targets"], dtype=np.float32)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        dataset_name = str(np.asarray(payload["dataset"]).item())
        target_name = str(np.asarray(payload["target_name"]).item())
    with np.load(args.features_npz, allow_pickle=True) as payload:
        single_tokens = np.asarray(payload["single_tokens"], dtype=np.float16)
        multiview_tokens = np.asarray(payload["multiview_tokens"], dtype=np.float16)
    band_mean = raw_spectra[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    band_std = np.maximum(
        raw_spectra[train_idx].std(axis=0, dtype=np.float64).astype(np.float32),
        1.0e-6,
    )
    spectra = ((raw_spectra - band_mean) / band_std).astype(np.float32)

    candidates = CANDIDATES[:2] if args.smoke else CANDIDATES
    seeds = tuple(args.seeds[:1]) if args.smoke else tuple(args.seeds)
    search: list[dict[str, Any]] = []
    for candidate in candidates:
        tokens = single_tokens if candidate.profile == "single" else multiview_tokens
        seed_results = [
            train_seed(
                candidate=candidate,
                seed=seed,
                tokens=tokens,
                spectra=spectra,
                targets=targets,
                train_idx=train_idx,
                val_idx=val_idx,
                output_dir=args.output_dir,
                args=args,
                device=device,
            )
            for seed in seeds
        ]
        validation_predictions = []
        validation_targets = None
        validation_indices = None
        for seed in seeds:
            payload = np.load(
                args.output_dir / candidate.name / f"seed{seed}" / "val_predictions.npz"
            )
            validation_predictions.append(payload["predictions"])
            validation_targets = payload["targets"]
            validation_indices = payload["indices"]
        ensemble_prediction = np.mean(validation_predictions, axis=0)
        ensemble_metrics = metrics(validation_targets, ensemble_prediction)
        row = {
            "candidate": candidate.__dict__,
            "seeds": list(seeds),
            "seed_results": seed_results,
            "ensemble_val": ensemble_metrics,
        }
        search.append(row)
        np.savez_compressed(
            args.output_dir / candidate.name / "ensemble_val_predictions.npz",
            targets=validation_targets,
            predictions=ensemble_prediction,
            indices=validation_indices,
        )
        print(
            f"[candidate] {candidate.name} ensemble_val_rmse={ensemble_metrics['rmse']:.8g}",
            flush=True,
        )

    selected = min(search, key=lambda row: row["ensemble_val"]["rmse"])
    selected_candidate = next(
        candidate for candidate in candidates if candidate.name == selected["candidate"]["name"]
    )
    selected_tokens = single_tokens if selected_candidate.profile == "single" else multiview_tokens
    transformed_train = transform_targets(targets[train_idx], selected_candidate.target_transform)
    target_mean = float(transformed_train.mean())
    target_std = float(max(transformed_train.std(), 1.0e-6))
    test_dataset = FixedFeatureDataset(
        selected_tokens,
        spectra,
        targets,
        test_idx,
        target_mean=target_mean,
        target_std=target_std,
        target_transform=selected_candidate.target_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    test_predictions: list[np.ndarray] = []
    test_targets = None
    test_indices = None
    for seed in seeds:
        checkpoint = torch.load(
            args.output_dir / selected_candidate.name / f"seed{seed}" / "best_val_rmse.pt",
            map_location=device,
            weights_only=False,
        )
        model = AwareMultiviewRegressor(
            bands=int(spectra.shape[1]),
            spectral_branch=selected_candidate.spectral_branch,
            dropout=float(args.dropout),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        test_targets, prediction, test_indices = predict(
            model,
            test_loader,
            device=device,
            precision=args.precision,
            target_mean=float(checkpoint["target_mean"]),
            target_std=float(checkpoint["target_std"]),
            target_transform=selected_candidate.target_transform,
        )
        test_predictions.append(prediction)
    ensemble_test_prediction = np.mean(test_predictions, axis=0)
    test_metrics = metrics(test_targets, ensemble_test_prediction)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        targets=test_targets,
        predictions=ensemble_test_prediction,
        test_idx=test_indices,
    )
    (args.output_dir / "validation_search.json").write_text(
        json.dumps(sorted(search, key=lambda row: row["ensemble_val"]["rmse"]), indent=2),
        encoding="utf-8",
    )
    result = {
        "model": "aware_multiview_spectral_ensemble_v3",
        "dataset": dataset_name,
        "target": target_name,
        "best_val_rmse": selected["ensemble_val"]["rmse"],
        "selection": selected,
        "test": test_metrics,
        "protocol": (
            "Aware-HSI pretraining-matched per-sample normalization; fixed single/multiview "
            "intermediate-token features; raw-spectrum residual head; candidate selected by "
            "three-seed ensemble validation RMSE; test evaluated once for the selected candidate"
        ),
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
