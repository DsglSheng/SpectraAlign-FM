from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class Candidate:
    name: str
    experts: int
    loss: str
    target_transform: str
    spectral_cnn: bool
    aux_weight: float
    rank_weight: float
    learning_rate: float


REGISTRY = {
    "karly_v5": (
        Candidate("retrieval_mse", 1, "mse", "raw", False, 0.0, 0.0, 5.0e-4),
        Candidate("retrieval_huber", 1, "smoothl1", "raw", True, 0.0, 0.0, 4.0e-4),
        Candidate("retrieval_rank05", 1, "mse", "raw", True, 0.0, 0.05, 4.0e-4),
        Candidate("retrieval_rank15", 1, "smoothl1", "raw", True, 0.0, 0.15, 4.0e-4),
    ),
    "karly_v6": (
        Candidate("local_moe3", 3, "mse", "raw", True, 0.10, 0.05, 4.0e-4),
        Candidate("local_moe5", 5, "mse", "raw", True, 0.10, 0.05, 4.0e-4),
        Candidate("local_moe3_huber", 3, "smoothl1", "raw", True, 0.25, 0.10, 4.0e-4),
        Candidate("local_moe5_huber", 5, "smoothl1", "raw", True, 0.25, 0.10, 4.0e-4),
    ),
    "gloria_v5": (
        Candidate("water_stack_log", 1, "mse", "log1p", False, 0.0, 0.0, 5.0e-4),
        Candidate("water_stack_huber", 1, "smoothl1", "log1p", True, 0.0, 0.0, 4.0e-4),
        Candidate("water_stack_tail", 1, "tail_mse", "log1p", True, 0.0, 0.0, 4.0e-4),
        Candidate("water_stack_rawtail", 1, "tail_mse", "raw", True, 0.0, 0.0, 4.0e-4),
    ),
    "gloria_v6": (
        Candidate("water_moe3", 3, "mse", "log1p", True, 0.15, 0.0, 4.0e-4),
        Candidate("water_moe5", 5, "mse", "log1p", True, 0.15, 0.0, 4.0e-4),
        Candidate("water_moe3_tail", 3, "tail_mse", "log1p", True, 0.30, 0.0, 4.0e-4),
        Candidate("water_moe5_tail", 5, "tail_mse", "log1p", True, 0.30, 0.0, 4.0e-4),
    ),
}


def transform_array(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values.astype(np.float32)
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def inverse_array(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    return np.maximum(np.expm1(values), 0.0)


class BranchDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        domain: np.ndarray,
        spectral: np.ndarray,
        stacks: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        stack_lookup: dict[int, int],
        *,
        target_mean: float,
        target_std: float,
        target_transform: str,
        thresholds: np.ndarray,
    ) -> None:
        self.tokens = tokens
        self.domain = domain
        self.spectral = spectral
        self.stacks = stacks
        self.targets = targets
        self.indices = indices
        self.stack_lookup = stack_lookup
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_transform = target_transform
        self.thresholds = thresholds

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        stack_row = self.stack_lookup[index]
        raw_target = float(self.targets[index])
        transformed_target = float(transform_array(np.asarray([raw_target]), self.target_transform)[0])
        transformed_stack = transform_array(self.stacks[stack_row], self.target_transform)
        normalized_stack = (transformed_stack - self.target_mean) / self.target_std
        target_bin = int(np.searchsorted(self.thresholds, raw_target, side="right"))
        return {
            "tokens": torch.from_numpy(self.tokens[index].astype(np.float32)),
            "domain": torch.from_numpy(self.domain[index]),
            "spectrum": torch.from_numpy(self.spectral[index]),
            "stack": torch.from_numpy(normalized_stack.astype(np.float32)),
            "target": torch.tensor(
                (transformed_target - self.target_mean) / self.target_std,
                dtype=torch.float32,
            ),
            "raw_target": torch.tensor(raw_target, dtype=torch.float32),
            "target_bin": torch.tensor(target_bin, dtype=torch.int64),
            "index": torch.tensor(index, dtype=torch.int64),
        }


class FusionModel(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        domain_dim: int,
        stack_dim: int,
        experts: int,
        spectral_cnn: bool,
        bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = 256
        self.token_project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU())
        self.token_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.domain_encoder = nn.Sequential(
            nn.LayerNorm(domain_dim),
            nn.Linear(domain_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stack_encoder = nn.Sequential(
            nn.LayerNorm(stack_dim),
            nn.Linear(stack_dim, 128),
            nn.GELU(),
        )
        self.stack_weights = nn.Sequential(nn.Linear(hidden + 128 + 128, stack_dim), nn.Softmax(dim=-1))
        self.spectral_cnn = spectral_cnn
        if spectral_cnn:
            self.spectral_encoder = nn.Sequential(
                nn.Conv1d(3, 64, 7, padding=3, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv1d(64, 128, 11, padding=5, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
            )
        context_dim = hidden + 128 + 128 + (128 if spectral_cnn else 0)
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in range(experts)
            ]
        )
        self.gate = nn.Linear(hidden, experts)
        self.bin_head = nn.Linear(hidden, bins)

    def forward(
        self,
        tokens: torch.Tensor,
        domain: torch.Tensor,
        spectrum: torch.Tensor,
        stack: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_features = self.token_project(tokens)
        token_weights = torch.softmax(self.token_score(token_features).squeeze(-1), dim=1)
        token_pooled = torch.sum(token_features * token_weights.unsqueeze(-1), dim=1)
        domain_encoded = self.domain_encoder(domain)
        stack_encoded = self.stack_encoder(stack)
        stack_context = torch.cat((token_pooled, domain_encoded, stack_encoded), dim=1)
        base_weights = self.stack_weights(stack_context)
        base_prediction = torch.sum(base_weights * stack, dim=1)
        branches = [token_pooled, domain_encoded, stack_encoded]
        if self.spectral_cnn:
            first = F.pad(spectrum[:, 1:] - spectrum[:, :-1], (1, 0))
            second = F.pad(first[:, 1:] - first[:, :-1], (1, 0))
            encoded = self.spectral_encoder(torch.stack((spectrum, first, second), dim=1))
            branches.append(encoded.mean(dim=-1) + encoded.amax(dim=-1))
        context = self.context(torch.cat(branches, dim=1))
        expert_outputs = torch.cat([expert(context) for expert in self.experts], dim=1)
        gate = torch.softmax(self.gate(context), dim=1)
        residual = torch.sum(gate * expert_outputs, dim=1)
        return base_prediction + residual, self.bin_head(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KarLy/GLORIA retrieval-stack and mixture-of-expert branches.")
    parser.add_argument("--mode", choices=tuple(REGISTRY), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--aware-features-npz", type=Path, required=True)
    parser.add_argument("--domain-features-npz", type=Path, required=True)
    parser.add_argument("--stack-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260701, 20260702, 20260703))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float("nan") if denominator <= 0 else 1.0 - float(np.sum(error**2)) / denominator,
    }


def regression_loss(prediction: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "smoothl1":
        return F.smooth_l1_loss(prediction, target, beta=0.5)
    if name == "tail_mse":
        weight = 1.0 + 0.35 * target.detach().abs().pow(2).clamp(max=10.0)
        return torch.sum(weight * (prediction - target) ** 2) / weight.sum().clamp_min(1.0e-6)
    raise ValueError(name)


def pairwise_rank_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    order = torch.randperm(prediction.numel(), device=prediction.device)
    target_difference = target - target[order]
    prediction_difference = prediction - prediction[order]
    sign = torch.sign(target_difference)
    valid = sign.ne(0)
    if not torch.any(valid):
        return prediction.new_zeros(())
    return F.softplus(-sign[valid] * prediction_difference[valid]).mean()


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
        tensors = {
            key: batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
            for key in ("tokens", "domain", "spectrum", "stack")
        }
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            prediction, _ = model(**tensors)
        transformed = prediction.float().cpu().numpy() * target_std + target_mean
        predictions.append(inverse_array(transformed, transform))
        truth.append(batch["raw_target"].numpy())
        indices.append(batch["index"].numpy())
    return np.concatenate(truth), np.concatenate(predictions), np.concatenate(indices)


def run_seed(
    candidate: Candidate,
    seed: int,
    *,
    arrays: dict[str, Any],
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
    targets = arrays["targets"]
    train_idx = arrays["train_idx"]
    transformed_train = transform_array(targets[train_idx], candidate.target_transform)
    target_mean = float(transformed_train.mean())
    target_std = float(max(transformed_train.std(), 1.0e-6))
    quantiles = np.linspace(0.0, 1.0, candidate.experts + 1)[1:-1]
    thresholds = np.quantile(targets[train_idx], quantiles).astype(np.float32)
    datasets = {}
    for split in ("train", "val"):
        indices = arrays[f"{split}_idx"]
        stacks = arrays[f"{split}_stack"]
        lookup = {int(index): position for position, index in enumerate(indices)}
        datasets[split] = BranchDataset(
            arrays["tokens"],
            arrays["domain"],
            arrays["spectral"],
            stacks,
            targets,
            indices,
            lookup,
            target_mean=target_mean,
            target_std=target_std,
            target_transform=candidate.target_transform,
            thresholds=thresholds,
        )
    if args.mode.startswith("gloria"):
        train_bins = np.digitize(targets[train_idx], np.quantile(targets[train_idx], [0.2, 0.4, 0.6, 0.8]))
        counts = np.bincount(train_bins, minlength=5)
        sample_weights = 1.0 / np.maximum(counts[train_bins], 1)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights.astype(np.float64)),
            num_samples=len(train_idx),
            replacement=True,
        )
        train_loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = FusionModel(
        token_dim=int(arrays["tokens"].shape[-1]),
        domain_dim=int(arrays["domain"].shape[1]),
        stack_dim=int(arrays["train_stack"].shape[1]),
        experts=candidate.experts,
        spectral_cnn=candidate.spectral_cnn,
        bins=candidate.experts,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=candidate.learning_rate * 0.01,
    )
    best = float("inf")
    stale = 0
    max_epochs = 3 if args.smoke else args.epochs
    patience = 3 if args.smoke else args.patience
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            tensors = {
                key: batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
                for key in ("tokens", "domain", "spectrum", "stack")
            }
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            target_bin = batch["target_bin"].to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bf16",
            ):
                prediction, bin_logits = model(**tensors)
                loss = regression_loss(prediction, target, candidate.loss)
                if candidate.aux_weight > 0.0:
                    loss = loss + candidate.aux_weight * F.cross_entropy(bin_logits, target_bin)
                if candidate.rank_weight > 0.0:
                    loss = loss + candidate.rank_weight * pairwise_rank_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        y_true, y_pred, _ = predict(
            model,
            val_loader,
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
                    "thresholds": thresholds,
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
        val_loader,
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
        domain = np.asarray(payload["domain_features"], dtype=np.float32)
        spectral = np.asarray(payload["spectral_view"], dtype=np.float32)
    domain = ((domain - domain[train_idx].mean(axis=0)) / np.maximum(domain[train_idx].std(axis=0), 1.0e-6)).astype(np.float32)
    spectral = ((spectral - spectral[train_idx].mean(axis=0)) / np.maximum(spectral[train_idx].std(axis=0), 1.0e-6)).astype(np.float32)
    with np.load(args.stack_npz, allow_pickle=True) as payload:
        stack_train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        stack_val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        stack_test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        train_stack = np.asarray(payload["train_stack"], dtype=np.float32)
        val_stack = np.asarray(payload["val_stack"], dtype=np.float32)
        test_stack = np.asarray(payload["test_stack"], dtype=np.float32)
    if not (
        np.array_equal(train_idx, stack_train_idx)
        and np.array_equal(val_idx, stack_val_idx)
        and np.array_equal(test_idx, stack_test_idx)
    ):
        raise ValueError("Stack indices do not match dataset splits.")
    arrays = {
        "tokens": tokens,
        "domain": domain,
        "spectral": spectral,
        "targets": targets,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_stack": train_stack,
        "val_stack": val_stack,
        "test_stack": test_stack,
    }
    candidates = REGISTRY[args.mode][:2] if args.smoke else REGISTRY[args.mode]
    seeds = tuple(args.seeds[:1]) if args.smoke else tuple(args.seeds)
    search = []
    for candidate in candidates:
        seed_results = [
            run_seed(
                candidate,
                seed,
                arrays=arrays,
                output_dir=args.output_dir,
                args=args,
                device=device,
            )
            for seed in seeds
        ]
        val_predictions, val_targets = [], None
        for seed in seeds:
            payload = np.load(args.output_dir / candidate.name / f"seed{seed}" / "val_predictions.npz")
            val_predictions.append(payload["predictions"])
            val_targets = payload["targets"]
        ensemble_prediction = np.mean(val_predictions, axis=0)
        row = {
            "candidate": candidate.__dict__,
            "seed_results": seed_results,
            "ensemble_val": metrics(val_targets, ensemble_prediction),
        }
        search.append(row)
        print(f"[candidate] {candidate.name} val_rmse={row['ensemble_val']['rmse']:.8g}", flush=True)
    selected = min(search, key=lambda row: row["ensemble_val"]["rmse"])
    candidate = next(item for item in candidates if item.name == selected["candidate"]["name"])
    transformed_train = transform_array(targets[train_idx], candidate.target_transform)
    target_mean = float(transformed_train.mean())
    target_std = float(max(transformed_train.std(), 1.0e-6))
    quantiles = np.linspace(0.0, 1.0, candidate.experts + 1)[1:-1]
    thresholds = np.quantile(targets[train_idx], quantiles).astype(np.float32)
    test_lookup = {int(index): position for position, index in enumerate(test_idx)}
    test_dataset = BranchDataset(
        tokens,
        domain,
        spectral,
        test_stack,
        targets,
        test_idx,
        test_lookup,
        target_mean=target_mean,
        target_std=target_std,
        target_transform=candidate.target_transform,
        thresholds=thresholds,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_predictions, test_targets, test_indices = [], None, None
    for seed in seeds:
        checkpoint = torch.load(
            args.output_dir / candidate.name / f"seed{seed}" / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model = FusionModel(
            token_dim=int(tokens.shape[-1]),
            domain_dim=int(domain.shape[1]),
            stack_dim=int(train_stack.shape[1]),
            experts=candidate.experts,
            spectral_cnn=candidate.spectral_cnn,
            bins=candidate.experts,
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
        "model": f"aware_{args.mode}",
        "dataset": dataset_name,
        "target": target_name,
        "best_val_rmse": selected["ensemble_val"]["rmse"],
        "selection": selected,
        "test": metrics(test_targets, ensemble_test),
        "protocol": (
            "strict-loaded frozen Aware features plus train-only OOF nonlinear retrieval stack; "
            "three-seed validation ensemble selection; test evaluated only after selection"
        ),
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
