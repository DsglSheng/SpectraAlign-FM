from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset

from aware_open.config import AwareEncoderConfig, FoundationConfig, FoundationDecoderConfig, FoundationLossConfig
from aware_open.data.pretrain import Sigma450KShardDataset, Sigma450KShardDistributedSampler, collate_modality_samples, move_batch_to_device
from aware_open.models.foundation import MetadataConditionedFoundationModel
from aware_open.spectral_prototypes import load_prototype_bank


@dataclass
class ResumeState:
    epoch: int = 0
    step: int = 0
    best_ssl: float = math.inf
    best_probe: float = -math.inf
    loaded_optimizer: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-stage metadata-conditioned foundation pretraining.")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/Data/SpectralData"))
    parser.add_argument("--sigma-shard-root", type=Path, default=None)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--probe-every-epochs", type=int, default=1)
    parser.add_argument("--probe-samples", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-workers", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--progress-refresh-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--space-crop-size", type=int, default=64)
    parser.add_argument("--space-crop-mode", choices=("none", "random", "center"), default="center")
    parser.add_argument("--space-normalization", choices=("none", "mean_std", "robust"), default="mean_std")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--band-dim", type=int, default=192)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--spectral-dim", type=int, default=96)
    parser.add_argument("--spatial-dim", type=int, default=96)
    parser.add_argument("--num-routing-prototypes", type=int, default=12)
    parser.add_argument("--prototype-heads", type=int, default=8)
    parser.add_argument("--prototype-depth", type=int, default=2)
    parser.add_argument("--local-patch-size", type=int, default=8)
    parser.add_argument("--token-encoder-depth", type=int, default=12)
    parser.add_argument("--token-encoder-heads", type=int, default=12)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--decoder-dim", type=int, default=384)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=6)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--tdfa-depth", type=int, default=2)
    parser.add_argument("--frequency-groups", type=int, default=5)
    parser.add_argument("--frequency-cutoff-low", type=float, default=0.15)
    parser.add_argument("--frequency-cutoff-mid", type=float, default=0.40)
    parser.add_argument("--prototype-temperature-teacher", type=float, default=0.07)
    parser.add_argument("--prototype-temperature-student", type=float, default=0.07)
    parser.add_argument("--proto-kl-weight", type=float, default=0.10)
    parser.add_argument("--proto-balance-weight", type=float, default=0.02)
    parser.add_argument("--warmup-steps-loss", type=int, default=40_000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}.")
    if args.save_every_epochs <= 0:
        raise ValueError(f"--save-every-epochs must be positive, got {args.save_every_epochs}.")
    if args.probe_every_epochs <= 0:
        raise ValueError(f"--probe-every-epochs must be positive, got {args.probe_every_epochs}.")
    if args.progress_refresh_seconds <= 0:
        raise ValueError(f"--progress-refresh-seconds must be positive, got {args.progress_refresh_seconds}.")
    if args.grad_accum_steps <= 0:
        raise ValueError(f"--grad-accum-steps must be positive, got {args.grad_accum_steps}.")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError(f"--mask-ratio must be in (0, 1), got {args.mask_ratio}.")
    if not 0.0 < args.frequency_cutoff_low < args.frequency_cutoff_mid < 1.0:
        raise ValueError("--frequency-cutoff-low and --frequency-cutoff-mid must satisfy 0 < low < mid < 1.")


def distributed_state() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def choose_device(local_rank: int) -> torch.device:
    return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")


def maybe_init_distributed(device: torch.device) -> None:
    _, world_size, local_rank = distributed_state()
    if world_size == 1:
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def crop_size_from_args(size: int, mode: str) -> tuple[int | None, bool]:
    if mode == "none":
        return None, False
    return size, mode == "random"


def build_encoder_config(args: argparse.Namespace) -> AwareEncoderConfig:
    return AwareEncoderConfig(
        model_dim=args.model_dim,
        band_dim=args.band_dim,
        context_dim=args.context_dim,
        spectral_dim=args.spectral_dim,
        spatial_dim=args.spatial_dim,
        num_prototypes=args.num_routing_prototypes,
        local_patch_size=args.local_patch_size,
        prototype_heads=args.prototype_heads,
        prototype_depth=args.prototype_depth,
        token_encoder_depth=args.token_encoder_depth,
        token_encoder_heads=args.token_encoder_heads,
        backbone_out_indices=tuple(range(max(0, args.token_encoder_depth - 4), args.token_encoder_depth)),
        backbone_pretrained=False,
        drop_rate=args.drop_rate,
    )


def build_foundation_config(args: argparse.Namespace, prototype_bank_size: int) -> FoundationConfig:
    return FoundationConfig(
        encoder=build_encoder_config(args),
        decoder=FoundationDecoderConfig(
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads,
            decoder_dropout=args.decoder_dropout,
            tdfa_depth=args.tdfa_depth,
            frequency_groups=args.frequency_groups,
            low_cutoff_ratio=args.frequency_cutoff_low,
            mid_cutoff_ratio=args.frequency_cutoff_mid,
        ),
        loss=FoundationLossConfig(
            reconstruction_low=0.1,
            reconstruction_mid=0.2,
            reconstruction_full=0.7,
            proto_kl=args.proto_kl_weight,
            proto_balance=args.proto_balance_weight,
        ),
        mask_ratio=args.mask_ratio,
        prototype_bank_size=prototype_bank_size,
        prototype_temperature_teacher=args.prototype_temperature_teacher,
        prototype_temperature_student=args.prototype_temperature_student,
    )


def build_dataset(args: argparse.Namespace) -> Sigma450KShardDataset:
    sigma_root = args.sigma_shard_root if args.sigma_shard_root is not None else args.data_root / "sigma450k_shards"
    crop_size, random_crop = crop_size_from_args(args.space_crop_size, args.space_crop_mode)
    return Sigma450KShardDataset(
        sigma_root,
        crop_size=crop_size,
        random_crop=random_crop,
        normalization=args.space_normalization,
    )


def build_probe_loader(args: argparse.Namespace, dataset: Sigma450KShardDataset) -> DataLoader | None:
    if args.probe_samples <= 0:
        return None
    sample_count = min(args.probe_samples, len(dataset))
    if sample_count <= 0:
        return None
    rng = random.Random(f"{args.seed}:probe")
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:sample_count])
    return DataLoader(
        subset,
        batch_size=min(args.probe_batch_size, sample_count),
        shuffle=False,
        num_workers=args.probe_workers,
        pin_memory=True,
        persistent_workers=args.probe_workers > 0,
        collate_fn=collate_modality_samples,
        drop_last=False,
    )


def add_scalar_sum(target: dict[str, torch.Tensor], key: str, value: torch.Tensor | float, *, device: torch.device) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if key not in target:
        target[key] = torch.zeros((), device=device, dtype=torch.float32)
    target[key] = target[key] + tensor


def reduce_scalar_sums(values: dict[str, torch.Tensor], *, world_size: int) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key, value in values.items():
        tensor = value.clone()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= float(world_size)
        reduced[key] = float(tensor.item())
    return reduced


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress_bar(fraction: float, width: int = 24) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(fraction * width)
    if filled >= width:
        return "[" + "=" * width + "]"
    if filled <= 0:
        return "[" + " " * width + "]"
    return "[" + "=" * max(0, filled - 1) + ">" + " " * (width - filled) + "]"


def write_live_status(message: str) -> None:
    sys.stdout.write("\r" + message.ljust(180))
    sys.stdout.flush()


def clear_live_status() -> None:
    sys.stdout.write("\r" + (" " * 180) + "\r")
    sys.stdout.flush()


def build_train_loader(
    args: argparse.Namespace,
    dataset: Sigma450KShardDataset,
    *,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, Sampler | None]:
    if world_size > 1:
        sampler: Sampler | None = Sigma450KShardDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_modality_samples,
        drop_last=True,
    )
    return loader, sampler


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _is_shallow_backbone_parameter(name: str, depth: int) -> bool:
    prefix = "backbone.transformer.blocks."
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix) :]
    index_text = suffix.split(".", 1)[0]
    if not index_text.isdigit():
        return False
    return int(index_text) < max(1, depth // 3)


def build_optimizer(args: argparse.Namespace, model: MetadataConditionedFoundationModel) -> tuple[torch.optim.Optimizer, list[dict[str, object]]]:
    groups: dict[tuple[str, bool], dict[str, object]] = {}

    def _group_name(name: str) -> str:
        if _is_shallow_backbone_parameter(name, args.token_encoder_depth):
            return "shallow_backbone"
        if name.startswith("backbone."):
            return "deep_backbone"
        return "base"

    def _lr_scale(group_name: str) -> float:
        if group_name == "shallow_backbone":
            return 0.2
        if group_name == "deep_backbone":
            return 0.6
        return 1.0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
        group_name = _group_name(name)
        key = (group_name, no_decay)
        if key not in groups:
            groups[key] = {
                "name": f"{group_name}_{'no_decay' if no_decay else 'decay'}",
                "params": [],
                "weight_decay": 0.0 if no_decay else args.weight_decay,
                "lr_scale": _lr_scale(group_name),
            }
        groups[key]["params"].append(parameter)

    optimizer = torch.optim.AdamW(list(groups.values()), lr=args.lr, betas=(args.beta1, args.beta2))
    info = [
        {
            "name": group["name"],
            "count": len(group["params"]),
            "weight_decay": group["weight_decay"],
            "lr_scale": group["lr_scale"],
        }
        for group in groups.values()
    ]
    return optimizer, info


def set_learning_rate(optimizer: torch.optim.Optimizer, base_lr: float) -> dict[str, float]:
    applied: dict[str, float] = {}
    for group in optimizer.param_groups:
        group_lr = base_lr * float(group.get("lr_scale", 1.0))
        group["lr"] = group_lr
        applied[str(group.get("name", "model"))] = group_lr
    return applied


def cosine_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def resolve_loss_schedule(args: argparse.Namespace, step: int) -> dict[str, float]:
    if step < args.warmup_steps_loss:
        alpha = float(step + 1) / float(max(1, args.warmup_steps_loss))
        return {
            "reconstruction_low": 0.5,
            "reconstruction_mid": 0.3,
            "reconstruction_full": 0.2,
            "proto_kl": args.proto_kl_weight * alpha,
            "proto_balance": args.proto_balance_weight * alpha,
        }
    return {
        "reconstruction_low": 0.1,
        "reconstruction_mid": 0.2,
        "reconstruction_full": 0.7,
        "proto_kl": args.proto_kl_weight,
        "proto_balance": args.proto_balance_weight,
    }


def checkpoint_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: checkpoint_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_safe_value(item) for item in value]
    return value


def build_checkpoint_payload(
    *,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    best_ssl: float,
    best_probe: float,
) -> dict[str, object]:
    return {
        "epoch": int(epoch),
        "step": step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage_state": {
            "stage": "foundation",
            "epoch": int(epoch),
            "best_ssl": float(best_ssl),
            "best_probe": float(best_probe),
        },
        "args": checkpoint_safe_value(vars(args)),
    }


def write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_checkpoint(output_dir: Path, payload: dict[str, object], epoch: int) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_checkpoint(checkpoint_dir / "latest.pt", payload)
    write_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", payload)


def save_named_checkpoint(output_dir: Path, payload: dict[str, object], name: str) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_checkpoint(checkpoint_dir / f"{name}.pt", payload)


def append_log(output_dir: Path, payload: dict[str, float | int | str | bool]) -> None:
    log_path = output_dir / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_foundation_state_dict(model: MetadataConditionedFoundationModel, state_dict: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}."
        )


def maybe_resume(args: argparse.Namespace, model: MetadataConditionedFoundationModel, optimizer: torch.optim.Optimizer) -> ResumeState:
    if args.resume is None:
        return ResumeState()
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    load_foundation_state_dict(model, payload["model"])
    checkpoint_stage_state = payload.get("stage_state", {})
    epoch = payload.get("epoch", checkpoint_stage_state.get("epoch"))
    step = int(payload.get("step", 0))
    if epoch is None:
        if step != 0:
            raise ValueError("Resume checkpoint is missing epoch metadata and cannot be resumed with the epoch-based trainer.")
        epoch = 0
    best_ssl = float(checkpoint_stage_state.get("best_ssl", math.inf))
    best_probe = float(checkpoint_stage_state.get("best_probe", -math.inf))
    loaded_optimizer = False
    try:
        optimizer.load_state_dict(payload["optimizer"])
        loaded_optimizer = True
    except ValueError:
        loaded_optimizer = False
    return ResumeState(
        epoch=int(epoch),
        step=step,
        best_ssl=best_ssl,
        best_probe=best_probe,
        loaded_optimizer=loaded_optimizer,
    )


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def encode_labels(labels: list[str]) -> tuple[torch.Tensor, int]:
    mapping = {label: index for index, label in enumerate(sorted(set(labels)))}
    encoded = torch.tensor([mapping[label] for label in labels], dtype=torch.long)
    return encoded, len(mapping)


def leave_one_out_centroid_accuracy(features: torch.Tensor, labels: list[str]) -> float:
    encoded, num_classes = encode_labels(labels)
    if features.shape[0] < 4 or num_classes < 2:
        return float("nan")
    normalized = F.normalize(features, dim=-1)
    class_sums = torch.zeros(num_classes, normalized.shape[1], dtype=normalized.dtype)
    class_counts = torch.zeros(num_classes, dtype=normalized.dtype)
    class_sums.index_add_(0, encoded, normalized)
    class_counts.index_add_(0, encoded, torch.ones_like(encoded, dtype=normalized.dtype))
    valid = 0
    correct = 0
    base_centroids = class_sums / class_counts.unsqueeze(-1).clamp_min(1.0)
    base_centroids = F.normalize(base_centroids, dim=-1)
    for index in range(normalized.shape[0]):
        label = int(encoded[index].item())
        if class_counts[label] < 2:
            continue
        centroids = base_centroids.clone()
        centroids[label] = F.normalize(
            (class_sums[label] - normalized[index]) / (class_counts[label] - 1.0),
            dim=-1,
        )
        prediction = int((normalized[index] @ centroids.T).argmax().item())
        correct += int(prediction == label)
        valid += 1
    if valid == 0:
        return float("nan")
    return correct / valid


def ridge_probe_score(features: torch.Tensor, targets: torch.Tensor, l2: float = 1.0e-3) -> float:
    if features.shape[0] < 6:
        return float("nan")
    targets = targets.float()
    if float(targets.std(unbiased=False).item()) < 1.0e-6:
        return float("nan")
    index = torch.arange(features.shape[0])
    splits = ((index[::2], index[1::2]), (index[1::2], index[::2]))
    scores: list[float] = []
    ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
    features = torch.cat([features, ones], dim=-1)
    for train_index, test_index in splits:
        if train_index.numel() == 0 or test_index.numel() == 0:
            continue
        x_train = features[train_index]
        x_test = features[test_index]
        y_train = targets[train_index]
        y_test = targets[test_index]
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        eye = torch.eye(x_train.shape[1], dtype=x_train.dtype)
        weights = torch.linalg.solve(x_train.T @ x_train + l2 * eye, x_train.T @ y_train)
        prediction = x_test @ weights
        mae = (prediction - y_test).abs().mean()
        baseline = (y_test - y_train.mean()).abs().mean().clamp_min(1.0e-6)
        scores.append(float((1.0 - mae / baseline).item()))
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def evaluate_probe(
    model: MetadataConditionedFoundationModel,
    loader: DataLoader | None,
    *,
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    if loader is None:
        return {}
    metrics: dict[str, float] = {}
    ssl_losses: list[float] = []
    features_list: list[torch.Tensor] = []
    sensor_labels: list[str] = []
    level_labels: list[str] = []
    gsd_targets: list[torch.Tensor] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                output = model(space=batch, loss_weights_override=loss_weights)
                ssl_losses.append(float(output["loss"].detach().item()))
                features_list.append(output["normalized_embedding"].detach().float().cpu())
                sensor_labels.extend(list(batch.metadata.sensor))
                level_labels.extend(list(batch.metadata.level))
                gsd_targets.append(batch.metadata.gsd.detach().float().cpu())
    finally:
        model.train()
    if not features_list:
        return metrics
    features = torch.cat(features_list, dim=0)
    metrics["probe_ssl"] = float(sum(ssl_losses) / len(ssl_losses))
    sensor_acc = leave_one_out_centroid_accuracy(features, sensor_labels)
    if not math.isnan(sensor_acc):
        metrics["probe_sensor_acc"] = sensor_acc
    level_acc = leave_one_out_centroid_accuracy(features, level_labels)
    if not math.isnan(level_acc):
        metrics["probe_level_acc"] = level_acc
    gsd_score = ridge_probe_score(features, torch.log1p(torch.cat(gsd_targets, dim=0)))
    if not math.isnan(gsd_score):
        metrics["probe_gsd_score"] = gsd_score
    components = [metrics[key] for key in ("probe_sensor_acc", "probe_level_acc", "probe_gsd_score") if key in metrics]
    metrics["probe_score"] = float(sum(components) / len(components)) if components else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)

    rank, world_size, local_rank = distributed_state()
    device = choose_device(local_rank)
    maybe_init_distributed(device)
    try:
        seed_everything(args.seed, rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        args.output_dir.mkdir(parents=True, exist_ok=True)

        dataset = build_dataset(args)
        train_loader, train_sampler = build_train_loader(args, dataset, world_size=world_size, rank=rank)
        steps_per_epoch = len(train_loader)
        if steps_per_epoch <= 0:
            raise ValueError("Training loader is empty after drop_last=True. Reduce --batch-size or inspect the dataset.")
        total_steps = args.epochs * steps_per_epoch
        probe_loader = build_probe_loader(args, dataset)

        eo1_bank = load_prototype_bank(args.prototype_bank_eo1)
        gf5_bank = load_prototype_bank(args.prototype_bank_gf5)
        if eo1_bank.centers.shape[0] != gf5_bank.centers.shape[0]:
            raise ValueError("EO1 and GF5 prototype banks must share the same prototype count.")

        config = build_foundation_config(args, int(eo1_bank.centers.shape[0]))
        model = MetadataConditionedFoundationModel(
            config,
            eo1_bank_centers=eo1_bank.centers,
            gf5_bank_centers=gf5_bank.centers,
            eo1_bank_wavelengths=eo1_bank.wavelengths_nm,
            gf5_bank_wavelengths=gf5_bank.wavelengths_nm,
        ).to(device)
        optimizer, optimizer_group_info = build_optimizer(args, model)
        resume_state = maybe_resume(args, model, optimizer)
        if resume_state.epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint epoch {resume_state.epoch} is already at or beyond requested --epochs={args.epochs}."
            )
        if resume_state.step != resume_state.epoch * steps_per_epoch:
            raise ValueError(
                "Resume checkpoint step/epoch metadata does not match the current loader length. "
                "Start a new run or resume from a checkpoint written by the epoch-based trainer on the same dataset setup."
            )
        if args.compile:
            model = torch.compile(model)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        if is_main_process():
            run_config = vars(args).copy()
            run_config["optimizer_groups"] = optimizer_group_info
            run_config["resume_loaded_optimizer"] = resume_state.loaded_optimizer
            run_config["steps_per_epoch"] = steps_per_epoch
            run_config["total_steps"] = total_steps
            with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, ensure_ascii=True, indent=2, default=str)

        raw_model = unwrap_model(model)
        best_ssl = resume_state.best_ssl
        best_probe = resume_state.best_probe
        baseline_probe: dict[str, float] = {}
        initial_weights = resolve_loss_schedule(args, 0)
        if probe_loader is not None:
            if world_size > 1:
                dist.barrier()
            if is_main_process():
                baseline_probe = evaluate_probe(raw_model, probe_loader, device=device, loss_weights=initial_weights)
            if world_size > 1:
                dist.barrier()
            if is_main_process():
                with (args.output_dir / "probe_baseline.json").open("w", encoding="utf-8") as handle:
                    json.dump(baseline_probe, handle, ensure_ascii=True, indent=2)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        global_step = resume_state.step
        session_start_time = time.time()
        for epoch in range(resume_state.epoch, args.epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            last_progress_time = epoch_start_time
            live_status_active = False
            epoch_totals: dict[str, torch.Tensor] = {}
            epoch_loss_totals: dict[str, torch.Tensor] = {}
            epoch_metric_totals: dict[str, torch.Tensor] = {}
            epoch_feature_totals: dict[str, torch.Tensor] = {}
            last_lr = 0.0
            last_applied_lrs: dict[str, float] = {}
            last_weight_schedule = resolve_loss_schedule(args, global_step)
            model.train()
            for step_in_epoch, batch in enumerate(train_loader):
                weight_schedule = resolve_loss_schedule(args, global_step)
                lr = cosine_lr(
                    global_step,
                    base_lr=args.lr,
                    min_lr=args.min_lr,
                    warmup_steps=args.warmup_steps,
                    total_steps=total_steps,
                )
                applied_lrs = set_learning_rate(optimizer, lr)
                batch = move_batch_to_device(batch, device)
                window_start = (step_in_epoch // args.grad_accum_steps) * args.grad_accum_steps
                window_end = min(window_start + args.grad_accum_steps, steps_per_epoch)
                window_size = max(1, window_end - window_start)
                with autocast_context(device, args.precision):
                    output = model(space=batch, loss_weights_override=weight_schedule)
                    loss = output["loss"] / float(window_size)
                loss.backward()
                if (step_in_epoch + 1) == window_end:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                add_scalar_sum(epoch_totals, "loss", output["loss"], device=device)
                for name, value in output["losses"].items():
                    add_scalar_sum(epoch_loss_totals, name, value, device=device)
                for name, value in output["metrics"].items():
                    add_scalar_sum(epoch_metric_totals, name, value, device=device)
                for name, value in output["feature_stats"].items():
                    add_scalar_sum(epoch_feature_totals, name, value, device=device)

                last_lr = lr
                last_applied_lrs = applied_lrs
                last_weight_schedule = weight_schedule
                global_step += 1
                if is_main_process():
                    now = time.time()
                    should_refresh_progress = (
                        (step_in_epoch == 0)
                        or ((step_in_epoch + 1) == steps_per_epoch)
                        or ((now - last_progress_time) >= args.progress_refresh_seconds)
                    )
                    if should_refresh_progress:
                        epoch_fraction = float(step_in_epoch + 1) / float(steps_per_epoch)
                        epoch_elapsed = now - epoch_start_time
                        epoch_eta_seconds = (epoch_elapsed / max(epoch_fraction, 1.0e-6)) - epoch_elapsed
                        completed_fraction = (epoch - resume_state.epoch) + epoch_fraction
                        remaining_fraction = max(0.0, float(args.epochs - resume_state.epoch) - completed_fraction)
                        total_eta_seconds = (now - session_start_time) / max(completed_fraction, 1.0e-6) * remaining_fraction
                        current_loss = float(output["loss"].detach().item())
                        progress_message = (
                            f"[Epoch {epoch + 1}/{args.epochs}] "
                            f"{step_in_epoch + 1}/{steps_per_epoch} "
                            f"{render_progress_bar(epoch_fraction)} "
                            f"{epoch_fraction * 100:5.1f}% "
                            f"loss~{current_loss:.4f} "
                            f"lr {lr:.2e} "
                            f"eta_epoch {format_duration(epoch_eta_seconds)} "
                            f"eta_total {format_duration(total_eta_seconds)}"
                        )
                        write_live_status(progress_message)
                        last_progress_time = now
                        live_status_active = True

            reduced_epoch = reduce_scalar_sums(epoch_totals, world_size=world_size)
            reduced_losses = reduce_scalar_sums(epoch_loss_totals, world_size=world_size)
            reduced_metrics = reduce_scalar_sums(epoch_metric_totals, world_size=world_size)
            reduced_feature_stats = reduce_scalar_sums(epoch_feature_totals, world_size=world_size)
            epoch_seconds = time.time() - epoch_start_time
            elapsed_seconds = time.time() - session_start_time
            completed_epochs = epoch - resume_state.epoch + 1
            mean_epoch_seconds = elapsed_seconds / max(1, completed_epochs)
            eta_seconds = mean_epoch_seconds * max(0, args.epochs - (epoch + 1))
            payload: dict[str, float | int | str | bool] = {
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "stage": "foundation",
                "lr": last_lr,
                "loss": reduced_epoch.get("loss", 0.0) / float(steps_per_epoch),
                "epoch_seconds": epoch_seconds,
                "steps_per_second": steps_per_epoch / max(epoch_seconds, 1.0e-6),
                "eta_seconds": eta_seconds,
                "eta": format_duration(eta_seconds),
                **{f"loss_{name}": value / float(steps_per_epoch) for name, value in reduced_losses.items()},
                **{f"metric_{name}": value / float(steps_per_epoch) for name, value in reduced_metrics.items()},
                **{f"feature_{name}": value / float(steps_per_epoch) for name, value in reduced_feature_stats.items()},
                **{f"lr_{name}": value for name, value in last_applied_lrs.items()},
                **{f"weight_{name}": value for name, value in last_weight_schedule.items()},
            }

            should_probe = probe_loader is not None and (
                ((epoch + 1) % args.probe_every_epochs == 0) or (epoch + 1) == args.epochs
            )
            if should_probe:
                if is_main_process() and live_status_active:
                    write_live_status(
                        f"[Epoch {epoch + 1}/{args.epochs}] training complete, running probe..."
                    )
                if world_size > 1:
                    dist.barrier()
                probe_metrics = (
                    evaluate_probe(raw_model, probe_loader, device=device, loss_weights=last_weight_schedule)
                    if is_main_process()
                    else {}
                )
                if world_size > 1:
                    dist.barrier()
                if is_main_process():
                    payload.update(probe_metrics)
                    for key, value in baseline_probe.items():
                        payload[f"baseline_{key}"] = value
                    checkpoint_names: list[str] = []
                    ssl_value = probe_metrics.get("probe_ssl", float("nan"))
                    if not math.isnan(ssl_value) and ssl_value < best_ssl:
                        best_ssl = ssl_value
                        checkpoint_names.append("best_ssl")
                    probe_score = probe_metrics.get("probe_score", float("nan"))
                    if not math.isnan(probe_score) and probe_score > best_probe:
                        best_probe = probe_score
                        checkpoint_names.append("best_probe")
                    if checkpoint_names:
                        best_payload = build_checkpoint_payload(
                            epoch=epoch + 1,
                            step=global_step,
                            model=model,
                            optimizer=optimizer,
                            args=args,
                            best_ssl=best_ssl,
                            best_probe=best_probe,
                        )
                        for name in checkpoint_names:
                            save_named_checkpoint(args.output_dir, best_payload, name)
            if is_main_process():
                if live_status_active:
                    clear_live_status()
                print(json.dumps(payload, ensure_ascii=True))
                append_log(args.output_dir, payload)

            should_save = ((epoch + 1) % args.save_every_epochs == 0) or (epoch + 1) == args.epochs
            if should_save and is_main_process():
                checkpoint_payload = build_checkpoint_payload(
                    epoch=epoch + 1,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    best_ssl=best_ssl,
                    best_probe=best_probe,
                )
                save_checkpoint(args.output_dir, checkpoint_payload, epoch + 1)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
