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
from aware_open.data.pretrain import (
    SensorTemperatureDistributedSampler,
    Sigma450KShardDataset,
    Sigma450KShardDistributedSampler,
    collate_modality_samples,
    move_batch_to_device,
)
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
    parser = argparse.ArgumentParser(description="Spectral-fidelity continued foundation pretraining.")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/Data/SpectralData"))
    parser.add_argument("--sigma-shard-root", type=Path, default=None)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--prototype-bank-dz1-vnir", type=Path, default=None)
    parser.add_argument("--prototype-bank-dz1-swir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Load model weights but reset epoch, step, and optimizer.")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument(
        "--allow-resume-loader-length-mismatch",
        action="store_true",
        help=(
            "Allow epoch checkpoint resume after changing batch size or world-size, "
            "and remap global step to epoch * current_steps_per_epoch."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="Fixed optimizer-step budget; 0 keeps epoch-based behavior.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--sensor-sampling-temperature",
        type=float,
        default=0.5,
        help=(
            "Sensor exponent alpha. alpha=1 uses the shard-local natural sampler "
            "to preserve sequential I/O; other non-negative values use temperature resampling."
        ),
    )
    parser.add_argument("--sampler-epoch-size", type=int, default=0)
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
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help=(
            "Enable DDP unused-parameter detection. Keep this off for normal pretraining, "
            "but turn it on for ablations that disable prototype, metadata, or frequency branches."
        ),
    )
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
    parser.add_argument("--reconstruction-low-weight", type=float, default=0.10)
    parser.add_argument("--reconstruction-mid-weight", type=float, default=0.20)
    parser.add_argument("--reconstruction-full-weight", type=float, default=0.70)
    parser.add_argument("--metadata-ablation", choices=("full", "no_sample", "no_band", "none"), default="full")
    parser.add_argument("--prototype-temperature-teacher", type=float, default=0.07)
    parser.add_argument("--prototype-temperature-student", type=float, default=0.07)
    parser.add_argument("--proto-kl-weight", type=float, default=0.10)
    parser.add_argument("--proto-balance-weight", type=float, default=0.02)
    parser.add_argument("--warmup-steps-loss", type=int, default=40_000)
    parser.add_argument("--original-loss-weight", type=float, default=0.55)
    parser.add_argument("--sam-loss-weight", type=float, default=0.20)
    parser.add_argument("--charbonnier-loss-weight", type=float, default=0.15)
    parser.add_argument("--derivative-loss-weight", type=float, default=0.05)
    parser.add_argument("--band-dropout-loss-weight", type=float, default=0.05)
    parser.add_argument("--band-dropout-ratio", type=float, default=0.15)
    parser.add_argument("--spectral-loss-eps", type=float, default=1.0e-3)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")
    if (args.prototype_bank_dz1_vnir is None) != (args.prototype_bank_dz1_swir is None):
        raise ValueError("Provide both DZ1 prototype banks or neither.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative.")
    if args.sensor_sampling_temperature < 0.0:
        raise ValueError("--sensor-sampling-temperature must be non-negative.")
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
    reconstruction_weights = (
        args.reconstruction_low_weight,
        args.reconstruction_mid_weight,
        args.reconstruction_full_weight,
    )
    if any(weight < 0.0 for weight in reconstruction_weights):
        raise ValueError("Reconstruction loss weights must be non-negative.")
    if sum(reconstruction_weights) <= 0.0:
        raise ValueError("At least one reconstruction loss weight must be positive.")
    spectral_weights = (
        args.original_loss_weight,
        args.sam_loss_weight,
        args.charbonnier_loss_weight,
        args.derivative_loss_weight,
        args.band_dropout_loss_weight,
    )
    if any(weight < 0.0 for weight in spectral_weights):
        raise ValueError("Spectral continued-pretraining loss weights must be non-negative.")
    if sum(spectral_weights) <= 0.0:
        raise ValueError("At least one continued-pretraining loss weight must be positive.")
    if not 0.0 < args.band_dropout_ratio <= 1.0:
        raise ValueError(f"--band-dropout-ratio must be in (0, 1], got {args.band_dropout_ratio}.")
    if args.spectral_loss_eps <= 0.0:
        raise ValueError(f"--spectral-loss-eps must be positive, got {args.spectral_loss_eps}.")


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
        metadata_ablation=args.metadata_ablation,
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
            reconstruction_low=args.reconstruction_low_weight,
            reconstruction_mid=args.reconstruction_mid_weight,
            reconstruction_full=args.reconstruction_full_weight,
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
        manifest_path=args.manifest_path,
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
    if math.isclose(args.sensor_sampling_temperature, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        # Natural sampling does not require individual-index resampling.  Keep
        # samples contiguous inside shuffled shards so large mmap files are not
        # reopened for nearly every item in a batch.
        sampler: Sampler | None = Sigma450KShardDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    elif args.sensor_sampling_temperature >= 0.0:
        sampler = SensorTemperatureDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            temperature=args.sensor_sampling_temperature,
            epoch_size=None if args.sampler_epoch_size <= 0 else args.sampler_epoch_size,
            seed=args.seed,
            drop_last=True,
        )
    elif world_size > 1:
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
        "reconstruction_low": args.reconstruction_low_weight,
        "reconstruction_mid": args.reconstruction_mid_weight,
        "reconstruction_full": args.reconstruction_full_weight,
        "proto_kl": args.proto_kl_weight,
        "proto_balance": args.proto_balance_weight,
    }


def masked_patch_mean(values: torch.Tensor, mask: torch.Tensor, pixel_mask: torch.Tensor | None = None) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1).expand_as(values)
    if pixel_mask is not None:
        weights = weights * pixel_mask.float()
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def spectral_channel_mask(batch) -> torch.Tensor:
    if batch.valid_mask is None:
        return torch.ones(batch.batch_size, batch.values.shape[1], dtype=torch.bool, device=batch.values.device)
    return batch.valid_mask.bool().to(batch.values.device)


def wavelength_delta_scale(batch, channel_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair_mask = channel_mask[:, 1:] & channel_mask[:, :-1]
    deltas = torch.diff(batch.wavelengths.float().to(batch.values.device), dim=1).abs().clamp_min(1.0)
    valid_deltas = deltas * pair_mask.float()
    mean_delta = valid_deltas.sum(dim=1) / pair_mask.float().sum(dim=1).clamp_min(1.0)
    scale = (mean_delta.unsqueeze(1) / deltas).clamp(0.25, 4.0)
    scale = torch.where(pair_mask, scale, torch.zeros_like(scale))
    return scale.unsqueeze(1).unsqueeze(1), pair_mask.unsqueeze(1).unsqueeze(1)


def random_band_focus_mask(channel_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    scores = torch.rand(channel_mask.shape, device=channel_mask.device)
    selected = (scores < ratio) & channel_mask
    has_selection = selected.any(dim=1, keepdim=True)
    masked_scores = scores.masked_fill(~channel_mask, float("inf"))
    fallback = masked_scores == masked_scores.min(dim=1, keepdim=True).values
    return selected | (~has_selection & fallback & channel_mask)


def spectral_reconstruction_losses(
    model: MetadataConditionedFoundationModel,
    output: dict[str, torch.Tensor | dict[str, torch.Tensor] | tuple[int, int]],
    batch,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prediction = output["prediction"]["full"]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Expected output['prediction']['full'] to be a tensor.")
    mask = output["mask"]
    if not isinstance(mask, torch.Tensor):
        raise TypeError("Expected output['mask'] to be a tensor.")

    channels = int(batch.values.shape[1])
    target = model.patchify(batch.values, model.config.encoder.local_patch_size)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target patch shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}.")
    if prediction.shape[-1] % channels != 0:
        raise ValueError(f"Patch feature dimension {prediction.shape[-1]} is not divisible by channel count {channels}.")

    batch_size, token_count, _ = prediction.shape
    patch_area = prediction.shape[-1] // channels
    pred = prediction.float().reshape(batch_size, token_count, patch_area, channels)
    tgt = target.float().reshape(batch_size, token_count, patch_area, channels)
    channel_mask = spectral_channel_mask(batch)
    valid = channel_mask.unsqueeze(1).unsqueeze(1)
    pixel_mask = valid.any(dim=-1).expand(batch_size, token_count, patch_area)
    eps = float(args.spectral_loss_eps)

    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}

    pred_valid = pred * valid.float()
    tgt_valid = tgt * valid.float()
    dot = (pred_valid * tgt_valid).sum(dim=-1)
    pred_norm = pred_valid.square().sum(dim=-1).clamp_min(eps).sqrt()
    tgt_norm = tgt_valid.square().sum(dim=-1).clamp_min(eps).sqrt()
    cosine = (dot / (pred_norm * tgt_norm).clamp_min(eps)).clamp(-1.0, 1.0)
    sam_loss = 1.0 - cosine
    losses["sam"] = masked_patch_mean(sam_loss, mask, pixel_mask=pixel_mask)
    metrics["spectral_cosine"] = masked_patch_mean(cosine, mask, pixel_mask=pixel_mask)

    charbonnier = torch.sqrt((pred - tgt).square() + eps * eps)
    channel_count = valid.float().sum(dim=-1).clamp_min(1.0)
    charbonnier = (charbonnier * valid.float()).sum(dim=-1) / channel_count
    losses["charbonnier_spectral"] = masked_patch_mean(charbonnier, mask, pixel_mask=pixel_mask)

    if channels > 1:
        scale, pair_mask = wavelength_delta_scale(batch, channel_mask)
        derivative_delta = ((pred[..., 1:] - pred[..., :-1]) - (tgt[..., 1:] - tgt[..., :-1])) * scale
        derivative = torch.sqrt(derivative_delta.square() + eps * eps)
        pair_count = pair_mask.float().sum(dim=-1).clamp_min(1.0)
        derivative = (derivative * pair_mask.float()).sum(dim=-1) / pair_count
        losses["derivative"] = masked_patch_mean(derivative, mask, pixel_mask=pair_mask.any(dim=-1).expand_as(pixel_mask))
    else:
        losses["derivative"] = prediction.new_zeros(())

    band_mask = random_band_focus_mask(channel_mask, args.band_dropout_ratio)
    band_valid = band_mask.unsqueeze(1).unsqueeze(1)
    band_count = band_valid.float().sum(dim=-1).clamp_min(1.0)
    band_charbonnier = (torch.sqrt((pred - tgt).square() + eps * eps) * band_valid.float()).sum(dim=-1) / band_count
    losses["band_dropout_denoising"] = masked_patch_mean(
        band_charbonnier,
        mask,
        pixel_mask=band_valid.any(dim=-1).expand_as(pixel_mask),
    )
    metrics["band_dropout_fraction"] = band_mask.float().sum() / channel_mask.float().sum().clamp_min(1.0)

    return losses, metrics


def continued_pretraining_loss(
    original_loss: torch.Tensor,
    spectral_losses: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weighted_components = {
        "original": original_loss * args.original_loss_weight,
        "sam": spectral_losses["sam"] * args.sam_loss_weight,
        "charbonnier_spectral": spectral_losses["charbonnier_spectral"] * args.charbonnier_loss_weight,
        "derivative": spectral_losses["derivative"] * args.derivative_loss_weight,
        "band_dropout_denoising": spectral_losses["band_dropout_denoising"] * args.band_dropout_loss_weight,
    }
    total = None
    for value in weighted_components.values():
        total = value if total is None else total + value
    if total is None:
        total = original_loss.new_zeros(())
    return total, weighted_components


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
    metadata = {
        "checkpoint": str(path),
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
        "stage_state": checkpoint_safe_value(payload.get("stage_state", {})),
    }
    with path.with_suffix(path.suffix + ".meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)


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


def load_best_value_from_log(output_dir: Path, key: str, *, mode: str, default: float) -> float:
    log_path = output_dir / "train_log.jsonl"
    if not log_path.exists():
        return default
    best = default
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line).get(key)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            value = float(value)
            if mode == "min":
                best = min(best, value)
            elif mode == "max":
                best = max(best, value)
            else:
                raise ValueError(f"Unsupported best-value mode {mode!r}.")
    return best


def load_foundation_state_dict(model: MetadataConditionedFoundationModel, state_dict: dict[str, torch.Tensor]) -> None:
    migrated = dict(state_dict)
    target = model.state_dict()
    sensor_keys = [key for key in migrated if key.endswith("sample_encoder.sensor_embed.weight")]
    if len(sensor_keys) == 1:
        key = sensor_keys[0]
        source = migrated[key]
        expected = target[key]
        if source.shape[1:] == expected.shape[1:] and source.shape[0] < expected.shape[0]:
            expanded = expected.clone()
            expanded[: source.shape[0]].copy_(source)
            others_index = min(10, source.shape[0] - 1)
            expanded[source.shape[0] :].copy_(
                source[others_index].view(1, -1).expand(expected.shape[0] - source.shape[0], -1)
            )
            migrated[key] = expanded
    result = model.load_state_dict(migrated, strict=False)
    allowed_missing = {
        key
        for key in result.missing_keys
        if key.startswith("extra_dz1_") and (key.endswith("_bank_centers") or key.endswith("_bank_wavelengths"))
    }
    missing = [key for key in result.missing_keys if key not in allowed_missing]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing: {missing}, Unexpected: {result.unexpected_keys}."
        )


def maybe_resume(args: argparse.Namespace, model: MetadataConditionedFoundationModel, optimizer: torch.optim.Optimizer) -> ResumeState:
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_foundation_state_dict(model, payload["model"])
        return ResumeState()
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
    if not args.reset_optimizer and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
            loaded_optimizer = True
        except (KeyError, ValueError):
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
        total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
        effective_epochs = math.ceil(total_steps / steps_per_epoch) if args.max_steps > 0 else args.epochs
        probe_loader = build_probe_loader(args, dataset)

        eo1_bank = load_prototype_bank(args.prototype_bank_eo1)
        gf5_bank = load_prototype_bank(args.prototype_bank_gf5)
        if eo1_bank.centers.shape[0] != gf5_bank.centers.shape[0]:
            raise ValueError("EO1 and GF5 prototype banks must share the same prototype count.")
        extra_banks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if args.prototype_bank_dz1_vnir is not None:
            dz1_vnir_bank = load_prototype_bank(args.prototype_bank_dz1_vnir)
            dz1_swir_bank = load_prototype_bank(args.prototype_bank_dz1_swir)
            for sensor, bank in (("dz1_vnir", dz1_vnir_bank), ("dz1_swir", dz1_swir_bank)):
                if bank.centers.shape[0] != eo1_bank.centers.shape[0]:
                    raise ValueError(f"{sensor} prototype count does not match EO1/GF5.")
                extra_banks[sensor] = (bank.centers, bank.wavelengths_nm)

        config = build_foundation_config(args, int(eo1_bank.centers.shape[0]))
        model = MetadataConditionedFoundationModel(
            config,
            eo1_bank_centers=eo1_bank.centers,
            gf5_bank_centers=gf5_bank.centers,
            eo1_bank_wavelengths=eo1_bank.wavelengths_nm,
            gf5_bank_wavelengths=gf5_bank.wavelengths_nm,
            extra_prototype_banks=extra_banks,
        ).to(device)
        optimizer, optimizer_group_info = build_optimizer(args, model)
        resume_state = maybe_resume(args, model, optimizer)
        if args.max_steps <= 0 and resume_state.epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint epoch {resume_state.epoch} is already at or beyond requested --epochs={args.epochs}."
            )
        expected_resume_step = resume_state.epoch * steps_per_epoch
        if args.resume is not None and resume_state.step != expected_resume_step:
            if not args.allow_resume_loader_length_mismatch:
                raise ValueError(
                    "Resume checkpoint step/epoch metadata does not match the current loader length. "
                    "Start a new run or resume from a checkpoint written by the epoch-based trainer on the same dataset setup."
                )
            if is_main_process():
                print(
                    "Resume checkpoint step/epoch metadata does not match the current loader length; "
                    f"remapping global_step {resume_state.step} -> {expected_resume_step} "
                    f"for epoch {resume_state.epoch}.",
                    flush=True,
                )
            resume_state.step = expected_resume_step
        if args.compile:
            model = torch.compile(model)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=args.find_unused_parameters,
            )

        if is_main_process():
            run_config = vars(args).copy()
            run_config["optimizer_groups"] = optimizer_group_info
            run_config["resume_loaded_optimizer"] = resume_state.loaded_optimizer
            run_config["steps_per_epoch"] = steps_per_epoch
            run_config["total_steps"] = total_steps
            run_config["effective_epochs"] = effective_epochs
            run_config["dataset_sensor_counts"] = {
                sensor: sum(shard.count for shard in dataset.shards if shard.sensor == sensor)
                for sensor in sorted({shard.sensor for shard in dataset.shards})
            }
            with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, ensure_ascii=True, indent=2, default=str)

        raw_model = unwrap_model(model)
        best_ssl = resume_state.best_ssl
        best_probe = resume_state.best_probe
        best_total = load_best_value_from_log(args.output_dir, "loss", mode="min", default=math.inf)
        best_sam = load_best_value_from_log(args.output_dir, "loss_sam", mode="min", default=math.inf)
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
        stop_training = False
        for epoch in range(resume_state.epoch, effective_epochs):
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
            steps_this_epoch = 0
            for step_in_epoch, batch in enumerate(train_loader):
                if global_step >= total_steps:
                    stop_training = True
                    break
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
                    spectral_losses, spectral_metrics = spectral_reconstruction_losses(raw_model, output, batch, args)
                    total_loss, weighted_components = continued_pretraining_loss(output["loss"], spectral_losses, args)
                    loss = total_loss / float(window_size)
                loss.backward()
                if (step_in_epoch + 1) == window_end:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                add_scalar_sum(epoch_totals, "loss", total_loss, device=device)
                add_scalar_sum(epoch_loss_totals, "original", output["loss"], device=device)
                for name, value in output["losses"].items():
                    add_scalar_sum(epoch_loss_totals, name, value, device=device)
                for name, value in spectral_losses.items():
                    add_scalar_sum(epoch_loss_totals, name, value, device=device)
                for name, value in weighted_components.items():
                    add_scalar_sum(epoch_loss_totals, f"weighted_{name}", value, device=device)
                for name, value in output["metrics"].items():
                    add_scalar_sum(epoch_metric_totals, name, value, device=device)
                for name, value in spectral_metrics.items():
                    add_scalar_sum(epoch_metric_totals, name, value, device=device)
                for name, value in output["feature_stats"].items():
                    add_scalar_sum(epoch_feature_totals, name, value, device=device)

                last_lr = lr
                last_applied_lrs = applied_lrs
                last_weight_schedule = weight_schedule
                global_step += 1
                steps_this_epoch += 1
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
                        remaining_fraction = max(0.0, float(effective_epochs - resume_state.epoch) - completed_fraction)
                        total_eta_seconds = (now - session_start_time) / max(completed_fraction, 1.0e-6) * remaining_fraction
                        current_loss = float(total_loss.detach().item())
                        progress_message = (
                            f"[Epoch {epoch + 1}/{effective_epochs}] "
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

            if steps_this_epoch <= 0:
                break
            reduced_epoch = reduce_scalar_sums(epoch_totals, world_size=world_size)
            reduced_losses = reduce_scalar_sums(epoch_loss_totals, world_size=world_size)
            reduced_metrics = reduce_scalar_sums(epoch_metric_totals, world_size=world_size)
            reduced_feature_stats = reduce_scalar_sums(epoch_feature_totals, world_size=world_size)
            epoch_seconds = time.time() - epoch_start_time
            elapsed_seconds = time.time() - session_start_time
            completed_epochs = epoch - resume_state.epoch + 1
            mean_epoch_seconds = elapsed_seconds / max(1, completed_epochs)
            eta_seconds = mean_epoch_seconds * max(0, effective_epochs - (epoch + 1))
            is_final_step = global_step >= total_steps
            payload: dict[str, float | int | str | bool] = {
                "epoch": epoch + 1,
                "epochs": effective_epochs,
                "step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "steps_this_epoch": steps_this_epoch,
                "stage": "foundation",
                "lr": last_lr,
                "loss": reduced_epoch.get("loss", 0.0) / float(steps_this_epoch),
                "epoch_seconds": epoch_seconds,
                "steps_per_second": steps_this_epoch / max(epoch_seconds, 1.0e-6),
                "eta_seconds": eta_seconds,
                "eta": format_duration(eta_seconds),
                **{f"loss_{name}": value / float(steps_this_epoch) for name, value in reduced_losses.items()},
                **{f"metric_{name}": value / float(steps_this_epoch) for name, value in reduced_metrics.items()},
                **{f"feature_{name}": value / float(steps_this_epoch) for name, value in reduced_feature_stats.items()},
                **{f"lr_{name}": value for name, value in last_applied_lrs.items()},
                **{f"weight_{name}": value for name, value in last_weight_schedule.items()},
                "weight_continue_original": args.original_loss_weight,
                "weight_continue_sam": args.sam_loss_weight,
                "weight_continue_charbonnier_spectral": args.charbonnier_loss_weight,
                "weight_continue_derivative": args.derivative_loss_weight,
                "weight_continue_band_dropout_denoising": args.band_dropout_loss_weight,
                "band_dropout_ratio": args.band_dropout_ratio,
            }

            should_probe = probe_loader is not None and (
                ((epoch + 1) % args.probe_every_epochs == 0) or is_final_step
            )
            if should_probe:
                if is_main_process() and live_status_active:
                    write_live_status(
                        f"[Epoch {epoch + 1}/{effective_epochs}] training complete, running probe..."
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
                epoch_checkpoint_names: list[str] = []
                total_value = float(payload.get("loss", float("nan")))
                if not math.isnan(total_value) and total_value < best_total:
                    best_total = total_value
                    epoch_checkpoint_names.append("best_total")
                sam_value = float(payload.get("loss_sam", float("nan")))
                if not math.isnan(sam_value) and sam_value < best_sam:
                    best_sam = sam_value
                    epoch_checkpoint_names.append("best_sam")
                if epoch_checkpoint_names:
                    epoch_best_payload = build_checkpoint_payload(
                        epoch=epoch + 1,
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        best_ssl=best_ssl,
                        best_probe=best_probe,
                    )
                    for name in epoch_checkpoint_names:
                        save_named_checkpoint(args.output_dir, epoch_best_payload, name)
                if live_status_active:
                    clear_live_status()
                print(json.dumps(payload, ensure_ascii=True))
                append_log(args.output_dir, payload)

            should_save = ((epoch + 1) % args.save_every_epochs == 0) or is_final_step
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
            if is_final_step or stop_training:
                break
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
