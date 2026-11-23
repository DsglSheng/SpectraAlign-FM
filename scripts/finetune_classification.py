from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aware_open.downstream import (
    AwareFeatureExtractor,
    SCENE_CLASS_NAMES,
    build_downstream_metadata,
    build_segmentation_scene,
    merge_tile_predictions,
    metrics_from_predictions,
    resolve_scene_config,
    seed_everything,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPERSIGMA_ROOT = REPO_ROOT.parent / "HyperSIGMA-main" / "ImageClassification" / "data"


class AwareSSFusionSeg(nn.Module):
    """HyperSIGMA ss_fusion_seg-style dense classifier backed by Aware pretraining."""

    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        num_classes: int,
        *,
        feature_layers: int = 4,
        fusion_channels: int = 128,
        class_prototypes: torch.Tensor | None = None,
        class_prototype_weight: float = 0.0,
        class_prototype_temperature: float = 0.10,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.feature_layers = int(feature_layers)
        self.fusion_channels = int(fusion_channels)
        self.embed_dim = extractor.model_dim
        self.spectral_gate_dim = 128
        self.class_prototype_weight = float(class_prototype_weight)
        self.class_prototype_temperature = max(float(class_prototype_temperature), 1.0e-6)
        if class_prototypes is None or self.class_prototype_weight <= 0.0:
            self.register_buffer("class_prototypes", torch.empty(0), persistent=False)
        else:
            self.register_buffer("class_prototypes", class_prototypes.float(), persistent=True)

        self.DR1 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.DR2 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.DR3 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.DR4 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(fusion_channels * 4, fusion_channels, kernel_size=1, bias=True)
        self.cls = nn.Conv2d(fusion_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.spec_reduce = nn.Sequential(
            nn.LayerNorm(extractor.band_dim),
            nn.Linear(extractor.band_dim, self.spectral_gate_dim, bias=False),
            nn.GELU(),
        )
        self.meta_reduce = nn.Sequential(
            nn.LayerNorm(extractor.context_dim),
            nn.Linear(extractor.context_dim, self.spectral_gate_dim, bias=False),
            nn.GELU(),
        )
        self.fc_spec1 = self._make_gate()
        self.fc_spec2 = self._make_gate()
        self.fc_spec3 = self._make_gate()
        self.fc_spec4 = self._make_gate()

    def _make_gate(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.spectral_gate_dim, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embed_dim, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _token_map(tokens: torch.Tensor, grid_size: tuple[int, int], output_size: tuple[int, int]) -> torch.Tensor:
        grid_h, grid_w = grid_size
        feature = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], grid_h, grid_w)
        return F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, *, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        features = self.extractor.forward_features(x)
        intermediate = list(features["intermediate_tokens"])
        if len(intermediate) < self.feature_layers:
            raise RuntimeError(
                f"Aware checkpoint exposes {len(intermediate)} intermediate maps, "
                f"but the segmentation head needs {self.feature_layers}."
            )
        selected = intermediate[-self.feature_layers :]
        grid_size = features["grid_size"]
        output_size = (int(x.shape[-2]), int(x.shape[-1]))
        feature_maps = [self._token_map(tokens, grid_size, output_size) for tokens in selected]
        img_feature1, img_feature2, img_feature3, img_feature4 = feature_maps

        spectral_summary = features["band_tokens"].mean(dim=1)
        spectral_context = self.spec_reduce(spectral_summary) + self.meta_reduce(features["sample_context"])
        spec_weights1 = self.fc_spec1(spectral_context).view(x.shape[0], -1, 1, 1)
        spec_weights2 = self.fc_spec2(spectral_context).view(x.shape[0], -1, 1, 1)
        spec_weights3 = self.fc_spec3(spectral_context).view(x.shape[0], -1, 1, 1)
        spec_weights4 = self.fc_spec4(spectral_context).view(x.shape[0], -1, 1, 1)

        ss_feature1 = self.DR1((1.0 + spec_weights1) * img_feature1)
        ss_feature2 = self.DR2((1.0 + spec_weights2) * img_feature2)
        ss_feature3 = self.DR3((1.0 + spec_weights3) * img_feature3)
        ss_feature4 = self.DR4((1.0 + spec_weights4) * img_feature4)

        ss_feature = torch.cat((ss_feature1, ss_feature2, ss_feature3, ss_feature4), dim=1)
        ss_feature = self.conv(ss_feature)
        logits_map = self.cls(ss_feature)
        logits = logits_map.permute(0, 2, 3, 1).reshape(x.shape[0], output_size[0] * output_size[1], -1)
        if self.class_prototype_weight > 0.0 and self.class_prototypes.numel() > 0:
            pixel_spectra = x.permute(0, 2, 3, 1).reshape(x.shape[0], output_size[0] * output_size[1], x.shape[1]).float()
            pixel_spectra = F.normalize(pixel_spectra, dim=-1)
            prototypes = F.normalize(self.class_prototypes.float(), dim=-1)
            proto_logits = torch.matmul(pixel_spectra, prototypes.t()) / self.class_prototype_temperature
            logits = logits + self.class_prototype_weight * proto_logits.to(dtype=logits.dtype)
        if return_features:
            return logits, features
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Aware with HyperSIGMA-style dense HSI classification.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key or alias, e.g. ip, pu, hc, hh, hu, zy.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HYPERSIGMA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--overlap-size", type=int, default=None)
    parser.add_argument("--train-num", type=int, default=None)
    parser.add_argument("--val-num", type=int, default=None)
    parser.add_argument("--split-mode", choices=("number", "official", "ratio"), default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scheduler-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay-rate", type=float, default=0.9)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--feature-layers", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--gsd", type=float, default=None)
    parser.add_argument("--normalization", choices=("none", "minmax", "mean_std"), default="minmax")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--pca-whiten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weighting", choices=("none", "inverse", "inverse_sqrt"), default="none")
    parser.add_argument("--proto-bank", choices=("none", "eo1", "gf5"), default="none")
    parser.add_argument("--proto-kl-weight", type=float, default=0.01)
    parser.add_argument("--proto-balance-weight", type=float, default=0.001)
    parser.add_argument("--proto-warmup-epochs", type=int, default=10)
    parser.add_argument("--class-prototype-weight", type=float, default=0.0)
    parser.add_argument("--class-prototype-temperature", type=float, default=0.10)
    parser.add_argument("--train-repeat-factor", type=int, default=None)
    parser.add_argument("--train-spatial-jitter", type=int, default=None)
    parser.add_argument("--centered-training", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sampling", choices=("shuffle", "inverse", "inverse_sqrt"), default="shuffle")
    parser.add_argument("--train-drop-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-drop-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-val-ratio", type=float, default=None)
    parser.add_argument("--eval-tta", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--init-model", type=Path, default=None, help="Load compatible model weights before starting a fresh optimizer schedule.")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.freeze_encoder:
        print("[Aware] --freeze-encoder is disabled for fair full-parameter fine-tuning; using freeze_encoder=False.")
        args.freeze_encoder = False
    return args


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels != 0
    if int(valid.sum().item()) == 0:
        return None, flat_logits.new_empty(0, flat_logits.shape[-1]), flat_labels.new_empty(0)
    target = flat_labels[valid] - 1
    loss = F.cross_entropy(
        flat_logits[valid],
        target.long(),
        label_smoothing=label_smoothing,
        weight=class_weights,
    )
    return loss, flat_logits[valid], target


def build_class_weights_from_labels(
    label_map: np.ndarray,
    class_count: int,
    mode: str,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    resolved_mode = str(mode).lower()
    if resolved_mode == "none":
        return None, {"mode": "none"}
    labels = label_map.reshape(-1)
    valid = labels[labels > 0].astype(np.int64, copy=False)
    if valid.size == 0:
        return None, {"mode": "none", "note": "empty_train_labels"}
    counts = np.bincount(valid, minlength=class_count + 1)[1:].astype(np.float64, copy=False)
    counts[counts <= 0.0] = 1.0
    if resolved_mode == "inverse":
        weights = 1.0 / counts
    elif resolved_mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class weighting mode {mode!r}.")
    weights = weights / max(float(weights.mean()), 1.0e-12)
    info = {
        "mode": resolved_mode,
        "counts": counts.astype(np.int64).tolist(),
        "weights": weights.astype(np.float32).tolist(),
    }
    return torch.tensor(weights, dtype=torch.float32), info


def build_class_spectral_prototypes(
    cube: np.ndarray,
    train_label_map: np.ndarray,
    class_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    flat_cube = cube.reshape(-1, cube.shape[-1]).astype(np.float32, copy=False)
    flat_labels = train_label_map.reshape(-1).astype(np.int64, copy=False)
    global_mean = flat_cube[flat_labels > 0].mean(axis=0) if np.any(flat_labels > 0) else flat_cube.mean(axis=0)
    prototypes = []
    counts = []
    missing = []
    for class_id in range(1, class_count + 1):
        mask = flat_labels == class_id
        count = int(mask.sum())
        counts.append(count)
        if count > 0:
            prototypes.append(flat_cube[mask].mean(axis=0))
        else:
            missing.append(class_id)
            prototypes.append(global_mean)
    stacked = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(stacked), {
        "enabled": True,
        "source": "training_pixels_only",
        "counts": counts,
        "missing_classes": missing,
    }


@torch.no_grad()
def evaluate_tiles(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
    precision: str,
    *,
    class_names: tuple[str, ...],
) -> tuple[dict[str, Any], list[np.ndarray]]:
    model.eval()
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    tile_predictions: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(x)
            loss, selected_logits, selected_target = masked_cross_entropy(logits, y)
        pred = logits.argmax(dim=-1).reshape(y.shape) + 1
        tile_predictions.extend(pred.cpu().numpy())
        if loss is None:
            continue
        batch_count = int(selected_target.numel())
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
        all_pred.append(selected_logits.argmax(dim=1).cpu().numpy())
        all_label.append(selected_target.cpu().numpy())
    if all_pred:
        pred_np = np.concatenate(all_pred)
        label_np = np.concatenate(all_label)
    else:
        pred_np = np.empty((0,), dtype=np.int64)
        label_np = np.empty((0,), dtype=np.int64)
    return (
        metrics_from_predictions(
            label_np,
            pred_np,
            class_count,
            total_loss / max(total_count, 1),
            class_names=class_names,
        ),
        tile_predictions,
    )


def evaluate_merged_test(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
    precision: str,
    class_names: tuple[str, ...],
    starts: list[tuple[int, int]],
    height: int,
    width: int,
    padded_height: int,
    padded_width: int,
    img_size: int,
    overlap_size: int,
    test_gt: np.ndarray,
) -> dict[str, Any]:
    _, tile_predictions = evaluate_tiles(
        model,
        loader,
        device,
        class_count,
        precision,
        class_names=class_names,
    )
    merged = merge_tile_predictions(
        tile_predictions,
        starts,
        height=padded_height,
        width=padded_width,
        img_size=img_size,
        overlap_size=overlap_size,
    )
    merged = merged[:height, :width]
    valid = test_gt.reshape(-1) != 0
    labels = test_gt.reshape(-1)[valid] - 1
    predictions = merged.reshape(-1)[valid] - 1
    return metrics_from_predictions(labels, predictions, class_count, 0.0, class_names=class_names)


def logits_to_map(logits: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
    height, width = spatial_size
    return logits.view(logits.shape[0], height, width, logits.shape[-1])


def infer_tile_logits(
    model: nn.Module,
    x: torch.Tensor,
    *,
    precision: str,
    eval_tta: bool,
) -> torch.Tensor:
    autocast_enabled = x.device.type == "cuda" and precision == "bf16"
    transforms = [
        ("none", None),
        ("flip_w", (-1,)),
        ("flip_h", (-2,)),
        ("flip_hw", (-2, -1)),
    ] if eval_tta else [("none", None)]
    accumulated = None
    for _, dims in transforms:
        aug_x = torch.flip(x, dims=dims) if dims is not None else x
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(aug_x)
        logits_map = logits_to_map(logits, (int(x.shape[-2]), int(x.shape[-1]))).float()
        if dims is not None:
            reverse_dims = []
            if -2 in dims:
                reverse_dims.append(1)
            if -1 in dims:
                reverse_dims.append(2)
            logits_map = torch.flip(logits_map, dims=tuple(reverse_dims))
        accumulated = logits_map if accumulated is None else accumulated + logits_map
    assert accumulated is not None
    return accumulated / float(len(transforms))


def build_blend_window(img_size: int, overlap_size: int, *, device: torch.device) -> torch.Tensor:
    if overlap_size <= 0:
        return torch.ones(img_size, img_size, device=device, dtype=torch.float32)
    ramp = torch.hann_window(img_size, periodic=False, device=device, dtype=torch.float32).clamp_min(1.0e-3)
    return torch.outer(ramp, ramp).clamp_min(1.0e-3)


@torch.no_grad()
def evaluate_merged_scene(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
    precision: str,
    class_names: tuple[str, ...],
    starts: list[tuple[int, int]],
    height: int,
    width: int,
    padded_height: int,
    padded_width: int,
    img_size: int,
    overlap_size: int,
    label_map: np.ndarray,
    eval_tta: bool,
) -> dict[str, Any]:
    model.eval()
    score_sum = torch.zeros(padded_height, padded_width, class_count, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    blend = build_blend_window(img_size, overlap_size, device=device)

    offset = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits_map = infer_tile_logits(model, x, precision=precision, eval_tta=eval_tta)
        batch_size = int(logits_map.shape[0])
        for batch_index in range(batch_size):
            row, col = starts[offset + batch_index]
            score_sum[row : row + img_size, col : col + img_size] += logits_map[batch_index] * blend.unsqueeze(-1)
            weight_sum[row : row + img_size, col : col + img_size] += blend
        offset += batch_size

    merged_logits = score_sum[:height, :width] / weight_sum[:height, :width].unsqueeze(-1).clamp_min(1.0e-6)
    flat_logits = merged_logits.reshape(-1, class_count)
    flat_labels = torch.from_numpy(label_map.reshape(-1)).to(device=device, dtype=torch.long)
    valid = flat_labels != 0
    if int(valid.sum().item()) == 0:
        return metrics_from_predictions(
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            class_count,
            0.0,
            class_names=class_names,
        )
    target = flat_labels[valid] - 1
    selected_logits = flat_logits[valid]
    loss = F.cross_entropy(selected_logits, target.long())
    prediction = selected_logits.argmax(dim=1).cpu().numpy()
    return metrics_from_predictions(
        target.cpu().numpy(),
        prediction,
        class_count,
        float(loss.item()),
        class_names=class_names,
    )


def build_layer_decay_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    layer_decay_rate: float,
) -> torch.optim.Optimizer:
    blocks = getattr(model.extractor.encoder.backbone.transformer, "blocks", [])
    depth = len(blocks)
    pattern = re.compile(r"extractor\.encoder\.backbone\.transformer\.blocks\.(\d+)\.")
    groups: dict[tuple[float, float], dict[str, object]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match = pattern.search(name)
        if match is None or depth <= 0:
            scale = 1.0
        else:
            layer_id = int(match.group(1))
            scale = layer_decay_rate ** max(depth - layer_id - 1, 0)
        decay = 0.0 if param.ndim <= 1 or name.endswith(".bias") else weight_decay
        key = (scale, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": decay}
        groups[key]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()), lr=lr, betas=(0.9, 0.999))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "oa": float(metrics["oa"]),
        "aa": float(metrics["aa"]),
        "kappa": float(metrics["kappa"]),
    }


def restore_cosine_scheduler_epoch(
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    completed_epochs: int,
) -> None:
    """Approximate CosineAnnealingLR state when resuming an older checkpoint."""
    completed_epochs = max(0, int(completed_epochs))
    scheduler.last_epoch = completed_epochs
    lrs: list[float] = []
    for base_lr, group in zip(scheduler.base_lrs, scheduler.optimizer.param_groups):
        lr = scheduler.eta_min + (base_lr - scheduler.eta_min) * (
            1.0 + math.cos(math.pi * float(completed_epochs) / float(scheduler.T_max))
        ) / 2.0
        group["lr"] = lr
        lrs.append(float(lr))
    scheduler._last_lr = lrs


def load_resume_payload(
    requested_path: Path | None,
    checkpoint_paths: dict[str, Path],
    device: torch.device,
) -> tuple[dict[str, Any] | None, Path | None, list[dict[str, str]]]:
    candidates: list[Path] = []
    if requested_path is not None:
        candidates.append(requested_path)
    for key in ("best_train_loss", "best_val_oa", "best_val_loss"):
        path = checkpoint_paths[key]
        if path not in candidates:
            candidates.append(path)

    failures: list[dict[str, str]] = []
    for path in candidates:
        if path is None or not path.exists():
            continue
        try:
            return torch.load(path, map_location=device, weights_only=False), path, failures
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if path.name == "latest_classification.pt":
                bad_path = path.with_suffix(path.suffix + f".corrupt_{int(time.time())}")
                try:
                    path.rename(bad_path)
                    failures[-1]["quarantined_as"] = str(bad_path)
                except Exception as rename_exc:
                    failures[-1]["quarantine_error"] = f"{type(rename_exc).__name__}: {rename_exc}"
    return None, None, failures


def load_compatible_model_weights(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    current.update(compatible)
    model.load_state_dict(current)
    return {
        "init_model": str(path),
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
        "source_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp_{int(time.time() * 1000)}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def proto_weight(base: float, epoch: int, warmup_epochs: int) -> float:
    if base <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return float(base)
    return float(base) * min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_config = resolve_scene_config(args.dataset)
    resolved_split_mode = scene_config.split_mode if args.split_mode is None else str(args.split_mode)
    resolved_train_num = scene_config.train_num if args.train_num is None else int(args.train_num)
    use_centered_training = resolved_split_mode == "number" if args.centered_training is None else bool(args.centered_training)
    if resolved_split_mode == "number" and resolved_train_num is not None:
        estimated_train_samples = int(resolved_train_num) * len(scene_config.class_names)
    else:
        estimated_train_samples = len(scene_config.class_names) * 50
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1 if scene_config.dataset == "indian_pines" else 2
    train_repeat_factor = args.train_repeat_factor
    if train_repeat_factor is None:
        if use_centered_training:
            if estimated_train_samples <= 256:
                train_repeat_factor = 12
            elif estimated_train_samples <= 768:
                train_repeat_factor = 6
            elif estimated_train_samples <= 1536:
                train_repeat_factor = 4
            else:
                train_repeat_factor = 2
        else:
            train_repeat_factor = 1
    train_spatial_jitter = args.train_spatial_jitter
    if train_spatial_jitter is None:
        if use_centered_training:
            if scene_config.dataset == "indian_pines":
                train_spatial_jitter = 12 if resolved_train_num >= 30 else 8
            elif estimated_train_samples <= 512:
                train_spatial_jitter = 12
            else:
                train_spatial_jitter = 8
        else:
            train_spatial_jitter = 0
    scene = build_segmentation_scene(
        data_root=args.data_root,
        config=scene_config,
        split_mode=args.split_mode,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        train_num=args.train_num,
        val_num=args.val_num,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        normalization=args.normalization,
        pca_components=args.pca_components,
        pca_whiten=args.pca_whiten,
        train_repeat_factor=train_repeat_factor,
        train_spatial_jitter=train_spatial_jitter,
        centered_training=args.centered_training,
        official_val_ratio=args.official_val_ratio,
        train_shuffle=args.train_shuffle,
        train_sampling=args.train_sampling,
        train_drop_last=args.train_drop_last,
        eval_drop_last=args.eval_drop_last,
    )
    class_weights, class_weight_info = build_class_weights_from_labels(
        scene["train_gt"],
        int(scene["class_count"]),
        args.class_weighting,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device=device)
    class_prototypes = None
    class_prototype_info: dict[str, Any] = {"enabled": False}
    if args.class_prototype_weight > 0.0:
        class_prototypes, class_prototype_info = build_class_spectral_prototypes(
            scene["cube"],
            scene["train_gt"],
            int(scene["class_count"]),
        )
        class_prototypes = class_prototypes.to(device=device)
    channels = int(scene["channels"])
    resolved_gsd = scene_config.gsd_m if args.gsd is None else float(args.gsd)
    metadata = build_downstream_metadata(
        num_bands=channels,
        sensor=scene_config.sensor,
        level=scene_config.level,
        gsd=resolved_gsd,
        wavelengths_nm=scene_config.wavelengths if int(scene_config.wavelengths.numel()) == channels else None,
        fwhm_nm=scene_config.fwhm if int(scene_config.fwhm.numel()) == channels else None,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )
    img_size = int(scene["img_size"])
    if img_size % extractor.local_patch_size != 0:
        raise ValueError(
            f"Resolved img_size={img_size} must be divisible by checkpoint local_patch_size="
            f"{extractor.local_patch_size}. Set --img-size to a divisible value."
        )
    model = AwareSSFusionSeg(
        extractor,
        int(scene["class_count"]),
        feature_layers=args.feature_layers,
        fusion_channels=args.fusion_channels,
        class_prototypes=class_prototypes,
        class_prototype_weight=args.class_prototype_weight,
        class_prototype_temperature=args.class_prototype_temperature,
    ).to(device)
    init_model_info: dict[str, Any] | None = None
    if args.init_model is not None:
        if not args.init_model.exists():
            raise FileNotFoundError(f"Init model checkpoint does not exist: {args.init_model}")
        init_model_info = load_compatible_model_weights(model, args.init_model, device)
        print(json.dumps({"event": "init_model_loaded", **init_model_info}, ensure_ascii=True))
    optimizer = build_layer_decay_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay_rate=args.layer_decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_epochs, eta_min=0.0)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    checkpoint_paths = {
        "best_train_loss": args.output_dir / "best_train_loss.pt",
        "best_val_loss": args.output_dir / "best_val_loss.pt",
        "best_val_oa": args.output_dir / "best_val_oa.pt",
        "latest": args.output_dir / "latest_classification.pt",
    }
    best_values = {
        "best_train_loss": math.inf,
        "best_val_loss": math.inf,
        "best_val_oa": -math.inf,
    }
    log_path = args.output_dir / "classification_log.jsonl"
    class_names = scene_config.class_names
    start_epoch = 1

    for key in ("best_train_loss", "best_val_loss", "best_val_oa"):
        path = checkpoint_paths[key]
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if key == "best_train_loss":
            best_values[key] = float(payload.get("train_loss", math.inf))
        elif key == "best_val_loss":
            best_values[key] = float(payload.get("val", {}).get("loss", math.inf))
        elif key == "best_val_oa":
            best_values[key] = float(payload.get("val", {}).get("oa", -math.inf))

    resume_path = args.resume_from
    if args.auto_resume and resume_path is None:
        resume_path = checkpoint_paths["latest"]
        if not any(path.exists() for path in checkpoint_paths.values()):
            resume_path = None
    if resume_path is not None:
        if args.resume_from is not None and not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        payload, loaded_from, resume_failures = load_resume_payload(resume_path, checkpoint_paths, device)
        if payload is None or loaded_from is None:
            raise RuntimeError(f"Unable to load any resume checkpoint. Failures: {resume_failures}")
        model.load_state_dict(payload["model"])
        loaded_optimizer = "optimizer" in payload
        loaded_scheduler = "scheduler" in payload
        if loaded_optimizer:
            optimizer.load_state_dict(payload["optimizer"])
        if loaded_scheduler:
            scheduler.load_state_dict(payload["scheduler"])
        else:
            restore_cosine_scheduler_epoch(scheduler, int(payload.get("epoch", 0)))
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(
            json.dumps(
                {
                    "event": "resume_classification",
                    "resume_from": str(loaded_from),
                    "requested_resume_from": str(resume_path),
                    "start_epoch": start_epoch,
                    "loaded_optimizer": loaded_optimizer,
                    "loaded_scheduler": loaded_scheduler,
                    "resume_failures": resume_failures,
                    "best_values": best_values,
                },
                ensure_ascii=True,
            )
        )

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = vars(args).copy()
        payload["batch_size"] = batch_size
        payload["split_mode"] = scene["split_summary"]["split_mode"]
        payload["train_ratio"] = scene["split_summary"]["train_ratio"]
        payload["val_ratio"] = scene["split_summary"]["val_ratio"]
        payload["resolved_scene"] = {
            "dataset": scene_config.dataset,
            "sensor": scene_config.sensor,
            "level": scene_config.level,
            "gsd_m": scene_config.gsd_m,
            "metadata_note": scene_config.metadata_note,
            "class_names": list(scene_config.class_names),
            "wavelengths": metadata.wavelengths_nm.tolist(),
            "fwhm": metadata.fwhm_nm.tolist(),
        }
        payload["metadata"] = {
            **asdict(metadata),
            "wavelengths_nm": metadata.wavelengths_nm.tolist(),
            "fwhm_nm": metadata.fwhm_nm.tolist(),
        }
        payload["splits"] = scene["split_summary"]
        payload["class_weights"] = class_weight_info
        payload["class_prototypes"] = {
            **class_prototype_info,
            "weight": float(args.class_prototype_weight),
            "temperature": float(args.class_prototype_temperature),
        }
        payload["init_model_info"] = init_model_info
        payload["implementation_note"] = (
            "This is the HyperSIGMA demo_seg_hypersigma-style dense pixel classifier: "
            "for Indian Pines it uses centered segmentation crops for training and overlapping tile-logit fusion for evaluation; "
            "for larger scenes it falls back to scene tiles. "
            "Training uses masked pixel CE with spectral/metadata-gated four-level feature fusion. "
            "Optional HyperSIGMA-style PCA preprocessing is applied on the normalized full cube before splitting when requested. "
            "The Aware-specific additions are metadata-conditioned encoder initialization and optional "
            "frozen ground prototype auxiliary regularization on training tiles only, optional class-balanced CE, "
            "and optional train-label-only class spectral prototype logit fusion during fine-tuning."
        )
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_proto_kl = 0.0
        running_proto_balance = 0.0
        running_proto_entropy = 0.0
        running_count = 0
        running_correct = 0
        kl_weight = 0.0 if args.proto_bank == "none" else proto_weight(args.proto_kl_weight, epoch, args.proto_warmup_epochs)
        balance_weight = (
            0.0 if args.proto_bank == "none" else proto_weight(args.proto_balance_weight, epoch, args.proto_warmup_epochs)
        )

        for x, y in scene["train_loader"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits, features = model(x, return_features=True)
                ce_loss, selected_logits, selected_target = masked_cross_entropy(
                    logits,
                    y,
                    label_smoothing=args.label_smoothing,
                    class_weights=class_weights,
                )
                if ce_loss is None:
                    continue
                if args.proto_bank == "none" or (kl_weight == 0.0 and balance_weight == 0.0):
                    zero = ce_loss.new_zeros(())
                    proto_losses = {"proto_kl": zero, "proto_balance": zero, "proto_entropy": zero}
                    loss = ce_loss
                else:
                    proto_losses = model.extractor.prototype_auxiliary_loss(
                        x,
                        features,
                        target_bank=args.proto_bank,
                    )
                    loss = ce_loss + kl_weight * proto_losses["proto_kl"] + balance_weight * proto_losses["proto_balance"]
            loss.backward()
            optimizer.step()

            batch_count = int(selected_target.numel())
            running_loss += float(loss.item()) * batch_count
            running_ce += float(ce_loss.item()) * batch_count
            running_proto_kl += float(proto_losses["proto_kl"].item()) * batch_count
            running_proto_balance += float(proto_losses["proto_balance"].item()) * batch_count
            running_proto_entropy += float(proto_losses["proto_entropy"].item()) * batch_count
            running_count += batch_count
            running_correct += int((selected_logits.argmax(dim=1) == selected_target).sum().item())
        scheduler.step()

        train_loss = running_loss / max(running_count, 1)
        has_val_labels = bool(np.count_nonzero(scene["val_gt"]) > 0)
        if has_val_labels:
            val_metrics = evaluate_merged_scene(
                model=model,
                loader=scene["val_loader"],
                device=device,
                class_count=int(scene["class_count"]),
                precision=args.precision,
                class_names=class_names,
                starts=scene["val_starts"],
                height=int(scene["height"]),
                width=int(scene["width"]),
                padded_height=int(scene["padded_height"]),
                padded_width=int(scene["padded_width"]),
                img_size=int(scene["img_size"]),
                overlap_size=int(scene["overlap_size"]),
                label_map=scene["val_gt"],
                eval_tta=args.eval_tta,
            )
        else:
            val_metrics = metrics_from_predictions(
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                int(scene["class_count"]),
                0.0,
                class_names=class_names,
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": running_ce / max(running_count, 1),
            "train_acc": running_correct / max(running_count, 1),
            "train_proto_kl": running_proto_kl / max(running_count, 1),
            "train_proto_balance": running_proto_balance / max(running_count, 1),
            "train_proto_entropy": running_proto_entropy / max(running_count, 1),
            "proto_kl_weight": kl_weight,
            "proto_balance_weight": balance_weight,
            "val_loss": val_metrics["loss"],
            "val_oa": val_metrics["oa"],
            "val_aa": val_metrics["aa"],
            "val_kappa": val_metrics["kappa"],
            "val_skipped": not has_val_labels,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record, ensure_ascii=True))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val": compact_metrics(val_metrics),
            "args": vars(args),
            "class_weights": class_weight_info,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        atomic_torch_save(checkpoint, checkpoint_paths["latest"])
        if train_loss < best_values["best_train_loss"]:
            best_values["best_train_loss"] = train_loss
            atomic_torch_save(checkpoint, checkpoint_paths["best_train_loss"])
        if has_val_labels and float(val_metrics["loss"]) < best_values["best_val_loss"]:
            best_values["best_val_loss"] = float(val_metrics["loss"])
            atomic_torch_save(checkpoint, checkpoint_paths["best_val_loss"])
        if has_val_labels and float(val_metrics["oa"]) > best_values["best_val_oa"]:
            best_values["best_val_oa"] = float(val_metrics["oa"])
            atomic_torch_save(checkpoint, checkpoint_paths["best_val_oa"])

    test_results: dict[str, Any] = {}
    has_val_labels = bool(np.count_nonzero(scene["val_gt"]) > 0)
    test_keys = ("latest",) if not has_val_labels else ("best_train_loss", "best_val_loss", "best_val_oa", "latest")
    for key in test_keys:
        payload = torch.load(checkpoint_paths[key], map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        metrics = evaluate_merged_scene(
            model=model,
            loader=scene["test_loader"],
            device=device,
            class_count=int(scene["class_count"]),
            precision=args.precision,
            class_names=class_names,
            starts=scene["test_starts"],
            height=int(scene["height"]),
            width=int(scene["width"]),
            padded_height=int(scene["padded_height"]),
            padded_width=int(scene["padded_width"]),
            img_size=int(scene["img_size"]),
            overlap_size=int(scene["overlap_size"]),
            label_map=scene["test_gt"],
            eval_tta=args.eval_tta,
        )
        test_results[key] = {
            "epoch": int(payload["epoch"]),
            "test": metrics,
            "selection_val": payload.get("val", {}),
            "selection_train_loss": float(payload.get("train_loss", float("nan"))),
        }

    print(json.dumps(test_results, ensure_ascii=True))
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_results, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
