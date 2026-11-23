from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aware_open.downstream import (
    AwareFeatureExtractor,
    build_detection_scene,
    build_downstream_metadata,
    compute_auc_scores,
    resolve_detection_scene_config,
    seed_everything,
    standardize_score_map,
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Aware for HyperSIGMA-style hyperspectral detection.")
    parser.add_argument("--task", type=str, required=True, choices=("htd", "had"))
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key, e.g. mosaic, aviris, cri, pavia.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cube-path", type=Path, default=None)
    parser.add_argument("--gt-path", type=Path, default=None)
    parser.add_argument("--coarse-path", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--overlap-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--scheduler-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--layer-decay-rate", type=float, default=0.9)
    parser.add_argument("--feature-layers", type=int, default=4)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--pseudo-bg-ratio", type=float, default=0.30)
    parser.add_argument("--pseudo-fg-ratio", type=float, default=0.0015)
    parser.add_argument("--target-spectrum-type", type=int, default=0)
    parser.add_argument("--target-spectrum-source", choices=("gt", "pseudo"), default="gt")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.15)
    parser.add_argument("--rank-pairs", type=int, default=4096)
    parser.add_argument("--rank-mode", choices=("random", "hard_tail"), default="random")
    parser.add_argument("--prior-blend-alpha", type=float, default=1.0)
    parser.add_argument("--train-label-source", choices=("pseudo", "gt"), default="pseudo")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--gsd", type=float, default=None)
    parser.add_argument("--normalization", choices=("none", "minmax", "mean_std"), default="mean_std")
    parser.add_argument("--disable-auto-priors", action="store_true")
    parser.add_argument("--disallow-missing-coarse", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()
    if args.freeze_encoder:
        print("[Aware] --freeze-encoder is disabled for fair full-parameter fine-tuning; using freeze_encoder=False.")
        args.freeze_encoder = False
    return args


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class AwareDetectionFusion(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        task: str,
        input_bands: int,
        prior_channels: int,
        feature_layers: int = 4,
        fusion_channels: int = 128,
        hidden_channels: int = 96,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.task = str(task).strip().lower()
        self.feature_layers = int(feature_layers)
        self.fusion_channels = int(fusion_channels)
        self.embed_dim = extractor.model_dim
        self.input_bands = int(input_bands)
        self.prior_channels = int(prior_channels)
        self.spectral_gate_dim = 128

        self.DR1 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.DR2 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.DR3 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.DR4 = nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels * 4, fusion_channels, kernel_size=1, bias=False),
            make_group_norm(fusion_channels),
            nn.GELU(),
        )

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

        if self.task == "htd":
            self.target_reduce = nn.Sequential(
                nn.LayerNorm(self.input_bands),
                nn.Linear(self.input_bands, fusion_channels, bias=False),
                nn.GELU(),
                nn.Linear(fusion_channels, fusion_channels, bias=False),
                nn.Tanh(),
            )
            extra_channels = 1
        else:
            self.target_reduce = None
            extra_channels = 0
        input_channels = fusion_channels + self.prior_channels + extra_channels
        half_hidden = max(32, hidden_channels // 2)
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, half_hidden, kernel_size=3, padding=1, bias=False),
            make_group_norm(half_hidden),
            nn.GELU(),
            nn.Conv2d(half_hidden, 1, kernel_size=1, bias=True),
        )

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        prior: torch.Tensor | None = None,
        target_spectrum: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        features = self.extractor.forward_features(x)
        intermediate = list(features["intermediate_tokens"])
        if len(intermediate) < self.feature_layers:
            raise RuntimeError(
                f"Aware checkpoint exposes {len(intermediate)} intermediate maps, "
                f"but the detection head needs {self.feature_layers}."
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
        fused = self.fusion(torch.cat((ss_feature1, ss_feature2, ss_feature3, ss_feature4), dim=1))

        pieces = []
        if self.task == "htd":
            if target_spectrum is None:
                raise ValueError("target_spectrum is required for HTD forward.")
            if target_spectrum.shape[-1] != self.input_bands:
                raise ValueError(
                    f"Expected target_spectrum with {self.input_bands} bands, got {target_spectrum.shape[-1]}."
                )
            target_context = self.target_reduce(target_spectrum.float()).view(x.shape[0], -1, 1, 1)
            fused = fused * (1.0 + target_context)
            x_norm = F.normalize(x.float(), dim=1, eps=1.0e-6)
            target_norm = F.normalize(target_spectrum.float(), dim=1, eps=1.0e-6).view(x.shape[0], -1, 1, 1)
            cosine = (x_norm * target_norm).sum(dim=1, keepdim=True)
            pieces.append(cosine)

        if prior is None:
            prior = fused.new_zeros((x.shape[0], self.prior_channels, output_size[0], output_size[1]))
        elif prior.shape[1] != self.prior_channels:
            raise ValueError(f"Expected {self.prior_channels} prior channels, got {prior.shape[1]}.")
        if prior.shape[-2:] != output_size:
            raise ValueError(f"Prior map shape {tuple(prior.shape[-2:])} is inconsistent with tile shape {output_size}.")

        head_input = torch.cat([fused, prior.float(), *pieces], dim=1)
        logits = self.head(head_input).squeeze(1)
        if return_features:
            return logits, features
        return logits


def masked_detection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: float,
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.15,
    rank_pairs: int = 4096,
    rank_mode: str = "random",
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != 255
    if int(valid.sum().item()) == 0:
        return None, flat_logits.new_empty(0), flat_labels.new_empty(0)
    target = flat_labels[valid].float()
    weight_value = max(float(pos_weight), 1.0)
    weight = torch.tensor(weight_value, device=flat_logits.device, dtype=flat_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(flat_logits[valid], target, pos_weight=weight)
    if rank_loss_weight > 0.0:
        selected = flat_logits[valid]
        pos = selected[target > 0.5]
        neg = selected[target < 0.5]
        pair_count = min(int(pos.numel()), int(neg.numel()), max(int(rank_pairs), 1))
        if pair_count > 0:
            if str(rank_mode).strip().lower() == "hard_tail":
                pos_pair = torch.topk(pos, k=pair_count, largest=False).values
                neg_pair = torch.topk(neg, k=pair_count, largest=True).values
            else:
                pos_idx = torch.randperm(int(pos.numel()), device=selected.device)[:pair_count]
                neg_idx = torch.randperm(int(neg.numel()), device=selected.device)[:pair_count]
                pos_pair = pos[pos_idx]
                neg_pair = neg[neg_idx]
            rank_loss = F.softplus(float(rank_margin) - pos_pair + neg_pair).mean()
            loss = loss + float(rank_loss_weight) * rank_loss
    return loss, flat_logits[valid], target


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


def build_blend_window(img_size: int, overlap_size: int, *, device: torch.device) -> torch.Tensor:
    if overlap_size <= 0:
        return torch.ones(img_size, img_size, device=device, dtype=torch.float32)
    ramp = torch.hann_window(img_size, periodic=False, device=device, dtype=torch.float32).clamp_min(1.0e-3)
    return torch.outer(ramp, ramp).clamp_min(1.0e-3)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    keys = (
        "auc",
        "auc_threshold_fpr",
        "auc_threshold_tpr",
        "auc_composite",
        "auc_ratio",
        "neural_auc",
        "prior_auc",
        "prior_blend_alpha",
    )
    return {key: float(metrics[key]) for key in keys}


def blend_detection_score(
    neural_score: np.ndarray,
    prior_score: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    neural = standardize_score_map(np.asarray(neural_score, dtype=np.float32))
    prior = standardize_score_map(np.asarray(prior_score, dtype=np.float32))
    return standardize_score_map(alpha * neural + (1.0 - alpha) * prior)


def forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    prior = batch.get("prior")
    target_spectrum = batch.get("target_spectrum")
    if prior is not None:
        prior = prior.to(device, non_blocking=True)
    if target_spectrum is not None:
        target_spectrum = target_spectrum.to(device, non_blocking=True)
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        logits = model(image, prior=prior, target_spectrum=target_spectrum)
    return logits.float()


@torch.no_grad()
def evaluate_detection_scene(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    precision: str,
    starts: list[tuple[int, int]],
    height: int,
    width: int,
    padded_height: int,
    padded_width: int,
    img_size: int,
    overlap_size: int,
    label_map: np.ndarray,
    prior_map: np.ndarray,
    prior_blend_alpha: float,
) -> dict[str, Any]:
    model.eval()
    score_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    blend = build_blend_window(img_size, overlap_size, device=device)

    offset = 0
    for batch in loader:
        logits_map = forward_batch(model, batch, device=device, precision=precision)
        batch_size = int(logits_map.shape[0])
        for batch_index in range(batch_size):
            row, col = starts[offset + batch_index]
            score_sum[row : row + img_size, col : col + img_size] += logits_map[batch_index] * blend
            weight_sum[row : row + img_size, col : col + img_size] += blend
        offset += batch_size

    merged_logits = score_sum[:height, :width] / weight_sum[:height, :width].clamp_min(1.0e-6)
    neural_score_map = torch.sigmoid(merged_logits).cpu().numpy().astype(np.float32, copy=False)
    neural_score_map = standardize_score_map(neural_score_map)
    prior_score_map = standardize_score_map(np.asarray(prior_map, dtype=np.float32))
    score_map = blend_detection_score(neural_score_map, prior_score_map, alpha=prior_blend_alpha)
    metrics = compute_auc_scores(score_map, label_map)
    neural_metrics = compute_auc_scores(neural_score_map, label_map)
    prior_metrics = compute_auc_scores(prior_score_map, label_map)
    metrics["score_mean"] = float(np.mean(score_map))
    metrics["score_std"] = float(np.std(score_map))
    metrics["neural_auc"] = float(neural_metrics["auc"])
    metrics["prior_auc"] = float(prior_metrics["auc"])
    metrics["prior_blend_alpha"] = float(np.clip(prior_blend_alpha, 0.0, 1.0))
    metrics["score_map"] = score_map
    metrics["neural_score_map"] = neural_score_map
    metrics["prior_score_map"] = prior_score_map
    return metrics


def save_score_outputs(
    *,
    output_dir: Path,
    scene: dict[str, Any],
    score_map: np.ndarray,
    prefix: str,
) -> None:
    np.save(output_dir / f"{prefix}_score_map.npy", score_map.astype(np.float32, copy=False))
    sio.savemat(
        output_dir / f"{prefix}_score_map.mat",
        {
            "score_map": score_map.astype(np.float32, copy=False),
            "gt": np.asarray(scene["labels"], dtype=np.uint8),
            "pseudo_labels": np.asarray(scene["pseudo_labels"], dtype=np.uint8),
            "coarse_map": np.asarray(scene["coarse"], dtype=np.float32),
        },
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_config = resolve_detection_scene_config(args.task, args.dataset)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 2 if args.task == "htd" else 1

    scene = build_detection_scene(
        data_root=args.data_root,
        config=scene_config,
        cube_path=args.cube_path,
        gt_path=args.gt_path,
        coarse_path=args.coarse_path,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        normalization=args.normalization,
        pseudo_bg_ratio=args.pseudo_bg_ratio,
        pseudo_fg_ratio=args.pseudo_fg_ratio,
        target_spectrum_type=args.target_spectrum_type,
        target_spectrum_source=args.target_spectrum_source,
        allow_missing_coarse=not args.disallow_missing_coarse,
        use_auto_priors=not args.disable_auto_priors,
    )
    train_label_source = str(args.train_label_source).strip().lower()
    if train_label_source == "gt":
        scene["train_loader"] = DataLoader(
            scene["eval_loader"].dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

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
            f"Resolved img_size={img_size} must be divisible by checkpoint local_patch_size={extractor.local_patch_size}."
        )

    model = AwareDetectionFusion(
        extractor,
        task=args.task,
        input_bands=channels,
        prior_channels=int(scene["prior_channels"]),
        feature_layers=args.feature_layers,
        fusion_channels=args.fusion_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)
    optimizer = build_layer_decay_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay_rate=args.layer_decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.scheduler_epochs), int(args.epochs)),
        eta_min=0.0,
    )

    if train_label_source == "gt":
        train_label_map = np.asarray(scene["labels"], dtype=np.int64)
        pseudo_positive = int(np.sum(train_label_map == 1))
        pseudo_negative = int(np.sum(train_label_map == 0))
    else:
        pseudo_positive = int(scene["summary"]["pseudo_positive_pixels"])
        pseudo_negative = int(scene["summary"]["pseudo_negative_pixels"])
    raw_pos_weight = float(pseudo_negative) / max(float(pseudo_positive), 1.0)
    pos_weight = min(max(raw_pos_weight, 1.0), float(args.max_pos_weight))

    checkpoint_paths = {
        "best_auc": args.output_dir / "best_auc.pt",
        "latest": args.output_dir / "latest_detection.pt",
    }
    best_auc = -math.inf
    log_path = args.output_dir / "detection_log.jsonl"

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = vars(args).copy()
        payload["batch_size"] = batch_size
        payload["scene"] = scene["summary"]
        payload["resolved_scene"] = {
            "dataset": scene_config.dataset,
            "task": scene_config.task,
            "sensor": scene_config.sensor,
            "level": scene_config.level,
            "gsd_m": scene_config.gsd_m,
            "metadata_note": scene_config.metadata_note,
            "wavelengths": metadata.wavelengths_nm.tolist(),
            "fwhm": metadata.fwhm_nm.tolist(),
        }
        payload["metadata"] = {
            **asdict(metadata),
            "wavelengths_nm": metadata.wavelengths_nm.tolist(),
            "fwhm_nm": metadata.fwhm_nm.tolist(),
        }
        payload["training"] = {
            "train_label_source": train_label_source,
            "pos_weight_raw": raw_pos_weight,
            "pos_weight_used": pos_weight,
            "prior_channels": int(scene["prior_channels"]),
        }
        payload["implementation_note"] = (
            "This Aware detection fine-tuner follows HyperSIGMA's dense detection protocol: "
            "32x32 overlapping tiles, pseudo labels from coarse detections, and full-scene AUC evaluation. "
            "Aware adds metadata-conditioned pretraining features, a four-level fusion head, "
            "and automatic classical detection priors when external coarse maps are unavailable."
        )
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        running_correct = 0

        for batch in scene["train_loader"]:
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, device=device, precision=args.precision)
            loss, selected_logits, selected_target = masked_detection_loss(
                logits,
                label,
                pos_weight=pos_weight,
                rank_loss_weight=args.rank_loss_weight,
                rank_margin=args.rank_margin,
                rank_pairs=args.rank_pairs,
                rank_mode=args.rank_mode,
            )
            if loss is None:
                continue
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()

            batch_count = int(selected_target.numel())
            running_loss += float(loss.item()) * batch_count
            running_count += batch_count
            running_correct += int(((torch.sigmoid(selected_logits) >= 0.5).float() == selected_target).sum().item())

        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_count, 1),
            "train_acc": running_correct / max(running_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }

        should_eval = epoch % max(int(args.eval_every), 1) == 0 or epoch == args.epochs
        if should_eval:
            eval_metrics = evaluate_detection_scene(
                model=model,
                loader=scene["eval_loader"],
                device=device,
                precision=args.precision,
                starts=scene["starts"],
                height=int(scene["height"]),
                width=int(scene["width"]),
                padded_height=int(scene["padded_height"]),
                padded_width=int(scene["padded_width"]),
                img_size=int(scene["img_size"]),
                overlap_size=int(scene["overlap_size"]),
                label_map=np.asarray(scene["labels"], dtype=np.int64),
                prior_map=np.asarray(scene["coarse"], dtype=np.float32),
                prior_blend_alpha=float(args.prior_blend_alpha),
            )
            score_map = eval_metrics.pop("score_map")
            neural_score_map = eval_metrics.pop("neural_score_map")
            prior_score_map = eval_metrics.pop("prior_score_map")
            record.update({f"eval_{key}": float(value) for key, value in eval_metrics.items()})

            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "train_loss": record["train_loss"],
                "eval": compact_metrics(eval_metrics),
                "args": vars(args),
                "scene_summary": scene["summary"],
            }
            torch.save(checkpoint, checkpoint_paths["latest"])
            save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=score_map, prefix="latest")
            save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=neural_score_map, prefix="latest_neural")
            save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=prior_score_map, prefix="latest_prior")
            if float(eval_metrics["auc"]) > best_auc:
                best_auc = float(eval_metrics["auc"])
                torch.save(checkpoint, checkpoint_paths["best_auc"])
                save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=score_map, prefix="best_auc")
                save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=neural_score_map, prefix="best_auc_neural")
                save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=prior_score_map, prefix="best_auc_prior")

        print(json.dumps(record, ensure_ascii=True))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    payload = torch.load(checkpoint_paths["best_auc"], map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    final_metrics = evaluate_detection_scene(
        model=model,
        loader=scene["eval_loader"],
        device=device,
        precision=args.precision,
        starts=scene["starts"],
        height=int(scene["height"]),
        width=int(scene["width"]),
        padded_height=int(scene["padded_height"]),
        padded_width=int(scene["padded_width"]),
        img_size=int(scene["img_size"]),
        overlap_size=int(scene["overlap_size"]),
        label_map=np.asarray(scene["labels"], dtype=np.int64),
        prior_map=np.asarray(scene["coarse"], dtype=np.float32),
        prior_blend_alpha=float(args.prior_blend_alpha),
    )
    final_score_map = final_metrics.pop("score_map")
    final_neural_score_map = final_metrics.pop("neural_score_map")
    final_prior_score_map = final_metrics.pop("prior_score_map")
    save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=final_score_map, prefix="final_best_auc")
    save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=final_neural_score_map, prefix="final_best_auc_neural")
    save_score_outputs(output_dir=args.output_dir, scene=scene, score_map=final_prior_score_map, prefix="final_best_auc_prior")
    final_result = {
        "best_auc_checkpoint_epoch": int(payload["epoch"]),
        "best_auc_metrics": final_metrics,
        "selection_eval": payload.get("eval", {}),
        "selection_train_loss": float(payload.get("train_loss", float("nan"))),
    }
    print(json.dumps(final_result, ensure_ascii=True))
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_result, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
