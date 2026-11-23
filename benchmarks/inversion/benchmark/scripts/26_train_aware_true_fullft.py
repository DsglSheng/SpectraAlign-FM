from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from aware_open.downstream.adapter import (
    AwareFeatureExtractor,
    build_downstream_metadata,
    metadata_to_batch,
)


DEFAULTS = {
    "checkpoint": Path(
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


@dataclass(frozen=True)
class Mode:
    dataset: str
    grid_size: int
    target_transform: str
    loss: str
    rank_weight: float
    feature_weight: float
    group_dro: bool
    coral_weight: float
    heteroscedastic: bool
    max_epochs: int
    patience: int
    warmup_epochs: int
    mask_ratio: float
    reconstruction_weight: float
    prototype_weight: float
    attention_pooling: bool = False
    spectral_augmentation: str = "none"
    balanced_groups: bool = False
    rex_weight: float = 0.0
    adversarial_weight: float = 0.0
    adversarial_levels: tuple[str, ...] = ()
    temporal_weight: float = 0.0
    calibration: str = "none"
    relaxed_features: str = "none"
    tta_shifts: tuple[int, ...] = (0,)
    min_epochs: int = 0
    schedule_epochs: int = 0
    soup_start_epoch: int = 0
    soup_topk: int = 0
    spectral_decoder: str = "none"
    num_experts: int = 1
    gloria_tail_weight: float = 0.25


MODES = {
    "karly_v5_full": Mode(
        "karly", 2, "raw", "mse", 0.05, 0.0, False, 0.0, False, 320, 70, 15, 0.25, 0.02, 0.01
    ),
    "karly_v6_full": Mode(
        "karly", 3, "raw", "smoothl1", 0.15, 0.03, False, 0.0, False, 360, 75, 18, 0.22, 0.03, 0.015
    ),
    "gloria_v5_full": Mode(
        "gloria", 2, "log1p", "smoothl1", 0.0, 0.0, False, 0.0, False, 280, 60, 12, 0.25, 0.02, 0.01
    ),
    "gloria_v6_full": Mode(
        "gloria", 3, "log1p", "hetero", 0.0, 0.0, True, 0.01, True, 320, 70, 15, 0.22, 0.03, 0.015
    ),
    "karly_v7_shift": Mode(
        "karly", 3, "raw", "smoothl1", 0.10, 0.02, False, 0.0, False,
        420, 90, 20, 0.20, 0.025, 0.012,
        attention_pooling=True,
        spectral_augmentation="karly_drift",
        balanced_groups=True,
        rex_weight=0.08,
        adversarial_weight=0.025,
        adversarial_levels=("primary",),
        temporal_weight=0.08,
        calibration="global_affine",
    ),
    "karly_v8_shift": Mode(
        "karly", 3, "raw", "smoothl1", 0.12, 0.03, False, 0.0, False,
        440, 100, 22, 0.18, 0.025, 0.012,
        attention_pooling=True,
        spectral_augmentation="karly_mixstyle",
        balanced_groups=True,
        rex_weight=0.12,
        adversarial_weight=0.035,
        adversarial_levels=("primary", "secondary"),
        temporal_weight=0.14,
        calibration="hierarchical",
    ),
    "gloria_v7_shift": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.02, 0.025, False, 0.0, False,
        380, 85, 18, 0.20, 0.025, 0.012,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.08,
        adversarial_weight=0.025,
        adversarial_levels=("primary",),
        calibration="global_affine",
    ),
    "gloria_v8_shift": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.03, 0.04, False, 0.0, False,
        420, 95, 20, 0.18, 0.025, 0.012,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument_strong",
        balanced_groups=True,
        rex_weight=0.12,
        adversarial_weight=0.035,
        adversarial_levels=("primary", "secondary", "tertiary"),
        calibration="hierarchical",
    ),
    "karly_v9_long": Mode(
        "karly", 3, "raw", "smoothl1", 0.12, 0.035, False, 0.0, False,
        650, 160, 35, 0.16, 0.02, 0.01,
        attention_pooling=True,
        spectral_augmentation="karly_drift",
        balanced_groups=True,
        rex_weight=0.035,
        temporal_weight=0.10,
        calibration="none",
        tta_shifts=(-1, 0, 1),
    ),
    "karly_v10_temporal": Mode(
        "karly", 3, "raw", "smoothl1", 0.10, 0.025, False, 0.0, False,
        720, 190, 40, 0.14, 0.018, 0.008,
        attention_pooling=True,
        spectral_augmentation="karly_drift",
        balanced_groups=True,
        rex_weight=0.025,
        temporal_weight=0.12,
        calibration="none",
        relaxed_features="karly_temporal",
        tta_shifts=(-1, 0, 1),
    ),
    "gloria_v9_long": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.025, 0.035, False, 0.0, False,
        600, 150, 32, 0.16, 0.02, 0.01,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.035,
        calibration="none",
        tta_shifts=(-1, 0, 1),
    ),
    "gloria_v10_source": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.02, 0.025, False, 0.0, False,
        680, 180, 36, 0.14, 0.018, 0.008,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.025,
        calibration="none",
        relaxed_features="gloria_source",
        tta_shifts=(-1, 0, 1),
    ),
    "gloria_v11_late_soup": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.03, 0.04, False, 0.0, False,
        540, 240, 24, 0.17, 0.022, 0.010,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument_strong",
        balanced_groups=True,
        rex_weight=0.08,
        adversarial_weight=0.025,
        adversarial_levels=("primary", "secondary", "tertiary"),
        calibration="none",
        tta_shifts=(-1, 0, 1),
        min_epochs=360,
        schedule_epochs=420,
        soup_start_epoch=220,
        soup_topk=5,
    ),
    "gloria_v12_source_soup": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.025, 0.035, False, 0.0, False,
        580, 260, 28, 0.15, 0.020, 0.009,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.05,
        calibration="none",
        relaxed_features="gloria_identity",
        tta_shifts=(-1, 0, 1),
        min_epochs=400,
        schedule_epochs=460,
        soup_start_epoch=240,
        soup_topk=5,
    ),
    "gloria_v13_water_decoder": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.025, 0.03, False, 0.0, False,
        460, 130, 24, 0.17, 0.020, 0.010,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.04,
        calibration="none",
        tta_shifts=(-1, 0, 1),
        min_epochs=220,
        schedule_epochs=360,
        spectral_decoder="water_multiscale",
        gloria_tail_weight=0.40,
    ),
    "gloria_v14_water_moe": Mode(
        "gloria", 3, "log1p", "smoothl1", 0.02, 0.025, False, 0.0, False,
        480, 140, 26, 0.16, 0.020, 0.010,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.035,
        calibration="none",
        tta_shifts=(-1, 0, 1),
        min_epochs=240,
        schedule_epochs=380,
        spectral_decoder="water_multiscale",
        num_experts=3,
        gloria_tail_weight=0.50,
    ),
    "gloria_v15_water_2x2": Mode(
        "gloria", 2, "log1p", "smoothl1", 0.015, 0.02, False, 0.0, False,
        440, 130, 22, 0.18, 0.018, 0.008,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument",
        balanced_groups=True,
        rex_weight=0.03,
        calibration="none",
        tta_shifts=(-1, 0, 1),
        min_epochs=220,
        schedule_epochs=340,
        spectral_decoder="water_multiscale",
        num_experts=2,
        gloria_tail_weight=0.45,
    ),
    "gloria_v16_tail_asinh": Mode(
        "gloria", 3, "asinh10", "smoothl1", 0.03, 0.03, False, 0.0, False,
        500, 150, 28, 0.15, 0.018, 0.008,
        attention_pooling=True,
        spectral_augmentation="gloria_instrument_strong",
        balanced_groups=True,
        rex_weight=0.04,
        calibration="none",
        tta_shifts=(-1, 0, 1),
        min_epochs=260,
        schedule_epochs=400,
        spectral_decoder="water_multiscale",
        num_experts=3,
        gloria_tail_weight=0.60,
    ),
}


class FullFTDataset(Dataset):
    def __init__(
        self,
        spectra: np.ndarray,
        domain_features: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray,
        secondary_groups: np.ndarray,
        tertiary_groups: np.ndarray,
        indices: np.ndarray,
        *,
        target_mean: float,
        target_std: float,
        transform: str,
    ) -> None:
        self.spectra = spectra
        self.domain_features = domain_features
        self.targets = targets
        self.groups = groups
        self.secondary_groups = secondary_groups
        self.tertiary_groups = tertiary_groups
        self.indices = indices
        self.target_mean = target_mean
        self.target_std = target_std
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        raw = float(self.targets[index])
        if self.transform == "log1p":
            transformed = math.log1p(max(raw, 0.0))
        elif self.transform == "asinh10":
            transformed = math.asinh(max(raw, 0.0) / 10.0)
        else:
            transformed = raw
        return {
            "spectrum": torch.from_numpy(self.spectra[index]),
            "domain": torch.from_numpy(self.domain_features[index]),
            "target": torch.tensor((transformed - self.target_mean) / self.target_std, dtype=torch.float32),
            "raw_target": torch.tensor(raw, dtype=torch.float32),
            "group": torch.tensor(int(self.groups[index]), dtype=torch.int64),
            "secondary_group": torch.tensor(int(self.secondary_groups[index]), dtype=torch.int64),
            "tertiary_group": torch.tensor(int(self.tertiary_groups[index]), dtype=torch.int64),
            "index": torch.tensor(index, dtype=torch.int64),
        }


def sample_normalize(spectrum: torch.Tensor) -> torch.Tensor:
    mean = spectrum.mean(dim=1, keepdim=True)
    std = spectrum.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (spectrum - mean) / std


def smooth(spectrum: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    return F.avg_pool1d(
        F.pad(spectrum[:, None, :], (pad, pad), mode="replicate"),
        kernel_size=kernel,
        stride=1,
    ).squeeze(1)


def difference(spectrum: torch.Tensor, order: int) -> torch.Tensor:
    output = spectrum
    for _ in range(order):
        output = F.pad(output[:, 1:] - output[:, :-1], (1, 0))
    return output


def build_view_cube(spectrum: torch.Tensor, patch_size: int, grid_size: int) -> torch.Tensor:
    native = sample_normalize(spectrum)
    smooth5 = sample_normalize(smooth(native, 5))
    smooth15 = sample_normalize(smooth(native, 15))
    first = sample_normalize(difference(native, 1))
    second = sample_normalize(difference(native, 2))
    continuum = sample_normalize(native - smooth15)
    ratio_view = sample_normalize(native / smooth15.abs().clamp_min(0.1))
    absolute_first = sample_normalize(difference(native, 1).abs())
    curvature = sample_normalize(smooth(second, 7))
    views = [native, smooth5, first, second]
    if grid_size == 3:
        views = [native, smooth5, smooth15, first, second, continuum, ratio_view, absolute_first, curvature]
    cells = torch.stack(views, dim=-1).reshape(spectrum.shape[0], spectrum.shape[1], grid_size, grid_size)
    return cells.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class WaterOpticsDecoder(nn.Module):
    def __init__(self, output_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        branch_width = 48
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(3, branch_width, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_width,
                        branch_width,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        groups=branch_width,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for kernel in (3, 11, 31)
            ]
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(branch_width * 3),
            nn.Linear(branch_width * 3, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        positive = spectrum.clamp_min(1.0e-7)
        log_spectrum = torch.log(positive)
        normalized = sample_normalize(spectrum)
        log_normalized = sample_normalize(log_spectrum)
        first = sample_normalize(difference(log_spectrum, 1))
        channels = torch.stack((normalized, log_normalized, first), dim=1)
        encoded = torch.cat(
            [branch(channels).squeeze(-1) for branch in self.branches],
            dim=1,
        )
        return self.projection(encoded)


class TrueFullFTRegressor(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        domain_dim: int,
        heteroscedastic: bool,
        dropout: float,
        attention_pooling: bool = False,
        adversarial_classes: dict[str, int] | None = None,
        spectral_decoder: str = "none",
        num_experts: int = 1,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.heteroscedastic = heteroscedastic
        self.attention_pooling = attention_pooling
        self.num_experts = num_experts
        hidden = 256
        self.view_attention = nn.Sequential(
            nn.LayerNorm(extractor.model_dim),
            nn.Linear(extractor.model_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.domain_encoder = nn.Sequential(
            nn.LayerNorm(domain_dim),
            nn.Linear(domain_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        spectral_dim = 192 if spectral_decoder == "water_multiscale" else 0
        self.spectral_encoder = (
            WaterOpticsDecoder(spectral_dim, dropout)
            if spectral_dim > 0
            else None
        )
        fused_dim = extractor.model_dim + hidden + spectral_dim
        output_dim = 2 if heteroscedastic else 1
        self.head = (
            nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output_dim),
            )
            if num_experts == 1
            else nn.Identity()
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(fused_dim),
                    nn.Linear(fused_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in range(num_experts if num_experts > 1 else 0)
            ]
        )
        self.expert_gate = (
            nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, num_experts),
            )
            if num_experts > 1
            else None
        )
        self.adversarial_heads = nn.ModuleDict(
            {
                level: nn.Sequential(
                    nn.LayerNorm(extractor.model_dim),
                    nn.Linear(extractor.model_dim, 128),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, classes),
                )
                for level, classes in (adversarial_classes or {}).items()
                if classes > 1
            }
        )

    def pool_tokens(
        self,
        tokens: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.attention_pooling:
            if visible is None:
                return tokens.mean(dim=1)
            return (tokens * visible.unsqueeze(-1)).sum(dim=1) / visible.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        scores = self.view_attention(tokens).squeeze(-1)
        if visible is not None:
            scores = scores.masked_fill(~visible.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).to(dtype=tokens.dtype)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    def adversarial_logits(
        self,
        embedding: torch.Tensor,
        *,
        scale: float,
    ) -> dict[str, torch.Tensor]:
        reversed_embedding = GradientReversal.apply(embedding, scale)
        return {
            level: head(reversed_embedding)
            for level, head in self.adversarial_heads.items()
        }

    def supervised_output(
        self,
        pooled: torch.Tensor,
        domain: torch.Tensor,
        spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        components = [pooled, self.domain_encoder(domain)]
        if self.spectral_encoder is not None:
            components.append(self.spectral_encoder(spectrum))
        fused = torch.cat(components, dim=1)
        if self.expert_gate is not None:
            expert_outputs = torch.cat(
                [expert(fused) for expert in self.experts],
                dim=1,
            )
            weights = torch.softmax(self.expert_gate(fused).float(), dim=1).to(
                dtype=expert_outputs.dtype
            )
            return (weights * expert_outputs).sum(dim=1), None
        output = self.head(fused)
        if self.heteroscedastic:
            return output[:, 0], output[:, 1].clamp(-5.0, 4.0)
        return output.squeeze(-1), None

    def forward_train(
        self,
        spectrum: torch.Tensor,
        domain: torch.Tensor,
        *,
        grid_size: int,
        mask_ratio: float,
        reconstruction_weight: float,
        prototype_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        cube = build_view_cube(spectrum, self.extractor.local_patch_size, grid_size)
        device = next(self.extractor.parameters()).device
        batch = metadata_to_batch(values=cube, metadata=self.extractor.metadata, device=device)
        encoder = self.extractor.encoder
        routing = encoder._build_routing(batch)
        tokenized = encoder.tokenizer(batch.values, routing)
        spatial_grid = (grid_size, grid_size)
        mask = encoder.mask_generator(
            batch_size=batch.batch_size,
            num_tokens=tokenized.local_tokens.shape[1],
            mask_ratio=mask_ratio,
            device=device,
        )
        backbone_output = encoder.backbone(tokenized.local_tokens, grid_size=spatial_grid, masks=mask)
        decoder_prediction = encoder.decoder(
            backbone_output["patch_tokens"],
            grid_size=spatial_grid,
            mask=mask,
            routing=routing,
        )
        reconstruction_losses, _ = encoder._frequency_losses(decoder_prediction, batch, mask)
        visible = (~mask).to(dtype=backbone_output["patch_tokens"].dtype)
        pooled = self.pool_tokens(backbone_output["patch_tokens"], visible)
        features = {
            "patch_tokens": backbone_output["patch_tokens"],
            "sample_context": routing.sample_context,
            "grid_size": spatial_grid,
        }
        prototype_losses = self.extractor.prototype_auxiliary_loss(
            cube,
            features,
            target_bank="eo1",
            center_window=3,
        )
        reconstruction = (
            0.1 * reconstruction_losses["reconstruction_low"]
            + 0.2 * reconstruction_losses["reconstruction_mid"]
            + 0.7 * reconstruction_losses["reconstruction_full"]
        )
        prototype = prototype_losses["proto_kl"] + 0.02 * prototype_losses["proto_balance"]
        auxiliary = reconstruction_weight * reconstruction + prototype_weight * prototype
        mean, log_variance = self.supervised_output(pooled, domain, spectrum)
        return mean, log_variance, {
            "reconstruction": reconstruction.detach(),
            "prototype": prototype.detach(),
            "embedding": pooled,
        }, auxiliary

    def forward_eval(
        self,
        spectrum: torch.Tensor,
        domain: torch.Tensor,
        *,
        grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cube = build_view_cube(spectrum, self.extractor.local_patch_size, grid_size)
        features = self.extractor.forward_features(cube)
        pooled = self.pool_tokens(features["patch_tokens"])
        return self.supervised_output(pooled, domain, spectrum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True full-parameter Aware-HSI inversion fine-tuning.")
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--domain-features-npz", type=Path, required=True)
    parser.add_argument("--groups-npz", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=2.0e-4)
    parser.add_argument("--adapter-lr", type=float, default=2.0e-5)
    parser.add_argument("--backbone-lr", type=float, default=4.0e-6)
    parser.add_argument("--decoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260704, 20260705))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-coverage-min", type=float, default=0.995)
    parser.add_argument("--smoke", action="store_true")
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=value)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_targets(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values.astype(np.float32)
    if transform == "asinh10":
        return np.arcsinh(np.maximum(values, 0.0) / 10.0).astype(np.float32)
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def inverse_targets(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "asinh10":
        return np.maximum(10.0 * np.sinh(values), 0.0)
    return np.maximum(np.expm1(values), 0.0)


def smoothed_group_features(
    values: np.ndarray,
    train_idx: np.ndarray,
    group_levels: list[np.ndarray],
    *,
    smoothing: float = 12.0,
) -> np.ndarray:
    train_mask = np.zeros(values.size, dtype=bool)
    train_mask[train_idx] = True
    global_mean = float(values[train_idx].mean())
    global_std = float(max(values[train_idx].std(), 1.0e-6))
    columns: list[np.ndarray] = []
    for groups in group_levels:
        prior = np.empty(values.size, dtype=np.float32)
        spread = np.empty(values.size, dtype=np.float32)
        support = np.empty(values.size, dtype=np.float32)
        one_hot_categories = np.unique(groups[train_idx])
        for index in range(values.size):
            selected = train_mask & (groups == groups[index])
            if train_mask[index]:
                selected[index] = False
            count = int(selected.sum())
            selected_values = values[selected]
            total = float(selected_values.sum()) if count else 0.0
            prior[index] = (total + smoothing * global_mean) / (count + smoothing)
            spread[index] = (
                float(selected_values.std()) if count >= 2 else global_std
            )
            support[index] = math.log1p(count)
        columns.extend((prior, spread, support))
        for category in one_hot_categories:
            columns.append((groups == category).astype(np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def prepare_karly_timestamps(raw_root: Path, targets: np.ndarray) -> np.ndarray:
    table = pd.read_csv(raw_root / "soilmoisture_dataset.csv")
    spectral_columns = [column for column in table.columns if str(column).isdigit()]
    raw_targets = pd.to_numeric(table["soil_moisture"], errors="coerce").to_numpy()
    spectra = table[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid = np.isfinite(raw_targets) & np.isfinite(spectra).all(axis=1)
    filtered = table.loc[valid].reset_index(drop=True)
    aligned_targets = filtered["soil_moisture"].to_numpy(dtype=np.float32)
    if aligned_targets.shape != targets.shape or not np.allclose(
        aligned_targets, targets, atol=1.0e-5
    ):
        raise ValueError("KarLy raw timestamps do not align with processed targets.")
    timestamps = pd.to_datetime(filtered["datetime"])
    return (timestamps.astype("int64").to_numpy(dtype=np.float64) / 3.6e12).astype(
        np.float32
    )


def temporal_neighbor_features(
    values: np.ndarray,
    train_idx: np.ndarray,
    groups: np.ndarray,
    timestamps_hours: np.ndarray,
    *,
    neighbors: int = 8,
) -> np.ndarray:
    train_mask = np.zeros(values.size, dtype=bool)
    train_mask[train_idx] = True
    global_mean = float(values[train_idx].mean())
    prior = np.empty(values.size, dtype=np.float32)
    nearest_gap = np.empty(values.size, dtype=np.float32)
    local_spread = np.empty(values.size, dtype=np.float32)
    for index in range(values.size):
        candidates = np.flatnonzero(train_mask & (groups == groups[index]))
        candidates = candidates[candidates != index]
        if candidates.size == 0:
            prior[index] = global_mean
            nearest_gap[index] = 24.0
            local_spread[index] = float(values[train_idx].std())
            continue
        gaps = np.abs(timestamps_hours[candidates] - timestamps_hours[index])
        selected = candidates[np.argsort(gaps)[:neighbors]]
        selected_gaps = np.abs(timestamps_hours[selected] - timestamps_hours[index])
        weights = 1.0 / np.maximum(selected_gaps, 1.0 / 60.0)
        weights = weights / weights.sum()
        prior[index] = float(np.sum(weights * values[selected]))
        nearest_gap[index] = float(selected_gaps.min())
        local_spread[index] = (
            float(values[selected].std()) if selected.size >= 2 else 0.0
        )
    return np.stack(
        (prior, np.log1p(nearest_gap), local_spread),
        axis=1,
    ).astype(np.float32)


def build_relaxed_domain_features(
    mode: Mode,
    *,
    targets: np.ndarray,
    train_idx: np.ndarray,
    groups: np.ndarray,
    secondary_groups: np.ndarray,
    tertiary_groups: np.ndarray,
    raw_root: Path | None,
) -> np.ndarray | None:
    if mode.relaxed_features == "none":
        return None
    transformed = transform_targets(targets, mode.target_transform)
    if mode.relaxed_features == "karly_temporal":
        if raw_root is None:
            raise ValueError("--raw-root is required for KarLy temporal features.")
        timestamps = prepare_karly_timestamps(raw_root, targets)
        group_features = smoothed_group_features(
            transformed,
            train_idx,
            [groups, secondary_groups],
            smoothing=10.0,
        )
        temporal_features = temporal_neighbor_features(
            transformed,
            train_idx,
            groups,
            timestamps,
            neighbors=8,
        )
        time_of_day = np.mod(timestamps, 24.0)
        cyclical = np.stack(
            (
                np.sin(2.0 * np.pi * time_of_day / 24.0),
                np.cos(2.0 * np.pi * time_of_day / 24.0),
            ),
            axis=1,
        ).astype(np.float32)
        return np.concatenate((group_features, temporal_features, cyclical), axis=1)
    if mode.relaxed_features == "gloria_identity":
        columns: list[np.ndarray] = []
        for level in (groups, secondary_groups, tertiary_groups):
            categories = np.unique(level[train_idx])
            train_counts = {
                int(category): int(np.sum(level[train_idx] == category))
                for category in categories
            }
            columns.append(
                np.asarray(
                    [math.log1p(train_counts.get(int(value), 0)) for value in level],
                    dtype=np.float32,
                )
            )
            for category in categories:
                columns.append((level == category).astype(np.float32))
        return np.stack(columns, axis=1).astype(np.float32)
    return smoothed_group_features(
        transformed,
        train_idx,
        [groups, secondary_groups, tertiary_groups],
        smoothing=15.0,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float("nan") if denominator <= 0 else 1.0 - float(np.sum(error**2)) / denominator,
    }


def augment_spectrum(spectrum: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "none":
        return spectrum
    batch, bands = spectrum.shape
    device = spectrum.device
    dtype = spectrum.dtype
    axis = torch.linspace(-1.0, 1.0, bands, device=device, dtype=dtype).unsqueeze(0)
    scale = spectrum.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    if kind.startswith("karly"):
        gain = 1.0 + 0.025 * torch.randn(batch, 1, device=device, dtype=dtype)
        slope = 0.025 * torch.randn(batch, 1, device=device, dtype=dtype) * axis
        baseline = 0.012 * torch.randn(batch, 1, device=device, dtype=dtype) * scale
        output = spectrum * (gain + slope) + baseline
        if kind == "karly_mixstyle" and batch > 1:
            order = torch.randperm(batch, device=device)
            own_mean = output.mean(dim=1, keepdim=True)
            own_std = output.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
            other_mean = own_mean[order]
            other_std = own_std[order]
            mix = torch.distributions.Beta(0.4, 0.4).sample((batch, 1)).to(device=device, dtype=dtype)
            mixed_mean = mix * own_mean + (1.0 - mix) * other_mean
            mixed_std = mix * own_std + (1.0 - mix) * other_std
            output = (output - own_mean) / own_std * mixed_std + mixed_mean
        noise = 0.004 * scale * torch.randn_like(output)
    else:
        strength = 1.0 if kind == "gloria_instrument" else 1.6
        gain = 1.0 + strength * 0.02 * torch.randn(batch, 1, device=device, dtype=dtype)
        tilt = strength * 0.025 * torch.randn(batch, 1, device=device, dtype=dtype) * axis
        curvature = strength * 0.012 * torch.randn(batch, 1, device=device, dtype=dtype) * (
            axis.square() - axis.square().mean()
        )
        output = spectrum * (gain + tilt + curvature)
        baseline = strength * 0.006 * torch.randn(batch, 1, device=device, dtype=dtype) * scale
        output = output + baseline
        noise = strength * 0.003 * scale * torch.randn_like(output)
    output = output + noise
    shifts = torch.randint(-1, 2, (batch,), device=device)
    shifted = output.clone()
    for shift in (-1, 1):
        selected = shifts == shift
        if torch.any(selected):
            rolled = torch.roll(output[selected], shifts=shift, dims=1)
            if shift < 0:
                rolled[:, -1] = rolled[:, -2]
            else:
                rolled[:, 0] = rolled[:, 1]
            shifted[selected] = rolled
    return shifted


def shift_spectrum(spectrum: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return spectrum
    shifted = torch.roll(spectrum, shifts=shift, dims=1)
    if shift < 0:
        shifted[:, shift:] = shifted[:, shift - 1 : shift]
    else:
        shifted[:, :shift] = shifted[:, shift : shift + 1]
    return shifted


def pairwise_rank_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    order = torch.randperm(prediction.numel(), device=prediction.device)
    target_difference = target - target[order]
    sign = torch.sign(target_difference)
    valid = sign.ne(0)
    if not torch.any(valid):
        return prediction.new_zeros(())
    return F.softplus(-sign[valid] * (prediction - prediction[order])[valid]).mean()


def feature_geometry_loss(embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embedding, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    distance = torch.abs(target[:, None] - target[None, :])
    desired = torch.exp(-distance / 0.5)
    return F.mse_loss((similarity + 1.0) * 0.5, desired)


def coral_group_loss(embedding: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(groups)
    means = []
    for group in unique:
        selected = embedding[groups == group]
        if selected.shape[0] >= 2:
            means.append(selected.mean(dim=0))
    if len(means) < 2:
        return embedding.new_zeros(())
    stacked = torch.stack(means)
    return stacked.var(dim=0, unbiased=False).mean()


def group_rex_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for group in torch.unique(groups):
        selected = groups == group
        if int(selected.sum()) >= 2:
            losses.append(F.smooth_l1_loss(prediction[selected], target[selected], beta=0.5))
    if len(losses) < 2:
        return prediction.new_zeros(())
    return torch.stack(losses).var(unbiased=False)


def temporal_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for group in torch.unique(groups):
        selected = torch.nonzero(groups == group, as_tuple=False).flatten()
        if selected.numel() < 2:
            continue
        order = selected[torch.argsort(indices[selected])]
        losses.append(
            F.smooth_l1_loss(
                prediction[order[1:]] - prediction[order[:-1]],
                target[order[1:]] - target[order[:-1]],
                beta=0.25,
            )
        )
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def fit_calibration(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "slope": 1.0, "intercept": 0.0, "group_offsets": {}}
    x = predictions.astype(np.float64)
    y = targets.astype(np.float64)
    variance = float(np.var(x))
    slope = 1.0 if variance < 1.0e-12 else float(np.cov(x, y, bias=True)[0, 1] / variance)
    slope = float(np.clip(slope, 0.5, 1.5))
    intercept = float(np.mean(y - slope * x))
    calibration: dict[str, Any] = {
        "mode": mode,
        "slope": slope,
        "intercept": intercept,
        "group_offsets": {},
    }
    if mode == "hierarchical":
        residual = y - (slope * x + intercept)
        offsets = {}
        for group in np.unique(groups):
            selected = groups == group
            count = int(selected.sum())
            shrinkage = count / (count + 12.0)
            offsets[str(int(group))] = float(shrinkage * residual[selected].mean())
        calibration["group_offsets"] = offsets
    return calibration


def apply_calibration(
    predictions: np.ndarray,
    groups: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    output = (
        float(calibration.get("slope", 1.0)) * predictions
        + float(calibration.get("intercept", 0.0))
    )
    offsets = calibration.get("group_offsets", {})
    if offsets:
        output = output + np.asarray(
            [float(offsets.get(str(int(group)), 0.0)) for group in groups],
            dtype=np.float32,
        )
    return output.astype(np.float32)


def regression_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor | None,
    target: torch.Tensor,
    mode: Mode,
    groups: torch.Tensor,
) -> torch.Tensor:
    if mode.heteroscedastic:
        per_item = 0.5 * (torch.exp(-log_variance) * (mean - target) ** 2 + log_variance)
    elif mode.loss == "smoothl1":
        per_item = F.smooth_l1_loss(mean, target, beta=0.5, reduction="none")
    else:
        per_item = F.mse_loss(mean, target, reduction="none")
    if mode.dataset == "gloria":
        per_item = per_item * (
            1.0
            + mode.gloria_tail_weight
            * target.detach().clamp_min(0.0).pow(2).clamp(max=8.0)
        )
    if mode.group_dro:
        group_losses = []
        for group in torch.unique(groups):
            group_losses.append(per_item[groups == group].mean())
        return torch.stack(group_losses).max()
    return per_item.mean()


def gradient_audit(model: nn.Module) -> dict[str, Any]:
    prediction_modules: list[nn.Module] = [
        model.domain_encoder,
        model.head,
        model.view_attention,
        model.adversarial_heads,
        model.experts,
    ]
    if model.expert_gate is not None:
        prediction_modules.append(model.expert_gate)
    if model.spectral_encoder is not None:
        prediction_modules.append(model.spectral_encoder)
    modules = {
        "band_encoder": model.extractor.encoder.band_encoder,
        "sample_encoder": model.extractor.encoder.sample_encoder,
        "hypernet": model.extractor.encoder.shared_hypernet,
        "tokenizer": model.extractor.encoder.tokenizer,
        "backbone": model.extractor.encoder.backbone,
        "prototype_head": model.extractor.encoder.prototype_query_head,
        "decoder": model.extractor.encoder.decoder,
        "domain_head": nn.ModuleList(prediction_modules),
    }
    report: dict[str, Any] = {}
    total_parameters = 0
    gradient_parameters = 0
    for name, module in modules.items():
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        total = sum(parameter.numel() for parameter in parameters)
        with_gradient = sum(
            parameter.numel()
            for parameter in parameters
            if parameter.grad is not None and torch.isfinite(parameter.grad).all()
        )
        total_parameters += total
        gradient_parameters += with_gradient
        report[name] = {
            "trainable": int(total),
            "with_finite_gradient": int(with_gradient),
            "coverage": 0.0 if total == 0 else float(with_gradient / total),
        }
    report["overall"] = {
        "trainable": int(total_parameters),
        "with_finite_gradient": int(gradient_parameters),
        "coverage": float(gradient_parameters / max(total_parameters, 1)),
    }
    return report


def make_optimizer(model: TrueFullFTRegressor, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = {
        "head": [],
        "adapter": [],
        "backbone": [],
        "decoder": [],
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        if (
            name.startswith("head.")
            or name.startswith("domain_encoder.")
            or name.startswith("view_attention.")
            or name.startswith("adversarial_heads.")
            or name.startswith("spectral_encoder.")
            or name.startswith("experts.")
            or name.startswith("expert_gate.")
        ):
            groups["head"].append(parameter)
        elif ".backbone." in name:
            groups["backbone"].append(parameter)
        elif ".decoder." in name:
            groups["decoder"].append(parameter)
        else:
            groups["adapter"].append(parameter)
    return torch.optim.AdamW(
        [
            {"params": groups["head"], "lr": args.head_lr, "name": "head"},
            {"params": groups["adapter"], "lr": args.adapter_lr, "name": "adapter"},
            {"params": groups["backbone"], "lr": args.backbone_lr, "name": "backbone"},
            {"params": groups["decoder"], "lr": args.decoder_lr, "name": "decoder"},
        ],
        weight_decay=args.weight_decay,
    )


def lr_multiplier(epoch_index: int, warmup: int, total: int) -> float:
    if epoch_index < warmup:
        return float(epoch_index + 1) / max(1, warmup)
    progress = (epoch_index - warmup) / max(1, total - warmup)
    return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def average_checkpoint_models(paths: list[Path]) -> dict[str, torch.Tensor]:
    if not paths:
        raise ValueError("No checkpoints were provided for averaging.")
    averaged: dict[str, torch.Tensor] = {}
    for checkpoint_index, path in enumerate(paths):
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        for name, value in state.items():
            cpu_value = value.detach().cpu()
            if checkpoint_index == 0:
                averaged[name] = (
                    cpu_value.float().clone()
                    if torch.is_floating_point(cpu_value)
                    else cpu_value.clone()
                )
            elif torch.is_floating_point(cpu_value):
                averaged[name].add_(cpu_value.float())
        del state
    divisor = float(len(paths))
    for name, value in averaged.items():
        if torch.is_floating_point(value):
            averaged[name] = value.div_(divisor)
    return averaged


@torch.no_grad()
def evaluate(
    model: TrueFullFTRegressor,
    loader: DataLoader,
    *,
    device: torch.device,
    mode: Mode,
    precision: str,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truth, predictions, indices = [], [], []
    for batch in loader:
        spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
        domain = batch["domain"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            means = [
                model.forward_eval(
                    shift_spectrum(spectrum, shift),
                    domain,
                    grid_size=mode.grid_size,
                )[0]
                for shift in mode.tta_shifts
            ]
            mean = torch.stack(means, dim=0).mean(dim=0)
        transformed = mean.float().cpu().numpy() * target_std + target_mean
        predictions.append(inverse_targets(transformed, mode.target_transform))
        truth.append(batch["raw_target"].numpy())
        indices.append(batch["index"].numpy())
    y_true = np.concatenate(truth).astype(np.float32)
    y_pred = np.concatenate(predictions).astype(np.float32)
    return metrics(y_true, y_pred), y_true, y_pred, np.concatenate(indices)


def train_seed(
    seed: int,
    *,
    arrays: dict[str, Any],
    metadata: Any,
    mode: Mode,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = args.output_dir / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "val_metrics.json"
    if final_path.exists() and (run_dir / "best_model.pt").exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    seed_everything(seed)
    transformed_train = transform_targets(arrays["targets"][arrays["train_idx"]], mode.target_transform)
    target_mean = float(transformed_train.mean())
    target_std = float(max(transformed_train.std(), 1.0e-6))
    datasets = {
        split: FullFTDataset(
            arrays["spectra"],
            arrays["domain_features"],
            arrays["targets"],
            arrays["groups"],
            arrays["secondary_groups"],
            arrays["tertiary_groups"],
            arrays[f"{split}_idx"],
            target_mean=target_mean,
            target_std=target_std,
            transform=mode.target_transform,
        )
        for split in ("train", "val", "test")
    }
    common = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    if mode.balanced_groups:
        train_indices = arrays["train_idx"]
        train_targets = arrays["targets"][train_indices]
        target_bins = np.digitize(
            train_targets,
            np.quantile(train_targets, [0.2, 0.4, 0.6, 0.8]),
        )
        train_groups = arrays["groups"][train_indices]
        joint = train_groups * 5 + target_bins
        counts = np.bincount(joint)
        weights = 1.0 / np.maximum(counts[joint], 1)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights.astype(np.float64)),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(datasets["train"], sampler=sampler, **common)
    elif mode.dataset == "gloria":
        train_targets = arrays["targets"][arrays["train_idx"]]
        bins = np.digitize(train_targets, np.quantile(train_targets, [0.2, 0.4, 0.6, 0.8]))
        counts = np.bincount(bins, minlength=5)
        weights = 1.0 / np.maximum(counts[bins], 1)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights.astype(np.float64)),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(datasets["train"], sampler=sampler, **common)
    else:
        train_loader = DataLoader(datasets["train"], shuffle=True, **common)
    val_loader = DataLoader(datasets["val"], shuffle=False, **common)
    test_loader = DataLoader(datasets["test"], shuffle=False, **common)

    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=False,
        device=device,
    )
    model = TrueFullFTRegressor(
        extractor,
        domain_dim=int(arrays["domain_features"].shape[1]),
        heteroscedastic=mode.heteroscedastic,
        dropout=args.dropout,
        attention_pooling=mode.attention_pooling,
        spectral_decoder=mode.spectral_decoder,
        num_experts=mode.num_experts,
        adversarial_classes={
            level: int(
                arrays[
                    {
                        "primary": "groups",
                        "secondary": "secondary_groups",
                        "tertiary": "tertiary_groups",
                    }[level]
                ].max()
                + 1
            )
            for level in mode.adversarial_levels
        },
    ).to(device)
    optimizer = make_optimizer(model, args)
    max_epochs = 2 if args.smoke else mode.max_epochs
    patience = 2 if args.smoke else mode.patience
    min_epochs = 0 if args.smoke else mode.min_epochs
    schedule_epochs = (
        max_epochs
        if args.smoke or mode.schedule_epochs <= 0
        else mode.schedule_epochs
    )
    soup_start_epoch = (
        1 if args.smoke and mode.soup_topk > 0 else mode.soup_start_epoch
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_multiplier(
            epoch,
            mode.warmup_epochs,
            schedule_epochs,
        ),
    )
    total_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config = {
        "mode": args.mode,
        "seed": seed,
        "mode_config": mode.__dict__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(total_trainable),
        "trainable_fraction": float(
            total_trainable / max(sum(parameter.numel() for parameter in model.parameters()), 1)
        ),
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)

    best = float("inf")
    stale = 0
    best_epoch = 0
    gradient_report = None
    late_candidates: list[tuple[float, int, Path]] = []
    log_path = run_dir / "metrics.jsonl"
    for epoch in range(1, max_epochs + 1):
        model.train()
        started = time.time()
        losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            spectrum = batch["spectrum"].to(device=device, dtype=torch.float32, non_blocking=True)
            domain = batch["domain"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            groups = batch["group"].to(device=device, dtype=torch.long, non_blocking=True)
            secondary_groups = batch["secondary_group"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            tertiary_groups = batch["tertiary_group"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            indices = batch["index"].to(device=device, dtype=torch.long, non_blocking=True)
            spectrum = augment_spectrum(spectrum, mode.spectral_augmentation)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bf16",
            ):
                mean, log_variance, auxiliary_metrics, auxiliary = model.forward_train(
                    spectrum,
                    domain,
                    grid_size=mode.grid_size,
                    mask_ratio=mode.mask_ratio,
                    reconstruction_weight=mode.reconstruction_weight,
                    prototype_weight=mode.prototype_weight,
                )
                supervised = regression_loss(mean, log_variance, target, mode, groups)
                loss = supervised + auxiliary
                embedding = auxiliary_metrics["embedding"]
                if mode.rank_weight > 0.0:
                    loss = loss + mode.rank_weight * pairwise_rank_loss(mean, target)
                if mode.feature_weight > 0.0:
                    loss = loss + mode.feature_weight * feature_geometry_loss(embedding, target)
                if mode.coral_weight > 0.0:
                    loss = loss + mode.coral_weight * coral_group_loss(embedding, groups)
                if mode.rex_weight > 0.0:
                    loss = loss + mode.rex_weight * group_rex_loss(mean, target, groups)
                if mode.temporal_weight > 0.0:
                    loss = loss + mode.temporal_weight * temporal_difference_loss(
                        mean, target, groups, indices
                    )
                if mode.adversarial_weight > 0.0:
                    labels = {
                        "primary": groups,
                        "secondary": secondary_groups,
                        "tertiary": tertiary_groups,
                    }
                    logits = model.adversarial_logits(embedding, scale=1.0)
                    adversarial = torch.stack(
                        [F.cross_entropy(value, labels[level]) for level, value in logits.items()]
                    ).mean()
                    loss = loss + mode.adversarial_weight * adversarial
            loss.backward()
            if gradient_report is None:
                gradient_report = gradient_audit(model)
                (run_dir / "gradient_audit.json").write_text(
                    json.dumps(gradient_report, indent=2),
                    encoding="utf-8",
                )
                coverage = gradient_report["overall"]["coverage"]
                print(json.dumps({"gradient_audit": gradient_report}, indent=2), flush=True)
                if coverage < args.gradient_coverage_min:
                    raise RuntimeError(
                        f"Gradient coverage {coverage:.6f} is below required {args.gradient_coverage_min:.6f}."
                    )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        val_metrics_raw, val_targets_epoch, val_predictions_epoch, val_indices_epoch = evaluate(
            model,
            val_loader,
            device=device,
            mode=mode,
            precision=args.precision,
            target_mean=target_mean,
            target_std=target_std,
        )
        calibration = fit_calibration(
            val_targets_epoch,
            val_predictions_epoch,
            arrays["groups"][val_indices_epoch],
            mode.calibration,
        )
        calibrated_val_predictions = apply_calibration(
            val_predictions_epoch,
            arrays["groups"][val_indices_epoch],
            calibration,
        )
        val_metrics = metrics(val_targets_epoch, calibrated_val_predictions)
        improved = val_metrics["rmse"] < best
        if improved:
            best = val_metrics["rmse"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_rmse": best,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "calibration": calibration,
                    "raw_val_metrics": val_metrics_raw,
                    "config": config,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale += 1
        if mode.soup_topk > 0 and epoch >= soup_start_epoch:
            candidate_path = run_dir / f"late_candidate_epoch{epoch:04d}.pt"
            should_save = (
                len(late_candidates) < mode.soup_topk
                or val_metrics["rmse"] < late_candidates[-1][0]
            )
            if should_save:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_val_rmse": val_metrics["rmse"],
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "calibration": calibration,
                        "raw_val_metrics": val_metrics_raw,
                        "config": config,
                    },
                    candidate_path,
                )
                late_candidates.append(
                    (float(val_metrics["rmse"]), epoch, candidate_path)
                )
                late_candidates.sort(key=lambda item: item[0])
                while len(late_candidates) > mode.soup_topk:
                    _, _, removed_path = late_candidates.pop()
                    removed_path.unlink(missing_ok=True)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": val_metrics,
            "best_val_rmse": best,
            "best_epoch": best_epoch,
            "stale": stale,
            "seconds": time.time() - started,
            "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_rmse={val_metrics['rmse']:.8g} best={best:.8g}@{best_epoch} "
            f"stale={stale} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if epoch >= min_epochs and stale >= patience:
            break

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    val_metrics_raw, val_targets, val_predictions_raw, val_indices = evaluate(
        model,
        val_loader,
        device=device,
        mode=mode,
        precision=args.precision,
        target_mean=target_mean,
        target_std=target_std,
    )
    calibration = checkpoint.get(
        "calibration",
        {"mode": "none", "slope": 1.0, "intercept": 0.0, "group_offsets": {}},
    )
    val_predictions = apply_calibration(
        val_predictions_raw,
        arrays["groups"][val_indices],
        calibration,
    )
    val_metrics = metrics(val_targets, val_predictions)
    selected_kind = "best_single"
    soup_summary: dict[str, Any] | None = None
    if late_candidates:
        soup_paths = [item[2] for item in late_candidates]
        soup_epochs = [item[1] for item in late_candidates]
        averaged_state = average_checkpoint_models(soup_paths)
        model.load_state_dict(averaged_state, strict=True)
        soup_raw_metrics, soup_targets, soup_predictions_raw, soup_indices = evaluate(
            model,
            val_loader,
            device=device,
            mode=mode,
            precision=args.precision,
            target_mean=target_mean,
            target_std=target_std,
        )
        soup_calibration = fit_calibration(
            soup_targets,
            soup_predictions_raw,
            arrays["groups"][soup_indices],
            mode.calibration,
        )
        soup_predictions = apply_calibration(
            soup_predictions_raw,
            arrays["groups"][soup_indices],
            soup_calibration,
        )
        soup_metrics = metrics(soup_targets, soup_predictions)
        soup_summary = {
            "epochs": soup_epochs,
            "val": soup_metrics,
            "raw_val": soup_raw_metrics,
        }
        if soup_metrics["rmse"] < val_metrics["rmse"]:
            selected_kind = "late_checkpoint_soup"
            checkpoint = {
                "model": averaged_state,
                "epoch": soup_epochs,
                "best_val_rmse": soup_metrics["rmse"],
                "target_mean": target_mean,
                "target_std": target_std,
                "calibration": soup_calibration,
                "raw_val_metrics": soup_raw_metrics,
                "config": config,
            }
            torch.save(checkpoint, run_dir / "selected_model.pt")
            val_metrics_raw = soup_raw_metrics
            val_targets = soup_targets
            val_predictions_raw = soup_predictions_raw
            val_indices = soup_indices
            val_predictions = soup_predictions
            val_metrics = soup_metrics
            calibration = soup_calibration
        for _, _, candidate_path in late_candidates:
            candidate_path.unlink(missing_ok=True)
    np.savez_compressed(
        run_dir / "val_predictions.npz",
        targets=val_targets,
        predictions=val_predictions,
        raw_predictions=val_predictions_raw,
        indices=val_indices,
    )
    result = {
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "best_val_rmse": checkpoint["best_val_rmse"],
        "val": val_metrics,
        "raw_val": val_metrics_raw,
        "calibration": calibration,
        "selected_kind": selected_kind,
        "late_soup": soup_summary,
        "gradient_coverage": gradient_report["overall"]["coverage"] if gradient_report else None,
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    del optimizer
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    mode = MODES[args.mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "test_metrics.json"
    if final_path.exists() and not args.smoke:
        print(f"[skip] completed: {final_path}")
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        spectra = np.asarray(payload["spectra"], dtype=np.float32)
        targets = np.asarray(payload["targets"], dtype=np.float32)
        wavelengths = np.asarray(payload["wavelengths_nm"], dtype=np.float32)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        dataset_name = str(np.asarray(payload["dataset"]).item())
        target_name = str(np.asarray(payload["target_name"]).item())
    with np.load(args.domain_features_npz, allow_pickle=True) as payload:
        domain_features = np.asarray(payload["domain_features"], dtype=np.float32)
    with np.load(args.groups_npz, allow_pickle=True) as payload:
        groups = np.asarray(payload["groups"], dtype=np.int64)
        metadata_json = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if mode.dataset == "karly":
        secondary_groups = np.asarray(metadata_json["day_ids"], dtype=np.int64)
        tertiary_groups = secondary_groups.copy()
    else:
        secondary_groups = np.asarray(metadata_json["organization_ids"], dtype=np.int64)
        tertiary_groups = np.asarray(metadata_json["site_ids"], dtype=np.int64)
    if not (
        groups.shape
        == secondary_groups.shape
        == tertiary_groups.shape
        == targets.shape
    ):
        raise ValueError("Domain-label arrays do not align with dataset targets.")
    relaxed_features = build_relaxed_domain_features(
        mode,
        targets=targets,
        train_idx=train_idx,
        groups=groups,
        secondary_groups=secondary_groups,
        tertiary_groups=tertiary_groups,
        raw_root=args.raw_root,
    )
    if relaxed_features is not None:
        domain_features = np.concatenate(
            (domain_features, relaxed_features),
            axis=1,
        ).astype(np.float32)
    domain_mean = domain_features[train_idx].mean(axis=0)
    domain_std = np.maximum(domain_features[train_idx].std(axis=0), 1.0e-6)
    domain_features = ((domain_features - domain_mean) / domain_std).astype(np.float32)
    arrays = {
        "spectra": spectra,
        "domain_features": domain_features,
        "targets": targets,
        "groups": groups,
        "secondary_groups": secondary_groups,
        "tertiary_groups": tertiary_groups,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    metadata = build_downstream_metadata(
        num_bands=int(wavelengths.size),
        sensor="field",
        level="field",
        gsd=1.0,
        wavelengths_nm=wavelengths,
    )
    seeds = tuple(args.seeds[:1]) if args.smoke else tuple(args.seeds)
    seed_results = [
        train_seed(
            seed,
            arrays=arrays,
            metadata=metadata,
            mode=mode,
            args=args,
            device=device,
        )
        for seed in seeds
    ]
    transformed_train = transform_targets(targets[train_idx], mode.target_transform)
    target_mean = float(transformed_train.mean())
    target_std = float(max(transformed_train.std(), 1.0e-6))
    test_dataset = FullFTDataset(
        spectra,
        domain_features,
        targets,
        groups,
        secondary_groups,
        tertiary_groups,
        test_idx,
        target_mean=target_mean,
        target_std=target_std,
        transform=mode.target_transform,
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
        extractor = AwareFeatureExtractor(
            checkpoint_path=args.checkpoint,
            prototype_bank_eo1=args.prototype_bank_eo1,
            prototype_bank_gf5=args.prototype_bank_gf5,
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        model = TrueFullFTRegressor(
            extractor,
            domain_dim=int(domain_features.shape[1]),
            heteroscedastic=mode.heteroscedastic,
            dropout=args.dropout,
            attention_pooling=mode.attention_pooling,
            spectral_decoder=mode.spectral_decoder,
            num_experts=mode.num_experts,
            adversarial_classes={
                level: int(
                    {
                        "primary": groups,
                        "secondary": secondary_groups,
                        "tertiary": tertiary_groups,
                    }[level].max()
                    + 1
                )
                for level in mode.adversarial_levels
            },
        ).to(device)
        seed_dir = args.output_dir / f"seed{seed}"
        selected_path = seed_dir / "selected_model.pt"
        checkpoint = torch.load(
            selected_path if selected_path.exists() else seed_dir / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        _, test_targets, prediction_raw, test_indices = evaluate(
            model,
            test_loader,
            device=device,
            mode=mode,
            precision=args.precision,
            target_mean=checkpoint["target_mean"],
            target_std=checkpoint["target_std"],
        )
        prediction = apply_calibration(
            prediction_raw,
            groups[test_indices],
            checkpoint.get(
                "calibration",
                {"mode": "none", "slope": 1.0, "intercept": 0.0, "group_offsets": {}},
            ),
        )
        test_predictions.append(prediction)
        del model
        torch.cuda.empty_cache()
    ensemble_prediction = np.mean(test_predictions, axis=0)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        targets=test_targets,
        predictions=ensemble_prediction,
        test_idx=test_indices,
    )
    result = {
        "model": f"aware_{args.mode}",
        "dataset": dataset_name,
        "target": target_name,
        "seeds": list(seeds),
        "seed_results": seed_results,
        "test": metrics(test_targets, ensemble_prediction),
        "protocol": (
            "true full-parameter Aware-HSI fine-tuning with tokenizer, band/sample encoders, hypernet, "
            "backbone, decoder, prototype head and regression head receiving gradients; multi-seed "
            "validation-checkpoint ensemble; test evaluated only after all seeds finish"
        ),
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
