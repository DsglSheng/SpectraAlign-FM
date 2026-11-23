from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FoundationConfig
from ..data.frequency import decompose_frequency_targets, radial_frequency_masks
from ..data_types import ModalityBatch
from ..spectral_prototypes import compute_patch_mean_spectra, entropy_balance_loss, prototype_teacher_distribution
from .backbone import TokenTransformerEncoder
from .common import FeedForward, SelfAttentionBlock
from .hypernet import BandRouting, SharedBandHyperNet
from .metadata import BandMetadataEncoder, MetadataVocabulary, SampleMetadataEncoder
from .tokenizers import AirSpaceTokenizer2D


@dataclass
class PreparedSpaceBatch:
    tokens: torch.Tensor
    grid_size: tuple[int, int]
    mask: torch.Tensor
    routing: BandRouting
    values: torch.Tensor


class RandomMaskGenerator(nn.Module):
    def forward(self, batch_size: int, num_tokens: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        if num_tokens < 2:
            raise ValueError("Masked modeling requires at least two patch tokens.")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}.")

        num_masked = int(round(num_tokens * mask_ratio))
        num_masked = max(1, min(num_tokens - 1, num_masked))

        noise = torch.rand(batch_size, num_tokens, device=device)
        order = torch.argsort(noise, dim=1)
        masked = order[:, :num_masked]

        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
        mask.scatter_(1, masked, True)
        return mask


class LowRankPatchReconstructionHead(nn.Module):
    def __init__(self, decoder_dim: int, band_dim: int, patch_size: int, rank: int) -> None:
        super().__init__()
        patch_area = patch_size * patch_size
        self.rank = rank
        self.token_to_basis = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, rank * patch_area),
        )
        self.band_to_coeff = nn.Sequential(
            nn.LayerNorm(band_dim),
            nn.Linear(band_dim, band_dim),
            nn.GELU(),
            nn.Linear(band_dim, rank),
        )
        self.band_to_bias = nn.Sequential(
            nn.LayerNorm(band_dim),
            nn.Linear(band_dim, band_dim),
            nn.GELU(),
            nn.Linear(band_dim, patch_area),
        )

    def forward(self, decoded_tokens: torch.Tensor, band_tokens: torch.Tensor) -> torch.Tensor:
        basis = self.token_to_basis(decoded_tokens).view(decoded_tokens.shape[0], decoded_tokens.shape[1], self.rank, -1)
        coeff = self.band_to_coeff(band_tokens)
        bias = self.band_to_bias(band_tokens)
        pred = torch.einsum("blrp,bcr->blcp", basis, coeff)
        pred = pred + bias.unsqueeze(1)
        return pred.flatten(-2)


def masked_token_pool(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    visible = (~mask).float().unsqueeze(-1)
    denom = visible.sum(dim=1).clamp_min(1.0)
    return (tokens * visible).sum(dim=1) / denom


class PrototypeQueryHead(nn.Module):
    def __init__(self, model_dim: int, context_dim: int, prototype_count: int) -> None:
        super().__init__()
        self.metadata_proj = nn.Linear(context_dim, model_dim)
        self.query = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, prototype_count),
        )

    def forward(self, patch_tokens: torch.Tensor, sample_context: torch.Tensor) -> torch.Tensor:
        conditioned = patch_tokens + self.metadata_proj(sample_context).unsqueeze(1)
        return self.query(conditioned)


class FrequencyEmbedding(nn.Module):
    def __init__(self, decoder_dim: int, context_dim: int) -> None:
        super().__init__()
        self.register_buffer("low_seed", self._build_seed(decoder_dim, math.pi))
        self.register_buffer("mid_seed", self._build_seed(decoder_dim, 3.0 * math.pi))
        self.register_buffer("high_seed", self._build_seed(decoder_dim, 6.0 * math.pi))
        self.refine = nn.ModuleDict(
            {
                "low": nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, decoder_dim)),
                "mid": nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, decoder_dim)),
                "full": nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, decoder_dim)),
            }
        )
        self.context_proj = nn.Linear(context_dim, decoder_dim)
        self.cls_proj = nn.Sequential(
            nn.LayerNorm(decoder_dim * 3),
            nn.Linear(decoder_dim * 3, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    @staticmethod
    def _build_seed(decoder_dim: int, multiplier: float) -> torch.Tensor:
        position = torch.arange(decoder_dim, dtype=torch.float32)
        return torch.cos(multiplier * position / max(decoder_dim - 1, 1)) * 0.1

    def forward(self, sample_context: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context_proj(sample_context)
        low = self.refine["low"](self.low_seed.view(1, -1).expand(sample_context.shape[0], -1)) + context
        mid = self.refine["mid"](self.mid_seed.view(1, -1).expand(sample_context.shape[0], -1)) + context
        full = self.refine["full"](self.high_seed.view(1, -1).expand(sample_context.shape[0], -1)) + context
        cls = self.cls_proj(torch.cat([low, mid, full], dim=-1))
        return {
            "cls": cls,
            "mask_tokens": {
                "low": low,
                "mid": mid,
                "full": full,
            },
        }


def _safe_group_count(channels: int, requested: int) -> int:
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SpectralSpatialContextInjection(nn.Module):
    def __init__(self, dim: int, context_dim: int, spectral_dim: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.context_proj = nn.Linear(context_dim, dim)
        self.spectral_proj = nn.Linear(spectral_dim, dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, sample_context: torch.Tensor, spectral_context: torch.Tensor) -> torch.Tensor:
        base = self.token_norm(tokens)
        context = self.context_proj(sample_context).unsqueeze(1).expand_as(base)
        spectral = self.spectral_proj(spectral_context).unsqueeze(1).expand_as(base)
        gate = self.gate(torch.cat([base, context, spectral], dim=-1))
        return base + gate * (context + spectral)


class GroupedFrequencyAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spectral_context_dim: int,
        *,
        num_groups: int,
        low_cutoff_ratio: float,
        mid_cutoff_ratio: float,
    ) -> None:
        super().__init__()
        self.low_cutoff_ratio = low_cutoff_ratio
        self.mid_cutoff_ratio = mid_cutoff_ratio
        groups = _safe_group_count(dim, num_groups)
        self.ssci = SpectralSpatialContextInjection(dim, context_dim, spectral_context_dim)
        self.amplitude_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=groups),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, groups=groups),
        )
        self.phase_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=groups),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, groups=groups),
        )
        self.band_attention = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=groups),
            nn.GELU(),
            nn.Conv2d(dim, 3, kernel_size=1),
        )
        self.context_bias = nn.Linear(context_dim, 3)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        *,
        sample_context: torch.Tensor,
        spectral_context: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        height, width = grid_size
        if height * width != num_tokens:
            raise ValueError(f"grid_size {grid_size} does not match {num_tokens} tokens.")

        injected = self.ssci(tokens, sample_context, spectral_context)
        feature_map = injected.transpose(1, 2).reshape(batch_size, dim, height, width)
        fft = torch.fft.fft2(feature_map.float(), dim=(-2, -1))
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)

        amplitude = amplitude * torch.sigmoid(self.amplitude_proj(amplitude))
        phase = phase + torch.tanh(self.phase_proj(phase))
        complex_feature = torch.polar(amplitude.clamp_min(1.0e-6), phase)

        low_mask, mid_mask, high_mask = radial_frequency_masks(
            height,
            width,
            low_cutoff_ratio=self.low_cutoff_ratio,
            mid_cutoff_ratio=self.mid_cutoff_ratio,
            device=tokens.device,
        )

        def _ifft(mask: torch.Tensor) -> torch.Tensor:
            masked = complex_feature * mask.view(1, 1, height, width)
            return torch.fft.ifft2(masked, dim=(-2, -1)).real

        low_component = _ifft(low_mask)
        mid_component = _ifft(mid_mask)
        high_component = _ifft(high_mask)

        weights = self.band_attention(feature_map) + self.context_bias(sample_context).view(batch_size, 3, 1, 1)
        weights = torch.softmax(weights, dim=1)

        if stage == "low":
            fused = low_component * weights[:, 0:1]
        elif stage == "mid":
            fused = low_component * weights[:, 0:1] + mid_component * weights[:, 1:2]
        elif stage == "full":
            fused = (
                low_component * weights[:, 0:1]
                + mid_component * weights[:, 1:2]
                + high_component * weights[:, 2:3]
            )
        else:
            raise ValueError(f"Unsupported progressive stage {stage!r}.")

        return fused.flatten(2).transpose(1, 2).to(dtype=tokens.dtype)


class TriDomainFusionAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        spectral_context_dim: int,
        *,
        num_groups: int,
        low_cutoff_ratio: float,
        mid_cutoff_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gfa = GroupedFrequencyAttention(
            dim=dim,
            context_dim=context_dim,
            spectral_context_dim=spectral_context_dim,
            num_groups=num_groups,
            low_cutoff_ratio=low_cutoff_ratio,
            mid_cutoff_ratio=mid_cutoff_ratio,
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        *,
        sample_context: torch.Tensor,
        spectral_context: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        normed = self.norm(tokens)
        self_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        gfa_out = self.gfa(
            normed,
            grid_size,
            sample_context=sample_context,
            spectral_context=spectral_context,
            stage=stage,
        )
        tokens = tokens + self.fuse(torch.cat([self_out, gfa_out], dim=-1))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class ProgressiveFrequencyDecoder(nn.Module):
    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        encoder = config.encoder
        decoder = config.decoder
        decoder_dim = decoder.decoder_dim
        self.decoder_dim = decoder_dim
        self.patch_size = encoder.local_patch_size
        self.shared_input = nn.Linear(encoder.model_dim, decoder_dim)
        self.shared_blocks = nn.ModuleList(
            [SelfAttentionBlock(decoder_dim, decoder.decoder_heads, dropout=decoder.decoder_dropout) for _ in range(decoder.decoder_depth)]
        )
        self.frequency_embedding = FrequencyEmbedding(decoder_dim, encoder.context_dim)
        self.stage_blocks = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        TriDomainFusionAttentionBlock(
                            decoder_dim,
                            decoder.decoder_heads,
                            encoder.context_dim,
                            encoder.band_dim,
                            num_groups=decoder.frequency_groups,
                            low_cutoff_ratio=decoder.low_cutoff_ratio,
                            mid_cutoff_ratio=decoder.mid_cutoff_ratio,
                            dropout=decoder.decoder_dropout,
                        )
                        for _ in range(decoder.tdfa_depth)
                    ]
                )
                for name in ("low", "mid", "full")
            }
        )
        rank = max(8, decoder_dim // 12)
        self.heads = nn.ModuleDict(
            {
                name: LowRankPatchReconstructionHead(
                    decoder_dim=decoder_dim,
                    band_dim=encoder.band_dim,
                    patch_size=encoder.local_patch_size,
                    rank=rank,
                )
                for name in ("low", "mid", "full")
            }
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        grid_size: tuple[int, int],
        mask: torch.Tensor,
        routing: BandRouting,
    ) -> dict[str, torch.Tensor]:
        tokens = self.shared_input(patch_tokens)
        for block in self.shared_blocks:
            tokens = block(tokens)

        frequency_tokens = self.frequency_embedding(routing.sample_context)
        spectral_context = routing.band_tokens.mean(dim=1)
        stage_tokens = tokens
        outputs: dict[str, torch.Tensor] = {}
        for name in ("low", "mid", "full"):
            stage_tokens = stage_tokens + frequency_tokens["cls"].unsqueeze(1)
            stage_tokens = stage_tokens + torch.where(
                mask.unsqueeze(-1),
                frequency_tokens["mask_tokens"][name].unsqueeze(1),
                torch.zeros_like(stage_tokens),
            )
            for block in self.stage_blocks[name]:
                stage_tokens = block(
                    stage_tokens,
                    grid_size,
                    sample_context=routing.sample_context,
                    spectral_context=spectral_context,
                    stage=name,
                )
            outputs[name] = self.heads[name](stage_tokens, routing.band_tokens)
        return outputs


class MetadataConditionedFoundationModel(nn.Module):
    def __init__(
        self,
        config: FoundationConfig = FoundationConfig(),
        *,
        eo1_bank_centers: torch.Tensor,
        gf5_bank_centers: torch.Tensor,
        eo1_bank_wavelengths: torch.Tensor,
        gf5_bank_wavelengths: torch.Tensor,
        extra_prototype_banks: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        encoder = config.encoder

        self.vocabulary = MetadataVocabulary(encoder.vocabulary)
        self.band_encoder = BandMetadataEncoder(
            band_dim=encoder.band_dim,
            wavelength_range_nm=encoder.wavelength_range_nm,
            fwhm_range_nm=encoder.fwhm_range_nm,
        )
        self.sample_encoder = SampleMetadataEncoder(
            vocabulary=self.vocabulary,
            context_dim=encoder.context_dim,
            gsd_range_m=encoder.gsd_range_m,
            area_range_m2=encoder.area_range_m2,
            timestamp_range_days=encoder.timestamp_range_days,
        )
        self.shared_hypernet = SharedBandHyperNet(
            band_dim=encoder.band_dim,
            context_dim=encoder.context_dim,
            spectral_dim=encoder.spectral_dim,
            spatial_dim=encoder.spatial_dim,
            num_prototypes=encoder.num_prototypes,
            num_heads=encoder.prototype_heads,
            dropout=encoder.drop_rate,
        )
        self.tokenizer = AirSpaceTokenizer2D(
            band_dim=encoder.band_dim,
            spectral_dim=encoder.spectral_dim,
            spatial_dim=encoder.spatial_dim,
            model_dim=encoder.model_dim,
            context_dim=encoder.context_dim,
            patch_size=encoder.local_patch_size,
            num_heads=encoder.prototype_heads,
            prototype_depth=encoder.prototype_depth,
            dropout=encoder.drop_rate,
        )
        self.backbone = TokenTransformerEncoder(
            embed_dim=encoder.model_dim,
            depth=encoder.token_encoder_depth,
            num_heads=encoder.token_encoder_heads,
            out_indices=encoder.backbone_out_indices,
            patch_size=encoder.local_patch_size,
            pretrained=encoder.backbone_pretrained,
            weights=encoder.backbone_weights,
            drop_path_rate=encoder.backbone_drop_path_rate,
            rope_dtype=encoder.backbone_rope_dtype,
            freeze=encoder.freeze_backbone,
        )
        self.mask_generator = RandomMaskGenerator()
        self.prototype_query_head = PrototypeQueryHead(
            model_dim=encoder.model_dim,
            context_dim=encoder.context_dim,
            prototype_count=config.prototype_bank_size,
        )
        self.decoder = ProgressiveFrequencyDecoder(config)

        self.register_buffer("eo1_bank_centers", F.normalize(eo1_bank_centers.float(), dim=-1))
        self.register_buffer("gf5_bank_centers", F.normalize(gf5_bank_centers.float(), dim=-1))
        self.register_buffer("eo1_bank_wavelengths", eo1_bank_wavelengths.float())
        self.register_buffer("gf5_bank_wavelengths", gf5_bank_wavelengths.float())
        self._extra_prototype_bank_names: dict[str, tuple[str, str]] = {}
        for sensor, (centers, wavelengths) in sorted((extra_prototype_banks or {}).items()):
            normalized_sensor = sensor.lower()
            if normalized_sensor in {"eo1", "gf5"}:
                raise ValueError(f"Pass {normalized_sensor!r} through the legacy prototype arguments.")
            safe_sensor = re.sub(r"[^a-z0-9_]", "_", normalized_sensor)
            centers_name = f"extra_{safe_sensor}_bank_centers"
            wavelengths_name = f"extra_{safe_sensor}_bank_wavelengths"
            self.register_buffer(centers_name, F.normalize(centers.float(), dim=-1))
            self.register_buffer(wavelengths_name, wavelengths.float())
            self._extra_prototype_bank_names[normalized_sensor] = (centers_name, wavelengths_name)

    @staticmethod
    def patchify(values: torch.Tensor, patch_size: int) -> torch.Tensor:
        if values.dim() != 4:
            raise ValueError(f"Expected image-like tensor (B, C, H, W), got {tuple(values.shape)}.")
        batch_size, channels, height, width = values.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Input spatial size {(height, width)} must be divisible by patch_size={patch_size}.")
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = values.view(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(
            batch_size,
            grid_h * grid_w,
            patch_size * patch_size * channels,
        )
        return patches

    def _build_routing(self, batch: ModalityBatch) -> BandRouting:
        batch_size = batch.batch_size
        metadata_ablation = self.config.encoder.metadata_ablation
        if metadata_ablation not in {"full", "no_sample", "no_band", "none"}:
            raise ValueError(f"Unsupported metadata_ablation={metadata_ablation!r}.")

        if metadata_ablation in {"no_band", "none"}:
            if batch.wavelengths.dim() == 1:
                band_count = int(batch.wavelengths.shape[0])
            else:
                band_count = int(batch.wavelengths.shape[1])
            band_tokens = batch.values.new_zeros(batch_size, band_count, self.config.encoder.band_dim)
        else:
            band_tokens = self.band_encoder(
                wavelengths=batch.wavelengths.to(batch.values.device),
                batch_size=batch_size,
                fwhm=None if batch.fwhm is None else batch.fwhm.to(batch.values.device),
                valid_mask=None if batch.valid_mask is None else batch.valid_mask.to(batch.values.device),
            )

        if metadata_ablation in {"no_sample", "none"}:
            sample_context = batch.values.new_zeros(batch_size, self.config.encoder.context_dim)
        else:
            sample_context = self.sample_encoder(batch.metadata, batch_size, batch.values.device)
        return self.shared_hypernet(
            band_tokens,
            sample_context,
            valid_mask=None if batch.valid_mask is None else batch.valid_mask.to(batch.values.device),
        )

    def _prepare_space(self, batch: ModalityBatch, mask_ratio: float | None = None) -> PreparedSpaceBatch:
        if batch.values.dim() != 4:
            raise ValueError(f"Space batch must be a cube tensor, got {tuple(batch.values.shape)}.")
        routing = self._build_routing(batch)
        encoded = self.tokenizer(batch.values, routing)
        patch_size = self.config.encoder.local_patch_size
        grid_size = (batch.values.shape[-2] // patch_size, batch.values.shape[-1] // patch_size)
        mask = self.mask_generator(
            batch_size=batch.batch_size,
            num_tokens=encoded.local_tokens.shape[1],
            mask_ratio=self.config.mask_ratio if mask_ratio is None else mask_ratio,
            device=batch.values.device,
        )
        return PreparedSpaceBatch(
            tokens=encoded.local_tokens,
            grid_size=grid_size,
            mask=mask,
            routing=routing,
            values=batch.values,
        )

    def forward_features(self, space: ModalityBatch) -> dict[str, torch.Tensor | tuple[int, int]]:
        """Encode a 2D hyperspectral batch for downstream tasks without pretraining losses."""
        if space.values.dim() != 4:
            raise ValueError(f"Space batch must be a cube tensor, got {tuple(space.values.shape)}.")
        patch_size = self.config.encoder.local_patch_size
        height, width = int(space.values.shape[-2]), int(space.values.shape[-1])
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Downstream input spatial size {(height, width)} must be divisible by local_patch_size={patch_size}."
            )

        routing = self._build_routing(space)
        encoded = self.tokenizer(space.values, routing)
        grid_size = (height // patch_size, width // patch_size)
        backbone_output = self.backbone(encoded.local_tokens, grid_size=grid_size, masks=None)
        patch_tokens = backbone_output["patch_tokens"]
        pooled_embedding = masked_token_pool(patch_tokens)
        return {
            "grid_size": grid_size,
            "patch_tokens": patch_tokens,
            "pooled_embedding": pooled_embedding,
            "normalized_embedding": F.normalize(pooled_embedding, dim=-1),
            "sample_context": routing.sample_context,
            "band_tokens": routing.band_tokens,
            "intermediate_tokens": backbone_output["intermediate_tokens"],
            "cls_token": backbone_output["cls_token"],
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float()
        denom = weights.sum().clamp_min(1.0)
        return (values * weights).sum() / denom

    def _bank_for_sensor(self, sensor: str) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_sensor = sensor.lower()
        if normalized_sensor == "eo1":
            return self.eo1_bank_centers, self.eo1_bank_wavelengths
        if normalized_sensor == "gf5":
            return self.gf5_bank_centers, self.gf5_bank_wavelengths
        names = self._extra_prototype_bank_names.get(normalized_sensor)
        if names is not None:
            return getattr(self, names[0]), getattr(self, names[1])
        raise ValueError(f"Unsupported sensor {sensor!r} for prototype supervision.")

    def _prototype_targets(self, batch: ModalityBatch, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_means = compute_patch_mean_spectra(batch.values, self.config.encoder.local_patch_size)
        teacher_probs: list[torch.Tensor] = []
        patch_visibility: list[torch.Tensor] = []
        valid_mask = batch.valid_mask
        for batch_index, sensor in enumerate(batch.metadata.sensor):
            bank_centers, bank_wavelengths = self._bank_for_sensor(sensor)
            current_valid = None
            if valid_mask is not None:
                current_valid = valid_mask[batch_index]
            elif batch.values.shape[1] != bank_centers.shape[1]:
                raise ValueError(
                    f"Missing valid_mask for sensor {sensor!r}; expected band count {bank_centers.shape[1]}, "
                    f"got padded count {batch.values.shape[1]}."
                )

            if current_valid is None:
                selected_patch = patch_means[batch_index]
            else:
                selected_patch = patch_means[batch_index][:, current_valid.bool()]
            if selected_patch.shape[1] != bank_centers.shape[1]:
                raise ValueError(
                    f"Patch spectrum dimension {selected_patch.shape[1]} does not match prototype bank "
                    f"{bank_centers.shape[1]} for sensor {sensor!r}."
                )
            if bank_wavelengths.shape[0] != selected_patch.shape[1]:
                raise ValueError(f"Prototype bank wavelengths are inconsistent for sensor {sensor!r}.")
            teacher_probs.append(
                prototype_teacher_distribution(
                    selected_patch,
                    bank_centers=bank_centers,
                    temperature=self.config.prototype_temperature_teacher,
                )
            )
            patch_visibility.append((~mask[batch_index]).float())
        return torch.stack(teacher_probs, dim=0), torch.stack(patch_visibility, dim=0)

    def _prototype_losses(
        self,
        patch_tokens: torch.Tensor,
        routing: BandRouting,
        batch: ModalityBatch,
        mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        logits = self.prototype_query_head(patch_tokens, routing.sample_context)
        log_prob = F.log_softmax(logits / max(self.config.prototype_temperature_student, 1.0e-6), dim=-1)
        teacher_prob, visibility = self._prototype_targets(batch, mask)
        proto_kl = F.kl_div(log_prob, teacher_prob, reduction="none").sum(dim=-1)
        proto_kl = (proto_kl * visibility).sum() / visibility.sum().clamp_min(1.0)
        proto_balance = entropy_balance_loss(log_prob.exp())
        entropy = -(log_prob.exp() * log_prob).sum(dim=-1).mean()
        return (
            {
                "proto_kl": proto_kl,
                "proto_balance": proto_balance,
            },
            {
                "student_prototype_entropy": entropy,
            },
        )

    def _frequency_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: ModalityBatch,
        mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        targets = decompose_frequency_targets(
            batch.values,
            low_cutoff_ratio=self.config.decoder.low_cutoff_ratio,
            mid_cutoff_ratio=self.config.decoder.mid_cutoff_ratio,
        )
        patch_feature_mask = None
        if batch.valid_mask is not None:
            patch_area = self.config.encoder.local_patch_size * self.config.encoder.local_patch_size
            patch_feature_mask = (
                batch.valid_mask.bool()
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, 1, patch_area, -1)
                .reshape(batch.batch_size, 1, patch_area * batch.values.shape[1])
                .float()
            )
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        for stage_name, target_name in (("low", "low"), ("mid", "low_mid"), ("full", "full")):
            target_patch = self.patchify(targets[target_name], self.config.encoder.local_patch_size)
            error = F.smooth_l1_loss(predictions[stage_name], target_patch, reduction="none")
            mae = torch.abs(predictions[stage_name] - target_patch)
            if patch_feature_mask is None:
                error = error.mean(dim=-1)
                mae = mae.mean(dim=-1)
            else:
                denom = patch_feature_mask.sum(dim=-1).clamp_min(1.0)
                error = (error * patch_feature_mask).sum(dim=-1) / denom
                mae = (mae * patch_feature_mask).sum(dim=-1) / denom
            losses[f"reconstruction_{stage_name}"] = self._masked_mean(error, mask)
            metrics[f"reconstruction_mae_{stage_name}"] = self._masked_mean(mae, mask)
        return losses, metrics

    def forward(
        self,
        *,
        space: ModalityBatch,
        mask_ratio: float | None = None,
        loss_weights_override: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | tuple[int, int]]:
        prepared = self._prepare_space(space, mask_ratio=mask_ratio)
        backbone_output = self.backbone(
            prepared.tokens,
            grid_size=prepared.grid_size,
            masks=prepared.mask,
        )
        decoder_predictions = self.decoder(
            backbone_output["patch_tokens"],
            grid_size=prepared.grid_size,
            mask=prepared.mask,
            routing=prepared.routing,
        )
        frequency_losses, frequency_metrics = self._frequency_losses(decoder_predictions, space, prepared.mask)
        prototype_losses, prototype_metrics = self._prototype_losses(
            backbone_output["patch_tokens"],
            prepared.routing,
            space,
            prepared.mask,
        )
        losses = {**frequency_losses, **prototype_losses}
        metrics = {**frequency_metrics, **prototype_metrics}

        active_weights = {
            "reconstruction_low": self.config.loss.reconstruction_low,
            "reconstruction_mid": self.config.loss.reconstruction_mid,
            "reconstruction_full": self.config.loss.reconstruction_full,
            "proto_kl": self.config.loss.proto_kl,
            "proto_balance": self.config.loss.proto_balance,
        }
        if loss_weights_override is not None:
            active_weights.update({key: float(value) for key, value in loss_weights_override.items()})

        total = None
        for name, value in losses.items():
            weight = active_weights.get(name, 0.0)
            if weight == 0.0:
                continue
            total = value * weight if total is None else total + value * weight
        if total is None:
            total = backbone_output["patch_tokens"].new_zeros(())

        pooled_embedding = masked_token_pool(backbone_output["patch_tokens"], prepared.mask)
        normalized_embedding = F.normalize(pooled_embedding, dim=-1)
        feature_stats = {
            "embedding_norm_mean": pooled_embedding.norm(dim=-1).mean(),
            "embedding_norm_std": pooled_embedding.norm(dim=-1).std(unbiased=False),
            "visible_ratio": (~prepared.mask).float().mean(),
        }

        return {
            "grid_size": prepared.grid_size,
            "mask": prepared.mask,
            "student_tokens": backbone_output["patch_tokens"],
            "prediction": decoder_predictions,
            "losses": losses,
            "metrics": metrics,
            "feature_stats": feature_stats,
            "pooled_embedding": pooled_embedding,
            "normalized_embedding": normalized_embedding,
            "loss": total,
            "loss_weights": active_weights,
        }
