from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aware_open.config import AwareEncoderConfig, FoundationConfig, FoundationDecoderConfig, FoundationLossConfig
from aware_open.data_types import ModalityBatch, SampleMetadataBatch
from aware_open.models.foundation import MetadataConditionedFoundationModel
from aware_open.spectral_prototypes import PrototypeBank, entropy_balance_loss, load_prototype_bank, prototype_teacher_distribution


@dataclass(frozen=True)
class DownstreamMetadata:
    sensor: str
    level: str
    gsd: float
    wavelengths_nm: torch.Tensor
    fwhm_nm: torch.Tensor


def linear_wavelengths(num_bands: int, start_nm: float, end_nm: float) -> torch.Tensor:
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}.")
    return torch.linspace(float(start_nm), float(end_nm), int(num_bands), dtype=torch.float32)


def approximate_fwhm_from_wavelengths(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    wavelengths_nm = wavelengths_nm.float().flatten()
    if wavelengths_nm.numel() == 1:
        return torch.ones_like(wavelengths_nm)
    diffs = torch.diff(wavelengths_nm).abs()
    left = torch.cat([diffs[:1], diffs])
    right = torch.cat([diffs, diffs[-1:]])
    return ((left + right) * 0.5).clamp_min(1.0)


def build_downstream_metadata(
    *,
    num_bands: int,
    sensor: str = "eo1",
    level: str = "L1",
    gsd: float = 20.0,
    wavelength_start_nm: float = 400.0,
    wavelength_end_nm: float = 2500.0,
    wavelengths_nm: Sequence[float] | torch.Tensor | None = None,
    fwhm_nm: Sequence[float] | torch.Tensor | None = None,
) -> DownstreamMetadata:
    if wavelengths_nm is None:
        wavelengths = linear_wavelengths(num_bands, wavelength_start_nm, wavelength_end_nm)
    else:
        wavelengths = torch.as_tensor(wavelengths_nm, dtype=torch.float32).flatten()
    if wavelengths.numel() != num_bands:
        raise ValueError(f"Expected {num_bands} wavelengths, got {wavelengths.numel()}.")

    if fwhm_nm is None:
        fwhm = approximate_fwhm_from_wavelengths(wavelengths)
    else:
        fwhm = torch.as_tensor(fwhm_nm, dtype=torch.float32).flatten()
    if fwhm.numel() != num_bands:
        raise ValueError(f"Expected {num_bands} FWHM values, got {fwhm.numel()}.")

    return DownstreamMetadata(
        sensor=str(sensor).lower(),
        level=str(level),
        gsd=float(gsd),
        wavelengths_nm=wavelengths,
        fwhm_nm=fwhm,
    )


def metadata_to_batch(*, values: torch.Tensor, metadata: DownstreamMetadata, device: torch.device) -> ModalityBatch:
    if values.dim() != 4:
        raise ValueError(f"Expected cube tensor shaped (B, C, H, W), got {tuple(values.shape)}.")
    batch_size, channels = int(values.shape[0]), int(values.shape[1])
    if metadata.wavelengths_nm.numel() != channels:
        raise ValueError(f"Metadata has {metadata.wavelengths_nm.numel()} bands but input has {channels} channels.")

    values = values.to(device=device, dtype=torch.float32)
    wavelengths = metadata.wavelengths_nm.to(device).view(1, channels).expand(batch_size, -1)
    fwhm = metadata.fwhm_nm.to(device).view(1, channels).expand(batch_size, -1)
    valid_mask = torch.ones(batch_size, channels, dtype=torch.bool, device=device)
    sample_metadata = SampleMetadataBatch(
        modality=["space"] * batch_size,
        sensor=[metadata.sensor] * batch_size,
        level=[metadata.level] * batch_size,
        gsd=torch.full((batch_size,), metadata.gsd, dtype=torch.float32, device=device),
    )
    return ModalityBatch(
        values=values,
        wavelengths=wavelengths,
        fwhm=fwhm,
        metadata=sample_metadata,
        valid_mask=valid_mask,
    )


def config_from_checkpoint_args(checkpoint_args: dict[str, object] | None, prototype_count: int) -> FoundationConfig:
    args = checkpoint_args or {}

    def get(name: str, default):
        return args.get(name, default)

    depth = int(get("token_encoder_depth", 12))
    encoder = AwareEncoderConfig(
        model_dim=int(get("model_dim", 768)),
        band_dim=int(get("band_dim", 192)),
        context_dim=int(get("context_dim", 256)),
        spectral_dim=int(get("spectral_dim", 96)),
        spatial_dim=int(get("spatial_dim", 96)),
        num_prototypes=int(get("num_routing_prototypes", 12)),
        local_patch_size=int(get("local_patch_size", 8)),
        prototype_heads=int(get("prototype_heads", 8)),
        prototype_depth=int(get("prototype_depth", 2)),
        token_encoder_depth=depth,
        token_encoder_heads=int(get("token_encoder_heads", 12)),
        backbone_out_indices=tuple(range(max(0, depth - 4), depth)),
        backbone_pretrained=False,
        drop_rate=float(get("drop_rate", 0.1)),
    )
    decoder = FoundationDecoderConfig(
        decoder_dim=int(get("decoder_dim", 384)),
        decoder_depth=int(get("decoder_depth", 2)),
        decoder_heads=int(get("decoder_heads", 6)),
        decoder_dropout=float(get("decoder_dropout", 0.0)),
        tdfa_depth=int(get("tdfa_depth", 2)),
        frequency_groups=int(get("frequency_groups", 5)),
        low_cutoff_ratio=float(get("frequency_cutoff_low", 0.15)),
        mid_cutoff_ratio=float(get("frequency_cutoff_mid", 0.40)),
    )
    loss = FoundationLossConfig(
        reconstruction_low=0.1,
        reconstruction_mid=0.2,
        reconstruction_full=0.7,
        proto_kl=float(get("proto_kl_weight", 0.10)),
        proto_balance=float(get("proto_balance_weight", 0.02)),
    )
    return FoundationConfig(
        encoder=encoder,
        decoder=decoder,
        loss=loss,
        mask_ratio=float(get("mask_ratio", 0.75)),
        prototype_bank_size=int(prototype_count),
        prototype_temperature_teacher=float(get("prototype_temperature_teacher", 0.07)),
        prototype_temperature_student=float(get("prototype_temperature_student", 0.07)),
    )


def load_aware_foundation(
    *,
    checkpoint_path: str | Path,
    prototype_bank_eo1: str | Path,
    prototype_bank_gf5: str | Path,
    prototype_bank_dz1_vnir: str | Path | None = None,
    prototype_bank_dz1_swir: str | Path | None = None,
    device: torch.device,
) -> MetadataConditionedFoundationModel:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    eo1_bank = load_prototype_bank(prototype_bank_eo1)
    gf5_bank = load_prototype_bank(prototype_bank_gf5)
    if eo1_bank.centers.shape[0] != gf5_bank.centers.shape[0]:
        raise ValueError("EO1 and GF5 prototype banks must have the same prototype count.")

    extra_prototype_banks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if prototype_bank_dz1_vnir is not None:
        dz1_vnir_bank = load_prototype_bank(prototype_bank_dz1_vnir)
        extra_prototype_banks["dz1_vnir"] = (dz1_vnir_bank.centers, dz1_vnir_bank.wavelengths_nm)
    if prototype_bank_dz1_swir is not None:
        dz1_swir_bank = load_prototype_bank(prototype_bank_dz1_swir)
        extra_prototype_banks["dz1_swir"] = (dz1_swir_bank.centers, dz1_swir_bank.wavelengths_nm)

    config = config_from_checkpoint_args(checkpoint_args, int(eo1_bank.centers.shape[0]))
    model = MetadataConditionedFoundationModel(
        config,
        eo1_bank_centers=eo1_bank.centers,
        gf5_bank_centers=gf5_bank.centers,
        eo1_bank_wavelengths=eo1_bank.wavelengths_nm,
        gf5_bank_wavelengths=gf5_bank.wavelengths_nm,
        extra_prototype_banks=extra_prototype_banks or None,
    )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint load result: {result}")
    return model.to(device)


class AwareFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        prototype_bank_eo1: str | Path,
        prototype_bank_gf5: str | Path,
        prototype_bank_dz1_vnir: str | Path | None = None,
        prototype_bank_dz1_swir: str | Path | None = None,
        metadata: DownstreamMetadata,
        freeze_encoder: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = load_aware_foundation(
            checkpoint_path=checkpoint_path,
            prototype_bank_eo1=prototype_bank_eo1,
            prototype_bank_gf5=prototype_bank_gf5,
            prototype_bank_dz1_vnir=prototype_bank_dz1_vnir,
            prototype_bank_dz1_swir=prototype_bank_dz1_swir,
            device=target_device,
        )
        self.model_dim = self.encoder.config.encoder.model_dim
        self.context_dim = self.encoder.config.encoder.context_dim
        self.band_dim = self.encoder.config.encoder.band_dim
        self.decoder_dim = self.encoder.config.decoder.decoder_dim
        self.prototype_count = self.encoder.config.prototype_bank_size
        self.local_patch_size = self.encoder.config.encoder.local_patch_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        device = next(self.encoder.parameters()).device
        batch = metadata_to_batch(values=x, metadata=self.metadata, device=device)
        patch_size = self.local_patch_size
        height, width = int(batch.values.shape[-2]), int(batch.values.shape[-1])
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Downstream input spatial size {(height, width)} must be divisible by local_patch_size={patch_size}."
            )

        routing = self.encoder._build_routing(batch)
        encoded = self.encoder.tokenizer(batch.values, routing)
        grid_size = (height // patch_size, width // patch_size)
        backbone_output = self.encoder.backbone(encoded.local_tokens, grid_size=grid_size, masks=None)
        patch_tokens = backbone_output["patch_tokens"]
        pooled_embedding = patch_tokens.mean(dim=1)
        return {
            "grid_size": grid_size,
            "patch_tokens": patch_tokens,
            "pooled_embedding": pooled_embedding,
            "normalized_embedding": F.normalize(pooled_embedding, dim=-1),
            "sample_context": routing.sample_context,
            "band_tokens": routing.band_tokens,
            "routing_assignments": routing.assignments,
            "intermediate_tokens": backbone_output["intermediate_tokens"],
            "cls_token": backbone_output["cls_token"],
        }

    def _bank_for_name(self, name: str | PrototypeBank) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        if isinstance(name, PrototypeBank):
            return name.centers.to(device=device), name.wavelengths_nm.to(device=device)

        normalized = str(name).lower()
        if normalized == "eo1":
            return self.encoder.eo1_bank_centers, self.encoder.eo1_bank_wavelengths
        if normalized == "gf5":
            return self.encoder.gf5_bank_centers, self.encoder.gf5_bank_wavelengths
        extra_names = getattr(self.encoder, "_extra_prototype_bank_names", {})
        if normalized in extra_names:
            centers_name, wavelengths_name = extra_names[normalized]
            return getattr(self.encoder, centers_name), getattr(self.encoder, wavelengths_name)
        raise ValueError(f"Unsupported prototype bank {name!r}; expected 'eo1', 'gf5', or a registered extra sensor bank.")

    @staticmethod
    def _resample_patch_spectra_linear(
        patch_spectra: torch.Tensor,
        source_wavelengths_nm: torch.Tensor,
        target_wavelengths_nm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_spectra.dim() != 3:
            raise ValueError(f"Expected patch spectra shaped (B, N, C), got {tuple(patch_spectra.shape)}.")
        source = source_wavelengths_nm.to(device=patch_spectra.device, dtype=torch.float32).flatten()
        target = target_wavelengths_nm.to(device=patch_spectra.device, dtype=torch.float32).flatten()
        if source.numel() != patch_spectra.shape[-1]:
            raise ValueError(
                f"Input has {patch_spectra.shape[-1]} channels but metadata has {source.numel()} wavelengths."
            )
        if source.numel() < 2:
            raise ValueError("At least two source wavelengths are required for prototype resampling.")

        order = torch.argsort(source)
        source = source[order]
        spectra = patch_spectra[..., order]
        if torch.any(torch.diff(source) <= 0):
            raise ValueError("Downstream wavelengths must be strictly increasing after sorting.")

        valid = (target >= source[0]) & (target <= source[-1])
        right = torch.searchsorted(source, target).clamp(min=1, max=source.numel() - 1)
        left = right - 1
        left_w = source[left]
        right_w = source[right]
        alpha = ((target - left_w) / (right_w - left_w).clamp_min(1.0e-6)).to(spectra.dtype)
        left_values = spectra.index_select(dim=-1, index=left)
        right_values = spectra.index_select(dim=-1, index=right)
        resampled = left_values * (1.0 - alpha.view(1, 1, -1)) + right_values * alpha.view(1, 1, -1)
        resampled = torch.where(valid.view(1, 1, -1), resampled, torch.zeros_like(resampled))
        return resampled, valid

    def prototype_auxiliary_loss(
        self,
        x: torch.Tensor,
        features: dict[str, torch.Tensor | tuple[int, int]],
        *,
        target_bank: str | PrototypeBank = "eo1",
        teacher_temperature: float | None = None,
        student_temperature: float | None = None,
        center_window: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Training-only prototype regularizer for downstream batches.

        This uses only input spectra and the frozen ground prototype bank. It does not
        inspect validation/test labels or perform sample retrieval.
        """

        device = next(self.encoder.parameters()).device
        values = x.to(device=device, dtype=torch.float32)
        if values.dim() != 4:
            raise ValueError(f"Expected cube tensor shaped (B, C, H, W), got {tuple(values.shape)}.")

        patch_size = self.local_patch_size
        if values.shape[-2] % patch_size != 0 or values.shape[-1] % patch_size != 0:
            raise ValueError("Downstream input spatial size must be divisible by the encoder patch size.")
        patch_spectra = F.avg_pool2d(values, kernel_size=patch_size, stride=patch_size).flatten(2).transpose(1, 2)

        bank_centers, bank_wavelengths = self._bank_for_name(target_bank)
        resampled, valid_mask = self._resample_patch_spectra_linear(
            patch_spectra,
            self.metadata.wavelengths_nm,
            bank_wavelengths,
        )
        batch_size, token_count, band_count = resampled.shape
        if int(valid_mask.sum().item()) < max(8, min(32, band_count // 8)):
            raise ValueError(
                f"Only {int(valid_mask.sum().item())}/{band_count} prototype target bands are covered by "
                "the downstream wavelengths."
            )

        teacher_temp = (
            self.encoder.config.prototype_temperature_teacher
            if teacher_temperature is None
            else float(teacher_temperature)
        )
        student_temp = (
            self.encoder.config.prototype_temperature_student
            if student_temperature is None
            else float(student_temperature)
        )
        with torch.no_grad():
            teacher = prototype_teacher_distribution(
                resampled.reshape(batch_size * token_count, band_count),
                bank_centers=bank_centers,
                temperature=teacher_temp,
                valid_mask=valid_mask,
            ).view(batch_size, token_count, -1)

        patch_tokens = features["patch_tokens"]
        sample_context = features["sample_context"]
        grid_size = features["grid_size"]
        if (
            not isinstance(patch_tokens, torch.Tensor)
            or not isinstance(sample_context, torch.Tensor)
            or not isinstance(grid_size, tuple)
        ):
            raise RuntimeError("forward_features did not return patch_tokens, sample_context, and grid_size.")
        grid_h, grid_w = grid_size
        if grid_h * grid_w != token_count:
            raise RuntimeError(f"grid_size {grid_size} is inconsistent with token_count={token_count}.")
        if center_window <= 0 or center_window % 2 != 1:
            raise ValueError(f"center_window must be a positive odd integer, got {center_window}.")
        half = center_window // 2
        center_h = grid_h // 2
        center_w = grid_w // 2
        row_start = max(0, center_h - half)
        row_end = min(grid_h, center_h + half + 1)
        col_start = max(0, center_w - half)
        col_end = min(grid_w, center_w + half + 1)
        selected_indices = [
            row * grid_w + col
            for row in range(row_start, row_end)
            for col in range(col_start, col_end)
        ]
        index_tensor = torch.tensor(selected_indices, device=patch_tokens.device, dtype=torch.long)
        selected_teacher = teacher.index_select(1, index_tensor)
        selected_tokens = patch_tokens.index_select(1, index_tensor)
        logits = self.encoder.prototype_query_head(selected_tokens, sample_context)
        log_prob = F.log_softmax(logits / max(student_temp, 1.0e-6), dim=-1)
        proto_kl = F.kl_div(log_prob, selected_teacher.detach(), reduction="none").sum(dim=-1).mean()
        probabilities = log_prob.exp()
        return {
            "proto_kl": proto_kl,
            "proto_balance": entropy_balance_loss(probabilities),
            "proto_entropy": -(probabilities * log_prob).sum(dim=-1).mean(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["pooled_embedding"]
