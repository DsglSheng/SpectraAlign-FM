from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..config import MetadataVocabularyConfig
from ..data_types import SampleMetadataBatch


def _expand_or_validate(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if values.dim() == 1:
        return values.unsqueeze(0).expand(batch_size, -1)
    if values.dim() == 2 and values.shape[0] == batch_size:
        return values
    raise ValueError(f"Expected shape (C,) or (B, C), got {tuple(values.shape)}.")


def _broadcast_strings(values: Sequence[str] | str, batch_size: int) -> list[str]:
    if isinstance(values, str):
        return [values] * batch_size
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} metadata strings, got {len(values)}.")
    return list(values)


class MetadataVocabulary:
    def __init__(self, config: MetadataVocabularyConfig) -> None:
        self.modalities = {name: idx for idx, name in enumerate(config.modalities)}
        self.sensors = {name: idx for idx, name in enumerate(config.sensors)}
        self.levels = {name: idx for idx, name in enumerate(config.levels)}

    def encode(self, kind: str, values: Sequence[str] | str, batch_size: int, device: torch.device) -> torch.Tensor:
        mapping = getattr(self, kind)
        tokens = _broadcast_strings(values, batch_size)
        unknown = [token for token in tokens if token not in mapping]
        if unknown:
            raise KeyError(f"Unknown {kind} labels: {sorted(set(unknown))}")
        return torch.tensor([mapping[token] for token in tokens], device=device, dtype=torch.long)


class LogFourierEncoder(nn.Module):
    def __init__(self, lower: float, upper: float, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Fourier embedding_dim must be even.")
        self.lower = lower
        self.upper = upper
        self.register_buffer(
            "frequencies",
            torch.logspace(math.log10(lower), math.log10(upper), embedding_dim // 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(min=self.lower, max=self.upper).float()
        values = values.unsqueeze(-1)
        scaled = values * (2.0 * math.pi / self.frequencies)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class SymmetricFourierEncoder(nn.Module):
    def __init__(self, lower: float, upper: float, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Fourier embedding_dim must be even.")
        self.register_buffer(
            "frequencies",
            torch.logspace(math.log10(lower), math.log10(upper), embedding_dim // 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float().unsqueeze(-1)
        scaled = values * (2.0 * math.pi / self.frequencies)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class BandMetadataEncoder(nn.Module):
    def __init__(
        self,
        band_dim: int,
        wavelength_range_nm: tuple[float, float],
        fwhm_range_nm: tuple[float, float],
    ) -> None:
        super().__init__()
        fourier_dim = band_dim
        self.wavelength_encoder = LogFourierEncoder(wavelength_range_nm[0], wavelength_range_nm[1], fourier_dim)
        self.fwhm_encoder = LogFourierEncoder(fwhm_range_nm[0], fwhm_range_nm[1], fourier_dim)
        self.valid_embedding = nn.Embedding(2, band_dim)
        self.missing_fwhm = nn.Parameter(torch.zeros(band_dim))
        self.proj = nn.Sequential(
            nn.Linear(band_dim, band_dim),
            nn.GELU(),
            nn.Linear(band_dim, band_dim),
            nn.LayerNorm(band_dim),
        )

    def forward(
        self,
        wavelengths: torch.Tensor,
        batch_size: int,
        fwhm: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        wl = _expand_or_validate(wavelengths, batch_size)
        wl_token = self.wavelength_encoder(wl)

        if fwhm is not None:
            bw = _expand_or_validate(fwhm, batch_size)
            bw_present = torch.ones(batch_size, wl.shape[1], 1, device=wl.device)
        else:
            bw = torch.zeros_like(wl)
            bw_present = torch.zeros(batch_size, wl.shape[1], 1, device=wl.device)
        bw_encoded = self.fwhm_encoder(bw)
        bw_missing = self.missing_fwhm.view(1, 1, -1).expand(batch_size, wl.shape[1], -1)
        bw_token = bw_encoded * bw_present + bw_missing * (1.0 - bw_present)

        valid = torch.ones_like(wl, dtype=torch.long) if valid_mask is None else _expand_or_validate(valid_mask.long(), batch_size)
        token = wl_token + bw_token + self.valid_embedding(valid)
        return self.proj(token)


class SampleMetadataEncoder(nn.Module):
    def __init__(
        self,
        vocabulary: MetadataVocabulary,
        context_dim: int,
        gsd_range_m: tuple[float, float],
        area_range_m2: tuple[float, float],
        timestamp_range_days: tuple[float, float],
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.modality_embed = nn.Embedding(len(vocabulary.modalities), context_dim)
        self.sensor_embed = nn.Embedding(len(vocabulary.sensors), context_dim)
        self.level_embed = nn.Embedding(len(vocabulary.levels), context_dim)

        scalar_dim = context_dim // 2
        self.gsd_encoder = LogFourierEncoder(gsd_range_m[0], gsd_range_m[1], scalar_dim)
        self.area_encoder = LogFourierEncoder(area_range_m2[0], area_range_m2[1], scalar_dim)
        self.time_encoder = LogFourierEncoder(max(timestamp_range_days[0], 1e-6), timestamp_range_days[1], scalar_dim)
        self.latlon_encoder = SymmetricFourierEncoder(1e-4, 180.0, scalar_dim)

        self.gsd_proj = nn.Linear(scalar_dim, context_dim)
        self.area_proj = nn.Linear(scalar_dim, context_dim)
        self.time_proj = nn.Linear(scalar_dim, context_dim)
        self.location_proj = nn.Linear(scalar_dim * 2, context_dim)

        self.null_area = nn.Parameter(torch.zeros(context_dim))
        self.null_time = nn.Parameter(torch.zeros(context_dim))
        self.null_location = nn.Parameter(torch.zeros(context_dim))

        self.fusion = nn.Sequential(
            nn.Linear(context_dim * 7, context_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )

    def _encode_optional_scalar(
        self,
        values: Optional[torch.Tensor],
        encoder: nn.Module,
        proj: nn.Module,
        null_value: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if values is None:
            raw_values = torch.zeros(batch_size, device=device)
            present = torch.zeros(batch_size, 1, device=device)
        else:
            raw_values = values.to(device).view(batch_size)
            present = torch.ones(batch_size, 1, device=device)

        encoded = proj(encoder(raw_values))
        null = null_value.view(1, -1).expand(batch_size, -1)
        return encoded * present + null * (1.0 - present)

    def forward(self, metadata: SampleMetadataBatch, batch_size: int, device: torch.device) -> torch.Tensor:
        modality = self.modality_embed(self.vocabulary.encode("modalities", metadata.modality, batch_size, device))
        sensor = self.sensor_embed(self.vocabulary.encode("sensors", metadata.sensor, batch_size, device))
        level = self.level_embed(self.vocabulary.encode("levels", metadata.level, batch_size, device))
        gsd = self.gsd_proj(self.gsd_encoder(metadata.gsd.to(device).view(batch_size)))

        area = self._encode_optional_scalar(
            metadata.footprint_area,
            self.area_encoder,
            self.area_proj,
            self.null_area,
            batch_size,
            device,
        )
        timestamp = self._encode_optional_scalar(
            metadata.timestamp,
            self.time_encoder,
            self.time_proj,
            self.null_time,
            batch_size,
            device,
        )

        if metadata.latitude is None or metadata.longitude is None:
            latitude = torch.zeros(batch_size, device=device)
            longitude = torch.zeros(batch_size, device=device)
            location_present = torch.zeros(batch_size, 1, device=device)
        else:
            latitude = metadata.latitude.to(device).view(batch_size)
            longitude = metadata.longitude.to(device).view(batch_size)
            location_present = torch.ones(batch_size, 1, device=device)
        encoded_location = self.location_proj(
            torch.cat([self.latlon_encoder(latitude), self.latlon_encoder(longitude)], dim=-1)
        )
        location = (
            encoded_location * location_present
            + self.null_location.view(1, -1).expand(batch_size, -1) * (1.0 - location_present)
        )

        fused = torch.cat([modality, sensor, level, gsd, area, timestamp, location], dim=-1)
        return self.fusion(fused)
