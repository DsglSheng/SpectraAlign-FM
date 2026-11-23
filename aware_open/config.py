from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class MetadataVocabularyConfig:
    modalities: Tuple[str, ...] = ("air", "space", "ground")
    sensors: Tuple[str, ...] = (
        "avc",
        "avng",
        "av3",
        "eo1",
        "gf5",
        "gf5b",
        "headwall",
        "rosis",
        "casi",
        "field",
        "others",
        "dz1_vnir",
        "dz1_swir",
    )
    levels: Tuple[str, ...] = (
        "L1",
        "L2",
        "field",
        "surface",
        "toa",
        "boa",
        "unknown",
    )


@dataclass(frozen=True)
class AwareEncoderConfig:
    model_dim: int = 768
    band_dim: int = 192
    context_dim: int = 256
    spectral_dim: int = 96
    spatial_dim: int = 96
    num_prototypes: int = 12
    local_patch_size: int = 8
    prototype_heads: int = 8
    prototype_depth: int = 2
    token_encoder_depth: int = 12
    token_encoder_heads: int = 12
    backbone_out_indices: Tuple[int, ...] = (2, 5, 8, 11)
    backbone_pretrained: bool = False
    backbone_weights: str | None = None
    backbone_drop_path_rate: float = 0.0
    backbone_rope_dtype: str = "fp32"
    freeze_backbone: bool = False
    drop_rate: float = 0.1
    metadata_ablation: str = "full"
    wavelength_range_nm: Tuple[float, float] = (350.0, 2550.0)
    fwhm_range_nm: Tuple[float, float] = (1.0, 40.0)
    gsd_range_m: Tuple[float, float] = (0.01, 1000.0)
    area_range_m2: Tuple[float, float] = (1.0, 1.0e10)
    timestamp_range_days: Tuple[float, float] = (0.0, 365250.0)
    vocabulary: MetadataVocabularyConfig = field(default_factory=MetadataVocabularyConfig)


@dataclass(frozen=True)
class FoundationDecoderConfig:
    decoder_dim: int = 384
    decoder_depth: int = 2
    decoder_heads: int = 6
    decoder_dropout: float = 0.0
    tdfa_depth: int = 2
    frequency_groups: int = 5
    low_cutoff_ratio: float = 0.15
    mid_cutoff_ratio: float = 0.40


@dataclass(frozen=True)
class FoundationLossConfig:
    reconstruction_low: float = 0.1
    reconstruction_mid: float = 0.2
    reconstruction_full: float = 0.7
    proto_kl: float = 0.10
    proto_balance: float = 0.02


@dataclass(frozen=True)
class FoundationConfig:
    encoder: AwareEncoderConfig = field(default_factory=AwareEncoderConfig)
    decoder: FoundationDecoderConfig = field(default_factory=FoundationDecoderConfig)
    loss: FoundationLossConfig = field(default_factory=FoundationLossConfig)
    mask_ratio: float = 0.75
    prototype_bank_size: int = 256
    prototype_temperature_teacher: float = 0.07
    prototype_temperature_student: float = 0.07
