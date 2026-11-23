from __future__ import annotations

from dataclasses import dataclass

import torch


HYDICE_URBAN_REMOVED_BANDS = (
    1,
    2,
    3,
    4,
    76,
    87,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
)


@dataclass(frozen=True)
class UnmixingSceneConfig:
    dataset: str
    sensor: str
    level: str
    gsd_m: float
    wavelengths: torch.Tensor
    fwhm: torch.Tensor
    metadata_note: str


def _fwhm_from_wavelengths(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    wavelengths = wavelengths_nm.float().flatten()
    if wavelengths.numel() == 1:
        return torch.ones_like(wavelengths)
    diffs = torch.diff(wavelengths).abs()
    left = torch.cat([diffs[:1], diffs])
    right = torch.cat([diffs, diffs[-1:]])
    return ((left + right) * 0.5).clamp_min(1.0)


def hydice_urban_wavelengths() -> torch.Tensor:
    removed = set(HYDICE_URBAN_REMOVED_BANDS)
    full = [400.0 + 10.0 * index for index in range(210)]
    kept = [value for band_index, value in enumerate(full, start=1) if band_index not in removed]
    wavelengths = torch.tensor(kept, dtype=torch.float32)
    if wavelengths.numel() != 162:
        raise RuntimeError(f"Expected 162 retained HYDICE Urban wavelengths, got {wavelengths.numel()}.")
    return wavelengths


def resolve_unmixing_scene_config(dataset: str) -> UnmixingSceneConfig:
    normalized = str(dataset).lower()
    if normalized != "urban4":
        raise ValueError("Only Urban4 is currently supported for downstream unmixing.")

    wavelengths = hydice_urban_wavelengths()
    return UnmixingSceneConfig(
        dataset="Urban4",
        sensor="others",
        level="surface",
        gsd_m=2.0,
        wavelengths=wavelengths,
        fwhm=_fwhm_from_wavelengths(wavelengths),
        metadata_note=(
            "Urban4 uses approximate HYDICE metadata: 2 m GSD and retained wavelengths reconstructed from the "
            "standard 210-band 400-2490 nm, 10 nm grid after removing common low-SNR/water bands "
            "(1-4, 76, 87, 101-111, 136-153, 198-210)."
        ),
    )
