from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sigma450KSensorProfile:
    sensor: str
    sensor_name: str
    profile_id: str
    level: str
    wavelengths_nm: np.ndarray
    fwhm_nm: np.ndarray | None
    gsd_m: float
    wavelength_source: str
    fwhm_source: str | None
    gsd_source: str
    response_model: str | None = None
    official_band_ids: tuple[int, ...] | None = None
    notes: str = ""


def _float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _hyperion_valid_198_profile() -> Sigma450KSensorProfile:
    vnir_band_ids = np.arange(8, 58, dtype=np.int32)
    swir_band_ids = np.arange(77, 225, dtype=np.int32)
    band_ids = np.concatenate([vnir_band_ids, swir_band_ids])

    wavelengths_nm = np.concatenate(
        [
            np.linspace(426.82, 925.41, num=vnir_band_ids.size, dtype=np.float32),
            np.linspace(912.45, 2395.50, num=swir_band_ids.size, dtype=np.float32),
        ]
    )
    fwhm_nm = np.full(wavelengths_nm.shape, 10.0, dtype=np.float32)

    return Sigma450KSensorProfile(
        sensor="eo1",
        sensor_name="EO-1 Hyperion",
        profile_id="eo1_hyperion_valid198",
        level="L1",
        wavelengths_nm=wavelengths_nm,
        fwhm_nm=fwhm_nm,
        gsd_m=30.0,
        wavelength_source="sensor_catalog_valid_bands",
        fwhm_source="sensor_nominal",
        gsd_source="sensor_nominal",
        response_model="gaussian_fwhm_nominal",
        official_band_ids=tuple(int(band_id) for band_id in band_ids),
        notes=(
            "Valid-band Hyperion profile using the standard calibrated VNIR and SWIR ranges "
            "before the additional 175-band water/noise filtering used by sigma450k."
        ),
    )


def build_eo1_hyperion_good175_profile() -> Sigma450KSensorProfile:
    valid198 = _hyperion_valid_198_profile()
    if valid198.official_band_ids is None:
        raise ValueError("The Hyperion 198-band profile is missing official band ids.")

    drop_bands = set(range(121, 127)) | set(range(167, 181)) | set(range(222, 225))
    keep_mask = np.asarray([band_id not in drop_bands for band_id in valid198.official_band_ids], dtype=bool)
    kept_band_ids = tuple(
        int(band_id)
        for band_id, keep in zip(valid198.official_band_ids, keep_mask, strict=True)
        if keep
    )

    return Sigma450KSensorProfile(
        sensor=valid198.sensor,
        sensor_name=valid198.sensor_name,
        profile_id="eo1_hyperion_good175",
        level=valid198.level,
        wavelengths_nm=_float_array(valid198.wavelengths_nm[keep_mask]),
        fwhm_nm=None if valid198.fwhm_nm is None else _float_array(valid198.fwhm_nm[keep_mask]),
        gsd_m=valid198.gsd_m,
        wavelength_source=valid198.wavelength_source,
        fwhm_source=valid198.fwhm_source,
        gsd_source=valid198.gsd_source,
        response_model=valid198.response_model,
        official_band_ids=kept_band_ids,
        notes=(
            "sigma450k EO1 partitions contain 175 channels, which match the standard Hyperion "
            "good-band configuration obtained after removing water-absorption and low-quality "
            "bands from the calibrated 198-band profile."
        ),
    )


def build_gf5_ahsi_vnir150_profile() -> Sigma450KSensorProfile:
    band_ids = np.arange(1, 151, dtype=np.int32)
    wavelengths_nm = np.linspace(390.32, 1029.18, num=band_ids.size, dtype=np.float32)
    fwhm_nm = np.full(wavelengths_nm.shape, 5.0, dtype=np.float32)

    return Sigma450KSensorProfile(
        sensor="gf5",
        sensor_name="GF-5 AHSI",
        profile_id="gf5_ahsi_vnir150",
        level="L1",
        wavelengths_nm=wavelengths_nm,
        fwhm_nm=fwhm_nm,
        gsd_m=30.0,
        wavelength_source="sensor_catalog_vnir",
        fwhm_source="sensor_nominal",
        gsd_source="sensor_nominal",
        response_model="gaussian_fwhm_nominal",
        official_band_ids=tuple(int(band_id) for band_id in band_ids),
        notes=(
            "sigma450k GF5 partitions expose 150 channels. Upstream HyperSIGMA pretraining code "
            "treats this branch as a visible/NIR cube, so the repository maps it to the AHSI "
            "150-band VNIR sensor profile."
        ),
    )


def infer_sigma450k_sensor(partition_name: str, file_stem: str = "") -> str:
    partition_upper = partition_name.upper()
    stem_upper = file_stem.upper()

    if partition_upper.startswith("EO1") or stem_upper.startswith("EO1"):
        return "eo1"
    if partition_upper.startswith("GF5") or stem_upper.startswith("GF5") or file_stem.isdigit():
        return "gf5"
    raise ValueError(f"Could not infer sigma450k sensor from partition={partition_name!r}, stem={file_stem!r}")


def resolve_sigma450k_profile(partition_name: str, band_count: int, *, file_stem: str = "") -> Sigma450KSensorProfile:
    sensor = infer_sigma450k_sensor(partition_name, file_stem)

    if sensor == "eo1" and band_count == 175:
        return build_eo1_hyperion_good175_profile()
    if sensor == "gf5" and band_count == 150:
        return build_gf5_ahsi_vnir150_profile()

    raise ValueError(
        f"Unsupported sigma450k layout for sensor={sensor!r}, partition={partition_name!r}, "
        f"file_stem={file_stem!r}, band_count={band_count}."
    )


def resolve_sigma450k_profile_from_path(mat_path: Path, band_count: int) -> Sigma450KSensorProfile:
    return resolve_sigma450k_profile(mat_path.parent.name, band_count, file_stem=mat_path.stem)
