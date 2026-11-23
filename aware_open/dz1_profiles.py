from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DZ1SensorProfile:
    sensor: str
    sensor_name: str
    native_gsd_m: float
    effective_gsd_m: float
    source_window_size: int
    output_size: int
    wavelength_min_nm: np.ndarray
    wavelengths_nm: np.ndarray
    wavelength_max_nm: np.ndarray
    effective_bandwidth_nm: np.ndarray
    response_model: str = "boxcar_measured_span"
    bandwidth_source: str = "MTL_MEASURED_MAX_MINUS_MIN_PROXY"


_VNIR_MIN = np.asarray(
    [410, 445, 480, 521, 539, 561, 619, 668, 697, 718, 739, 759, 799, 841, 940, 982],
    dtype=np.float32,
)
_VNIR_CENTER = np.asarray(
    [421, 452.5, 490, 530.5, 550, 575.5, 633.5, 678.5, 701.5, 722.5, 743.5, 769, 814, 850.5, 960.5, 1000.5],
    dtype=np.float32,
)
_VNIR_MAX = np.asarray(
    [432, 460, 500, 540, 561, 590, 648, 689, 706, 727, 748, 779, 829, 860, 981, 1019],
    dtype=np.float32,
)

_SWIR_MIN = np.asarray([1178, 1581, 1623, 1737, 1974, 2128, 2193, 2243, 2314, 2390], dtype=np.float32)
_SWIR_CENTER = np.asarray(
    [1196.5, 1600, 1642, 1757.5, 1996.5, 2147, 2213.5, 2265, 2335, 2429],
    dtype=np.float32,
)
_SWIR_MAX = np.asarray([1215, 1619, 1661, 1778, 2019, 2166, 2234, 2287, 2356, 2468], dtype=np.float32)


def build_dz1_vnir_profile(*, native64: bool = False) -> DZ1SensorProfile:
    source_size = 64 if native64 else 128
    effective_gsd = 14.0 if native64 else 28.0
    return DZ1SensorProfile(
        sensor="dz1_vnir",
        sensor_name="DZ01 VNIR",
        native_gsd_m=14.0,
        effective_gsd_m=effective_gsd,
        source_window_size=source_size,
        output_size=64,
        wavelength_min_nm=_VNIR_MIN.copy(),
        wavelengths_nm=_VNIR_CENTER.copy(),
        wavelength_max_nm=_VNIR_MAX.copy(),
        effective_bandwidth_nm=(_VNIR_MAX - _VNIR_MIN).astype(np.float32),
    )


def build_dz1_swir_profile() -> DZ1SensorProfile:
    return DZ1SensorProfile(
        sensor="dz1_swir",
        sensor_name="DZ01 SWIR",
        native_gsd_m=30.0,
        effective_gsd_m=30.0,
        source_window_size=64,
        output_size=64,
        wavelength_min_nm=_SWIR_MIN.copy(),
        wavelengths_nm=_SWIR_CENTER.copy(),
        wavelength_max_nm=_SWIR_MAX.copy(),
        effective_bandwidth_nm=(_SWIR_MAX - _SWIR_MIN).astype(np.float32),
    )


def resolve_dz1_profile(sensor: str, *, native64: bool = False) -> DZ1SensorProfile:
    normalized = sensor.strip().lower()
    if normalized in {"v", "vnir", "dz1_vnir"}:
        return build_dz1_vnir_profile(native64=native64)
    if normalized in {"s", "swir", "dz1_swir"}:
        return build_dz1_swir_profile()
    raise ValueError(f"Unsupported DZ1 sensor: {sensor!r}")
