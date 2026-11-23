from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset-specific physical spectroscopy features.")
    parser.add_argument("--dataset", choices=("green", "qxrd", "karly", "gloria"), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def at(spectra: np.ndarray, wavelengths: np.ndarray, wavelength: float) -> np.ndarray:
    index = int(np.argmin(np.abs(wavelengths - wavelength)))
    return spectra[:, index]


def mean_window(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    mask = (wavelengths >= lower) & (wavelengths <= upper)
    if not np.any(mask):
        return at(spectra, wavelengths, 0.5 * (lower + upper))
    return spectra[:, mask].mean(axis=1)


def ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left / np.maximum(np.abs(right), 1.0e-6)


def normalized_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right) / np.maximum(np.abs(left) + np.abs(right), 1.0e-6)


def slope(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    return (at(spectra, wavelengths, upper) - at(spectra, wavelengths, lower)) / max(
        upper - lower, 1.0
    )


def continuum_depth(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    left_nm: float,
    center_nm: float,
    right_nm: float,
) -> np.ndarray:
    left = at(spectra, wavelengths, left_nm)
    center = at(spectra, wavelengths, center_nm)
    right = at(spectra, wavelengths, right_nm)
    alpha = (center_nm - left_nm) / max(right_nm - left_nm, 1.0e-6)
    continuum = left * (1.0 - alpha) + right * alpha
    return 1.0 - ratio(center, continuum)


def stack(features: list[np.ndarray]) -> np.ndarray:
    return np.stack(features, axis=1).astype(np.float32)


def green_features(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    features: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        features.append(value)

    for wavelength in (430, 450, 531, 550, 570, 670, 705, 720, 750, 800, 860, 970, 1200, 1240, 1450, 1650, 1940, 2100, 2200, 2300):
        add(f"r{wavelength}", at(spectra, wavelengths, wavelength))
    add("ndvi", normalized_difference(at(spectra, wavelengths, 800), at(spectra, wavelengths, 670)))
    add("red_edge_nd", normalized_difference(at(spectra, wavelengths, 750), at(spectra, wavelengths, 705)))
    add("pri", normalized_difference(at(spectra, wavelengths, 531), at(spectra, wavelengths, 570)))
    add("ndwi_860_1240", normalized_difference(at(spectra, wavelengths, 860), at(spectra, wavelengths, 1240)))
    add("swir_nd_1650_2200", normalized_difference(at(spectra, wavelengths, 1650), at(spectra, wavelengths, 2200)))
    add("ratio_705_670", ratio(at(spectra, wavelengths, 705), at(spectra, wavelengths, 670)))
    add("ratio_800_550", ratio(at(spectra, wavelengths, 800), at(spectra, wavelengths, 550)))
    for lower, upper in ((680, 750), (750, 900), (1000, 1300), (1500, 1800), (2000, 2300)):
        add(f"slope_{lower}_{upper}", slope(spectra, wavelengths, lower, upper))
        add(f"mean_{lower}_{upper}", mean_window(spectra, wavelengths, lower, upper))
    for left, center, right in ((900, 970, 1050), (1350, 1450, 1550), (1850, 1940, 2050), (2050, 2200, 2350)):
        add(f"depth_{center}", continuum_depth(spectra, wavelengths, left, center, right))
    spectral_view = np.log1p(np.maximum(spectra, 0.0)).astype(np.float32)
    return stack(features), spectral_view, names


def qxrd_features(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    features: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        features.append(value)

    for wavelength in (1000, 1200, 1400, 1900, 2050, 2100, 2160, 2180, 2200, 2210, 2250, 2300, 2350, 2450):
        add(f"r{wavelength}", at(spectra, wavelengths, wavelength))
    for center in (1400, 1900, 2160, 2180, 2200, 2210, 2250):
        if center < 2000:
            left, right = center - 80, center + 80
        else:
            left, right = 2050, 2350
        add(f"depth_{center}", continuum_depth(spectra, wavelengths, left, center, right))
    add("kaolinite_doublet", continuum_depth(spectra, wavelengths, 2050, 2160, 2250) + continuum_depth(spectra, wavelengths, 2100, 2200, 2350))
    add("ratio_2160_2200", ratio(at(spectra, wavelengths, 2160), at(spectra, wavelengths, 2200)))
    add("ratio_2200_2250", ratio(at(spectra, wavelengths, 2200), at(spectra, wavelengths, 2250)))
    for lower, upper in ((1300, 1500), (1800, 2000), (2050, 2160), (2160, 2200), (2200, 2300)):
        add(f"slope_{lower}_{upper}", slope(spectra, wavelengths, lower, upper))
        add(f"mean_{lower}_{upper}", mean_window(spectra, wavelengths, lower, upper))
    continuum = uniform_filter1d(maximum_filter1d(spectra, size=31, axis=1, mode="nearest"), size=9, axis=1, mode="nearest")
    spectral_view = (1.0 - spectra / np.maximum(continuum, 1.0e-6)).astype(np.float32)
    return stack(features), spectral_view, names


def karly_features(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    features: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        features.append(value)

    for wavelength in (454, 500, 550, 620, 670, 705, 740, 780, 820, 850, 900, 930, 950):
        add(f"r{wavelength}", at(spectra, wavelengths, wavelength))
    for left, right in ((780, 670), (850, 670), (900, 550), (930, 670), (950, 740)):
        add(f"nd_{left}_{right}", normalized_difference(at(spectra, wavelengths, left), at(spectra, wavelengths, right)))
        add(f"ratio_{left}_{right}", ratio(at(spectra, wavelengths, left), at(spectra, wavelengths, right)))
    for lower, upper in ((550, 670), (670, 740), (740, 850), (850, 950)):
        add(f"slope_{lower}_{upper}", slope(spectra, wavelengths, lower, upper))
        add(f"mean_{lower}_{upper}", mean_window(spectra, wavelengths, lower, upper))
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    spectral_view = (centered / np.maximum(spectra.std(axis=1, keepdims=True), 1.0e-6)).astype(np.float32)
    return stack(features), spectral_view, names


def gloria_features(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    features: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        features.append(value)

    for wavelength in (400, 412, 443, 490, 510, 531, 555, 620, 665, 681, 709, 740, 780, 865):
        add(f"rrs{wavelength}", at(spectra, wavelengths, wavelength))
    green = at(spectra, wavelengths, 555)
    for blue in (412, 443, 490, 510):
        add(f"ratio_{blue}_555", ratio(at(spectra, wavelengths, blue), green))
        add(f"logratio_{blue}_555", np.log(np.maximum(ratio(at(spectra, wavelengths, blue), green), 1.0e-6)))
    add("red_edge_nd", normalized_difference(at(spectra, wavelengths, 709), at(spectra, wavelengths, 665)))
    add("nir_red_ratio", ratio(at(spectra, wavelengths, 709), at(spectra, wavelengths, 665)))
    add("blue_green_max_ratio", np.maximum.reduce([ratio(at(spectra, wavelengths, blue), green) for blue in (443, 490, 510)]))
    continuum_681 = at(spectra, wavelengths, 665) + (681 - 665) / (709 - 665) * (
        at(spectra, wavelengths, 709) - at(spectra, wavelengths, 665)
    )
    add("flh_681", at(spectra, wavelengths, 681) - continuum_681)
    for lower, upper in ((400, 500), (500, 600), (620, 665), (665, 709), (709, 780)):
        add(f"slope_{lower}_{upper}", slope(spectra, wavelengths, lower, upper))
        add(f"mean_{lower}_{upper}", mean_window(spectra, wavelengths, lower, upper))
    spectral_view = np.log(np.maximum(spectra, 1.0e-7)).astype(np.float32)
    return stack(features), spectral_view, names


BUILDERS: dict[str, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, list[str]]]] = {
    "green": green_features,
    "qxrd": qxrd_features,
    "karly": karly_features,
    "gloria": gloria_features,
}


def main() -> int:
    args = parse_args()
    if args.output_npz.exists() and not args.force:
        print(f"[skip] exists: {args.output_npz}")
        return 0
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        spectra = np.asarray(payload["spectra"], dtype=np.float32)
        wavelengths = np.asarray(payload["wavelengths_nm"], dtype=np.float32)
        dataset_name = str(np.asarray(payload["dataset"]).item())
    domain_features, spectral_view, names = BUILDERS[args.dataset](spectra, wavelengths)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        dataset=np.array(dataset_name),
        branch=np.array(args.dataset),
        domain_features=domain_features,
        spectral_view=spectral_view,
        feature_names=np.asarray(names),
    )
    summary = {
        "dataset": dataset_name,
        "branch": args.dataset,
        "samples": int(spectra.shape[0]),
        "bands": int(spectra.shape[1]),
        "domain_features": int(domain_features.shape[1]),
        "wavelength_range_nm": [float(wavelengths.min()), float(wavelengths.max())],
    }
    args.output_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
