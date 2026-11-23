from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_SOURCE_ROOT = WORKSPACE / "hysens_pixel_regression_case_20260609" / "data" / "processed"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed"

DATASETS = {
    "gloria": {
        "source": "gloria_chlorophyll.npz",
        "output": "gloria_chlorophyll_256.npz",
    },
    "karly": {
        "source": "karly_soil_moisture.npz",
        "output": "karly_soil_moisture_256.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample GLORIA/KarLy spectra to a shared 256-band representation."
    )
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--bands", type=int, default=256)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def validate_splits(data: dict[str, np.ndarray], sample_count: int) -> None:
    split_arrays = [np.asarray(data[name], dtype=np.int64) for name in ("train_idx", "val_idx", "test_idx")]
    merged = np.concatenate(split_arrays)
    if merged.size != np.unique(merged).size:
        raise ValueError("Train/validation/test splits overlap.")
    if merged.min(initial=0) < 0 or merged.max(initial=-1) >= sample_count:
        raise ValueError("A split index is outside the sample range.")


def resample_spectra(
    spectra: np.ndarray,
    wavelengths_nm: np.ndarray,
    target_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(wavelengths_nm)
    sorted_wavelengths = wavelengths_nm[order]
    sorted_spectra = spectra[:, order]
    unique_wavelengths, unique_indices = np.unique(sorted_wavelengths, return_index=True)
    unique_spectra = sorted_spectra[:, unique_indices]
    if unique_wavelengths.size < 2:
        raise ValueError("At least two unique wavelengths are required.")
    target_wavelengths = np.linspace(
        float(unique_wavelengths[0]),
        float(unique_wavelengths[-1]),
        target_bands,
        dtype=np.float32,
    )
    output = np.empty((spectra.shape[0], target_bands), dtype=np.float32)
    for row_index, row in enumerate(unique_spectra):
        output[row_index] = np.interp(target_wavelengths, unique_wavelengths, row).astype(np.float32)
    return output, target_wavelengths


def prepare_one(name: str, args: argparse.Namespace) -> Path:
    spec = DATASETS[name]
    source = (args.source_root / spec["source"]).resolve()
    output = (args.output_root / spec["output"]).resolve()
    if output.exists() and not args.force:
        print(f"[skip] exists: {output}")
        return output
    if not source.is_file():
        raise FileNotFoundError(source)

    with np.load(source, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}
    spectra = np.asarray(data["spectra"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    wavelengths_nm = np.asarray(data["wavelengths_nm"], dtype=np.float32)
    if spectra.ndim != 2 or targets.shape != (spectra.shape[0],):
        raise ValueError(f"Unexpected spectra/target shapes: {spectra.shape}, {targets.shape}")
    if wavelengths_nm.shape != (spectra.shape[1],):
        raise ValueError(f"Unexpected wavelength shape: {wavelengths_nm.shape}")
    validate_splits(data, spectra.shape[0])

    resampled, target_wavelengths = resample_spectra(spectra, wavelengths_nm, int(args.bands))
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    metadata.update(
        {
            "sensor": str(metadata.get("sensor", "field")),
            "level": str(metadata.get("level", "field")),
            "gsd": float(metadata.get("gsd", 1.0)),
            "unified_resampling": {
                "method": "per-spectrum linear interpolation",
                "source_bands": int(spectra.shape[1]),
                "target_bands": int(args.bands),
                "source_wavelength_range_nm": [float(wavelengths_nm.min()), float(wavelengths_nm.max())],
                "target_wavelength_range_nm": [
                    float(target_wavelengths.min()),
                    float(target_wavelengths.max()),
                ],
                "source_npz": str(source),
            },
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dataset=data["dataset"],
        target_name=data["target_name"],
        spectra=resampled,
        targets=targets,
        wavelengths_nm=target_wavelengths,
        train_idx=np.asarray(data["train_idx"], dtype=np.int64),
        val_idx=np.asarray(data["val_idx"], dtype=np.int64),
        test_idx=np.asarray(data["test_idx"], dtype=np.int64),
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False, default=jsonable)),
    )
    summary = {
        "dataset": str(np.asarray(data["dataset"]).item()),
        "source": str(source),
        "output": str(output),
        "samples": int(resampled.shape[0]),
        "source_bands": int(spectra.shape[1]),
        "target_bands": int(resampled.shape[1]),
        "wavelength_range_nm": [float(target_wavelengths.min()), float(target_wavelengths.max())],
        "splits": {
            split: int(np.asarray(data[f"{split}_idx"]).size)
            for split in ("train", "val", "test")
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return output


def main() -> int:
    args = parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in selected:
        prepare_one(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
