from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


def load_hyperview(path: Path) -> np.ndarray:
    payload = sio.loadmat(path)
    if "HypeRvieW" not in payload:
        raise KeyError(f"{path} does not contain HypeRvieW.")
    return np.asarray(payload["HypeRvieW"])


def load_mat_label(path: Path) -> np.ndarray:
    payload = sio.loadmat(path)
    arrays = [(key, value) for key, value in payload.items() if not key.startswith("__")]
    if not arrays:
        raise KeyError(f"{path} does not contain a label array.")
    return np.asarray(arrays[0][1], dtype=np.uint8)


def load_bayarea_raw_label(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size < 4:
        raise ValueError(f"{path} is too small to contain a raw label map.")
    width, height, bands = [int(value) for value in raw[:3]]
    if bands != 1:
        raise ValueError(f"Expected one label band in {path}, got {bands}.")
    labels = raw[3:]
    expected = width * height
    if labels.size != expected:
        raise ValueError(f"Expected {expected} labels from {path}, got {labels.size}.")
    # The raw header stores width=500, height=600. The MATLAB HypeRvieW arrays
    # load as 600x500, and this C-order reshape aligns the label map to them.
    return labels.reshape(height, width).astype(np.uint8, copy=False)


def counts(array: np.ndarray) -> dict[str, int]:
    values, nums = np.unique(array, return_counts=True)
    return {str(int(value)): int(num) for value, num in zip(values, nums)}


def save_mat(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sio.savemat(tmp, payload, do_compression=True)
    tmp.replace(path)


def prepare_bayarea(source_root: Path, output_root: Path) -> dict[str, Any]:
    root = source_root / "bayArea"
    t1 = load_hyperview(root / "mat" / "Bay_Area_2013.mat")
    t2 = load_hyperview(root / "mat" / "Bay_Area_2015.mat")
    gt = load_bayarea_raw_label(root / "raw" / "bayarea_gtChanges.raw")
    if t1.shape != t2.shape or t1.shape[:2] != gt.shape:
        raise ValueError(f"BayArea shape mismatch: T1={t1.shape}, T2={t2.shape}, GT={gt.shape}")
    out = output_root / "bayarea" / "BayArea.mat"
    save_mat(out, {"T1": t1, "T2": t2, "GT": gt})
    return {
        "dataset": "bayarea",
        "output": str(out),
        "shape_t1": list(t1.shape),
        "shape_t2": list(t2.shape),
        "shape_gt": list(gt.shape),
        "gt_counts": counts(gt),
        "source": "CiTIUS ChangeDetectionDataset bayArea; GT from raw/bayarea_gtChanges.raw after three-int32 dimension header, reshaped to 600x500.",
    }


def prepare_barbara(source_root: Path, output_root: Path) -> dict[str, Any]:
    root = source_root / "santaBarbara"
    t1 = load_hyperview(root / "mat" / "barbara_2013.mat")
    t2 = load_hyperview(root / "mat" / "barbara_2014.mat")
    gt = load_mat_label(root / "mat" / "barbara_gtChanges.mat")
    if t1.shape != t2.shape or t1.shape[:2] != gt.shape:
        raise ValueError(f"Barbara shape mismatch: T1={t1.shape}, T2={t2.shape}, GT={gt.shape}")
    out = output_root / "barbara" / "Barbara.mat"
    save_mat(out, {"T1": t1, "T2": t2, "GT": gt})
    return {
        "dataset": "barbara",
        "output": str(out),
        "shape_t1": list(t1.shape),
        "shape_t2": list(t2.shape),
        "shape_gt": list(gt.shape),
        "gt_counts": counts(gt),
        "source": "CiTIUS ChangeDetectionDataset santaBarbara; converted from separated .mat files to HyperSIGMA single-file key layout.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CiTIUS CD datasets in HyperSIGMA-compatible .mat layout.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["bayarea", "barbara"], choices=["bayarea", "barbara"])
    args = parser.parse_args()

    results = []
    if "bayarea" in args.datasets:
        results.append(prepare_bayarea(args.source_root, args.output_root))
    if "barbara" in args.datasets:
        results.append(prepare_barbara(args.source_root, args.output_root))
    provenance = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "These files are converted from the public CiTIUS hyperspectral change detection dataset into "
            "the key layout used by HyperSIGMA: T1, T2, GT. HyperSIGMA's Baidu files have matching public "
            "scene identities for BayArea and Santa Barbara."
        ),
        "source_url": "https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset",
        "results": results,
    }
    out_path = args.output_root / "DATA_PROVENANCE_citius.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=True)
    print(json.dumps(provenance, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
