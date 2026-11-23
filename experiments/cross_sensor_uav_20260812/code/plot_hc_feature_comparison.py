from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("spectralearth", "hypersigma", "specaware", "ours")
TITLES = ("Pseudo-RGB", "SpectralEarth", "HyperSIGMA", "SpecAware", "Ours")


def robust01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a, b = np.percentile(x, (lo, hi))
    return np.clip((x - a) / max(float(b - a), 1.0e-8), 0.0, 1.0)


def pseudo_rgb(hsi: np.ndarray) -> np.ndarray:
    # WHU-Hi HanChuan spans approximately 400--1000 nm with 274 bands.
    wavelengths = np.linspace(400.0, 1000.0, hsi.shape[-1])
    indices = [int(np.argmin(np.abs(wavelengths - target))) for target in (660.0, 550.0, 470.0)]
    return np.stack([robust01(hsi[..., idx]) for idx in indices], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = {method: np.load(args.input_dir / f"{method}.npz", allow_pickle=False) for method in METHODS}
    reference = arrays["ours"]
    rois = [json.loads(str(v)) for v in reference["rois"]]
    raw_hsi = reference["raw_hsi"]
    maps = {method: arrays[method]["feature_norm"].astype(np.float32) for method in METHODS}
    # One robust range per model across all fixed ROIs: no panel-wise brightness tuning.
    maps = {method: robust01(value) for method, value in maps.items()}

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "font.weight": "bold",
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(len(rois), 5, figsize=(12.2, 7.3), constrained_layout=True)
    for row, roi in enumerate(rois):
        axes[row, 0].imshow(pseudo_rgb(raw_hsi[row]))
        for col, method in enumerate(METHODS, start=1):
            axes[row, col].imshow(maps[method][row], cmap="turbo", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(f"Region {row + 1}", fontsize=12, fontweight="bold")
        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.8)
    for col, title in enumerate(TITLES):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"png": str(args.output), "pdf": str(args.output.with_suffix('.pdf')), "rois": rois}))


if __name__ == "__main__":
    main()
