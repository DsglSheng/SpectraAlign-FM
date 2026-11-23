from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


METHODS = ("spectralearth", "hypersigma", "specaware", "ours")
TITLES = ("Pseudo-RGB", "Target mask", "SpectralEarth", "HyperSIGMA", "SpecAware", "Ours")


def robust01(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    a, b = np.percentile(x, (lo, hi))
    return np.clip((x - a) / max(float(b - a), 1.0e-8), 0.0, 1.0)


def pseudo_rgb(hsi: np.ndarray) -> np.ndarray:
    wavelengths = np.linspace(400.0, 1000.0, hsi.shape[-1])
    ids = [int(np.argmin(np.abs(wavelengths - w))) for w in (660.0, 550.0, 470.0)]
    rgb = np.stack([robust01(hsi[..., idx], 2, 98) for idx in ids], axis=-1)
    return np.clip(rgb ** 0.8, 0, 1)


def crop(a: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    half = size // 2
    pad = ((half, half), (half, half)) + ((0, 0),) * max(a.ndim - 2, 0)
    p = np.pad(a, pad, mode="reflect")
    return p[row : row + size, col : col + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    arrays = {m: np.load(args.input_dir / f"cam_{m}.npz", allow_pickle=False) for m in METHODS}
    root = Path("/mnt/Data/SpectralData/Aware_open/data/classification/hanchuan")
    raw = loadmat(root / "WHU_Hi_HanChuan.mat")["WHU_Hi_HanChuan"].astype(np.float32)
    labels = loadmat(root / "WHU_Hi_HanChuan_gt.mat")["WHU_Hi_HanChuan_gt"].astype(np.int16)
    size = int(selected["crop_size"])
    rgbs = [pseudo_rgb(crop(raw, int(v["row"]), int(v["col"]), size)) for v in selected["samples"]]
    label_crops = [crop(labels, int(v["row"]), int(v["col"]), size) for v in selected["samples"]]
    # One attribution range per model over all selected samples; no panel-wise tuning.
    normalized = {m: robust01(arrays[m]["attribution"].astype(np.float32), 2, 98) for m in METHODS}
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12, "font.weight": "bold", "axes.titleweight": "bold",
    })
    rows = len(selected["samples"])
    fig, axes = plt.subplots(rows, 6, figsize=(13.5, 2.32 * rows), constrained_layout=True)
    for r, sample in enumerate(selected["samples"]):
        rgb = rgbs[r]
        target = int(sample["class_id"])
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(label_crops[r] == target, cmap="gray", vmin=0, vmax=1)
        for c, method in enumerate(METHODS, start=2):
            axes[r, c].imshow(rgb)
            axes[r, c].imshow(normalized[method][r], cmap="magma", vmin=0, vmax=1, alpha=0.62)
        axes[r, 0].set_ylabel(sample["class_name"], fontsize=11.5, fontweight="bold")
        for c in range(6):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values(): spine.set_linewidth(0.8)
    for c, title in enumerate(TITLES):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold", pad=6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=450, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "figure_png": str(args.output), "figure_pdf": str(args.output.with_suffix('.pdf')),
        "method": "class-conditioned occlusion attribution",
        "normalization": "one robust 2-98 percentile range per model across all samples",
        "samples": selected["samples"],
    }
    args.output.with_name("class_activation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"png": str(args.output), "pdf": str(args.output.with_suffix('.pdf'))}))


if __name__ == "__main__":
    main()
