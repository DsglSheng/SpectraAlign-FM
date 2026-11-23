from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


METHODS = ("spectralearth", "hypersigma", "specaware", "ours")
ROW_NAMES = ("(a) Image", "(b) SpectralEarth", "(c) HyperSIGMA", "(d) SpecAware", "(e) Ours")
DATA_ROOT = Path("/mnt/Data/SpectralData/HyperSIGMA-main/ImageClassification/data")


def robust01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a, b = np.percentile(x, (lo, hi))
    return np.clip((x - a) / max(float(b - a), 1e-8), 0, 1)


def load_raw() -> np.ndarray:
    path = next(DATA_ROOT.rglob("PaviaU.mat"))
    d = loadmat(path)
    return next(v for k, v in d.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3).astype(np.float32)


def crop(a: np.ndarray, r: int, c: int, size: int = 128) -> np.ndarray:
    h = size // 2
    return a[r - h : r + h, c - h : c + h]


def pseudo_rgb(hsi: np.ndarray) -> np.ndarray:
    wavelengths = np.linspace(430.0, 860.0, hsi.shape[-1])
    ids = [int(np.argmin(abs(wavelengths - w))) for w in (650.0, 550.0, 470.0)]
    channels = [robust01(hsi[..., i], 1, 99) for i in ids]
    return np.clip(np.stack(channels, -1) ** 0.75, 0, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = spec["candidates"]
    arrays = {m: np.load(args.features_dir / f"{m}.npz", allow_pickle=False) for m in METHODS}
    lookup = {m: {str(i): j for j, i in enumerate(arrays[m]["ids"])} for m in METHODS}
    selected = []
    for class_id in range(1, 10):
        pool = [v for v in candidates if int(v["class_id"]) == class_id]
        joint = []
        for v in pool:
            ok = all(int(arrays[m]["predictions"][lookup[m][v["id"]]]) == class_id for m in METHODS)
            if ok:
                min_conf = min(float(arrays[m]["confidence"][lookup[m][v["id"]]]) for m in METHODS)
                joint.append((min_conf, float(v["local_purity_41"]), float(v["distance"]), v))
        source = joint if joint else [
            (sum(int(arrays[m]["predictions"][lookup[m][v["id"]]]) == class_id for m in METHODS),
             float(v["local_purity_41"]), float(v["distance"]), v) for v in pool
        ]
        source.sort(key=lambda x: x[:3], reverse=True)
        selected.append(source[0][3])

    raw = load_raw()
    rgbs = [pseudo_rgb(crop(raw, int(v["row"]), int(v["col"]))) for v in selected]
    responses = {}
    for method in METHODS:
        stack = np.stack([arrays[method]["response"][lookup[method][v["id"]]] for v in selected]).astype(np.float32)
        stack = np.stack([gaussian_filter(x, sigma=0.7) for x in stack])
        responses[method] = robust01(stack, 2, 98)

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12, "font.weight": "bold",
    })
    fig, axes = plt.subplots(5, 9, figsize=(15.6, 8.65), gridspec_kw={"wspace": 0.018, "hspace": 0.018})
    for col, sample in enumerate(selected):
        axes[0, col].imshow(rgbs[col])
        axes[0, col].text(
            0.04, 0.055, sample["class_name"], transform=axes[0, col].transAxes,
            color="white", fontsize=10.5, fontweight="bold", va="bottom", ha="left",
            bbox={"facecolor": "black", "alpha": 0.42, "edgecolor": "none", "pad": 1.5},
        )
        for row, method in enumerate(METHODS, 1):
            axes[row, col].imshow(responses[method][col], cmap="turbo", vmin=0, vmax=1, interpolation="bicubic")
    for row in range(5):
        axes[row, 0].set_ylabel(ROW_NAMES[row], fontsize=12.5, fontweight="bold", rotation=90, labelpad=10)
        for col in range(9):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            for s in axes[row, col].spines.values():
                s.set_color("white"); s.set_linewidth(0.65)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=450, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    manifest = {
        "dataset": "Pavia University", "samples": selected,
        "rows": list(ROW_NAMES), "feature_definition": "last-layer centered channel L2 norm",
        "normalization": "one 2-98 percentile range per model across all nine samples",
    }
    args.output.with_name("pu_feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"png": str(args.output), "pdf": str(args.output.with_suffix('.pdf')), "classes": [v["class_name"] for v in selected]}))


if __name__ == "__main__":
    main()
