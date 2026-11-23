from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray, bins: np.ndarray) -> dict[str, object]:
    quantiles = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles_0_10_25_50_75_90_100": [float(x) for x in quantiles],
        "hist_bins": [float(x) for x in bins],
        "hist_counts": np.histogram(values, bins=bins)[0].astype(int).tolist(),
        "count_ge_20": int(np.sum(values >= 20.0)),
        "count_ge_50": int(np.sum(values >= 50.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    targets = np.asarray(data["targets"], dtype=np.float32)
    bins = np.asarray([0, 0.5, 1, 2, 5, 10, 20, 50, 100], dtype=np.float32)
    out = {"all": summarize(targets, bins)}
    for key in ("train_idx", "val_idx", "test_idx"):
        out[key.replace("_idx", "")] = summarize(targets[np.asarray(data[key], dtype=np.int64)], bins)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
