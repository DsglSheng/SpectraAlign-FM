from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def auc_rank(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    pos = label == 1
    neg = label == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_f1(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    thresholds = np.quantile(score, np.linspace(0.01, 0.99, 199))
    best = (0.0, float(thresholds[0]))
    for threshold in thresholds:
        pred = score >= threshold
        tp = int(((label == 1) & pred).sum())
        fp = int(((label == 0) & pred).sum())
        fn = int(((label == 1) & ~pred).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        if f1 > best[0]:
            best = (float(f1), float(threshold))
    return best


def load_hyperview(path: Path) -> np.ndarray:
    return np.asarray(sio.loadmat(path)["HypeRvieW"], dtype=np.float32)


def load_mat_label(path: Path) -> np.ndarray:
    payload = sio.loadmat(path)
    for key, value in payload.items():
        if not key.startswith("__"):
            return np.asarray(value, dtype=np.uint8)
    raise KeyError(path)


def raw_label_variants(path: Path) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, dtype=np.int32)
    payload = raw[3:].astype(np.uint8, copy=False)
    return {
        "raw_500x600_T": payload.reshape(500, 600).T,
        "raw_600x500_C": payload.reshape(600, 500),
        "raw_600x500_F": payload.reshape(600, 500, order="F"),
        "raw_500x600_F_T": payload.reshape(500, 600, order="F").T,
    }


def summarize_variant(name: str, labels: np.ndarray, score: np.ndarray) -> None:
    if labels.shape != score.shape:
        print(f"{name}: skip shape={labels.shape} expected={score.shape}")
        return
    valid = labels > 0
    y = (labels[valid] == 1).astype(np.int64)
    s = score[valid]
    values, counts = np.unique(labels, return_counts=True)
    f1, threshold = best_f1(s, y)
    print(
        f"{name}: shape={labels.shape} counts={dict(zip(values.tolist(), counts.tolist()))} "
        f"auc={auc_rank(s, y):.4f} best_diff_f1={f1:.4f} threshold={threshold:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root / "bayArea"
    t1 = load_hyperview(root / "mat" / "Bay_Area_2013.mat")
    t2 = load_hyperview(root / "mat" / "Bay_Area_2015.mat")
    score = np.mean(np.abs(t1 - t2), axis=2)
    variants = raw_label_variants(root / "raw" / "bayarea_gtChanges.raw")
    variants["mat_gtChanges2"] = load_mat_label(root / "mat" / "bayArea_gtChanges2.mat.mat")
    variants["mat_gtChangesold"] = load_mat_label(root / "mat" / "bayArea_gtChangesolf.mat")
    for name, labels in variants.items():
        summarize_variant(name, labels, score)


if __name__ == "__main__":
    main()
