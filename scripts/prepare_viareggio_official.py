from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter
from sklearn import metrics as sk_metrics


SCENES = {
    "d1h1": {
        "cube_key": "img_1",
        "mask_rel": "OriginalDataSet/D1_F12_H1_Cropped/TGTs_INFO.mat",
        "source_name": "D1_F12_H1_Cropped",
    },
    "d1h2": {
        "cube_key": "img_2",
        "mask_rel": "OriginalDataSet/D1_F12_H2_Cropped_Aligned/TGTs_INFO.mat",
        "source_name": "D1_F12_H2_Cropped_Aligned",
    },
    "d2h2": {
        "cube_key": "img_3",
        "mask_rel": "OriginalDataSet/D2_F22_H2_Cropped_Aligned/TGTs_INFO.mat",
        "source_name": "D2_F22_H2_Cropped_Aligned",
    },
}


def standardize_score_map(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if not np.isfinite(lower) or not np.isfinite(upper) or abs(upper - lower) < 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lower) / (upper - lower)).astype(np.float32, copy=False)


def compute_auc(output: np.ndarray, target: np.ndarray) -> float:
    y_true = np.reshape(target, (-1, 1), order="F")
    y_score = np.reshape(output, (-1, 1), order="F")
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=False)
    return float(round(sk_metrics.auc(fpr[1:], tpr[1:]), 4))


def ridge_inverse(matrix: np.ndarray, ridge: float = 1.0e-3) -> np.ndarray:
    dimension = int(matrix.shape[0])
    trace = float(np.trace(matrix)) / max(dimension, 1)
    ridge_value = max(float(ridge) * max(trace, 1.0), 1.0e-6)
    return np.linalg.pinv(matrix + ridge_value * np.eye(dimension, dtype=np.float64))


def compute_rx_map(cube: np.ndarray, ridge: float = 1.0e-3) -> np.ndarray:
    flat = np.asarray(cube, dtype=np.float64).reshape(-1, cube.shape[-1])
    mean = np.mean(flat, axis=0, keepdims=True)
    centered = flat - mean
    covariance = (centered.T @ centered) / max(int(flat.shape[0]) - 1, 1)
    precision = ridge_inverse(covariance, ridge=ridge)
    scores = np.einsum("nc,cd,nd->n", centered, precision, centered, optimize=True)
    return scores.reshape(cube.shape[:2]).astype(np.float32, copy=False)


def compute_local_residual_map(cube: np.ndarray, window_size: int = 5) -> np.ndarray:
    values = np.asarray(cube, dtype=np.float32)
    local_mean = uniform_filter(values, size=(window_size, window_size, 1), mode="nearest")
    return np.mean((values - local_mean) ** 2, axis=-1).astype(np.float32, copy=False)


def load_scene(source_root: Path, scene_key: str) -> tuple[np.ndarray, np.ndarray]:
    scene = SCENES[scene_key]
    img_data_path = source_root / "ProcessDataForACDA" / "img_data.mat"
    mask_path = source_root / scene["mask_rel"]
    if not img_data_path.exists():
        raise FileNotFoundError(f"Missing img_data.mat: {img_data_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing target mask: {mask_path}")
    cube = sio.loadmat(img_data_path)[scene["cube_key"]].astype(np.float32, copy=False)
    labels = (sio.loadmat(mask_path)["MASK_TGT"] > 0).astype(np.uint8, copy=False)
    if cube.shape[:2] != labels.shape:
        raise ValueError(f"Scene shape mismatch for {scene_key}: cube={cube.shape}, gt={labels.shape}")
    return cube, labels


def write_variant(output_root: Path, scene_key: str, prior_name: str, cube: np.ndarray, labels: np.ndarray) -> dict:
    rx = standardize_score_map(compute_rx_map(cube))
    residual = standardize_score_map(compute_local_residual_map(cube))
    if prior_name == "rx":
        coarse = rx
    elif prior_name == "rx_res5":
        coarse = standardize_score_map(0.85 * rx + 0.15 * residual)
    else:
        raise ValueError(f"Unsupported prior: {prior_name}")

    run_root = output_root / f"viareggio_official_{scene_key}_{prior_name}"
    coarse_root = run_root / "coarse_det"
    coarse_root.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        run_root / "viareggio.mat",
        {
            "hsi": cube.astype(np.float32, copy=False),
            "hsi_gt": labels.astype(np.uint8, copy=False),
        },
        do_compression=True,
    )
    sio.savemat(
        coarse_root / "Viareggio_coarse_det_map.mat",
        {"show": coarse.astype(np.float32, copy=False)},
        do_compression=True,
    )
    metrics = {
        "scene": scene_key,
        "source_name": SCENES[scene_key]["source_name"],
        "prior": prior_name,
        "shape": list(cube.shape),
        "gt_positive_pixels": int(np.sum(labels > 0)),
        "coarse_auc": compute_auc(coarse, labels),
        "rx_auc": compute_auc(rx, labels),
        "residual_auc": compute_auc(residual, labels),
        "cube_key": SCENES[scene_key]["cube_key"],
        "mask_rel": SCENES[scene_key]["mask_rel"],
    }
    with (run_root / "source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official Viareggio 2013 HAD variants for Aware fine-tuning.")
    parser.add_argument("--source-root", type=Path, default=Path("Viareggio 2013"))
    parser.add_argument("--output-root", type=Path, default=Path("tmp/viareggio_official_prepared"))
    parser.add_argument("--scenes", nargs="+", default=["d1h1", "d1h2", "d2h2"], choices=sorted(SCENES))
    parser.add_argument("--priors", nargs="+", default=["rx", "rx_res5"], choices=["rx", "rx_res5"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for scene_key in args.scenes:
        cube, labels = load_scene(args.source_root, scene_key)
        for prior_name in args.priors:
            result = write_variant(args.output_root, scene_key, prior_name, cube, labels)
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
