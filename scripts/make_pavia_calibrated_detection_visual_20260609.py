from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\Aware_open")
REF_DIR = (
    ROOT
    / "SCI-Writing"
    / "hypersigma_style_single_prediction_maps_official_palette_corrected_rgb_20260515"
    / "run_20260515_rgbfix"
    / "detection"
    / "HAD_Pavia"
)
OUT_DIR = REF_DIR.parent / "HAD_Pavia_calibrated_20260609"
LOCAL_RESULT = ROOT / "result" / "pavia_calibrated_visual_20260609"
REMOTE_METRICS = (
    "/mnt/Data/SpectralData/Aware_open/result/latest200e_structural_20260602/results/"
    "had_pavia_semigt_calibrated_a008_gtlabel_l001_m006_lr12e5_bg045_fg0009_e100_f192h160_seed20260616/"
    "ours_v3wa_kl003_full450k/test_metrics.json"
)


def norm01(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr)
    if data.dtype != np.uint8:
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(data).save(path)


def save_heat(path: Path, arr: np.ndarray, cmap: str = "magma") -> None:
    rgba = plt.get_cmap(cmap)(norm01(arr))
    save_rgb(path, (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8))


def binary_rgb(mask: np.ndarray, fg=(230, 25, 75), bg=(20, 20, 20)) -> np.ndarray:
    m = np.asarray(mask).astype(np.float32) > 0
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[~m] = bg
    rgb[m] = fg
    return rgb


def outcome_rgb(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = np.asarray(label).astype(np.int32) > 0
    pr = np.asarray(pred).astype(np.int32) > 0
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[~gt & ~pr] = [25, 25, 25]
    rgb[gt & pr] = [255, 255, 255]
    rgb[~gt & pr] = [230, 25, 75]
    rgb[gt & ~pr] = [0, 130, 200]
    return rgb


def metric_card(path: Path, metrics: dict) -> None:
    img = Image.new("RGB", (900, 330), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        title_font = font = small = ImageFont.load_default()
    best = metrics["best_auc_metrics"]
    lines = [
        "HAD-Pavia calibrated detection",
        f"AUC: {best['auc'] * 100:.2f}%  |  HyperSIGMA: 99.96%",
        f"FPR@threshold: {best['auc_threshold_fpr'] * 100:.2f}%  |  TPR: {best['auc_threshold_tpr'] * 100:.2f}%",
        f"Composite: {best['auc_composite']:.4f}  |  Ratio: {best['auc_ratio']:.2f}",
        f"prior_blend_alpha: {best.get('prior_blend_alpha', 0.08):.2f}",
    ]
    draw.text((32, 28), lines[0], fill=(20, 20, 20), font=title_font)
    y = 92
    for line in lines[1:4]:
        draw.text((36, y), line, fill=(30, 30, 30), font=font)
        y += 52
    draw.text((36, y + 8), lines[4], fill=(90, 90, 90), font=small)
    img.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = np.load(REF_DIR / "raw_arrays.npz")
    score = np.load(LOCAL_RESULT / "best_auc_score_map.npy").astype(np.float32)
    neural = np.load(LOCAL_RESULT / "best_auc_neural_score_map.npy").astype(np.float32)
    prior = np.load(LOCAL_RESULT / "best_auc_prior_score_map.npy").astype(np.float32)
    metrics = json.loads((LOCAL_RESULT / "test_metrics.json").read_text(encoding="utf-8"))

    label = old["label"].astype(np.uint8)
    if tuple(score.shape) != tuple(label.shape):
        raise ValueError(f"score shape {score.shape} does not match label shape {label.shape}")

    pred_p995 = (score >= np.percentile(score, 99.5)).astype(np.uint8)
    neg = score[label == 0]
    fpr01_thr = float(np.quantile(neg, 0.99)) if neg.size else float(np.percentile(score, 99.0))
    pred_fpr01 = (score >= fpr01_thr).astype(np.uint8)

    for name in ["original_rgb.png", "groundtruth_binary.png", "groundtruth_target_map.png", "rgb_band_info.csv"]:
        src = REF_DIR / name
        if src.exists():
            (OUT_DIR / name).write_bytes(src.read_bytes())

    save_heat(OUT_DIR / "ours_score_map.png", score)
    save_heat(OUT_DIR / "ours_score_heatmap.png", score)
    save_heat(OUT_DIR / "ours_neural_score_heatmap.png", neural)
    save_heat(OUT_DIR / "ours_prior_score_heatmap.png", prior, cmap="viridis")
    save_rgb(OUT_DIR / "ours_prediction_binary_p995.png", binary_rgb(pred_p995, fg=(255, 255, 255)))
    save_rgb(OUT_DIR / "ours_prediction_target_map.png", binary_rgb(pred_p995))
    save_rgb(OUT_DIR / "ours_error_map.png", outcome_rgb(label, pred_p995))
    save_rgb(OUT_DIR / "ours_prediction_binary_fpr01.png", binary_rgb(pred_fpr01, fg=(255, 255, 255)))
    save_rgb(OUT_DIR / "ours_prediction_target_map_fpr01.png", binary_rgb(pred_fpr01))
    save_rgb(OUT_DIR / "ours_error_map_fpr01.png", outcome_rgb(label, pred_fpr01))
    metric_card(OUT_DIR / "pavia_calibrated_metric_card.png", metrics)

    np.savez_compressed(
        OUT_DIR / "raw_arrays.npz",
        original_cube=old["original_cube"],
        label=label,
        score_map=score,
        neural_score_map=neural,
        prior_score_map=prior,
        prediction_mask=pred_p995,
        prediction_mask_fpr01=pred_fpr01,
        fpr01_threshold=np.asarray(fpr01_thr, dtype=np.float32),
    )

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "HAD_Pavia",
                "visualization": "calibrated_20260609",
                "style_reference": str(REF_DIR),
                "metrics_json_remote": REMOTE_METRICS,
                "metrics_json_local": str(LOCAL_RESULT / "test_metrics.json"),
                "auc_percent": metrics["best_auc_metrics"]["auc"] * 100.0,
                "hyper_sigma_auc_percent": 99.96,
                "prediction_mask": "p99.5, matching the existing HyperSIGMA-style detection asset naming",
                "metric_aligned_mask": "negative-score 99th percentile, approximately FPR 1%",
                "fpr01_threshold": fpr01_thr,
                "files": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    with (OUT_DIR / "asset_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "bytes"])
        for p in sorted(OUT_DIR.iterdir()):
            if p.is_file():
                writer.writerow([p.name, p.stat().st_size])

    print(json.dumps({"output": str(OUT_DIR), "files": len(list(OUT_DIR.glob('*'))), "auc": metrics["best_auc_metrics"]["auc"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
