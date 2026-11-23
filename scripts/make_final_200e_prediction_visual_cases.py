#!/usr/bin/env python3
"""Generate paper-ready visual cases for the final 200-epoch result summary.

The script is intentionally read-only with respect to experiment outputs. It
collects saved prediction maps, score maps, confusion matrices, and metrics from
the selected final-200e downstream runs and writes a timestamped visualization
pack under result/ours_200e_paper_visual_cases_20260513.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover - scipy exists on the training server
    loadmat = None


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
DEFAULT_OUT_ROOT = PROJECT / "result" / "ours_200e_paper_visual_cases_20260513"


def p(*parts: str) -> Path:
    return PROJECT.joinpath(*parts)


def existing_result(candidates: Sequence[Path]) -> Path:
    """Return the first candidate that looks usable, else the first path."""
    for path in candidates:
        if not path:
            continue
        if any((path / name).exists() for name in (
            "test_metrics.json",
            "best_metrics.json",
            "final_best_f1_pred_map.npy",
            "best_f1_pred_map.npy",
            "final_best_auc_score_map.npy",
            "best_auc_score_map.npy",
            "unmixing_result.mat",
        )):
            return path
    return candidates[0]


DATASETS: List[Dict[str, Any]] = [
    # HyperSIGMA comparison: classification.
    {
        "group": "hypersigma_classification",
        "kind": "classification",
        "name": "IP",
        "result": existing_result([
            p("result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/classification_ip_recheck_seed3713_no_tta/ours_v3wa_kl003_full450k"),
        ]),
        "official": {"OA": 0.9618, "AA": 0.9666, "Kappa": 0.9565},
    },
    {
        "group": "hypersigma_classification",
        "kind": "classification",
        "name": "PU",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_pu/ours_v3wa_kl003_full450k"),
        "official": {"OA": 0.9854, "AA": 0.9833, "Kappa": 0.9806},
    },
    {
        "group": "hypersigma_classification",
        "kind": "classification",
        "name": "HC",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_hanchuan/ours_v3wa_kl003_full450k"),
        "official": {},
    },
    {
        "group": "hypersigma_classification",
        "kind": "classification",
        "name": "HH",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_hh/ours_v3wa_kl003_full450k"),
        "official": {},
    },
    {
        "group": "hypersigma_classification",
        "kind": "classification",
        "name": "HU",
        "result": p("result/ours_v3wa_kl003_full450k_history_epoch130cfg_rerun_20260509/fair_tune_v2/results/hu_160_inv_f192_lr6e5_e180/ours_v3wa_kl003_full450k"),
        "official": {},
    },
    {
        "group": "hypersigma_classification",
        "kind": "classification",
        "name": "ZY",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_zy/ours_v3wa_kl003_full450k"),
        "official": {},
    },
    # HyperSIGMA comparison: target detection.
    {
        "group": "hypersigma_detection",
        "kind": "detection",
        "name": "HTD-Mosaic",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/detection_htd_mosaic/ours_v3wa_kl003_full450k"),
        "official": {"AUC": 0.9989},
    },
    {
        "group": "hypersigma_detection",
        "kind": "detection",
        "name": "HTD-AVIRIS",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/detection_htd_aviris/ours_v3wa_kl003_full450k"),
        "official": {"AUC": 0.9989},
    },
    {
        "group": "hypersigma_detection",
        "kind": "detection",
        "name": "HAD-Pavia",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/detection_had_pavia/ours_v3wa_kl003_full450k"),
        "official": {"AUC": 0.9999},
    },
    {
        "group": "hypersigma_detection",
        "kind": "detection",
        "name": "HAD-CRI",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/detection_had_cri/ours_v3wa_kl003_full450k"),
        "official": {"AUC": 0.9989},
    },
    {
        "group": "hypersigma_detection",
        "kind": "detection",
        "name": "HAD-Viareggio",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/detection_had_viareggio_official/ours_v3wa_kl003_full450k"),
        "official": {"AUC": 0.9997},
    },
    # HyperSIGMA comparison: change detection.
    {
        "group": "hypersigma_change_detection",
        "kind": "change_detection",
        "name": "Farmland",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/change_farmland/ours_v3wa_kl003_full450k"),
        "official": {},
    },
    {
        "group": "hypersigma_change_detection",
        "kind": "change_detection",
        "name": "Hermiston",
        "result": p("result/ours_v3wa_kl003_full450k_history_epoch130cfg_rerun_20260509/fair_tune_v4_20260510/results/hermiston_freeze_spectral_lr1e4_e120/ours_v3wa_kl003_full450k"),
        "official": {},
    },
    {
        "group": "hypersigma_change_detection",
        "kind": "change_detection",
        "name": "BayArea",
        "result": existing_result([
            p("result/specaware_iterative_improve_20260512/results/bayarea_stats_lr5e5_drop0_e140/ours_v3wa_kl003_full450k"),
            p("result/specaware_iterative_improve_20260512/results/bayarea_stats_lr5e5_drop0_e100/ours_v3wa_kl003_full450k"),
        ]),
        "official": {"F1": 0.9911, "Kappa": 0.9809},
    },
    {
        "group": "hypersigma_change_detection",
        "kind": "change_detection",
        "name": "Barbara",
        "result": existing_result([
            p("result/specaware_iterative_improve_20260512/results/barbara_lr1e4_drop0_e120/ours_v3wa_kl003_full450k"),
            p("result/ours_v3wa_kl003_full450k_history_epoch130cfg_rerun_20260509/fair_tune_v4_20260510/results/barbara_freeze_spectral_lr1e4_e120/ours_v3wa_kl003_full450k"),
        ]),
        "official": {"F1": 0.9939, "Kappa": 0.9899},
    },
    {
        "group": "hypersigma_change_detection",
        "kind": "change_detection",
        "name": "OSCD",
        "result": existing_result([
            p("result/ours_200e_final_once_improve_oscd_houston_20260513/results/oscd_30k_low_lr_resume_from_recheck_best/ours_v3wa_kl003_full450k"),
            p("result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/oscd_30k/ours_v3wa_kl003_full450k"),
        ]),
        "official": {},
    },
    # HyperSIGMA comparison: unmixing, denoising, super-resolution.
    {
        "group": "hypersigma_unmixing",
        "kind": "unmixing",
        "name": "Urban4",
        "result": p("result/ours_v3wa_kl003_full450k_downstream_20260508/results/unmix_urban4_hypersigma_spreg6/ours_v3wa_kl003_full450k"),
        "official": {"avg_sad": 0.0547, "avg_mse": 0.0151},
    },
    {
        "group": "hypersigma_denoising",
        "kind": "denoising",
        "name": "WDC-Case5",
        "result": p("result/ours_v3wa_kl003_full450k_history_epoch130cfg_rerun_20260509/fair_tune_v5_20260510/results/wdc_stage5_freeze_l2sam_lr1e6_e30_from_stage4"),
        "official": {"PSNR": 28.503, "SSIM": 0.961, "SAM": 0.106},
    },
    {
        "group": "hypersigma_super_resolution",
        "kind": "super_resolution",
        "name": "Houston-Proxy-x4",
        "result": existing_result([
            p("result/ours_200e_final_once_improve_oscd_houston_20260513/results/houston_proxy_x4_low_lr_resume_from_recheck_best/ours_v3wa_kl003_full450k"),
            p("result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/houston_proxy_x4/ours_v3wa_kl003_full450k"),
        ]),
        "official": {},
    },
    {
        "group": "hypersigma_super_resolution",
        "kind": "super_resolution",
        "name": "Houston-Proxy-x8",
        "result": existing_result([
            p("result/ours_200e_final_once_improve_oscd_houston_20260513/results/houston_proxy_x8_low_lr_resume_from_recheck_best/ours_v3wa_kl003_full450k"),
            p("result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/houston_proxy_x8/ours_v3wa_kl003_full450k"),
        ]),
        "official": {},
    },
    # SpecAware comparison.
    {
        "group": "specaware",
        "kind": "segmentation_or_classification",
        "name": "AeroRIT",
        "result": p("result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/aerorit_recheck_seed20260513/ours_v3wa_kl003_full450k"),
        "official": {"mIoU": 0.6330},
    },
    {
        "group": "specaware",
        "kind": "segmentation_or_classification",
        "name": "Qingpu-HSI",
        "result": p("result/specaware_comparison_v1_20260510/qingpu_ours_upeernet_lr5e5_e100_crop128_acc16_20260511"),
        "official": {},
    },
    {
        "group": "specaware",
        "kind": "segmentation_or_classification",
        "name": "WHU-H2SR",
        "result": existing_result([
            p("result/specaware_iterative_improve_20260512/results/whu_h2sr_lr4e5_steps1400_e100_crop256_acc16"),
            p("result/specaware_iterative_improve_20260512/results/whu_h2sr_lr3e5_steps1400_e100_crop256_acc16"),
            p("result/specaware_comparison_v1_20260510/whu_h2sr_ours_lr3e5_e80_crop256_acc16_20260511"),
        ]),
        "official": {"OA": 0.8972, "AA": 0.8184, "Kappa": 0.8517, "mIoU": 0.6744},
    },
    {
        "group": "specaware",
        "kind": "change_detection",
        "name": "Bay-Area-SpecAware",
        "result": existing_result([
            p("result/specaware_iterative_improve_20260512/results/bayarea_stats_lr5e5_drop0_e140/ours_v3wa_kl003_full450k"),
            p("result/specaware_iterative_improve_20260512/results/bayarea_stats_lr5e5_drop0_e100/ours_v3wa_kl003_full450k"),
        ]),
        "official": {"OA": 0.9905, "Kappa": 0.9809, "F1": 0.9911},
    },
    {
        "group": "specaware",
        "kind": "change_detection",
        "name": "Santa-Barbara-SpecAware",
        "result": existing_result([
            p("result/specaware_iterative_improve_20260512/results/barbara_lr1e4_drop0_e120/ours_v3wa_kl003_full450k"),
            p("result/ours_v3wa_kl003_full450k_history_epoch130cfg_rerun_20260509/fair_tune_v4_20260510/results/barbara_freeze_spectral_lr1e4_e120/ours_v3wa_kl003_full450k"),
        ]),
        "official": {"OA": 0.9952, "Kappa": 0.9899, "F1": 0.9939},
    },
]


METRIC_ALIASES = {
    "OA": ["OA", "oa", "overall_accuracy", "accuracy"],
    "AA": ["AA", "aa", "average_accuracy", "mean_accuracy"],
    "Kappa": ["Kappa", "kappa"],
    "F1": ["F1", "f1", "f1_score"],
    "Precision": ["Precision", "precision"],
    "Recall": ["Recall", "recall"],
    "AUC": ["AUC", "auc", "roc_auc"],
    "mIoU": ["mIoU", "miou", "mean_iou"],
    "PSNR": ["PSNR", "psnr"],
    "SSIM": ["SSIM", "ssim"],
    "SAM": ["SAM", "sam"],
    "MPSNR": ["MPSNR", "mpsnr"],
    "MSSIM": ["MSSIM", "mssim"],
    "ERGAS": ["ERGAS", "ergas"],
    "CC": ["CC", "cc"],
    "RMSE": ["RMSE", "rmse"],
    "avg_sad": ["avg_sad", "sad"],
    "avg_mse": ["avg_mse", "mse"],
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def flatten_metrics(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_metrics(v, key))
            elif isinstance(v, (int, float, str, list)):
                out[key] = v
    return out


def load_metrics(result: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for name in ("test_metrics.json", "best_metrics.json", "metrics.json"):
        merged.update(flatten_metrics(read_json(result / name)))

    # Keep common top-level arrays available for confusion/per-class plots.
    for name in ("test_metrics.json", "best_metrics.json"):
        raw = read_json(result / name)
        for container in _metric_containers(raw):
            for key in ("confusion_matrix", "per_class_accuracy", "per_class_acc", "per_class_iou"):
                if key in container and key not in merged:
                    merged[key] = container[key]
    return merged


def _metric_containers(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, dict):
        yield raw
        for value in raw.values():
            if isinstance(value, dict):
                yield value
                if "test" in value and isinstance(value["test"], dict):
                    yield value["test"]


def metric_value(metrics: Dict[str, Any], canonical: str) -> Optional[float]:
    aliases = METRIC_ALIASES.get(canonical, [canonical])
    for key, value in metrics.items():
        tail = key.split(".")[-1]
        if tail in aliases or key in aliases:
            try:
                return float(value)
            except Exception:
                continue
    return None


def compact_metrics(metrics: Dict[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        value = metric_value(metrics, key)
        if value is not None and math.isfinite(value):
            out[key] = value
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def save_fig(fig: plt.Figure, out_base: Path) -> List[str]:
    ensure_dir(out_base.parent)
    written = []
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def norm01(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0, 1)


def find_file(result: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = result / name
        if path.exists():
            return path
    return None


def load_array(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def load_maps_mat(result: Path) -> Dict[str, np.ndarray]:
    if loadmat is None:
        return {}
    mat_path = find_file(result, [
        "final_best_f1_maps.mat",
        "best_f1_maps.mat",
        "final_best_auc_maps.mat",
        "best_auc_maps.mat",
        "maps.mat",
    ])
    if not mat_path:
        return {}
    try:
        raw = loadmat(mat_path)
    except Exception:
        return {}
    return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__") and isinstance(v, np.ndarray)}


def metrics_card(dataset: Dict[str, Any], metrics: Dict[str, Any], out_dir: Path) -> List[str]:
    keys = [
        "OA", "AA", "Kappa", "F1", "Precision", "Recall", "AUC", "mIoU",
        "PSNR", "SSIM", "SAM", "MPSNR", "MSSIM", "ERGAS", "CC", "RMSE",
        "avg_sad", "avg_mse",
    ]
    ours = compact_metrics(metrics, keys)
    official = dataset.get("official", {})
    rows = []
    for key in keys:
        if key in ours or key in official:
            rows.append([key, _fmt(ours.get(key)), _fmt(official.get(key))])
    if not rows:
        rows.append(["status", "metrics not found", ""])

    fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.45 * len(rows) + 1.2)))
    ax.axis("off")
    ax.set_title(f"{dataset['name']} metrics", fontsize=13, pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Ours", "Reference"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    return save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_metrics_card")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def draw_confusion_or_bars(dataset: Dict[str, Any], metrics: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    files: List[str] = []
    note = "classification/segmentation metrics visualized as confusion or per-class bars"

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        try:
            arr = np.asarray(cm, dtype=np.float64)
            if arr.ndim == 2 and arr.size:
                row_sum = arr.sum(axis=1, keepdims=True)
                norm = np.divide(arr, row_sum, out=np.zeros_like(arr), where=row_sum > 0)
                fig, ax = plt.subplots(figsize=(7.5, 6.4))
                im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_title(f"{dataset['name']} confusion matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Ground truth")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if arr.shape[0] <= 18:
                    ax.set_xticks(range(arr.shape[1]))
                    ax.set_yticks(range(arr.shape[0]))
                files.extend(save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_confusion_matrix"))
        except Exception:
            pass

    bar_items: List[Tuple[str, Sequence[float]]] = []
    for key in ("per_class_accuracy", "per_class_acc", "per_class_iou"):
        value = metrics.get(key)
        if value is not None:
            try:
                arr = np.asarray(value, dtype=np.float64).ravel()
                if arr.size:
                    bar_items.append((key, arr))
            except Exception:
                pass

    if not bar_items:
        # Build a compact metric bar chart if no per-class data was saved.
        selected = compact_metrics(metrics, ["OA", "AA", "Kappa", "F1", "AUC", "mIoU"])
        if selected:
            bar_items.append(("summary_metrics", list(selected.values())))

    for key, values in bar_items[:2]:
        values = np.asarray(values, dtype=np.float64).ravel()
        fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(values)), 3.8))
        x = np.arange(len(values))
        ax.bar(x, values, color="#3d6fb6")
        ax.set_ylim(0, max(1.0, min(1.05, float(np.nanmax(values)) * 1.08)))
        ax.set_title(f"{dataset['name']} {key.replace('_', ' ')}")
        ax.set_xlabel("Class / metric index")
        ax.set_ylabel("Value")
        if len(values) <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels([str(i + 1) for i in x], rotation=90 if len(values) > 16 else 0)
        files.extend(save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_{safe_name(key)}"))

    if not files:
        note = "no prediction maps or class-level statistics were saved; only metrics card was generated"
    return files, note


def draw_detection(dataset: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    result = dataset["result"]
    score = load_array(find_file(result, [
        "final_best_auc_score_map.npy",
        "best_auc_score_map.npy",
        "latest_score_map.npy",
    ]))
    if score is None:
        return [], "detection score map not found; only metrics card was generated"
    score = np.squeeze(score)
    if score.ndim != 2:
        return [], "detection score map has unexpected shape; only metrics card was generated"
    norm = norm01(score)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].imshow(norm, cmap="magma")
    axes[0].set_title("Anomaly score")
    axes[0].axis("off")
    thresh = float(np.percentile(score[np.isfinite(score)], 99.5)) if np.isfinite(score).any() else 0
    axes[1].imshow(score >= thresh, cmap="gray")
    axes[1].set_title("High-score mask")
    axes[1].axis("off")
    files = save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_score_case")
    return files, "score map and percentile high-score mask generated from saved detector output"


def draw_change_detection(dataset: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    result = dataset["result"]
    maps = load_maps_mat(result)
    pred = load_array(find_file(result, ["final_best_f1_pred_map.npy", "best_f1_pred_map.npy", "latest_pred_map.npy"]))
    prob = load_array(find_file(result, ["final_best_f1_prob_map.npy", "best_f1_prob_map.npy", "latest_prob_map.npy"]))

    for key in ("pred_map", "pred", "prediction"):
        if pred is None and key in maps:
            pred = np.squeeze(maps[key])
    for key in ("prob_map", "prob", "score_map", "score"):
        if prob is None and key in maps:
            prob = np.squeeze(maps[key])
    gt = None
    for key in ("gt", "label", "labels", "gt_map", "target"):
        if key in maps:
            gt = np.squeeze(maps[key])
            break

    panels: List[Tuple[str, np.ndarray, str]] = []
    if gt is not None and gt.ndim == 2:
        panels.append(("Ground truth", gt, "gray"))
    if pred is not None:
        pred = np.squeeze(pred)
        if pred.ndim == 2:
            panels.append(("Prediction", pred, "gray"))
    if prob is not None:
        prob = np.squeeze(prob)
        if prob.ndim == 2:
            panels.append(("Probability", norm01(prob), "viridis"))
    if gt is not None and pred is not None and gt.shape == pred.shape:
        err = (gt.astype(np.int32) != pred.astype(np.int32)).astype(np.float32)
        panels.append(("Error", err, "Reds"))

    if not panels and str(dataset["name"]).lower() == "oscd":
        return draw_oscd_prediction(dataset, out_dir)
    if not panels:
        return [], "change-detection prediction/probability maps not found; only metrics card was generated"

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    files = save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_change_case")
    return files, "change-detection prediction/probability case generated from saved maps"


def _scene_rank_from_oscd_metrics(result: Path) -> List[str]:
    raw = read_json(result / "test_metrics.json")
    scenes = raw.get("metrics", {}).get("per_scene", [])
    ranked = []
    for item in scenes:
        try:
            ranked.append((float(item.get("metrics", {}).get("f1", 0.0)), str(item.get("name", ""))))
        except Exception:
            continue
    ranked = [(score, name) for score, name in ranked if name]
    ranked.sort(reverse=True)
    return [name for _, name in ranked]


def draw_oscd_prediction(dataset: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    best = result / "best.pt"
    cfg = read_json(result / "config.json")
    if not best.exists() or not cfg:
        return [], "OSCD best.pt/config.json not found; only metrics card was generated"
    try:
        import torch

        sys.path.insert(0, str(PROJECT / "scripts"))
        import finetune_oscd_change_detection as oscd
    except Exception as exc:
        return [], f"could not import OSCD model for prediction export: {exc}"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_root = Path(cfg.get("data_root", PROJECT / "data/oscd/original_data"))
        crop_size = int(cfg.get("crop_size", 128))
        ranked = _scene_rank_from_oscd_metrics(result)
        test_names = [line.strip() for line in (data_root / "test.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        chosen = [name for name in ranked if name in test_names][:3]
        if len(chosen) < 3:
            chosen.extend([name for name in test_names if name not in chosen][: 3 - len(chosen)])

        metadata = oscd.build_downstream_metadata(
            num_bands=13,
            sensor="others",
            level="L1",
            gsd=10.0,
            wavelengths_nm=oscd.S2_WAVELENGTHS_NM,
            fwhm_nm=oscd.S2_FWHM_NM,
        )
        extractor = oscd.AwareFeatureExtractor(
            checkpoint_path=Path(cfg["checkpoint"]),
            prototype_bank_eo1=Path(cfg["prototype_bank_eo1"]),
            prototype_bank_gf5=Path(cfg["prototype_bank_gf5"]),
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        model = oscd.AwareOSCDFusion(
            extractor,
            feature_layers=int(cfg.get("feature_layers", 4)),
            fusion_channels=int(cfg.get("fusion_channels", 128)),
            hidden_channels=int(cfg.get("hidden_channels", 128)),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device)
        payload = torch.load(best, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        ds = oscd.OSCDDataset(data_root, chosen, crop_size=crop_size, max_samples=None, split="test")
        files: List[str] = []
        raw_payload: Dict[str, np.ndarray] = {}
        use_amp = str(cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"
        with torch.no_grad():
            for idx in range(len(ds)):
                batch = ds[idx]
                pre = batch["pre"].unsqueeze(0).to(device)
                post = batch["post"].unsqueeze(0).to(device)
                label = batch["label"].unsqueeze(0).to(device)
                pre_patches, coords = oscd.crop_tensor(pre, crop_size)
                post_patches, _ = oscd.crop_tensor(post, crop_size)
                preds = []
                for pp, qq in zip(pre_patches, post_patches):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        preds.append(model(pp, qq).float())
                logits = oscd.combine_patches(preds, coords, (label.shape[-2], label.shape[-1]), crop_size)
                prob = torch.softmax(logits, dim=1)[:, 1].squeeze(0).detach().cpu().numpy()
                pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
                gt = label.squeeze(0).detach().cpu().numpy().astype(np.int64)
                pre_np = pre.squeeze(0).detach().cpu().numpy()
                post_np = post.squeeze(0).detach().cpu().numpy()
                name = str(batch["name"])
                raw_payload[f"{name}_pre"] = pre_np
                raw_payload[f"{name}_post"] = post_np
                raw_payload[f"{name}_gt"] = gt
                raw_payload[f"{name}_pred"] = pred
                raw_payload[f"{name}_prob"] = prob
                valid = gt < 2
                err = np.zeros_like(prob, dtype=np.float32)
                err[valid] = (pred[valid] != gt[valid]).astype(np.float32)
                fig, axes = plt.subplots(1, 6, figsize=(17.0, 3.2))
                axes[0].imshow(cube_rgb(pre_np))
                axes[0].set_title("Pre")
                axes[1].imshow(cube_rgb(post_np))
                axes[1].set_title("Post")
                axes[2].imshow(np.where(gt == 255, 0, gt), cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Ground truth")
                axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
                axes[3].set_title("Prediction")
                axes[4].imshow(norm01(prob), cmap="viridis")
                axes[4].set_title("Change prob.")
                axes[5].imshow(err, cmap="Reds", vmin=0, vmax=1)
                axes[5].set_title("Error")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"OSCD {name}", y=1.03, fontsize=13)
                files.extend(save_fig(fig, out_dir / f"OSCD_{safe_name(name)}_prediction"))
        np.savez_compressed(out_dir / "OSCD_prediction_cases.npz", **raw_payload)
        return files + [str(out_dir / "OSCD_prediction_cases.npz")], "OSCD predictions exported from best.pt for top test scenes"
    except Exception as exc:
        return [], f"OSCD prediction export failed; only metrics card was generated: {exc}"


def draw_unmixing(dataset: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    if loadmat is None:
        return [], "scipy unavailable; only metrics card was generated"
    path = dataset["result"] / "unmixing_result.mat"
    if not path.exists():
        return [], "unmixing_result.mat not found; only metrics card was generated"
    try:
        raw = {k: np.asarray(v) for k, v in loadmat(path).items() if not k.startswith("__")}
    except Exception:
        return [], "failed to read unmixing_result.mat; only metrics card was generated"

    files: List[str] = []
    abundance = None
    spectra = None
    for key, arr in raw.items():
        if arr.ndim == 3 and abundance is None:
            abundance = arr
        if arr.ndim == 2 and spectra is None and min(arr.shape) <= 16 and max(arr.shape) > 16:
            spectra = arr

    if abundance is not None:
        maps = _abundance_to_last_channel(abundance)
        n = min(6, maps.shape[-1])
        cols = min(3, n)
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
        axes = np.asarray(axes).reshape(-1)
        for idx in range(n):
            axes[idx].imshow(norm01(maps[..., idx]), cmap="viridis")
            axes[idx].set_title(f"Abundance {idx + 1}")
            axes[idx].axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        files.extend(save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_abundance_maps"))

    if spectra is not None:
        spec = spectra
        if spec.shape[0] <= 16:
            spec = spec.T
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for idx in range(min(spec.shape[1], 8)):
            ax.plot(spec[:, idx], linewidth=1.3, label=f"EM {idx + 1}")
        ax.set_title(f"{dataset['name']} estimated endmembers")
        ax.set_xlabel("Band")
        ax.set_ylabel("Normalized abundance")
        ax.legend(fontsize=8, ncol=2)
        files.extend(save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_endmember_spectra"))

    if files:
        return files, "unmixing abundance/endmember visualizations generated from unmixing_result.mat"
    return [], "unmixing mat did not expose plottable abundance/spectral arrays; only metrics card was generated"


def _abundance_to_last_channel(arr: np.ndarray) -> np.ndarray:
    data = np.squeeze(arr)
    if data.ndim != 3:
        return data
    dims = data.shape
    channel_axis = int(np.argmin(dims))
    if dims[channel_axis] > 32:
        channel_axis = 2
    return np.moveaxis(data, channel_axis, -1)


def draw_metric_bars(dataset: Dict[str, Any], metrics: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    keys = [
        "OA", "AA", "Kappa", "F1", "AUC", "mIoU",
        "PSNR", "SSIM", "SAM", "MPSNR", "MSSIM", "ERGAS", "CC", "RMSE",
        "avg_sad", "avg_mse",
    ]
    ours = compact_metrics(metrics, keys)
    official = dataset.get("official", {})
    labels = [k for k in keys if k in ours or k in official]
    if not labels:
        return [], "no plottable scalar metrics found"

    x = np.arange(len(labels))
    width = 0.36
    ours_vals = [ours.get(k, np.nan) for k in labels]
    ref_vals = [official.get(k, np.nan) for k in labels]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.75 * len(labels)), 4.0))
    ax.bar(x - width / 2, ours_vals, width, label="Ours", color="#3d6fb6")
    ax.bar(x + width / 2, ref_vals, width, label="Reference", color="#d28b38")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(f"{dataset['name']} scalar metrics")
    ax.legend()
    files = save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_metric_bars")
    return files, "scalar metrics visualized because image-level predictions were not saved"


def cube_to_hwc(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {arr.shape}")
    if arr.shape[0] <= 512 and arr.shape[1] > 8 and arr.shape[2] > 8:
        return arr.transpose(1, 2, 0)
    return arr


def cube_rgb(cube: np.ndarray) -> np.ndarray:
    hwc = cube_to_hwc(cube)
    bands = int(hwc.shape[-1])
    if bands >= 3:
        idx = [
            min(bands - 1, max(0, int(round(0.65 * (bands - 1))))),
            min(bands - 1, max(0, int(round(0.35 * (bands - 1))))),
            min(bands - 1, max(0, int(round(0.15 * (bands - 1))))),
        ]
        rgb = hwc[..., idx]
    else:
        rgb = np.repeat(hwc[..., :1], 3, axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[..., channel]
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [1, 99])
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi > lo:
            out[..., channel] = np.clip((plane - lo) / (hi - lo), 0, 1)
    return out


def draw_cube_triplet(
    *,
    name: str,
    input_cube: np.ndarray,
    pred_cube: np.ndarray,
    gt_cube: np.ndarray,
    out_dir: Path,
    input_title: str,
    out_stem: str,
) -> List[str]:
    pred_hwc = cube_to_hwc(pred_cube)
    gt_hwc = cube_to_hwc(gt_cube)
    min_h = min(pred_hwc.shape[0], gt_hwc.shape[0])
    min_w = min(pred_hwc.shape[1], gt_hwc.shape[1])
    pred_hwc = pred_hwc[:min_h, :min_w]
    gt_hwc = gt_hwc[:min_h, :min_w]
    err = np.mean(np.abs(np.clip(pred_hwc, 0, 1) - np.clip(gt_hwc, 0, 1)), axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.5))
    axes[0].imshow(cube_rgb(input_cube))
    axes[0].set_title(input_title)
    axes[1].imshow(cube_rgb(pred_hwc))
    axes[1].set_title("Prediction")
    axes[2].imshow(cube_rgb(gt_hwc))
    axes[2].set_title("Ground truth")
    axes[3].imshow(norm01(err), cmap="magma")
    axes[3].set_title("Absolute error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name, y=1.02, fontsize=13)
    return save_fig(fig, out_dir / out_stem)


def draw_denoising_prediction(dataset: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    best = result / "best.pt"
    cfg = read_json(result / "config.json")
    if not best.exists() or not cfg:
        return draw_metric_bars(dataset, load_metrics(result), out_dir)
    try:
        import torch

        sys.path.insert(0, str(PROJECT / "scripts"))
        import finetune_denoising as den
    except Exception as exc:
        more, _ = draw_metric_bars(dataset, load_metrics(result), out_dir)
        return more, f"could not import denoising model for prediction export: {exc}"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata = den.build_downstream_metadata(
            num_bands=191,
            sensor="others",
            level="surface",
            gsd=2.8,
            wavelength_start_nm=400.0,
            wavelength_end_nm=2500.0,
        )
        extractor = den.AwareFeatureExtractor(
            checkpoint_path=Path(cfg["checkpoint"]),
            prototype_bank_eo1=Path(cfg["prototype_bank_eo1"]),
            prototype_bank_gf5=Path(cfg["prototype_bank_gf5"]),
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        model = den.AwareDenoisingNet(
            extractor,
            channels=191,
            feature_channels=int(cfg.get("feature_channels", 64)),
            hidden_channels=int(cfg.get("hidden_channels", 256)),
            use_intermediate=not bool(cfg.get("no_intermediate", False)),
        ).to(device)
        payload = torch.load(best, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        test_dir = Path(cfg["data_root"]) / "Testing" / "Patch_Cases" / "Case5"
        sample_count = min(3, len(den.WdcPatchMatDataset(test_dir)))
        files: List[str] = []
        raw_payload: Dict[str, np.ndarray] = {}
        dataset_obj = den.WdcPatchMatDataset(test_dir)
        use_amp = str(cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"
        with torch.no_grad():
            for index in range(sample_count):
                noisy, clean, file_name = dataset_obj[index]
                noisy_b = noisy.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred = den.predict_with_tta(model, noisy_b, enabled=bool(cfg.get("eval_tta", False)))
                pred_np = pred.squeeze(0).detach().cpu().float().numpy()
                noisy_np = noisy.detach().cpu().float().numpy()
                clean_np = clean.detach().cpu().float().numpy()
                raw_payload[f"case_{index}_noisy"] = noisy_np
                raw_payload[f"case_{index}_pred"] = pred_np
                raw_payload[f"case_{index}_gt"] = clean_np
                files.extend(
                    draw_cube_triplet(
                        name=f"{dataset['name']} {file_name}",
                        input_cube=noisy_np,
                        pred_cube=pred_np,
                        gt_cube=clean_np,
                        out_dir=out_dir,
                        input_title="Noisy input",
                        out_stem=f"{safe_name(dataset['name'])}_case{index:02d}_prediction",
                    )
                )
        np.savez_compressed(out_dir / f"{safe_name(dataset['name'])}_prediction_cases.npz", **raw_payload)
        return files + [str(out_dir / f"{safe_name(dataset['name'])}_prediction_cases.npz")], "denoising predictions exported from best.pt on Case5 test patches"
    except Exception as exc:
        more, _ = draw_metric_bars(dataset, load_metrics(result), out_dir)
        return more, f"denoising prediction export failed; metrics fallback generated: {exc}"


def draw_sr_prediction(dataset: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    best = result / "best.pt"
    cfg = read_json(result / "config.json")
    if not best.exists() or not cfg:
        return draw_metric_bars(dataset, load_metrics(result), out_dir)
    try:
        import torch

        sys.path.insert(0, str(PROJECT / "scripts"))
        import finetune_super_resolution as sr
    except Exception as exc:
        more, _ = draw_metric_bars(dataset, load_metrics(result), out_dir)
        return more, f"could not import SR model for prediction export: {exc}"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_root = Path(cfg["data_root"])
        dataset_name = str(cfg.get("dataset_name", "houston"))
        scale = int(cfg["scale"])
        test_set = sr.HSTestData(data_root / f"{dataset_name}_test.mat")
        ms, lms, gt = test_set[0]
        channels = int(cfg.get("channels") or gt.shape[0])
        metadata = sr.build_downstream_metadata(num_bands=channels, sensor="others", level="surface", gsd=1.0)
        extractor = sr.AwareFeatureExtractor(
            checkpoint_path=Path(cfg["checkpoint"]),
            prototype_bank_eo1=Path(cfg["prototype_bank_eo1"]),
            prototype_bank_gf5=Path(cfg["prototype_bank_gf5"]),
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        model = sr.AwareSRFusion(
            extractor,
            scale=scale,
            channels=channels,
            feature_layers=int(cfg.get("feature_layers", 4)),
            fusion_channels=int(cfg.get("fusion_channels", 128)),
        ).to(device)
        payload = torch.load(best, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        ms_b = ms.unsqueeze(0).to(device)
        lms_b = lms.unsqueeze(0).to(device)
        gt_b = gt.unsqueeze(0).to(device)
        pred = torch.empty_like(gt_b)
        block_size = 32
        use_amp = str(cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"
        with torch.no_grad():
            for row in range(0, ms_b.shape[-2], block_size):
                for col in range(0, ms_b.shape[-1], block_size):
                    r2 = min(row + block_size, ms_b.shape[-2])
                    c2 = min(col + block_size, ms_b.shape[-1])
                    r = max(0, r2 - block_size) if r2 - row < block_size else row
                    c = max(0, c2 - block_size) if c2 - col < block_size else col
                    hr0, hr1 = r * scale, r2 * scale
                    hc0, hc1 = c * scale, c2 * scale
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        out = model(ms_b[:, :, r:r2, c:c2], lms_b[:, :, hr0:hr1, hc0:hc1])
                    pred[:, :, hr0:hr1, hc0:hc1] = out.float()
        pred_np = pred.squeeze(0).detach().cpu().float().numpy()
        lms_np = lms.detach().cpu().float().numpy()
        gt_np = gt.detach().cpu().float().numpy()
        ms_np = ms.detach().cpu().float().numpy()
        np.savez_compressed(
            out_dir / f"{safe_name(dataset['name'])}_prediction_case.npz",
            low_resolution=ms_np,
            bicubic_input=lms_np,
            prediction=pred_np,
            ground_truth=gt_np,
        )
        files = draw_cube_triplet(
            name=dataset["name"],
            input_cube=lms_np,
            pred_cube=pred_np,
            gt_cube=gt_np,
            out_dir=out_dir,
            input_title="Bicubic input",
            out_stem=f"{safe_name(dataset['name'])}_prediction",
        )
        return files + [str(out_dir / f"{safe_name(dataset['name'])}_prediction_case.npz")], "SR prediction exported from best.pt on Houston Proxy test cube"
    except Exception as exc:
        more, _ = draw_metric_bars(dataset, load_metrics(result), out_dir)
        return more, f"SR prediction export failed; metrics fallback generated: {exc}"


def draw_segmentation_prediction(dataset: Dict[str, Any], metrics: Dict[str, Any], out_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    best = result / "best_val_miou.pt"
    cfg = read_json(result / "config.json")
    if not best.exists() or not cfg or not (result / "test_metrics.json").exists():
        return draw_confusion_or_bars(dataset, metrics, out_dir)
    try:
        import torch

        sys.path.insert(0, str(PROJECT / "scripts"))
        import finetune_specaware_segmentation as seg
    except Exception as exc:
        more, _ = draw_confusion_or_bars(dataset, metrics, out_dir)
        return more, f"could not import SpecAware segmentation model for prediction export: {exc}"

    try:
        args_cfg = cfg.get("args", {})
        data_kind = str(cfg.get("dataset") or args_cfg.get("dataset"))
        data_dir = Path(args_cfg.get("data_dir"))
        seed = int(args_cfg.get("seed", 20260511))
        crop_size = int(args_cfg.get("crop_size", 128))
        stats_samples = int(args_cfg.get("stats_samples", 80000))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if data_kind == "qingpu":
            image, image_info = seg.open_envi_image(data_dir / "Qingpu_HSI_image.dat", data_dir / "Qingpu_HSI_image.hdr")
            gt_mem, _ = seg.open_envi_image(data_dir / "Qingpu_HSI_gt.dat", data_dir / "Qingpu_HSI_gt.hdr")
            labels = np.asarray(gt_mem.data[0] if gt_mem.layout == "bsq" else gt_mem.data[:, 0, :], dtype=np.uint8)
            class_names = seg.QINGPU_CLASS_NAMES
            metadata_sensor, metadata_level, metadata_gsd = "eo1", "L1", 0.75
            wavelengths = torch.linspace(400.0, 1000.0, int(image_info["bands"]))
            class_count = len(class_names)
            split_path = result / f"splits_seed{seed}_tr{int(args_cfg.get('train_per_class', 2000))}_va{int(args_cfg.get('val_per_class', 500))}_te{int(args_cfg.get('test_per_class', 1000))}.npz"
            splits = seg.split_indices(
                labels,
                class_count=class_count,
                train_per_class=int(args_cfg.get("train_per_class", 2000)),
                val_per_class=int(args_cfg.get("val_per_class", 500)),
                test_per_class=int(args_cfg.get("test_per_class", 1000)),
                seed=seed,
                output_path=split_path,
            )
            test_labels = seg.make_sparse_label_map(labels, splits["test"])
            mean, std = seg.estimate_band_stats(
                image,
                labels,
                sample_count=stats_samples,
                seed=seed,
                output_path=result / f"band_stats_{stats_samples}_seed{seed}.npz",
            )
            test_dataset = seg.SpectralCropDataset(
                image=image,
                labels=test_labels,
                indices=splits["test"],
                mean=mean,
                std=std,
                crop_size=crop_size,
                train=False,
                seed=seed + 41,
            )
        elif data_kind == "aerorit":
            import tifffile

            refl = data_dir / "full_reflectance" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine-001_"
            hdr = data_dir / "headers" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
            if not hdr.exists():
                hdr = data_dir / "headers_extracted" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
            image, image_info = seg.open_envi_image(refl, hdr)
            rgb = np.asarray(tifffile.memmap(data_dir / "sample_10band" / "image_labels.tif"), dtype=np.uint8)
            labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for class_id, color in enumerate(seg.AERORIT_CLASS_COLORS, start=1):
                labels[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
            class_names = seg.AERORIT_CLASS_NAMES
            metadata_sensor, metadata_level, metadata_gsd = "others", "L1", 0.4
            wavelengths = torch.tensor(image_info["wavelengths"], dtype=torch.float32) if len(image_info.get("wavelengths", [])) == int(image_info["bands"]) else torch.linspace(400.0, 897.0, int(image_info["bands"]))
            class_count = len(class_names)
            split_path = result / f"splits_seed{seed}_tr{int(args_cfg.get('train_per_class', 2000))}_va{int(args_cfg.get('val_per_class', 500))}_te{int(args_cfg.get('test_per_class', 1000))}.npz"
            splits = seg.split_indices(
                labels,
                class_count=class_count,
                train_per_class=int(args_cfg.get("train_per_class", 2000)),
                val_per_class=int(args_cfg.get("val_per_class", 500)),
                test_per_class=int(args_cfg.get("test_per_class", 1000)),
                seed=seed,
                output_path=split_path,
            )
            test_labels = seg.make_sparse_label_map(labels, splits["test"])
            mean, std = seg.estimate_band_stats(
                image,
                labels,
                sample_count=stats_samples,
                seed=seed,
                output_path=result / f"band_stats_{stats_samples}_seed{seed}.npz",
            )
            test_dataset = seg.SpectralCropDataset(
                image=image,
                labels=test_labels,
                indices=splits["test"],
                mean=mean,
                std=std,
                crop_size=crop_size,
                train=False,
                seed=seed + 41,
            )
        elif data_kind == "whu":
            pairs = seg.build_whu_pairs(data_dir)
            class_names = seg.WHU_H2SR_CLASS_NAMES
            class_count = len(class_names)
            metadata_sensor, metadata_level, metadata_gsd = "others", "surface", 1.0
            image_info = {"bands": 249}
            wavelengths = torch.linspace(400.0, 1000.0, int(image_info["bands"]))
            split_path = result / f"tile_splits_seed{seed}_tr20_va10_te70.npz"
            splits = seg.split_tile_pairs(pairs, seed=seed, output_path=split_path)
            mean, std = seg.estimate_tile_band_stats(
                pairs,
                splits["train"],
                sample_count=stats_samples,
                seed=seed,
                output_path=result / f"tile_band_stats_{stats_samples}_seed{seed}.npz",
            )
            test_dataset = seg.WhuTileDataset(
                pairs=pairs,
                indices=splits["test"],
                mean=mean,
                std=std,
                crop_size=crop_size,
                class_count=class_count,
                train=False,
                seed=seed + 41,
            )
        else:
            return draw_confusion_or_bars(dataset, metrics, out_dir)

        metadata = seg.build_downstream_metadata(
            num_bands=int(image_info["bands"]),
            sensor=metadata_sensor,
            level=metadata_level,
            gsd=metadata_gsd,
            wavelengths_nm=wavelengths,
            fwhm_nm=None,
        )
        extractor = seg.AwareFeatureExtractor(
            checkpoint_path=Path(args_cfg["checkpoint"]),
            prototype_bank_eo1=Path(args_cfg["prototype_bank_eo1"]),
            prototype_bank_gf5=Path(args_cfg["prototype_bank_gf5"]),
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        model = seg.AwareUPerNetSeg(extractor, class_count).to(device)
        payload = torch.load(best, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        use_amp = str(args_cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"

        files: List[str] = []
        raw_payload: Dict[str, np.ndarray] = {}
        sample_count = min(3, len(test_dataset))
        with torch.no_grad():
            for index in range(sample_count):
                x, y = test_dataset[index]
                x_b = x.unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(x_b)
                pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64) + 1
                gt = y.detach().cpu().numpy().astype(np.int64)
                pred = np.where(gt > 0, pred, 0)
                err = ((pred != gt) & (gt > 0)).astype(np.float32)
                x_np = x.detach().cpu().numpy()
                raw_payload[f"case_{index}_input"] = x_np
                raw_payload[f"case_{index}_gt"] = gt
                raw_payload[f"case_{index}_pred"] = pred
                fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
                axes[0].imshow(cube_rgb(x_np))
                axes[0].set_title("Input")
                axes[1].imshow(gt, cmap="tab20", vmin=0, vmax=class_count)
                axes[1].set_title("Ground truth")
                axes[2].imshow(pred, cmap="tab20", vmin=0, vmax=class_count)
                axes[2].set_title("Prediction")
                axes[3].imshow(err, cmap="Reds", vmin=0, vmax=1)
                axes[3].set_title("Error")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"{dataset['name']} test crop {index}", y=1.03, fontsize=13)
                files.extend(save_fig(fig, out_dir / f"{safe_name(dataset['name'])}_crop{index:02d}_prediction"))
        np.savez_compressed(out_dir / f"{safe_name(dataset['name'])}_segmentation_prediction_crops.npz", **raw_payload)
        return files + [str(out_dir / f"{safe_name(dataset['name'])}_segmentation_prediction_crops.npz")], "SpecAware segmentation prediction crops exported from best_val_miou.pt"
    except Exception as exc:
        more, _ = draw_confusion_or_bars(dataset, metrics, out_dir)
        return more, f"segmentation prediction export failed; fallback generated: {exc}"


def process_dataset(dataset: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    rel_dir = Path(dataset["group"]) / safe_name(dataset["name"])
    out_dir = run_dir / rel_dir
    ensure_dir(out_dir)
    result = Path(dataset["result"])
    metrics = load_metrics(result)

    files: List[str] = []
    notes: List[str] = []
    files.extend(metrics_card(dataset, metrics, out_dir))

    kind = dataset["kind"]
    if kind == "classification":
        more, note = draw_confusion_or_bars(dataset, metrics, out_dir)
    elif kind == "detection":
        more, note = draw_detection(dataset, out_dir)
    elif kind == "change_detection":
        more, note = draw_change_detection(dataset, out_dir)
    elif kind == "unmixing":
        more, note = draw_unmixing(dataset, out_dir)
    elif kind == "segmentation_or_classification":
        more, note = draw_segmentation_prediction(dataset, metrics, out_dir)
    elif kind == "denoising":
        more, note = draw_denoising_prediction(dataset, out_dir)
    elif kind == "super_resolution":
        more, note = draw_sr_prediction(dataset, out_dir)
    else:
        more, note = draw_metric_bars(dataset, metrics, out_dir)
    files.extend(more)
    notes.append(note)

    if not result.exists():
        notes.append("source result directory does not exist yet")
    elif not metrics:
        notes.append("metrics JSON not found or not parseable")

    return {
        "group": dataset["group"],
        "dataset": dataset["name"],
        "kind": kind,
        "source_result": str(result),
        "output_dir": str(out_dir),
        "figures": files,
        "notes": "; ".join(notes),
    }


def write_manifest(rows: List[Dict[str, Any]], run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_path = run_dir / "figure_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "dataset", "kind", "source_result", "output_dir", "figures", "notes"])
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["figures"] = " | ".join(row["figures"])
            writer.writerow(flat)

    readme = run_dir / "README.md"
    with readme.open("w", encoding="utf-8") as f:
        f.write("# Final 200-epoch Paper Visualization Cases\n\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("These figures are generated from saved downstream outputs. ")
        f.write("For tasks where the fine-tuning script did not save pixel-level predictions, ")
        f.write("the pack uses confusion matrices, per-class bars, or scalar metric cards.\n\n")
        for row in rows:
            f.write(f"## {row['group']} / {row['dataset']}\n\n")
            f.write(f"- Source: `{row['source_result']}`\n")
            f.write(f"- Output: `{row['output_dir']}`\n")
            f.write(f"- Note: {row['notes']}\n")
            f.write(f"- Figures: {len(row['figures'])}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_root / run_id
    ensure_dir(run_dir)

    rows = []
    for dataset in DATASETS:
        try:
            rows.append(process_dataset(dataset, run_dir))
        except Exception as exc:
            rows.append({
                "group": dataset["group"],
                "dataset": dataset["name"],
                "kind": dataset["kind"],
                "source_result": str(dataset["result"]),
                "output_dir": str(run_dir / dataset["group"] / safe_name(dataset["name"])),
                "figures": [],
                "notes": f"visualization failed: {exc}",
            })

    write_manifest(rows, run_dir)
    with (args.out_root / "latest_run.txt").open("w", encoding="utf-8") as f:
        f.write(str(run_dir) + "\n")
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
