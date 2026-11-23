#!/usr/bin/env python3
"""Export raw prediction assets for SCI paper figure assembly.

This script is intentionally non-destructive. It reads completed downstream
results, writes a timestamped pack, and keeps both raw single-panel assets and
ready-to-use selected paper panels.

Output layout:
  raw/<track>/<dataset>/
      original*.png, label.png, prediction.png, score/probability.png, error.png
      raw_arrays.npz when arrays are available
  paper_panels/<track>/<dataset>/
      compact composed panels for paper drafting
  analysis/<track>/<dataset>/
      confusion matrices / metric cards for classification-style tasks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
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
except Exception:
    loadmat = None


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
DEFAULT_OUT_ROOT = PROJECT / "result" / "paper_raw_prediction_assets_20260514"

sys.path.insert(0, str(PROJECT / "scripts"))
import make_final_200e_prediction_visual_cases as final_vis  # noqa: E402
import make_v3wa_kl003_ablation_prediction_visual_cases as ab_vis  # noqa: E402


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return json.load(handle)


def norm01(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0, 1)


def cube_to_hwc(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D cube, got {arr.shape}")
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
        out[..., channel] = norm01(rgb[..., channel])
    return out


def save_image(path: Path, arr: np.ndarray, *, cmap: Optional[str] = None, vmin: Any = None, vmax: Any = None) -> str:
    ensure(path.parent)
    data = np.asarray(arr)
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        plt.imsave(path, np.clip(data, 0, 1))
    else:
        plt.imsave(path, data, cmap=cmap, vmin=vmin, vmax=vmax)
    return str(path)


def save_panel(path_base: Path, panels: Sequence[Tuple[str, np.ndarray, Optional[str]]], title: str = "") -> List[str]:
    ensure(path_base.parent)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.25))
    if n == 1:
        axes = [axes]
    for ax, (name, arr, cmap) in zip(axes, panels):
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            ax.imshow(np.clip(arr, 0, 1))
        else:
            ax.imshow(arr, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.03)
    written = []
    for ext in ("png", "pdf"):
        out = path_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(str(out))
    plt.close(fig)
    return written


def find_file(result: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = result / name
        if p.exists():
            return p
    return None


def load_npy(result: Path, names: Sequence[str]) -> Optional[np.ndarray]:
    p = find_file(result, names)
    if p is None:
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def load_result_maps(result: Path) -> Dict[str, np.ndarray]:
    if loadmat is None:
        return {}
    candidates = [
        "final_best_f1_maps.mat", "best_f1_maps.mat", "latest_maps.mat",
        "final_best_auc_score_map.mat", "best_auc_score_map.mat", "latest_score_map.mat",
        "unmixing_result.mat",
    ]
    maps: Dict[str, np.ndarray] = {}
    for name in candidates:
        path = result / name
        if not path.exists():
            continue
        try:
            raw = loadmat(path)
        except Exception:
            continue
        for k, v in raw.items():
            if not k.startswith("__") and isinstance(v, np.ndarray):
                maps[k] = np.asarray(v)
    return maps


def arrays_from_data_root(data_root: Path, target_shape: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    """Best-effort loader for original cubes/labels from common .mat files."""
    if loadmat is None or not data_root.exists():
        return {}
    mats = [p for p in data_root.glob("*.mat")]
    # Avoid huge raw archives when smaller prepared .mat files exist.
    mats = sorted(mats, key=lambda p: (("raw" in str(p).lower()), p.stat().st_size if p.exists() else 0))
    arrays: Dict[str, np.ndarray] = {}
    for mat in mats[:4]:
        try:
            raw = loadmat(mat)
        except Exception:
            continue
        for k, v in raw.items():
            if k.startswith("__") or not isinstance(v, np.ndarray):
                continue
            arr = np.squeeze(np.asarray(v))
            if arr.ndim not in (2, 3):
                continue
            if target_shape is not None and arr.ndim >= 2:
                shp = arr.shape[:2] if arr.ndim == 2 else (arr.shape[0], arr.shape[1])
                shp_alt = arr.shape[-2:] if arr.ndim == 3 else shp
                if shp != target_shape and shp_alt != target_shape:
                    continue
            arrays[f"{mat.stem}:{k}"] = arr
    return arrays


def choose_cube_and_label(arrays: Dict[str, np.ndarray], target_shape: Optional[Tuple[int, int]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    cube = None
    label = None
    for key, arr in arrays.items():
        low = key.lower()
        if arr.ndim == 3 and cube is None:
            cube = arr
        if arr.ndim == 2:
            uniq = np.unique(arr[np.isfinite(arr)]) if np.issubdtype(arr.dtype, np.number) else np.array([])
            looks_label = any(s in low for s in ("gt", "label", "map", "target", "mask")) or (0 < uniq.size <= 32)
            if looks_label and label is None:
                if target_shape is None or arr.shape == target_shape:
                    label = arr
    return cube, label


def write_npz(path: Path, **arrays: np.ndarray) -> Optional[str]:
    payload = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    if not payload:
        return None
    ensure(path.parent)
    np.savez_compressed(path, **payload)
    return str(path)


def export_classification(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path, analysis_dir: Path) -> List[str]:
    metrics = final_vis.load_metrics(Path(dataset["result"]))
    files = []
    more, _ = final_vis.draw_confusion_or_bars(dataset, metrics, analysis_dir)
    files.extend(more)
    files.extend(final_vis.metrics_card(dataset, metrics, analysis_dir))
    return files


def export_detection(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    score = load_npy(result, ["final_best_auc_score_map.npy", "best_auc_score_map.npy", "latest_score_map.npy"])
    if score is None:
        return [], "score map missing"
    score = np.squeeze(score)
    files: List[str] = []
    files.append(save_image(raw_dir / "score_map.png", norm01(score), cmap="magma"))
    threshold = float(np.percentile(score[np.isfinite(score)], 99.5)) if np.isfinite(score).any() else 0.0
    pred = (score >= threshold).astype(np.uint8)
    files.append(save_image(raw_dir / "prediction_mask_p995.png", pred, cmap="gray", vmin=0, vmax=1))
    cfg = read_json(result / "config.json")
    arrays = arrays_from_data_root(Path(cfg.get("data_root", "")), score.shape)
    cube, label = choose_cube_and_label(arrays, score.shape)
    rgb = cube_rgb(cube) if cube is not None else None
    if rgb is not None:
        files.append(save_image(raw_dir / "original_rgb.png", rgb))
    if label is not None:
        label = np.squeeze(label)
        files.append(save_image(raw_dir / "label.png", label, cmap="gray"))
        err = ((label > 0) != (pred > 0)).astype(np.uint8)
        files.append(save_image(raw_dir / "error.png", err, cmap="Reds", vmin=0, vmax=1))
    files.append(write_npz(raw_dir / "raw_arrays.npz", original_cube=cube, label=label, score_map=score, prediction_mask=pred) or "")
    panel_items = []
    if rgb is not None:
        panel_items.append(("Original", rgb, None))
    if label is not None:
        panel_items.append(("Label", label, "gray"))
    panel_items.extend([("Score", norm01(score), "magma"), ("Prediction", pred, "gray")])
    if label is not None:
        panel_items.append(("Error", ((label > 0) != (pred > 0)).astype(np.uint8), "Reds"))
    files.extend(save_panel(panel_dir / "selected_detection_panel", panel_items, dataset["name"]))
    return [f for f in files if f], "full score/prediction map exported"


def export_change_detection(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    maps = load_result_maps(result)
    pred = load_npy(result, ["final_best_f1_pred_map.npy", "best_f1_pred_map.npy", "latest_pred_map.npy"])
    prob = load_npy(result, ["final_best_f1_prob_map.npy", "best_f1_prob_map.npy", "latest_prob_map.npy"])
    for key in ("pred_map", "pred", "prediction"):
        if pred is None and key in maps:
            pred = np.squeeze(maps[key])
    for key in ("prob_map", "prob", "score_map", "score"):
        if prob is None and key in maps:
            prob = np.squeeze(maps[key])
    label = None
    for key in ("gt", "label", "labels", "gt_map", "target"):
        if key in maps:
            label = np.squeeze(maps[key])
            break
    if pred is None and prob is None and str(dataset["name"]).lower() == "oscd":
        return export_oscd_all(dataset, raw_dir, panel_dir)
    if pred is None and prob is None:
        return [], "prediction/probability maps missing"
    if pred is None and prob is not None:
        pred = (np.squeeze(prob) >= 0.5).astype(np.uint8)
    pred = np.squeeze(pred)
    target_shape = pred.shape
    cfg = read_json(result / "config.json")
    arrays = arrays_from_data_root(Path(cfg.get("data_root", "")) / str(cfg.get("dataset", dataset["name"])).lower(), target_shape)
    if not arrays:
        arrays = arrays_from_data_root(Path(cfg.get("data_root", "")), target_shape)
    cubes = [arr for arr in arrays.values() if arr.ndim == 3]
    if label is None:
        _, label = choose_cube_and_label(arrays, target_shape)
    files: List[str] = []
    rgb1 = cube_rgb(cubes[0]) if len(cubes) >= 1 else None
    rgb2 = cube_rgb(cubes[1]) if len(cubes) >= 2 else None
    if rgb1 is not None:
        files.append(save_image(raw_dir / "original_t1_rgb.png", rgb1))
    if rgb2 is not None:
        files.append(save_image(raw_dir / "original_t2_rgb.png", rgb2))
    if label is not None:
        files.append(save_image(raw_dir / "label.png", label, cmap="gray", vmin=0, vmax=1))
    if prob is not None:
        files.append(save_image(raw_dir / "probability.png", norm01(np.squeeze(prob)), cmap="viridis"))
    files.append(save_image(raw_dir / "prediction.png", pred, cmap="gray", vmin=0, vmax=1))
    err = None
    if label is not None and np.squeeze(label).shape == pred.shape:
        err = (np.squeeze(label).astype(np.int32) != pred.astype(np.int32)).astype(np.uint8)
        files.append(save_image(raw_dir / "error.png", err, cmap="Reds", vmin=0, vmax=1))
    files.append(write_npz(raw_dir / "raw_arrays.npz", t1=cubes[0] if len(cubes) >= 1 else None, t2=cubes[1] if len(cubes) >= 2 else None, label=label, probability=prob, prediction=pred, error=err) or "")
    panels = []
    if rgb1 is not None:
        panels.append(("T1", rgb1, None))
    if rgb2 is not None:
        panels.append(("T2", rgb2, None))
    if label is not None:
        panels.append(("Label", label, "gray"))
    panels.append(("Prediction", pred, "gray"))
    if prob is not None:
        panels.append(("Probability", norm01(prob), "viridis"))
    if err is not None:
        panels.append(("Error", err, "Reds"))
    files.extend(save_panel(panel_dir / "selected_change_panel", panels, dataset["name"]))
    return [f for f in files if f], "full change map exported"


def export_unmixing(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path) -> Tuple[List[str], str]:
    result = Path(dataset["result"])
    path = result / "unmixing_result.mat"
    if loadmat is None or not path.exists():
        return [], "unmixing_result.mat missing"
    raw = {k: np.asarray(v) for k, v in loadmat(path).items() if not k.startswith("__")}
    abundance = None
    spectra = None
    for _, arr in raw.items():
        arr = np.squeeze(arr)
        if arr.ndim == 3 and abundance is None:
            abundance = arr
        if arr.ndim == 2 and spectra is None and min(arr.shape) <= 16 and max(arr.shape) > 16:
            spectra = arr
    if abundance is None:
        return [], "abundance array missing"
    maps = ab_vis._abundance_to_last_channel(abundance) if hasattr(ab_vis, "_abundance_to_last_channel") else np.moveaxis(abundance, 0, -1)
    files: List[str] = []
    for i in range(maps.shape[-1]):
        files.append(save_image(raw_dir / f"abundance_em{i+1:02d}.png", norm01(maps[..., i]), cmap="viridis"))
    if spectra is not None:
        np.savetxt(raw_dir / "endmember_spectra.csv", np.asarray(spectra), delimiter=",")
        files.append(str(raw_dir / "endmember_spectra.csv"))
    files.append(write_npz(raw_dir / "raw_arrays.npz", abundance=maps, endmember_spectra=spectra) or "")
    panels = [(f"EM {i+1}", norm01(maps[..., i]), "viridis") for i in range(min(6, maps.shape[-1]))]
    files.extend(save_panel(panel_dir / "selected_abundance_panel", panels, dataset["name"]))
    return [f for f in files if f], "abundance/endmember arrays exported"


def export_oscd_all(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path) -> Tuple[List[str], str]:
    # Reuse the existing exporter first; it creates the needed model and handles OSCD preprocessing.
    # It exports representative scenes. We then split the NPZ into separate raw PNGs.
    files, note = final_vis.draw_oscd_prediction(dataset, panel_dir)
    npz = panel_dir / "OSCD_prediction_cases.npz"
    if not npz.exists():
        return files, note
    z = np.load(npz)
    extra: List[str] = []
    scene_names = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_pred")})
    for scene in scene_names:
        sdir = raw_dir / safe_name(scene)
        ensure(sdir)
        pre = z[f"{scene}_pre"]
        post = z[f"{scene}_post"]
        gt = z[f"{scene}_gt"]
        pred = z[f"{scene}_pred"]
        prob = z[f"{scene}_prob"]
        extra.append(save_image(sdir / "pre_rgb.png", cube_rgb(pre)))
        extra.append(save_image(sdir / "post_rgb.png", cube_rgb(post)))
        extra.append(save_image(sdir / "label.png", np.where(gt == 255, 0, gt), cmap="gray", vmin=0, vmax=1))
        extra.append(save_image(sdir / "prediction.png", pred, cmap="gray", vmin=0, vmax=1))
        extra.append(save_image(sdir / "probability.png", norm01(prob), cmap="viridis"))
        extra.append(save_image(sdir / "error.png", ((pred != gt) & (gt < 2)).astype(np.uint8), cmap="Reds", vmin=0, vmax=1))
        extra.append(write_npz(sdir / "raw_arrays.npz", pre=pre, post=post, label=gt, prediction=pred, probability=prob) or "")
    return files + [f for f in extra if f], note + "; raw scene images split from NPZ"


def export_denoising(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path) -> Tuple[List[str], str]:
    files, note = final_vis.draw_denoising_prediction(dataset, panel_dir)
    # Split the exported cases into single images. The existing exporter already runs the model.
    stem = safe_name(dataset["name"])
    npz = panel_dir / f"{stem}_prediction_cases.npz"
    if not npz.exists():
        return files, note
    z = np.load(npz)
    extra: List[str] = []
    case_ids = sorted({k.split("_")[1] for k in z.files if k.startswith("case_") and k.endswith("_pred")})
    for cid in case_ids:
        sdir = raw_dir / f"case_{cid}"
        noisy = z[f"case_{cid}_noisy"]
        pred = z[f"case_{cid}_pred"]
        gt = z[f"case_{cid}_gt"]
        err = np.mean(np.abs(cube_to_hwc(pred) - cube_to_hwc(gt)), axis=-1)
        extra.append(save_image(sdir / "noisy_rgb.png", cube_rgb(noisy)))
        extra.append(save_image(sdir / "prediction_rgb.png", cube_rgb(pred)))
        extra.append(save_image(sdir / "ground_truth_rgb.png", cube_rgb(gt)))
        extra.append(save_image(sdir / "error.png", norm01(err), cmap="magma"))
        extra.append(write_npz(sdir / "raw_arrays.npz", noisy=noisy, prediction=pred, ground_truth=gt, error=err) or "")
    return files + [f for f in extra if f], note + "; single-case raw images split from NPZ"


def export_sr(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path) -> Tuple[List[str], str]:
    files, note = final_vis.draw_sr_prediction(dataset, panel_dir)
    stem = safe_name(dataset["name"])
    npz = panel_dir / f"{stem}_prediction_case.npz"
    if not npz.exists():
        return files, note
    z = np.load(npz)
    extra: List[str] = []
    lr = z["low_resolution"]
    bic = z["bicubic_input"]
    pred = z["prediction"]
    gt = z["ground_truth"]
    err = np.mean(np.abs(cube_to_hwc(pred) - cube_to_hwc(gt)), axis=-1)
    extra.append(save_image(raw_dir / "low_resolution_rgb.png", cube_rgb(lr)))
    extra.append(save_image(raw_dir / "bicubic_input_rgb.png", cube_rgb(bic)))
    extra.append(save_image(raw_dir / "prediction_rgb.png", cube_rgb(pred)))
    extra.append(save_image(raw_dir / "ground_truth_rgb.png", cube_rgb(gt)))
    extra.append(save_image(raw_dir / "error.png", norm01(err), cmap="magma"))
    extra.append(write_npz(raw_dir / "raw_arrays.npz", low_resolution=lr, bicubic_input=bic, prediction=pred, ground_truth=gt, error=err) or "")
    return files + [f for f in extra if f], note + "; single raw images split from NPZ"


def export_segmentation(dataset: Dict[str, Any], raw_dir: Path, panel_dir: Path, analysis_dir: Path) -> Tuple[List[str], str]:
    metrics = final_vis.load_metrics(Path(dataset["result"]))
    files, note = final_vis.draw_segmentation_prediction(dataset, metrics, panel_dir)
    # Split representative crop NPZ into individual raw image files. For these large datasets, the
    # selected crops are the practical image-level assets; the full metrics/confusion remain in analysis.
    stem = safe_name(dataset["name"])
    npz = panel_dir / f"{stem}_segmentation_prediction_crops.npz"
    extra: List[str] = []
    if npz.exists():
        z = np.load(npz)
        case_ids = sorted({k.split("_")[1] for k in z.files if k.startswith("case_") and k.endswith("_pred")})
        for cid in case_ids:
            sdir = raw_dir / f"crop_{cid}"
            x = z[f"case_{cid}_input"]
            gt = z[f"case_{cid}_gt"]
            pred = z[f"case_{cid}_pred"]
            extra.append(save_image(sdir / "original_rgb.png", cube_rgb(x)))
            extra.append(save_image(sdir / "label.png", gt, cmap="tab20"))
            extra.append(save_image(sdir / "prediction.png", pred, cmap="tab20"))
            extra.append(save_image(sdir / "error.png", ((pred != gt) & (gt > 0)).astype(np.uint8), cmap="Reds", vmin=0, vmax=1))
            extra.append(write_npz(sdir / "raw_arrays.npz", input=x, label=gt, prediction=pred) or "")
    more, _ = final_vis.draw_confusion_or_bars(dataset, metrics, analysis_dir)
    extra.extend(more)
    return files + [f for f in extra if f], note + "; representative raw crop images split from NPZ"


def process_final_dataset(dataset: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    group = safe_name(dataset["group"])
    name = safe_name(dataset["name"])
    raw_dir = run_dir / "final_200epoch_comparison" / "raw" / group / name
    panel_dir = run_dir / "final_200epoch_comparison" / "paper_panels" / group / name
    analysis_dir = run_dir / "final_200epoch_comparison" / "analysis" / group / name
    ensure(raw_dir); ensure(panel_dir); ensure(analysis_dir)
    kind = dataset["kind"]
    files: List[str] = []
    note = ""
    if kind == "classification":
        files = export_classification(dataset, raw_dir, panel_dir, analysis_dir)
        note = "classification analysis only; no pixel prediction maps were saved"
    elif kind == "detection":
        files, note = export_detection(dataset, raw_dir, panel_dir)
    elif kind == "change_detection":
        files, note = export_change_detection(dataset, raw_dir, panel_dir)
    elif kind == "unmixing":
        files, note = export_unmixing(dataset, raw_dir, panel_dir)
    elif kind == "denoising":
        files, note = export_denoising(dataset, raw_dir, panel_dir)
    elif kind == "super_resolution":
        files, note = export_sr(dataset, raw_dir, panel_dir)
    elif kind == "segmentation_or_classification":
        files, note = export_segmentation(dataset, raw_dir, panel_dir, analysis_dir)
    else:
        files = export_classification(dataset, raw_dir, panel_dir, analysis_dir)
        note = "fallback analysis"
    return {
        "track": "final_200epoch_comparison",
        "group": dataset["group"],
        "dataset": dataset["name"],
        "kind": kind,
        "source_result": str(dataset["result"]),
        "raw_dir": str(raw_dir),
        "panel_dir": str(panel_dir),
        "analysis_dir": str(analysis_dir),
        "file_count": len(files),
        "files": files,
        "note": note,
    }


def process_ablation(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Classification analysis for each branch.
    for model in ab_vis.MODELS:
        dataset = {
            "name": f"HanChuan_{model}",
            "kind": "classification",
            "result": ab_vis.TASK_DIRS["classification_hanchuan"] / model,
            "official": {},
        }
        raw_dir = run_dir / "v3wa_kl003_ablation" / "raw" / "classification_hanchuan" / model
        panel_dir = run_dir / "v3wa_kl003_ablation" / "paper_panels" / "classification_hanchuan" / model
        analysis_dir = run_dir / "v3wa_kl003_ablation" / "analysis" / "classification_hanchuan" / model
        ensure(raw_dir); ensure(panel_dir); ensure(analysis_dir)
        files = []
        try:
            files.extend(export_classification(dataset, raw_dir, panel_dir, analysis_dir))
            note = "classification analysis only"
        except Exception as exc:
            note = f"failed: {exc}"
        rows.append({"track": "v3wa_kl003_ablation", "dataset": "HanChuan", "model": model, "task": "classification", "file_count": len(files), "files": files, "note": note})
    # Detection maps.
    for model in ab_vis.MODELS:
        dataset = {"name": f"HTD-AVIRIS_{model}", "kind": "detection", "result": ab_vis.TASK_DIRS["detection_htd_aviris"] / model, "official": {}}
        raw_dir = run_dir / "v3wa_kl003_ablation" / "raw" / "detection_htd_aviris" / model
        panel_dir = run_dir / "v3wa_kl003_ablation" / "paper_panels" / "detection_htd_aviris" / model
        ensure(raw_dir); ensure(panel_dir)
        try:
            files, note = export_detection(dataset, raw_dir, panel_dir)
        except Exception as exc:
            files, note = [], f"failed: {exc}"
        rows.append({"track": "v3wa_kl003_ablation", "dataset": "HTD-AVIRIS", "model": model, "task": "detection", "file_count": len(files), "files": files, "note": note})
    # Unmixing maps.
    for model in ab_vis.MODELS:
        dataset = {"name": f"Urban4_{model}", "kind": "unmixing", "result": ab_vis.TASK_DIRS["unmix_urban4_hypersigma_spreg6"] / model, "official": {}}
        raw_dir = run_dir / "v3wa_kl003_ablation" / "raw" / "unmix_urban4" / model
        panel_dir = run_dir / "v3wa_kl003_ablation" / "paper_panels" / "unmix_urban4" / model
        ensure(raw_dir); ensure(panel_dir)
        try:
            files, note = export_unmixing(dataset, raw_dir, panel_dir)
        except Exception as exc:
            files, note = [], f"failed: {exc}"
        rows.append({"track": "v3wa_kl003_ablation", "dataset": "Urban4", "model": model, "task": "unmixing", "file_count": len(files), "files": files, "note": note})
    # Denoising branches: use existing ablation exporter then split its NPZ.
    try:
        temp_dir = run_dir / "v3wa_kl003_ablation" / "paper_panels" / "denoising_wdc_case5" / "_all_models"
        ensure(temp_dir)
        den_files, den_note = ab_vis.draw_denoising_predictions(run_dir / "v3wa_kl003_ablation")
        npz = run_dir / "v3wa_kl003_ablation" / "denoising_wdc_case5" / "wdc_case5_ablation_prediction_cases.npz"
        files = list(den_files)
        if npz.exists():
            z = np.load(npz)
            for model in ab_vis.MODELS:
                for k in sorted([x for x in z.files if x.startswith(f"{model}_case") and x.endswith("_pred")]):
                    case = k[len(model) + 5: -5]
                    raw_dir = run_dir / "v3wa_kl003_ablation" / "raw" / "denoising_wdc_case5" / model / f"case_{case}"
                    noisy = z[f"case{case}_noisy"]
                    gt = z[f"case{case}_gt"]
                    pred = z[k]
                    err = np.mean(np.abs(cube_to_hwc(pred) - cube_to_hwc(gt)), axis=-1)
                    files.append(save_image(raw_dir / "noisy_rgb.png", cube_rgb(noisy)))
                    files.append(save_image(raw_dir / "prediction_rgb.png", cube_rgb(pred)))
                    files.append(save_image(raw_dir / "ground_truth_rgb.png", cube_rgb(gt)))
                    files.append(save_image(raw_dir / "error.png", norm01(err), cmap="magma"))
                    files.append(write_npz(raw_dir / "raw_arrays.npz", noisy=noisy, prediction=pred, ground_truth=gt, error=err) or "")
        rows.append({"track": "v3wa_kl003_ablation", "dataset": "WDC Case5", "model": "all", "task": "denoising", "file_count": len(files), "files": [f for f in files if f], "note": den_note})
    except Exception as exc:
        rows.append({"track": "v3wa_kl003_ablation", "dataset": "WDC Case5", "model": "all", "task": "denoising", "file_count": 0, "files": [], "note": f"failed: {exc}"})
    return rows


def write_index(rows: List[Dict[str, Any]], run_dir: Path) -> None:
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (run_dir / "asset_index.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["track", "group", "dataset", "kind", "model", "task", "source_result", "raw_dir", "panel_dir", "analysis_dir", "file_count", "note"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = [
        "# Raw Prediction Assets for SCI Writing",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This pack separates raw single-panel assets from selected composed paper panels.",
        "",
        "- `final_200epoch_comparison/raw`: original/label/prediction/score/error single files when available.",
        "- `final_200epoch_comparison/paper_panels`: selected composed panels.",
        "- `final_200epoch_comparison/analysis`: confusion matrices and metric plots for classification-style tasks.",
        "- `v3wa_kl003_ablation/raw`: raw assets for the final ablation branches.",
        "- `v3wa_kl003_ablation/paper_panels`: selected composed panels for ablation figures.",
        "",
        "Large classification tasks do not have image-like prediction maps saved; confusion matrices and per-class metrics are provided instead.",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_root / run_id
    ensure(run_dir)
    rows: List[Dict[str, Any]] = []
    for dataset in final_vis.DATASETS:
        try:
            rows.append(process_final_dataset(dataset, run_dir))
        except Exception as exc:
            rows.append({
                "track": "final_200epoch_comparison",
                "group": dataset.get("group"),
                "dataset": dataset.get("name"),
                "kind": dataset.get("kind"),
                "source_result": str(dataset.get("result")),
                "file_count": 0,
                "files": [],
                "note": f"failed: {exc}",
            })
    rows.extend(process_ablation(run_dir))
    write_index(rows, run_dir)
    ensure(args.out_root)
    (args.out_root / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
