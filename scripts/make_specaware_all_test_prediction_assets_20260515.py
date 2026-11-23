#!/usr/bin/env python3
"""Export all-test visual assets for the SpecAware comparison track.

The official SpecAware repository exposes model/example code but not a
standalone prediction-visualization exporter or public downstream checkpoints.
This script therefore exports Ours predictions and stores paper-reference
placeholders for official SpecAware figures separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
if not PROJECT.exists():
    PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from scripts import finetune_specaware_segmentation as seg  # noqa: E402


OUT_ROOT = PROJECT / "result" / "specaware_all_test_prediction_visual_assets_corrected_rgb_20260515"


RESULTS = {
    "AeroRIT_initial_complete": PROJECT / "result/specaware_comparison_v1_20260510/aerorit_ours_upeernet_lr5e5_e100_crop128_acc16_20260511",
    "AeroRIT_recheck_partial": PROJECT / "result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/aerorit_recheck_seed20260513/ours_v3wa_kl003_full450k",
    "Qingpu_HSI": PROJECT / "result/specaware_comparison_v1_20260510/qingpu_ours_upeernet_lr5e5_e100_crop128_acc16_20260511",
    "WHU_H2SR": PROJECT / "result/specaware_iterative_improve_20260512/results/whu_h2sr_lr4e5_steps1400_e100_crop256_acc16",
}

CD_RESULTS = {
    "Bay_Area": PROJECT / "result/specaware_iterative_improve_20260512/results/bayarea_stats_lr5e5_drop0_e140/ours_v3wa_kl003_full450k",
    "Santa_Barbara": PROJECT / "result/specaware_iterative_improve_20260512/results/barbara_spectral_lr8e5_drop002_e140/ours_v3wa_kl003_full450k",
}

OFFICIAL_REFS = {
    "AeroRIT": {"SpecAware": "OA 92.85 / AA 91.65 / Kappa 88.63 / mIoU 78.78"},
    "Qingpu-HSI": {"SpecAware": "OA 96.68 / AA 78.23 / Kappa 95.23 / mIoU 59.57"},
    "WHU-H2SR": {"SpecAware": "OA 89.72 / AA 81.84 / Kappa 85.17 / mIoU 67.44"},
    "Bay Area": {"SpecAware": "OA 99.05 / Kappa 98.09 / F1 99.11 / P 99.02 / R 99.20"},
    "Santa Barbara": {"SpecAware": "OA 99.52 / Kappa 98.99 / F1 99.39 / P 99.35 / R 99.42"},
}


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def norm01(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def band_indices_for_rgb(bands: int, wavelengths: list[float] | np.ndarray | None = None) -> tuple[list[int], str]:
    """Use a conventional remote-sensing false color: NIR, red, green."""
    targets = [800.0, 660.0, 550.0]
    if wavelengths is not None and len(wavelengths) == bands:
        wave = np.asarray(wavelengths, dtype=np.float32)
    else:
        wave = np.linspace(400.0, 1000.0, int(bands), dtype=np.float32)
    indices = [int(np.argmin(np.abs(wave - t))) for t in targets]
    desc = "false_color_NIR_R_G"
    return indices, desc


def cube_to_hwc(cube: np.ndarray, *, bands_hint: int | None = None) -> np.ndarray:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D spectral cube, got {arr.shape}")
    if bands_hint is not None:
        if arr.shape[-1] == bands_hint:
            return arr
        if arr.shape[0] == bands_hint:
            return np.transpose(arr, (1, 2, 0))
    # Most scene cubes are HxWxC. Treat the smallest plausible spectral axis as bands.
    if arr.shape[-1] <= 512 and arr.shape[0] > 512 and arr.shape[1] > 128:
        return arr
    if arr.shape[0] <= 512:
        return np.transpose(arr, (1, 2, 0))
    return arr


def false_color_from_hwc(cube: np.ndarray, wavelengths: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, list[int], str]:
    hwc = cube_to_hwc(cube)
    bands, desc = band_indices_for_rgb(int(hwc.shape[-1]), wavelengths)
    rgb = np.stack([norm01(hwc[..., b]) for b in bands], axis=-1)
    return rgb, bands, desc


def save_rgb(path: Path, rgb: np.ndarray) -> str:
    ensure(path.parent)
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    plt.imsave(path, arr)
    return str(path)


def save_gray(path: Path, arr: np.ndarray, *, cmap: str = "gray", vmin: float | None = None, vmax: float | None = None) -> str:
    ensure(path.parent)
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)
    return str(path)


def palette_for(dataset: str, class_count: int) -> np.ndarray:
    pal = np.zeros((class_count + 1, 3), dtype=np.uint8)
    if dataset == "aerorit":
        pal[1:] = np.asarray(seg.AERORIT_CLASS_COLORS, dtype=np.uint8)
        return pal
    if dataset == "whu":
        colors = [
            (170, 210, 80),
            (230, 200, 90),
            (35, 135, 60),
            (120, 190, 95),
            (205, 70, 70),
            (120, 120, 120),
            (210, 120, 55),
            (55, 140, 220),
        ]
        pal[1:] = np.asarray(colors[:class_count], dtype=np.uint8)
        return pal
    tab = plt.get_cmap("tab20", max(class_count, 1))
    for i in range(1, class_count + 1):
        pal[i] = (np.asarray(tab((i - 1) % 20)[:3]) * 255).astype(np.uint8)
    return pal


def save_label(path: Path, label: np.ndarray, class_count: int, dataset: str) -> list[str]:
    pal = palette_for(dataset, class_count)
    arr = np.asarray(label, dtype=np.int64)
    arr = np.where((arr >= 0) & (arr <= class_count), arr, 0)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    save_rgb(png, pal[arr])
    fig, ax = plt.subplots(figsize=(arr.shape[1] / 160, arr.shape[0] / 160), dpi=160)
    ax.imshow(pal[arr])
    ax.axis("off")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return [str(png), str(pdf)]


def save_label_png_only(path: Path, label: np.ndarray, class_count: int, dataset: str) -> str:
    pal = palette_for(dataset, class_count)
    arr = np.asarray(label, dtype=np.int64)
    arr = np.where((arr >= 0) & (arr <= class_count), arr, 0)
    png = path.with_suffix(".png")
    save_rgb(png, pal[arr])
    return str(png)


def dilate_sparse_labels(label: np.ndarray, radius: int = 4) -> np.ndarray:
    arr = np.asarray(label, dtype=np.uint8)
    try:
        from scipy.ndimage import maximum_filter

        return maximum_filter(arr, size=max(1, radius * 2 + 1))
    except Exception:
        out = arr.copy()
        ys, xs = np.nonzero(arr > 0)
        h, w = arr.shape
        for y, x in zip(ys, xs):
            out[max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)] = arr[y, x]
        return out


def dilate_sparse_float(values: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    filled = np.where(mask, np.nan_to_num(values, nan=0.0), 0.0).astype(np.float32)
    try:
        from scipy.ndimage import maximum_filter

        dil = maximum_filter(filled, size=max(1, radius * 2 + 1))
        dil_mask = maximum_filter(mask.astype(np.uint8), size=max(1, radius * 2 + 1)) > 0
        return np.where(dil_mask, dil, np.nan).astype(np.float32)
    except Exception:
        out = np.full(values.shape, np.nan, dtype=np.float32)
        ys, xs = np.nonzero(mask)
        h, w = values.shape
        for y, x in zip(ys, xs):
            out[max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)] = values[y, x]
        return out


def save_sparse_error_rgb(path: Path, gt: np.ndarray, pred: np.ndarray) -> str:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    mask = gt > 0
    out[mask & (gt == pred)] = (40, 180, 70)
    out[mask & (gt != pred)] = (220, 60, 60)
    save_rgb(path, out)
    return str(path)


def save_palette_csv(path: Path, class_names: list[str] | tuple[str, ...], dataset: str) -> str:
    pal = palette_for(dataset, len(class_names))
    ensure(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "r", "g", "b"])
        writer.writerow([0, "background", *pal[0].tolist()])
        for i, name in enumerate(class_names, start=1):
            writer.writerow([i, name, *pal[i].tolist()])
    return str(path)


def image_view_rgb(image: seg.SpectralImageView, wavelengths: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, list[int], str]:
    bands, height, width = image.shape
    rgb_bands, desc = band_indices_for_rgb(bands, wavelengths)
    if image.layout == "bsq":
        cube = np.stack([np.asarray(image.data[b]) for b in rgb_bands], axis=-1)
    elif image.layout == "bil":
        cube = np.stack([np.asarray(image.data[:, b, :]) for b in rgb_bands], axis=-1)
    else:
        cube = np.zeros((height, width, 3), dtype=np.float32)
    return np.stack([norm01(cube[..., i]) for i in range(3)], axis=-1), rgb_bands, desc


def tile_rgb_from_chw(chw: np.ndarray, wavelengths: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, list[int], str]:
    arr = np.asarray(chw, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] >= 3:
        bands, desc = band_indices_for_rgb(int(arr.shape[0]), wavelengths)
        return np.stack([norm01(arr[b]) for b in bands], axis=-1), bands, desc
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        bands, desc = band_indices_for_rgb(int(arr.shape[-1]), wavelengths)
        return np.stack([norm01(arr[..., b]) for b in bands], axis=-1), bands, desc
    return np.repeat(norm01(np.squeeze(arr))[..., None], 3, axis=-1), [0, 0, 0], "gray_repeat"


def load_specaware_model(result: Path, *, device: torch.device) -> tuple[nn_model, dict[str, Any]]:  # type: ignore[valid-type]
    cfg = read_json(result / "config.json")
    args_cfg = cfg.get("args", {})
    data_kind = str(cfg.get("dataset") or args_cfg.get("dataset"))
    if data_kind == "qingpu":
        bands = int(cfg["image_info"]["bands"])
        wavelengths = torch.linspace(400.0, 1000.0, bands)
        metadata = seg.build_downstream_metadata(num_bands=bands, sensor="eo1", level="L1", gsd=0.75, wavelengths_nm=wavelengths, fwhm_nm=None)
        class_count = len(seg.QINGPU_CLASS_NAMES)
    elif data_kind == "aerorit":
        bands = int(cfg["image_info"]["bands"])
        wavelengths_raw = cfg.get("image_info", {}).get("wavelengths", [])
        wavelengths = torch.tensor(wavelengths_raw, dtype=torch.float32) if len(wavelengths_raw) == bands else torch.linspace(400.0, 897.0, bands)
        metadata = seg.build_downstream_metadata(num_bands=bands, sensor="others", level="L1", gsd=0.4, wavelengths_nm=wavelengths, fwhm_nm=None)
        class_count = len(seg.AERORIT_CLASS_NAMES)
    elif data_kind == "whu":
        bands = 249
        metadata = seg.build_downstream_metadata(num_bands=bands, sensor="others", level="surface", gsd=1.0, wavelengths_nm=torch.linspace(400.0, 1000.0, bands), fwhm_nm=None)
        class_count = len(seg.WHU_H2SR_CLASS_NAMES)
    else:
        raise ValueError(data_kind)
    extractor = seg.AwareFeatureExtractor(
        checkpoint_path=Path(args_cfg["checkpoint"]),
        prototype_bank_eo1=Path(args_cfg["prototype_bank_eo1"]),
        prototype_bank_gf5=Path(args_cfg["prototype_bank_gf5"]),
        metadata=metadata,
        freeze_encoder=False,
        device=device,
    )
    model = seg.AwareUPerNetSeg(
        extractor,
        class_count,
        decoder_channels=int(args_cfg.get("decoder_channels", 256)),
        segmentation_channels=int(args_cfg.get("segmentation_channels", 64)),
    ).to(device)
    ckpt = result / "best_val_miou.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def build_scene_dataset(result: Path) -> dict[str, Any]:
    cfg = read_json(result / "config.json")
    args_cfg = cfg["args"]
    data_kind = str(cfg.get("dataset") or args_cfg.get("dataset"))
    data_dir = Path(args_cfg["data_dir"])
    seed = int(args_cfg.get("seed", 20260511))
    crop_size = int(args_cfg.get("crop_size", 128))
    stats_samples = int(args_cfg.get("stats_samples", 80000))
    if data_kind == "qingpu":
        image, image_info = seg.open_envi_image(data_dir / "Qingpu_HSI_image.dat", data_dir / "Qingpu_HSI_image.hdr")
        wavelengths = np.linspace(400.0, 1000.0, int(image_info["bands"]), dtype=np.float32)
        gt_mem, _ = seg.open_envi_image(data_dir / "Qingpu_HSI_gt.dat", data_dir / "Qingpu_HSI_gt.hdr")
        labels = np.asarray(gt_mem.data[0] if gt_mem.layout == "bsq" else gt_mem.data[:, 0, :], dtype=np.uint8)
        class_names = seg.QINGPU_CLASS_NAMES
        dataset_tag = "qingpu"
        split_path = result / f"splits_seed{seed}_tr{int(args_cfg.get('train_per_class', 2000))}_va{int(args_cfg.get('val_per_class', 500))}_te{int(args_cfg.get('test_per_class', 1000))}.npz"
        splits = seg.split_indices(
            labels,
            class_count=len(class_names),
            train_per_class=int(args_cfg.get("train_per_class", 2000)),
            val_per_class=int(args_cfg.get("val_per_class", 500)),
            test_per_class=int(args_cfg.get("test_per_class", 1000)),
            seed=seed,
            output_path=split_path,
        )
        mean, std = seg.estimate_band_stats(image, labels, sample_count=stats_samples, seed=seed, output_path=result / f"band_stats_{stats_samples}_seed{seed}.npz")
        return dict(kind=data_kind, image=image, labels=labels, splits=splits, mean=mean, std=std, crop_size=crop_size, class_names=class_names, dataset_tag=dataset_tag, wavelengths=wavelengths)
    if data_kind == "aerorit":
        import tifffile

        refl = data_dir / "full_reflectance" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine-001_"
        hdr = data_dir / "headers" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
        if not hdr.exists():
            hdr = data_dir / "headers_extracted" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
        image, image_info = seg.open_envi_image(refl, hdr)
        bands = int(image_info["bands"])
        wavelengths = image_info.get("wavelengths", [])
        if len(wavelengths) != bands:
            wavelengths = np.linspace(400.0, 897.0, bands, dtype=np.float32)
        rgb_label = np.asarray(tifffile.memmap(data_dir / "sample_10band" / "image_labels.tif"), dtype=np.uint8)
        labels = np.zeros(rgb_label.shape[:2], dtype=np.uint8)
        for class_id, color in enumerate(seg.AERORIT_CLASS_COLORS, start=1):
            labels[np.all(rgb_label == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
        class_names = seg.AERORIT_CLASS_NAMES
        dataset_tag = "aerorit"
        split_path = result / f"splits_seed{seed}_tr{int(args_cfg.get('train_per_class', 3000))}_va{int(args_cfg.get('val_per_class', 1000))}_te{int(args_cfg.get('test_per_class', 3000))}.npz"
        splits = seg.split_indices(
            labels,
            class_count=len(class_names),
            train_per_class=int(args_cfg.get("train_per_class", 3000)),
            val_per_class=int(args_cfg.get("val_per_class", 1000)),
            test_per_class=int(args_cfg.get("test_per_class", 3000)),
            seed=seed,
            output_path=split_path,
        )
        mean, std = seg.estimate_band_stats(image, labels, sample_count=stats_samples, seed=seed, output_path=result / f"band_stats_{stats_samples}_seed{seed}.npz")
        return dict(kind=data_kind, image=image, labels=labels, splits=splits, mean=mean, std=std, crop_size=crop_size, class_names=class_names, dataset_tag=dataset_tag, wavelengths=wavelengths)
    raise ValueError(data_kind)


@torch.no_grad()
def export_sparse_scene(name: str, result: Path, out_dir: Path, *, device: torch.device, batch_size: int = 16) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model, cfg = load_specaware_model(result, device=device)
    scene = build_scene_dataset(result)
    image: seg.SpectralImageView = scene["image"]
    labels: np.ndarray = scene["labels"]
    splits = scene["splits"]
    mean = scene["mean"].astype(np.float32)
    std = np.maximum(scene["std"].astype(np.float32), 1.0e-6)
    crop_size = int(scene["crop_size"])
    class_count = len(scene["class_names"])
    dataset_tag = str(scene["dataset_tag"])
    target = ensure(out_dir / name)
    rgb, bands, rgb_desc = image_view_rgb(image, scene.get("wavelengths"))
    files = [
        save_rgb(target / "original_pseudo_rgb.png", rgb),
        save_palette_csv(target / "class_palette.csv", scene["class_names"], dataset_tag),
    ]
    files += save_label(target / "groundtruth_full_map.png", labels, class_count, dataset_tag)
    test_indices = np.asarray(splits["test"], dtype=np.int64)
    h, w = labels.shape
    pred_map = np.zeros((h, w), dtype=np.uint8)
    gt_map = np.zeros((h, w), dtype=np.uint8)
    confidence_map = np.full((h, w), np.nan, dtype=np.float32)
    half = crop_size // 2
    use_amp = str(cfg.get("args", {}).get("precision", "bf16")) == "bf16" and device.type == "cuda"
    center_rows: list[int] = []
    center_cols: list[int] = []
    center_pred: list[int] = []
    center_gt: list[int] = []
    center_conf: list[float] = []
    for start in range(0, test_indices.size, batch_size):
        batch_indices = test_indices[start : start + batch_size]
        tensors = []
        coords: list[tuple[int, int, int, int]] = []
        for flat in batch_indices:
            row = int(flat // w)
            col = int(flat % w)
            top = int(np.clip(row - half, 0, max(h - crop_size, 0)))
            left = int(np.clip(col - half, 0, max(w - crop_size, 0)))
            patch = image.read_patch(top, left, crop_size)
            patch = (patch - mean[:, None, None]) / std[:, None, None]
            tensors.append(torch.from_numpy(patch.copy()).float())
            coords.append((row, col, row - top, col - left))
        x = torch.stack(tensors, dim=0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        prob = F.softmax(logits.float(), dim=1).detach().cpu().numpy()
        pred = prob.argmax(axis=1).astype(np.uint8) + 1
        conf = prob.max(axis=1)
        for i, (row, col, rr, cc) in enumerate(coords):
            p = int(pred[i, rr, cc])
            g = int(labels[row, col])
            pred_map[row, col] = p
            gt_map[row, col] = g
            confidence_map[row, col] = float(conf[i, rr, cc])
            center_rows.append(row)
            center_cols.append(col)
            center_pred.append(p)
            center_gt.append(g)
            center_conf.append(float(conf[i, rr, cc]))
    visible_radius = 5
    gt_visible = dilate_sparse_labels(gt_map, radius=visible_radius)
    pred_visible = dilate_sparse_labels(pred_map, radius=visible_radius)
    conf_visible = dilate_sparse_float(confidence_map, gt_map > 0, radius=visible_radius)
    files.append(save_label_png_only(target / "test_groundtruth_single_pixel_sparse_map.png", gt_map, class_count, dataset_tag))
    files.append(save_label_png_only(target / "ours_prediction_single_pixel_sparse_map.png", pred_map, class_count, dataset_tag))
    files += save_label(target / "test_groundtruth_sparse_map.png", gt_visible, class_count, dataset_tag)
    files += save_label(target / "ours_prediction_test_sparse_map.png", pred_visible, class_count, dataset_tag)
    files.append(save_sparse_error_rgb(target / "ours_error_test_sparse_map.png", gt_visible, pred_visible))
    files.append(save_gray(target / "ours_confidence_test_sparse_map.png", np.nan_to_num(conf_visible, nan=0.0), cmap="viridis", vmin=0, vmax=1))
    np.savez_compressed(
        target / "all_test_center_predictions.npz",
        test_indices=test_indices,
        rows=np.asarray(center_rows, dtype=np.int32),
        cols=np.asarray(center_cols, dtype=np.int32),
        gt=np.asarray(center_gt, dtype=np.uint8),
        pred=np.asarray(center_pred, dtype=np.uint8),
        confidence=np.asarray(center_conf, dtype=np.float32),
        prediction_sparse_map=pred_map,
        groundtruth_sparse_map=gt_map,
        confidence_sparse_map=confidence_map,
    )
    files.append(str(target / "all_test_center_predictions.npz"))
    with (target / "rgb_band_info.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        wavelengths = np.asarray(scene.get("wavelengths", []), dtype=np.float32)
        writer.writerow(["channel", "zero_based_band_index", "one_based_band_index", "wavelength_nm", "render", "note"])
        for channel, band in zip(["R", "G", "B"], bands):
            wave = float(wavelengths[int(band)]) if wavelengths.size > int(band) else ""
            writer.writerow([channel, int(band), int(band) + 1, wave, rgb_desc, "R=NIR, G=red, B=green; percentile-normalized"])
    rows.append(
        {
            "dataset": name,
            "type": "single_scene_sparse_test",
            "result_dir": str(result),
            "test_centers": int(test_indices.size),
            "files": len(files),
            "note": "All test centers are saved in the sparse maps and NPZ. Per-center crop PNGs are intentionally not expanded to avoid tens of thousands of redundant files.",
        }
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def export_whu_all_tiles(result: Path, out_dir: Path, *, device: torch.device, batch_size: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model, cfg = load_specaware_model(result, device=device)
    args_cfg = cfg["args"]
    data_dir = Path(args_cfg["data_dir"])
    seed = int(args_cfg.get("seed", 20260511))
    crop_size = int(args_cfg.get("crop_size", 256))
    stats_samples = int(args_cfg.get("stats_samples", 120000))
    train_fraction = float(args_cfg.get("whu_train_fraction", 0.20))
    val_fraction = float(args_cfg.get("whu_val_fraction", 0.10))
    train_pct = int(round(train_fraction * 100))
    val_pct = int(round(val_fraction * 100))
    test_pct = max(0, 100 - train_pct - val_pct)
    pairs = seg.build_whu_pairs(data_dir)
    splits = seg.split_tile_pairs(
        pairs,
        seed=seed,
        output_path=result / f"tile_splits_seed{seed}_tr{train_pct}_va{val_pct}_te{test_pct}.npz",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    mean, std = seg.estimate_tile_band_stats(
        pairs,
        splits["train"],
        sample_count=stats_samples,
        seed=seed,
        output_path=result / f"tile_band_stats_{stats_samples}_seed{seed}.npz",
    )
    dataset = seg.WhuTileDataset(
        pairs=pairs,
        indices=splits["test"],
        mean=mean,
        std=std,
        crop_size=crop_size,
        class_count=len(seg.WHU_H2SR_CLASS_NAMES),
        train=False,
        seed=seed + 41,
    )
    target = ensure(out_dir / "WHU_H2SR" / "all_test_tiles")
    summary = ensure(out_dir / "WHU_H2SR")
    files = [
        save_palette_csv(summary / "class_palette.csv", seg.WHU_H2SR_CLASS_NAMES, "whu"),
    ]
    whu_wavelengths = np.linspace(400.0, 1000.0, 249, dtype=np.float32)
    whu_rgb_bands, whu_rgb_desc = band_indices_for_rgb(249, whu_wavelengths)
    with (summary / "rgb_band_info.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "zero_based_band_index", "one_based_band_index", "wavelength_nm", "render", "note"])
        for channel, band in zip(["R", "G", "B"], whu_rgb_bands):
            writer.writerow([channel, int(band), int(band) + 1, float(whu_wavelengths[int(band)]), whu_rgb_desc, "R=NIR, G=red, B=green; percentile-normalized"])
    use_amp = str(args_cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"
    tile_rows: list[dict[str, Any]] = []
    for start in range(0, len(dataset), batch_size):
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        indices: list[int] = []
        for local_idx in range(start, min(start + batch_size, len(dataset))):
            x, y = dataset[local_idx]
            xs.append(x)
            ys.append(y)
            indices.append(int(splits["test"][local_idx]))
        x_b = torch.stack(xs).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x_b)
        pred_b = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8) + 1
        for j, local_idx in enumerate(range(start, min(start + batch_size, len(dataset)))):
            pair_idx = indices[j]
            image_path, label_path = pairs[pair_idx]
            tile_dir = ensure(target / f"test_{local_idx:04d}_{image_path.stem}")
            x_orig = xs[j].numpy() * std[:, None, None] + mean[:, None, None]
            gt = ys[j].numpy().astype(np.uint8)
            pred = np.where(gt > 0, pred_b[j], 0).astype(np.uint8)
            err = ((pred != gt) & (gt > 0)).astype(np.float32)
            input_rgb, _, _ = tile_rgb_from_chw(x_orig, whu_wavelengths)
            files.append(save_rgb(tile_dir / "input_pseudo_rgb.png", input_rgb))
            files.append(save_label_png_only(tile_dir / "groundtruth.png", gt, len(seg.WHU_H2SR_CLASS_NAMES), "whu"))
            files.append(save_label_png_only(tile_dir / "ours_prediction.png", pred, len(seg.WHU_H2SR_CLASS_NAMES), "whu"))
            files.append(save_gray(tile_dir / "ours_error.png", err, cmap="Reds", vmin=0, vmax=1))
            np.savez_compressed(tile_dir / "prediction_arrays.npz", label=gt, prediction=pred, image_path=str(image_path), label_path=str(label_path), split_pair_index=pair_idx)
            files.append(str(tile_dir / "prediction_arrays.npz"))
            tile_rows.append({"local_test_index": local_idx, "split_pair_index": pair_idx, "image": str(image_path), "label": str(label_path), "dir": str(tile_dir)})
        print(json.dumps({"event": "whu_export_progress", "done": min(start + batch_size, len(dataset)), "total": len(dataset)}, ensure_ascii=True), flush=True)
    with (summary / "tile_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["local_test_index", "split_pair_index", "image", "label", "dir"])
        writer.writeheader()
        writer.writerows(tile_rows)
    rows.append({"dataset": "WHU-H2SR", "type": "all_test_tiles", "result_dir": str(result), "test_tiles": len(dataset), "files": len(files)})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def change_rgb_from_mat(mat_path: Path) -> tuple[np.ndarray | None, np.ndarray | None, list[int], str, int]:
    if loadmat is None or not mat_path.exists():
        return None, None, [], "", 0
    raw = {k: np.asarray(v) for k, v in loadmat(mat_path).items() if not k.startswith("__")}
    cubes = [v for v in raw.values() if v.ndim == 3]
    if len(cubes) < 2:
        return None, None, [], "", 0

    def first_existing(names: list[str]) -> np.ndarray | None:
        lower = {k.lower(): v for k, v in raw.items()}
        for name in names:
            arr = lower.get(name.lower())
            if arr is not None and arr.ndim == 3:
                return arr
        return None

    cube1_candidate = first_existing(["T1", "img_t1", "image_t1", "data_t1", "im1"])
    cube2_candidate = first_existing(["T2", "img_t2", "image_t2", "data_t2", "im2"])
    cube1 = cube1_candidate if cube1_candidate is not None else cubes[0]
    cube2 = cube2_candidate if cube2_candidate is not None else cubes[1]
    band_count = int(cube_to_hwc(cube1).shape[-1])
    rgb1, bands, desc = false_color_from_hwc(cube1)
    rgb2, _, _ = false_color_from_hwc(cube2)
    return rgb1, rgb2, bands, desc, band_count


def save_binary(path: Path, arr: np.ndarray, *, changed_color=(255, 255, 255)) -> str:
    data = np.asarray(arr)
    out = np.zeros((*data.shape, 3), dtype=np.uint8)
    out[data > 0] = np.asarray(changed_color, dtype=np.uint8)
    save_rgb(path, out)
    return str(path)


def save_cd_error(path: Path, gt: np.ndarray, pred: np.ndarray) -> str:
    gt1 = np.asarray(gt) > 0
    pr1 = np.asarray(pred) > 0
    out = np.zeros((*gt1.shape, 3), dtype=np.uint8)
    out[(~gt1) & (~pr1)] = (30, 30, 30)
    out[gt1 & pr1] = (40, 180, 70)
    out[(~gt1) & pr1] = (220, 60, 60)
    out[gt1 & (~pr1)] = (60, 120, 230)
    save_rgb(path, out)
    return str(path)


def export_change_dataset(name: str, result: Path, mat_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    target = ensure(out_dir / name)
    files: list[str] = []
    mat = result / "best_f1_maps.mat"
    if not mat.exists():
        mat = result / "final_best_f1_maps.mat"
    if loadmat is None or not mat.exists():
        return [{"dataset": name, "type": "change_detection", "status": "missing_maps"}]
    raw = loadmat(mat)
    gt = np.asarray(raw.get("gt"))
    pred = np.asarray(raw.get("pred_map"))
    prob = np.asarray(raw.get("prob_map"))
    gt = np.where(gt > 0, 1, 0).astype(np.uint8)
    pred = np.where(pred > 0, 1, 0).astype(np.uint8)
    rgb1, rgb2, rgb_bands, rgb_desc, rgb_band_count = change_rgb_from_mat(mat_path)
    if rgb1 is not None:
        files.append(save_rgb(target / "original_t1_pseudo_rgb.png", rgb1))
    if rgb2 is not None:
        files.append(save_rgb(target / "original_t2_pseudo_rgb.png", rgb2))
    if rgb_bands:
        wavelengths = np.linspace(400.0, 1000.0, max(1, rgb_band_count), dtype=np.float32)
        with (target / "rgb_band_info.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["channel", "zero_based_band_index", "one_based_band_index", "wavelength_nm", "render", "note"])
            for channel, band in zip(["R", "G", "B"], rgb_bands):
                wave = float(wavelengths[int(band)]) if int(band) < wavelengths.size else ""
                writer.writerow([channel, int(band), int(band) + 1, wave, rgb_desc, "R=NIR, G=red, B=green; percentile-normalized"])
    files.append(save_binary(target / "groundtruth_change_mask.png", gt))
    files.append(save_binary(target / "ours_prediction_change_mask.png", pred))
    if prob.size:
        files.append(save_gray(target / "ours_probability_map.png", norm01(prob), cmap="viridis"))
    files.append(save_cd_error(target / "ours_error_tp_fp_fn.png", gt, pred))
    np.savez_compressed(target / "change_detection_arrays.npz", gt=gt, prediction=pred, probability=prob)
    files.append(str(target / "change_detection_arrays.npz"))
    return [{"dataset": name, "type": "change_detection_full_scene", "result_dir": str(result), "files": len(files)}]


def write_tables(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (run_dir / "asset_index.csv").open("w", encoding="utf-8", newline="") as f:
        fields = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SpecAware Comparison All-Test Prediction Assets",
        "",
        "This pack stores Ours prediction assets for the SpecAware comparison track.",
        "",
        "- AeroRIT/Qingpu-HSI are single-scene datasets in this setup: all test-center predictions are stored as sparse full-scene maps and NPZ arrays.",
        "- WHU-H2SR is tile-based: every test tile is exported as standalone input/GT/prediction/error files.",
        "- Bay Area and Santa Barbara are full-scene change-detection maps.",
        "- RGB quicklooks use remote-sensing false color: R=NIR (~800 nm), G=red (~660 nm), B=green (~550 nm), with percentile normalization. When exact wavelengths are unavailable, bands are selected from a linear 400-1000 nm assumption and recorded in each `rgb_band_info.csv`.",
        "- Sparse segmentation maps are rendered with radius-5 point dilation for visibility. The exact single-pixel maps and center predictions are preserved in `*_single_pixel_sparse_map.png` and `all_test_center_predictions.npz`.",
        "- The official SpecAware repository does not provide downstream checkpoints or a prediction-visualization exporter; official paper figure page renders can be placed under `paper_reference_from_specaware`.",
        "",
        "## Official References",
        "",
    ]
    for dataset, values in OFFICIAL_REFS.items():
        lines.append(f"- {dataset}: {values}")
    lines += ["", "## Export Rows", ""]
    for row in rows:
        lines.append(f"- {row.get('dataset')}: {row.get('type')} ({row.get('files', 0)} files)")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--skip-whu", action="store_true")
    parser.add_argument("--only-whu", action="store_true")
    parser.add_argument("--whu-result-dir", type=Path, default=None)
    parser.add_argument("--whu-batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_name = args.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    if not args.only_whu:
        # Completed AeroRIT result is retained; the later recheck checkpoint is also exported if present.
        for name in ["AeroRIT_initial_complete", "AeroRIT_recheck_partial", "Qingpu_HSI"]:
            result = RESULTS[name]
            if (result / "best_val_miou.pt").exists() and (result / "config.json").exists():
                try:
                    rows += export_sparse_scene(name, result, run_dir / "segmentation", device=device)
                except Exception as exc:
                    rows.append({"dataset": name, "type": "single_scene_sparse_test", "status": f"failed: {exc}", "result_dir": str(result)})
            else:
                rows.append({"dataset": name, "type": "single_scene_sparse_test", "status": "missing_checkpoint_or_config", "result_dir": str(result)})

    if not args.skip_whu:
        result = args.whu_result_dir or RESULTS["WHU_H2SR"]
        if (result / "best_val_miou.pt").exists() and (result / "config.json").exists():
            try:
                rows += export_whu_all_tiles(result, run_dir / "segmentation", device=device, batch_size=args.whu_batch_size)
            except Exception as exc:
                rows.append({"dataset": "WHU-H2SR", "type": "all_test_tiles", "status": f"failed: {exc}", "result_dir": str(result)})
        else:
            rows.append({"dataset": "WHU-H2SR", "type": "all_test_tiles", "status": "missing_checkpoint_or_config", "result_dir": str(result)})

    if not args.only_whu:
        rows += export_change_dataset("Bay_Area", CD_RESULTS["Bay_Area"], PROJECT / "data/change_detection/bayarea/BayArea.mat", run_dir / "change_detection")
        rows += export_change_dataset("Santa_Barbara", CD_RESULTS["Santa_Barbara"], PROJECT / "data/change_detection/barbara/Barbara.mat", run_dir / "change_detection")

    ensure(run_dir / "paper_reference_from_specaware")
    (run_dir / "paper_reference_from_specaware" / "README.md").write_text(
        "Official SpecAware prediction images are not reproducible from the public repo because downstream checkpoints/export scripts are not released. "
        "Rendered pages/crops from the SpecAware paper should be stored here for visual reference.\n",
        encoding="utf-8",
    )
    write_tables(run_dir, rows)
    (args.out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
