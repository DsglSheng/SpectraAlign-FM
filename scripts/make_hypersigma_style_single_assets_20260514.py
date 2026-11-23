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
from PIL import Image


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
if not PROJECT.exists():
    PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import finetune_classification as cls  # noqa: E402


OUT_ROOT = PROJECT / "result" / "hypersigma_style_single_prediction_maps_20260514"
PRETRAIN_CKPT = PROJECT / "runs/pretrain_ours_v3wa_kl003_full450k_20260505/branches/ours_v3wa_kl003_full450k/checkpoints/best_total.pt"
PROTO_EO1 = PROJECT / "artifacts/prototypes_v3_core_waterzero_artificialx4_20260503/prototype_bank_eo1.pt"
PROTO_GF5 = PROJECT / "artifacts/prototypes_v3_core_waterzero_artificialx4_20260503/prototype_bank_gf5.pt"


CLASSIFICATION_RUNS = {
    "IP": PROJECT / "result/ours_200e_recheck_oscd_houston_ip_aerorit_20260513/results/classification_ip_recheck_seed3713_no_tta/ours_v3wa_kl003_full450k",
    "PU": PROJECT / "result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_pu/ours_v3wa_kl003_full450k",
    "HC": PROJECT / "result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_hanchuan/ours_v3wa_kl003_full450k",
    "HH": PROJECT / "result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_hh/ours_v3wa_kl003_full450k",
    "HU": PROJECT / "result/ours_v3wa_kl003_full450k_history_epoch130cfg_rerun_20260509/fair_tune_v2/results/hu_160_inv_f192_lr6e5_e180/ours_v3wa_kl003_full450k",
    "ZY": PROJECT / "result/ours_v3wa_kl003_full450k_downstream_20260508/results/classification_zy/ours_v3wa_kl003_full450k",
}

CLASSIFICATION_CKPT = {
    "IP": "best_train_loss.pt",
    "PU": "best_train_loss.pt",
    "HC": "best_train_loss.pt",
    "HH": "best_train_loss.pt",
    "HU": "best_train_loss.pt",
    "ZY": "best_train_loss.pt",
}


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name)).strip("_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm01(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def cube_rgb(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube)
    if arr.ndim == 2:
        return np.repeat(norm01(arr)[..., None], 3, axis=-1)
    if arr.ndim != 3:
        return np.zeros((64, 64, 3), dtype=np.float32)
    if arr.shape[-1] >= 3:
        bands = [arr.shape[-1] // 6, arr.shape[-1] // 2, min(arr.shape[-1] - 1, 5 * arr.shape[-1] // 6)]
        return np.stack([norm01(arr[..., b]) for b in bands], axis=-1)
    return np.repeat(norm01(arr[..., 0])[..., None], 3, axis=-1)


def palette(class_count: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
            [128, 128, 128],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    if class_count + 1 <= base.shape[0]:
        return base[: class_count + 1]
    extra = (plt.get_cmap("tab20")(np.linspace(0, 1, class_count + 1))[:, :3] * 255).astype(np.uint8)
    extra[0] = 0
    return extra


def save_rgb(path: Path, arr: np.ndarray, *, also_pdf: bool = False) -> list[str]:
    ensure_dir(path.parent)
    data = np.asarray(arr)
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 1)
        data = (data * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(data).save(path)
    files = [str(path)]
    if also_pdf:
        pdf = path.with_suffix(".pdf")
        Image.fromarray(data).save(pdf)
        files.append(str(pdf))
    return files


def save_label(path: Path, labels: np.ndarray, class_count: int, *, also_pdf: bool = True) -> list[str]:
    pal = palette(class_count)
    lab = np.asarray(labels, dtype=np.int64)
    lab = np.where((lab >= 0) & (lab < pal.shape[0]), lab, 0)
    rgb = pal[lab]
    return save_rgb(path, rgb, also_pdf=also_pdf)


def save_gray(path: Path, arr: np.ndarray, *, also_pdf: bool = False) -> list[str]:
    return save_rgb(path, np.repeat(norm01(arr)[..., None], 3, axis=-1), also_pdf=also_pdf)


def save_heat(path: Path, arr: np.ndarray, cmap: str = "magma", *, also_pdf: bool = False) -> list[str]:
    ensure_dir(path.parent)
    rgba = plt.get_cmap(cmap)(norm01(arr))
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    return save_rgb(path, rgb, also_pdf=also_pdf)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_scene_from_config(config: dict[str, Any], *, batch_size: int | None = None) -> tuple[Any, dict[str, Any]]:
    class Obj:
        pass

    args = Obj()
    for key, value in config.items():
        setattr(args, key, value)
    for key in ("data_root", "output_dir", "checkpoint", "prototype_bank_eo1", "prototype_bank_gf5", "resume_from"):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)))
    scene_config = cls.resolve_scene_config(args.dataset)
    resolved_batch_size = int(batch_size or getattr(args, "batch_size", 1) or 1)
    scene = cls.build_segmentation_scene(
        data_root=args.data_root,
        config=scene_config,
        split_mode=args.split_mode,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        train_num=args.train_num,
        val_num=args.val_num,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=resolved_batch_size,
        num_workers=0,
        augment=False,
        normalization=args.normalization,
        pca_components=args.pca_components,
        pca_whiten=args.pca_whiten,
        train_repeat_factor=1,
        train_spatial_jitter=0,
        centered_training=args.centered_training,
        official_val_ratio=args.official_val_ratio,
        train_shuffle=False,
        train_sampling=args.train_sampling,
        train_drop_last=False,
        eval_drop_last=False,
    )
    return args, {"config": scene_config, "scene": scene, "batch_size": resolved_batch_size}


@torch.no_grad()
def predict_scene_map(model: torch.nn.Module, loader: Any, starts: list[tuple[int, int]], scene: dict[str, Any], precision: str, eval_tta: bool) -> np.ndarray:
    device = next(model.parameters()).device
    class_count = int(scene["class_count"])
    height, width = int(scene["height"]), int(scene["width"])
    padded_height, padded_width = int(scene["padded_height"]), int(scene["padded_width"])
    img_size, overlap_size = int(scene["img_size"]), int(scene["overlap_size"])
    score_sum = torch.zeros(padded_height, padded_width, class_count, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    blend = cls.build_blend_window(img_size, overlap_size, device=device)
    offset = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits_map = cls.infer_tile_logits(model, x, precision=precision, eval_tta=eval_tta)
        for batch_index in range(int(logits_map.shape[0])):
            row, col = starts[offset + batch_index]
            score_sum[row : row + img_size, col : col + img_size] += logits_map[batch_index] * blend.unsqueeze(-1)
            weight_sum[row : row + img_size, col : col + img_size] += blend
        offset += int(logits_map.shape[0])
    logits = score_sum[:height, :width] / weight_sum[:height, :width].unsqueeze(-1).clamp_min(1.0e-6)
    return (logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int16) + 1)


def export_classification_maps(run_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset, result_dir in CLASSIFICATION_RUNS.items():
        target = ensure_dir(out_dir / "classification" / dataset)
        if not result_dir.exists():
            rows.append({"task": "classification", "dataset": dataset, "status": "missing_result_dir", "path": str(result_dir)})
            continue
        config = load_json(result_dir / "config.json")
        args, bundle = build_scene_from_config(config)
        scene_config = bundle["config"]
        scene = bundle["scene"]
        metadata = cls.build_downstream_metadata(
            num_bands=int(scene["channels"]),
            sensor=scene_config.sensor,
            level=scene_config.level,
            gsd=scene_config.gsd_m if args.gsd is None else float(args.gsd),
            wavelengths_nm=scene_config.wavelengths if int(scene_config.wavelengths.numel()) == int(scene["channels"]) else None,
            fwhm_nm=scene_config.fwhm if int(scene_config.fwhm.numel()) == int(scene["channels"]) else None,
        )
        extractor = cls.AwareFeatureExtractor(
            checkpoint_path=PRETRAIN_CKPT,
            prototype_bank_eo1=PROTO_EO1,
            prototype_bank_gf5=PROTO_GF5,
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        model = cls.AwareSSFusionSeg(
            extractor,
            int(scene["class_count"]),
            feature_layers=int(args.feature_layers),
            fusion_channels=int(args.fusion_channels),
        ).to(device)
        ckpt_path = result_dir / CLASSIFICATION_CKPT[dataset]
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        # Use the test loader starts because it covers the full scene, while metrics
        # are evaluated only on nonzero test labels.
        pred_all = predict_scene_map(
            model,
            scene["test_loader"],
            scene["test_starts"],
            scene,
            precision=str(args.precision),
            eval_tta=bool(args.eval_tta),
        )
        labels = np.asarray(scene["labels"], dtype=np.int16)
        test_gt = np.asarray(scene["test_gt"], dtype=np.int16)
        pred_on_gt = np.where(labels > 0, pred_all, 0).astype(np.int16)
        pred_on_test = np.where(test_gt > 0, pred_all, 0).astype(np.int16)
        class_count = int(scene["class_count"])
        files: list[str] = []
        files += save_label(target / "groundtruth_full_map.png", labels, class_count)
        files += save_label(target / "test_groundtruth_mask.png", test_gt, class_count)
        files += save_label(target / "ours_prediction_full_scene.png", pred_all, class_count)
        files += save_label(target / "ours_prediction_on_groundtruth_mask.png", pred_on_gt, class_count)
        files += save_label(target / "ours_prediction_on_test_mask.png", pred_on_test, class_count)
        files += save_rgb(target / "original_rgb.png", cube_rgb(scene["cube"]))
        np.savez_compressed(
            target / "classification_maps_raw.npz",
            labels=labels,
            test_gt=test_gt,
            prediction_full_scene=pred_all,
            prediction_on_groundtruth_mask=pred_on_gt,
            prediction_on_test_mask=pred_on_test,
        )
        with (target / "class_palette.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "r", "g", "b"])
            pal = palette(class_count)
            writer.writerow([0, "background", *pal[0].tolist()])
            for i, name in enumerate(scene_config.class_names, start=1):
                writer.writerow([i, name, *pal[i].tolist()])
        rows.append(
            {
                "task": "classification",
                "dataset": dataset,
                "status": "ok",
                "checkpoint": str(ckpt_path),
                "class_count": class_count,
                "shape": [int(scene["height"]), int(scene["width"])],
                "file_count": len(files) + 2,
            }
        )
        del model, extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def latest_raw_root() -> Path:
    base = PROJECT / "result/paper_raw_prediction_assets_20260514"
    pointer = base / "latest_run.txt"
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if p.exists():
            return p / "final_200epoch_comparison" / "raw"
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No raw prediction asset run found under {base}")
    return runs[-1] / "final_200epoch_comparison" / "raw"


def copy_if_exists(src: Path, dst: Path) -> list[str]:
    if not src.exists():
        return []
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return [str(dst)]


def change_outcome(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = np.asarray(label).astype(np.int32) > 0
    pr = np.asarray(pred).astype(np.int32) > 0
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[~gt & ~pr] = [25, 25, 25]       # TN
    rgb[gt & pr] = [255, 255, 255]      # TP
    rgb[~gt & pr] = [230, 25, 75]       # FP
    rgb[gt & ~pr] = [0, 130, 200]       # FN
    return rgb


def binary_rgb(mask: np.ndarray, fg=(230, 25, 75), bg=(20, 20, 20)) -> np.ndarray:
    m = np.asarray(mask).astype(np.float32) > 0
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[~m] = bg
    rgb[m] = fg
    return rgb


def export_detection_from_raw(raw_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    src_base = raw_root / "hypersigma_detection"
    for dataset in ["HTD_Mosaic", "HTD_AVIRIS", "HAD_Pavia", "HAD_CRI", "HAD_Viareggio"]:
        src = src_base / dataset
        dst = ensure_dir(out_dir / "detection" / dataset)
        files: list[str] = []
        for old, new in [
            ("original_rgb.png", "original_rgb.png"),
            ("score_map.png", "ours_score_map.png"),
            ("label.png", "groundtruth_binary.png"),
            ("prediction_mask_p995.png", "ours_prediction_binary_p995.png"),
            ("error.png", "ours_error_map.png"),
        ]:
            files += copy_if_exists(src / old, dst / new)
        npz = src / "raw_arrays.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            if "label" in z:
                files += save_rgb(dst / "groundtruth_target_map.png", binary_rgb(z["label"], fg=(255, 255, 255)))
            if "prediction_mask" in z:
                files += save_rgb(dst / "ours_prediction_target_map.png", binary_rgb(z["prediction_mask"]))
            if "score_map" in z:
                files += save_heat(dst / "ours_score_heatmap.png", z["score_map"])
            copy_if_exists(npz, dst / "raw_arrays.npz")
        rows.append({"task": "detection", "dataset": dataset, "status": "ok" if files else "missing", "file_count": len(files)})
    return rows


def export_change_from_raw(raw_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    src_base = raw_root / "hypersigma_change_detection"
    for dataset in ["Farmland", "Hermiston", "BayArea", "Barbara", "OSCD"]:
        srcs = [src_base / dataset] if dataset != "OSCD" else [p for p in (src_base / "OSCD").iterdir() if p.is_dir()]
        for src in srcs:
            name = dataset if dataset != "OSCD" else f"OSCD_{src.name}"
            dst = ensure_dir(out_dir / "changedetection" / name)
            files: list[str] = []
            for old, new in [
                ("original_t1_rgb.png", "hsi_time1_rgb.png"),
                ("pre_rgb.png", "hsi_time1_rgb.png"),
                ("original_t2_rgb.png", "hsi_time2_rgb.png"),
                ("post_rgb.png", "hsi_time2_rgb.png"),
                ("label.png", "groundtruth_binary.png"),
                ("prediction.png", "ours_prediction_binary.png"),
                ("probability.png", "ours_probability_map.png"),
                ("error.png", "ours_error_map.png"),
            ]:
                files += copy_if_exists(src / old, dst / new)
            npz = src / "raw_arrays.npz"
            if npz.exists():
                z = np.load(npz, allow_pickle=True)
                lab = z["label"] if "label" in z else None
                pred = z["prediction"] if "prediction" in z else None
                if lab is not None and pred is not None:
                    files += save_rgb(dst / "ours_TN_TP_FP_FN_map.png", change_outcome(lab, pred), also_pdf=True)
                copy_if_exists(npz, dst / "raw_arrays.npz")
            rows.append({"task": "changedetection", "dataset": name, "status": "ok" if files else "missing", "file_count": len(files)})
    return rows


def export_denoise_sr_unmix_from_raw(raw_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Denoising selected Case5 patches.
    src = raw_root / "hypersigma_denoising" / "WDC_Case5"
    for case_dir in sorted(src.glob("case_*")):
        dst = ensure_dir(out_dir / "denoising" / "WDC_Case5" / case_dir.name)
        files = []
        for old, new in [
            ("noisy_rgb.png", "noisy_input_rgb.png"),
            ("prediction_rgb.png", "ours_denoised_rgb.png"),
            ("ground_truth_rgb.png", "groundtruth_clean_rgb.png"),
            ("error.png", "ours_error_heatmap.png"),
        ]:
            files += copy_if_exists(case_dir / old, dst / new)
        npz = case_dir / "raw_arrays.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            if {"noisy", "prediction", "ground_truth"}.issubset(set(z.files)):
                h, w = z["prediction"].shape[:2]
                r, c = h // 2, w // 2
                fig, ax = plt.subplots(figsize=(4.2, 3.0))
                ax.plot(np.asarray(z["ground_truth"])[r, c], label="Groundtruth", linewidth=1.5)
                ax.plot(np.asarray(z["noisy"])[r, c], label="Noisy", linewidth=1.0)
                ax.plot(np.asarray(z["prediction"])[r, c], label="Ours", linewidth=1.2)
                ax.set_xlabel("Band")
                ax.set_ylabel("Value")
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                fig.savefig(dst / "center_pixel_spectral_curve.png", dpi=300)
                fig.savefig(dst / "center_pixel_spectral_curve.pdf", bbox_inches="tight")
                plt.close(fig)
                files += [str(dst / "center_pixel_spectral_curve.png"), str(dst / "center_pixel_spectral_curve.pdf")]
            copy_if_exists(npz, dst / "raw_arrays.npz")
        rows.append({"task": "denoising", "dataset": f"WDC_Case5/{case_dir.name}", "status": "ok" if files else "missing", "file_count": len(files)})

    # Super-resolution Houston Proxy x4/x8.
    for dataset in ["Houston_Proxy_x4", "Houston_Proxy_x8"]:
        src = raw_root / "hypersigma_super_resolution" / dataset
        dst = ensure_dir(out_dir / "super_resolution" / dataset)
        files = []
        for old, new in [
            ("low_resolution_rgb.png", "low_resolution_rgb.png"),
            ("bicubic_input_rgb.png", "bicubic_rgb.png"),
            ("prediction_rgb.png", "ours_sr_rgb.png"),
            ("ground_truth_rgb.png", "groundtruth_hr_rgb.png"),
            ("error.png", "ours_error_heatmap.png"),
            ("raw_arrays.npz", "raw_arrays.npz"),
        ]:
            files += copy_if_exists(src / old, dst / new)
        rows.append({"task": "super_resolution", "dataset": dataset, "status": "ok" if files else "missing", "file_count": len(files)})

    # Unmixing abundance/endmembers.
    src = raw_root / "hypersigma_unmixing" / "Urban4"
    dst = ensure_dir(out_dir / "unmixing" / "Urban4")
    files = []
    for f in sorted(src.glob("abundance_em*.png")):
        files += copy_if_exists(f, dst / f"ours_{f.name}")
    for old, new in [("endmember_spectra.csv", "ours_endmember_spectra.csv"), ("raw_arrays.npz", "raw_arrays.npz")]:
        files += copy_if_exists(src / old, dst / new)
    npz = src / "raw_arrays.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        if "endmember_spectra" in z:
            spec = np.asarray(z["endmember_spectra"])
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            if spec.ndim == 2:
                if spec.shape[0] < spec.shape[1]:
                    spec_plot = spec.T
                else:
                    spec_plot = spec
                for i in range(min(spec_plot.shape[1], 8)):
                    ax.plot(spec_plot[:, i], linewidth=1.2, label=f"EM {i+1}")
            ax.set_xlabel("Band")
            ax.set_ylabel("Reflectance / abundance")
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(dst / "ours_endmember_spectra.png", dpi=300)
            fig.savefig(dst / "ours_endmember_spectra.pdf", bbox_inches="tight")
            plt.close(fig)
            files += [str(dst / "ours_endmember_spectra.png"), str(dst / "ours_endmember_spectra.pdf")]
    rows.append({"task": "unmixing", "dataset": "Urban4", "status": "ok" if files else "missing", "file_count": len(files)})
    return rows


def write_readme(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HyperSIGMA-Style Single Prediction Assets",
        "",
        "This folder contains individual, unstitched visual assets for paper assembly.",
        "The layout follows the task types in the HyperSIGMA appendix Figure folder:",
        "",
        "- `classification/<dataset>/`: full-scene pixel classification maps, GT masks, original RGB, palette CSV.",
        "- `detection/<dataset>/`: original RGB, GT target mask, Ours score map and binary target map.",
        "- `changedetection/<dataset>/`: time-1/time-2 RGB, GT, Ours binary map, probability, TN/TP/FP/FN map.",
        "- `denoising/WDC_Case5/case_*`: noisy input, Ours denoised result, clean GT, error heatmap, spectral curve.",
        "- `super_resolution/Houston_Proxy_x4|x8`: LR/bicubic/Ours/GT/error single images.",
        "- `unmixing/Urban4`: Ours abundance maps and endmember spectra.",
        "- `paper_reference_from_hypersigma_appendix/`: optional crops extracted from the HyperSIGMA appendix PDFs.",
        "",
        "No comparison panels are stitched here; every image is saved as a standalone asset.",
        "",
        "## Export Summary",
        "",
    ]
    for row in rows:
        lines.append(f"- {row.get('task')} / {row.get('dataset')}: {row.get('status')} ({row.get('file_count', 0)} files)")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--skip-classification", action="store_true")
    args = parser.parse_args()
    run_name = args.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_root = latest_raw_root()
    rows: list[dict[str, Any]] = [{"task": "source", "dataset": "raw_assets", "status": "ok", "path": str(raw_root)}]
    if not args.skip_classification:
        rows += export_classification_maps(run_dir, run_dir)
    rows += export_detection_from_raw(raw_root, run_dir)
    rows += export_change_from_raw(raw_root, run_dir)
    rows += export_denoise_sr_unmix_from_raw(raw_root, run_dir)
    write_readme(run_dir, rows)
    (args.out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "items": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
