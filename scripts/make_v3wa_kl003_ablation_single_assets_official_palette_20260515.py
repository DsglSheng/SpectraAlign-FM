#!/usr/bin/env python3
"""Create single-image official-palette assets for v3wa_kl003 ablation.

This is the ablation counterpart of the final HyperSIGMA-style single-image
pack. It keeps every image standalone so paper panels can be assembled later.
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


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
if not PROJECT.exists():
    PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from scripts import finetune_classification as cls  # noqa: E402
from scripts import make_hypersigma_style_single_assets_official_palette_20260514 as style  # noqa: E402


ROOT = PROJECT / "result" / "v3wa_kl003_ablation_final_downstream_20260504"
RAW_ROOT = PROJECT / "result" / "paper_raw_prediction_assets_20260514" / "run_20260514_133243" / "v3wa_kl003_ablation" / "raw"
OUT_ROOT = PROJECT / "result" / "v3wa_kl003_ablation_official_palette_single_prediction_maps_corrected_rgb_20260515"

MODELS = ["base", "full", "wo_ground_proto", "wo_metadata", "wo_frequency"]
MODEL_LABELS = {
    "base": "Base",
    "full": "Full",
    "wo_ground_proto": "w/o Ground",
    "wo_metadata": "w/o Metadata",
    "wo_frequency": "w/o Frequency",
}


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_if_exists(src: Path, dst: Path) -> list[str]:
    if not src.exists():
        return []
    ensure(dst.parent)
    shutil.copy2(src, dst)
    return [str(dst)]


def save_fig(fig: plt.Figure, path_base: Path) -> list[str]:
    ensure(path_base.parent)
    png = path_base.with_suffix(".png")
    pdf = path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [str(png), str(pdf)]


def cube_to_hwc(cube: np.ndarray) -> np.ndarray:
    """Return a spectral cube as H,W,C when the source is C,H,W."""
    arr = np.squeeze(np.asarray(cube))
    if arr.ndim != 3:
        return arr
    if arr.shape[0] <= 512 and arr.shape[1] > 8 and arr.shape[2] > 8:
        return np.transpose(arr, (1, 2, 0))
    return arr


def plot_endmember_spectra(src: Path, model_name: str, dst: Path) -> list[str]:
    spec: np.ndarray | None = None
    npz = src / "raw_arrays.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        if "endmember_spectra" in z.files:
            spec = np.asarray(z["endmember_spectra"])
    csv_path = src / "endmember_spectra.csv"
    if spec is None and csv_path.exists():
        spec = np.asarray(np.loadtxt(csv_path, delimiter=","))
    if spec is None or spec.ndim != 2:
        return []
    spec_plot = spec.T if spec.shape[0] < spec.shape[1] else spec
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for i in range(min(spec_plot.shape[1], 8)):
        ax.plot(spec_plot[:, i], linewidth=1.2, label=f"EM {i + 1}")
    ax.set_title(f"Urban4 {MODEL_LABELS[model_name]} Endmembers")
    ax.set_xlabel("Band")
    ax.set_ylabel("Reflectance / abundance scale")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return save_fig(fig, dst / "ours_endmember_spectra")


def export_classification(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in MODELS:
        result_dir = ROOT / "results" / "classification_hanchuan" / model_name
        target = ensure(run_dir / "classification" / "HanChuan" / model_name)
        if not result_dir.exists():
            rows.append({"task": "classification", "dataset": "HanChuan", "model": model_name, "status": "missing_result_dir"})
            continue
        config = read_json(result_dir / "config.json")
        args, bundle = style.build_scene_from_config(config)
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
        pretrain_ckpt = Path(config["checkpoint"])
        proto_eo1 = Path(config.get("prototype_bank_eo1") or style.PROTO_EO1)
        proto_gf5 = Path(config.get("prototype_bank_gf5") or style.PROTO_GF5)
        extractor = cls.AwareFeatureExtractor(
            checkpoint_path=pretrain_ckpt,
            prototype_bank_eo1=proto_eo1,
            prototype_bank_gf5=proto_gf5,
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        net = cls.AwareSSFusionSeg(
            extractor,
            int(scene["class_count"]),
            feature_layers=int(args.feature_layers),
            fusion_channels=int(args.fusion_channels),
        ).to(device)
        ckpt_path = result_dir / "best_train_loss.pt"
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(payload["model"])
        net.eval()

        pred_all = style.predict_scene_map(
            net,
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
        # HanChuan uses HyperSIGMA's official HC palette.
        files += style.save_label(target / "groundtruth_full_map.png", labels, class_count, dataset="HC")
        files += style.save_label(target / "test_groundtruth_mask.png", test_gt, class_count, dataset="HC")
        files += style.save_label(target / "ours_prediction_full_scene.png", pred_all, class_count, dataset="HC")
        files += style.save_label(target / "ours_prediction_on_groundtruth_mask.png", pred_on_gt, class_count, dataset="HC")
        files += style.save_label(target / "ours_prediction_on_test_mask.png", pred_on_test, class_count, dataset="HC")
        wavelengths = scene_config.wavelengths if int(scene_config.wavelengths.numel()) == int(scene["channels"]) else None
        files += style.save_rgb(target / "original_rgb.png", style.cube_rgb(scene["cube"], wavelengths))
        bands = style.save_rgb_band_info(target / "rgb_band_info.csv", scene["cube"], wavelengths)
        with (target / "class_palette.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "r", "g", "b"])
            pal = style.palette(class_count, "HC")
            writer.writerow([0, "background", *pal[0].tolist()])
            for i, name in enumerate(scene_config.class_names, start=1):
                writer.writerow([i, name, *pal[i].tolist()])
        np.savez_compressed(
            target / "classification_maps_raw.npz",
            labels=labels,
            test_gt=test_gt,
            prediction_full_scene=pred_all,
            prediction_on_groundtruth_mask=pred_on_gt,
            prediction_on_test_mask=pred_on_test,
        )
        files.append(str(target / "classification_maps_raw.npz"))
        rows.append(
            {
                "task": "classification",
                "dataset": "HanChuan",
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "status": "ok",
                "checkpoint": str(ckpt_path),
                "pretrain_checkpoint": str(pretrain_ckpt),
                "shape": [int(scene["height"]), int(scene["width"])],
                "rgb_zero_based_bands": bands,
                "file_count": len(files) + 2,
            }
        )
        del net, extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def export_detection(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        src = RAW_ROOT / "raw" / "detection_htd_aviris" / model_name
        # Backward-compatible path: RAW_ROOT already points at .../v3wa_kl003_ablation/raw.
        if not src.exists():
            src = RAW_ROOT / "detection_htd_aviris" / model_name
        dst = ensure(run_dir / "detection" / "HTD_AVIRIS" / model_name)
        files: list[str] = []
        for old, new in [
            ("label.png", "groundtruth_binary.png"),
            ("score_map.png", "ours_score_map.png"),
            ("prediction_mask_p995.png", "ours_prediction_binary_p995.png"),
            ("error.png", "ours_error_map.png"),
        ]:
            files += copy_if_exists(src / old, dst / new)
        npz = src / "raw_arrays.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            if "original_cube" in z:
                files += style.save_rgb(dst / "original_rgb.png", style.cube_rgb(z["original_cube"]))
                style.save_rgb_band_info(dst / "rgb_band_info.csv", z["original_cube"])
            files += copy_if_exists(npz, dst / "raw_arrays.npz")
        rows.append({"task": "detection", "dataset": "HTD-AVIRIS", "model": model_name, "status": "ok" if files else "missing", "file_count": len(files)})
    return rows


def export_denoising(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        src_model = RAW_ROOT / "raw" / "denoising_wdc_case5" / model_name
        if not src_model.exists():
            src_model = RAW_ROOT / "denoising_wdc_case5" / model_name
        for case_dir in sorted(src_model.glob("case_*")):
            dst = ensure(run_dir / "denoising" / "WDC_Case5" / model_name / case_dir.name)
            files: list[str] = []
            for old, new in [
                ("error.png", "ours_error_heatmap.png"),
            ]:
                files += copy_if_exists(case_dir / old, dst / new)
            npz = case_dir / "raw_arrays.npz"
            if npz.exists():
                z = np.load(npz, allow_pickle=True)
                if "noisy" in z:
                    files += style.save_rgb(dst / "noisy_input_rgb.png", style.cube_rgb(z["noisy"]))
                    style.save_rgb_band_info(dst / "rgb_band_info.csv", z["noisy"])
                if "prediction" in z:
                    files += style.save_rgb(dst / "ours_denoised_rgb.png", style.cube_rgb(z["prediction"]))
                if "ground_truth" in z:
                    files += style.save_rgb(dst / "groundtruth_clean_rgb.png", style.cube_rgb(z["ground_truth"]))
                if {"noisy", "prediction", "ground_truth"}.issubset(set(z.files)):
                    noisy = style.cube_to_hwc(z["noisy"])
                    pred = style.cube_to_hwc(z["prediction"])
                    gt = style.cube_to_hwc(z["ground_truth"])
                    h, w = pred.shape[:2]
                    r, c = h // 2, w // 2
                    fig, ax = plt.subplots(figsize=(4.2, 3.0))
                    ax.plot(gt[r, c], label="Groundtruth", linewidth=1.5)
                    ax.plot(noisy[r, c], label="Noisy", linewidth=1.0)
                    ax.plot(pred[r, c], label=MODEL_LABELS[model_name], linewidth=1.2)
                    ax.set_xlabel("Band")
                    ax.set_ylabel("Value")
                    ax.legend(frameon=False, fontsize=8)
                    fig.tight_layout()
                    files += save_fig(fig, dst / "center_pixel_spectral_curve")
                files += copy_if_exists(npz, dst / "raw_arrays.npz")
            rows.append(
                {
                    "task": "denoising",
                    "dataset": "WDC Case5",
                    "model": model_name,
                    "case": case_dir.name,
                    "status": "ok" if files else "missing",
                    "file_count": len(files),
                }
            )
    return rows


def export_unmixing(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        src = RAW_ROOT / "raw" / "unmix_urban4" / model_name
        if not src.exists():
            src = RAW_ROOT / "unmix_urban4" / model_name
        dst = ensure(run_dir / "unmixing" / "Urban4" / model_name)
        files: list[str] = []
        for abundance in sorted(src.glob("abundance_em*.png")):
            files += copy_if_exists(abundance, dst / f"ours_{abundance.name}")
        files += copy_if_exists(src / "endmember_spectra.csv", dst / "ours_endmember_spectra.csv")
        files += copy_if_exists(src / "raw_arrays.npz", dst / "raw_arrays.npz")
        files += plot_endmember_spectra(src, model_name, dst)
        rows.append({"task": "unmixing", "dataset": "Urban4", "model": model_name, "status": "ok" if files else "missing", "file_count": len(files)})
    return rows


def write_index(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (run_dir / "asset_index.csv").open("w", encoding="utf-8", newline="") as f:
        fields = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = [
        "# v3wa_kl003 Ablation Official-Palette Single Prediction Assets",
        "",
        "Standalone, non-stitched visual assets for the five ablation branches.",
        "",
        "Models: Base, Full, w/o Ground, w/o Metadata, w/o Frequency.",
        "",
        "Classification maps use the official HyperSIGMA HC palette from `ImageClassification/model/utils.py::imshow_HC`.",
        "HanChuan `original_rgb.png` is pseudo-color, not strict true-color; see each `rgb_band_info.csv`.",
        "RGB quicklooks use R=NIR (~800 nm), G=red (~660 nm), B=green (~550 nm), with percentile normalization.",
        "When exact wavelengths are unavailable, band selection assumes a linear 400-1000 nm range.",
        "",
        "Key classification files:",
        "- `groundtruth_full_map.png`: full labeled GT map.",
        "- `test_groundtruth_mask.png`: test split GT map only.",
        "- `original_rgb.png`: pseudo-color HSI quicklook.",
        "- `ours_prediction_full_scene.png`: whole-scene prediction.",
        "- `ours_prediction_on_groundtruth_mask.png`: prediction shown only on all GT-labeled pixels.",
        "- `ours_prediction_on_test_mask.png`: prediction shown only on test pixels.",
        "",
    ]
    for row in rows:
        lines.append(f"- {row.get('task')} / {row.get('dataset')} / {row.get('model')}: {row.get('status')} ({row.get('file_count', 0)} files)")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()
    run_name = args.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    rows += export_classification(run_dir)
    rows += export_detection(run_dir)
    rows += export_denoising(run_dir)
    rows += export_unmixing(run_dir)
    write_index(run_dir, rows)
    (args.out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "items": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
