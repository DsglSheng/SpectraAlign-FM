#!/usr/bin/env python3
"""Generate prediction/result visual cases for the v3wa_kl003 final ablation.

This is the paper-figure companion for:
  /mnt/Data/SpectralData/Aware_open/result/v3wa_kl003_ablation_final_downstream_20260504

It creates timestamped, non-overwriting output packs with per-model prediction
visuals for the five ablation branches:
Base, Full, w/o Ground, w/o Metadata, w/o Frequency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
ROOT = PROJECT / "result" / "v3wa_kl003_ablation_final_downstream_20260504"
RESULTS = ROOT / "results"
DEFAULT_OUT_ROOT = ROOT / "figures_prediction_cases_20260513"

MODELS = ["base", "full", "wo_ground_proto", "wo_metadata", "wo_frequency"]
MODEL_LABELS = {
    "base": "Base",
    "full": "Full",
    "wo_ground_proto": "w/o Ground",
    "wo_metadata": "w/o Metadata",
    "wo_frequency": "w/o Frequency",
}
MODEL_COLORS = {
    "base": "#6c757d",
    "full": "#d62728",
    "wo_ground_proto": "#1f77b4",
    "wo_metadata": "#2ca02c",
    "wo_frequency": "#9467bd",
}

PRETRAIN_WEIGHTS = {
    "base": PROJECT / "runs/pretrain_ablation_paper_v1/branches/base/checkpoints/best_total.pt",
    "full": PROJECT / "runs/pretrain_v3wa_weight_control_20260503/branches/full_v3wa_kl003/checkpoints/best_total.pt",
    "wo_ground_proto": PROJECT / "runs/pretrain_v3wa_kl003_ablation_v1_20260504/branches/wo_ground_proto/checkpoints/best_total.pt",
    "wo_metadata": PROJECT / "runs/pretrain_v3wa_kl003_ablation_v1_20260504/branches/wo_metadata/checkpoints/best_total.pt",
    "wo_frequency": PROJECT / "runs/pretrain_v3wa_kl003_ablation_v1_20260504/branches/wo_frequency/checkpoints/best_total.pt",
}

TASK_DIRS = {
    "classification_hanchuan": RESULTS / "classification_hanchuan",
    "detection_htd_aviris": RESULTS / "detection_htd_aviris",
    "denoising_wdc_case5_l2_e40": RESULTS / "denoising_wdc_case5_l2_e40",
    "unmix_urban4_hypersigma_spreg6": RESULTS / "unmix_urban4_hypersigma_spreg6",
}

SELECTED_DOWNSTREAM_WEIGHTS = {
    "classification_hanchuan": "best_val_oa.pt",
    "detection_htd_aviris": "best_auc.pt",
    "denoising_wdc_case5_l2_e40": "best.pt",
    "unmix_urban4_hypersigma_spreg6": "final.pt",
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return json.load(handle)


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def save_fig(fig: plt.Figure, out_base: Path) -> List[str]:
    ensure(out_base.parent)
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
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0, 1)


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
    for c in range(3):
        out[..., c] = norm01(rgb[..., c])
    return out


def class_node(model: str) -> Dict[str, Any]:
    data = read_json(TASK_DIRS["classification_hanchuan"] / model / "test_metrics.json")
    node = data.get("best_val_oa") or data.get("best_val_loss") or data.get("best_train_loss") or data.get("latest") or data
    return node.get("test", node)


def det_node(model: str) -> Dict[str, Any]:
    data = read_json(TASK_DIRS["detection_htd_aviris"] / model / "test_metrics.json")
    return data.get("selection_eval") or data.get("best_auc_metrics") or data.get("metrics") or data


def den_node(model: str) -> Dict[str, Any]:
    root = TASK_DIRS["denoising_wdc_case5_l2_e40"] / model
    path = root / "test_metrics.json"
    if not path.exists():
        path = root / "best_metrics.json"
    return read_json(path)


def unmix_node(model: str) -> Dict[str, Any]:
    return read_json(TASK_DIRS["unmix_urban4_hypersigma_spreg6"] / model / "test_metrics.json")


def collect_metric_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        c = class_node(model)
        d = det_node(model)
        n = den_node(model)
        u = unmix_node(model)
        rows.append({
            "model": model,
            "label": MODEL_LABELS[model],
            "hanchuan_oa": c.get("oa"),
            "hanchuan_aa": c.get("aa"),
            "hanchuan_kappa": c.get("kappa"),
            "htd_auc": d.get("auc"),
            "htd_auc_composite": d.get("auc_composite"),
            "htd_auc_ratio": d.get("auc_ratio"),
            "wdc_psnr": n.get("psnr"),
            "wdc_ssim": n.get("ssim"),
            "wdc_sam": n.get("sam"),
            "urban_sad": u.get("avg_sad"),
            "urban_mse": u.get("avg_mse"),
        })
    return rows


def write_metric_table(run_dir: Path) -> None:
    rows = collect_metric_rows()
    path = run_dir / "ablation_metric_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_weight_manifest(run_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model in MODELS:
        for task, task_root in TASK_DIRS.items():
            rows.append({
                "model": model,
                "label": MODEL_LABELS[model],
                "task": task,
                "pretraining_checkpoint": str(PRETRAIN_WEIGHTS[model]),
                "downstream_result_dir": str(task_root / model),
                "selected_downstream_weight": str(task_root / model / SELECTED_DOWNSTREAM_WEIGHTS[task]),
            })
    with (run_dir / "weight_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    with (run_dir / "weight_manifest.md").open("w", encoding="utf-8") as handle:
        handle.write("# v3wa_kl003 Ablation Weight Manifest\n\n")
        handle.write("| Model | Task | Pretraining checkpoint | Downstream selected weight |\n")
        handle.write("|---|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {row['task']} | `{row['pretraining_checkpoint']}` | `{row['selected_downstream_weight']}` |\n"
            )
    return rows


def make_summary_figures(run_dir: Path) -> List[str]:
    rows = collect_metric_rows()
    files: List[str] = []
    metrics = [
        ("HanChuan OA", "hanchuan_oa", True),
        ("HanChuan AA", "hanchuan_aa", True),
        ("HanChuan Kappa", "hanchuan_kappa", True),
        ("HTD AUC", "htd_auc", True),
        ("HTD AUC composite", "htd_auc_composite", True),
        ("HTD AUC ratio", "htd_auc_ratio", True),
        ("WDC PSNR", "wdc_psnr", True),
        ("WDC SSIM", "wdc_ssim", True),
        ("WDC SAM", "wdc_sam", False),
        ("Urban SAD", "urban_sad", False),
        ("Urban MSE", "urban_mse", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    reps = [
        ("HanChuan OA", "hanchuan_oa", True),
        ("HTD AUC composite", "htd_auc_composite", True),
        ("WDC PSNR", "wdc_psnr", True),
        ("Urban MSE", "urban_mse", False),
    ]
    for ax, (title, key, higher) in zip(axes.ravel(), reps):
        values = np.array([float(row[key]) for row in rows])
        x = np.arange(len(MODELS))
        bars = ax.bar(x, values, color=[MODEL_COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        pad = max((values.max() - values.min()) * 0.2, abs(values.mean()) * 0.002)
        ax.set_ylim(values.min() - pad, values.max() + pad)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.suptitle("v3wa_kl003 Final Ablation: Representative Metrics", fontsize=14)
    fig.tight_layout()
    files.extend(save_fig(fig, run_dir / "summary" / "ablation_representative_metrics"))

    mat = []
    raw = []
    labels = []
    for title, key, higher in metrics:
        values = np.array([float(row[key]) for row in rows])
        score = values if higher else -values
        norm = (score - score.min()) / (score.max() - score.min() + 1.0e-12)
        mat.append(norm)
        raw.append(values)
        labels.append(f"{title} {'up' if higher else 'down'}")
    mat = np.asarray(mat)
    raw = np.asarray(raw)
    fig, ax = plt.subplots(figsize=(8.9, 7.1))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{raw[i, j]:.4f}", ha="center", va="center", fontsize=7)
    ax.set_title("All Ablation Metrics (row-normalized; darker is better)")
    fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    fig.tight_layout()
    files.extend(save_fig(fig, run_dir / "summary" / "ablation_metric_heatmap"))
    return files


def make_classification_figures(run_dir: Path) -> List[str]:
    out = run_dir / "classification_hanchuan"
    files: List[str] = []
    data = {m: class_node(m) for m in MODELS}

    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.7))
    for ax, model in zip(axes, MODELS):
        cm = np.asarray(data[model]["confusion_matrix"], dtype=np.float64)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{MODEL_LABELS[model]}\nOA={data[model]['oa']:.4f}", fontsize=9)
        ax.set_xlabel("Pred.")
        if ax is axes[0]:
            ax.set_ylabel("GT")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle("HanChuan Classification: Normalized Confusion Matrices")
    files.extend(save_fig(fig, out / "all_models_confusion_matrices"))

    for model in MODELS:
        cm = np.asarray(data[model]["confusion_matrix"], dtype=np.float64)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.2, 5.3))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"HanChuan {MODEL_LABELS[model]} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        files.extend(save_fig(fig, out / safe_name(model) / "confusion_matrix"))

        per = data[model]["per_class_accuracy"]
        labels = [str(item.get("class_id", idx + 1)) for idx, item in enumerate(per)]
        values = [float(item["accuracy"]) for item in per]
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        x = np.arange(len(values))
        ax.bar(x, values, color=MODEL_COLORS[model])
        ax.set_ylim(0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_title(f"HanChuan {MODEL_LABELS[model]} per-class accuracy")
        ax.set_ylabel("Accuracy")
        files.extend(save_fig(fig, out / safe_name(model) / "per_class_accuracy"))
    return files


def make_detection_figures(run_dir: Path) -> List[str]:
    out = run_dir / "detection_htd_aviris"
    files: List[str] = []
    maps = {}
    for model in MODELS:
        maps[model] = np.load(TASK_DIRS["detection_htd_aviris"] / model / "final_best_auc_score_map.npy")

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.4))
    for ax, model in zip(axes, MODELS):
        im = ax.imshow(maps[model], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{MODEL_LABELS[model]}\nAUC={det_node(model)['auc']:.4f}", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.suptitle("HTD-AVIRIS Detection: Score Maps")
    files.extend(save_fig(fig, out / "all_models_score_maps"))

    full = maps["full"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4))
    for ax, model in zip(axes, [m for m in MODELS if m != "full"]):
        diff = full - maps[model]
        vmax = np.percentile(np.abs(diff), 99)
        im = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Full - {MODEL_LABELS[model]}", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    fig.suptitle("HTD-AVIRIS Detection: Full Score Difference")
    files.extend(save_fig(fig, out / "full_minus_ablation_score_maps"))

    for model in MODELS:
        score = maps[model]
        threshold = float(np.percentile(score[np.isfinite(score)], 99.5))
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
        axes[0].imshow(score, cmap="magma", vmin=0, vmax=1)
        axes[0].set_title("Score map")
        axes[1].imshow(score >= threshold, cmap="gray")
        axes[1].set_title("High-score mask")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"HTD-AVIRIS {MODEL_LABELS[model]}")
        files.extend(save_fig(fig, out / safe_name(model) / "score_case"))
    return files


def make_unmixing_figures(run_dir: Path) -> List[str]:
    out = run_dir / "unmix_urban4"
    files: List[str] = []
    mats = {m: sio.loadmat(TASK_DIRS["unmix_urban4_hypersigma_spreg6"] / m / "unmixing_result.mat") for m in MODELS}

    fig, axes = plt.subplots(4, 5, figsize=(14.0, 10.5))
    for em in range(4):
        stack = np.concatenate([mats[m]["abundances"][em].ravel() for m in MODELS])
        lo, hi = np.percentile(stack, [1, 99])
        for col, model in enumerate(MODELS):
            ax = axes[em, col]
            img = np.clip((mats[model]["abundances"][em] - lo) / (hi - lo + 1.0e-12), 0, 1)
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            if em == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"EM {em + 1}", fontsize=9)
    fig.suptitle("Urban4 Unmixing: Abundance Maps")
    fig.tight_layout()
    files.extend(save_fig(fig, out / "all_models_abundance_maps"))

    bands = np.arange(1, mats["full"]["endmembers"].shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    for em, ax in enumerate(axes.ravel()):
        for model in MODELS:
            lw = 2.4 if model == "full" else 1.25
            ax.plot(bands, mats[model]["endmembers"][em], color=MODEL_COLORS[model], linewidth=lw, label=MODEL_LABELS[model])
        ax.set_title(f"Endmember {em + 1}")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Urban4 Unmixing: Learned Endmember Spectra")
    fig.tight_layout()
    files.extend(save_fig(fig, out / "all_models_endmember_spectra"))

    for model in MODELS:
        fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
        for em, ax in enumerate(axes):
            ax.imshow(norm01(mats[model]["abundances"][em]), cmap="viridis")
            ax.set_title(f"EM {em + 1}")
            ax.axis("off")
        fig.suptitle(f"Urban4 {MODEL_LABELS[model]} abundance maps")
        files.extend(save_fig(fig, out / safe_name(model) / "abundance_maps"))
    return files


def draw_denoising_predictions(run_dir: Path) -> Tuple[List[str], str]:
    out = run_dir / "denoising_wdc_case5"
    files: List[str] = []
    try:
        import torch

        sys.path.insert(0, str(PROJECT / "scripts"))
        import finetune_denoising as den
    except Exception as exc:
        return files, f"denoising import failed: {exc}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_preds: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in MODELS}
    noisy_cases: Dict[int, np.ndarray] = {}
    clean_cases: Dict[int, np.ndarray] = {}
    names: Dict[int, str] = {}
    raw_payload: Dict[str, np.ndarray] = {}

    for model_name in MODELS:
        result = TASK_DIRS["denoising_wdc_case5_l2_e40"] / model_name
        cfg = read_json(result / "config.json")
        best = result / "best.pt"
        if not cfg or not best.exists():
            continue
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
        net = den.AwareDenoisingNet(
            extractor,
            channels=191,
            feature_channels=int(cfg.get("feature_channels", 64)),
            hidden_channels=int(cfg.get("hidden_channels", 256)),
            use_intermediate=not bool(cfg.get("no_intermediate", False)),
        ).to(device)
        payload = torch.load(best, map_location=device, weights_only=False)
        net.load_state_dict(payload["model"])
        net.eval()
        dataset = den.WdcPatchMatDataset(Path(cfg["data_root"]) / "Testing" / "Patch_Cases" / "Case5")
        sample_count = min(3, len(dataset))
        use_amp = str(cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"
        with torch.no_grad():
            for idx in range(sample_count):
                noisy, clean, file_name = dataset[idx]
                noisy_b = noisy.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred = den.predict_with_tta(net, noisy_b, enabled=bool(cfg.get("eval_tta", False)))
                pred_np = pred.squeeze(0).detach().cpu().float().numpy()
                noisy_np = noisy.detach().cpu().float().numpy()
                clean_np = clean.detach().cpu().float().numpy()
                model_preds[model_name][idx] = pred_np
                noisy_cases[idx] = noisy_np
                clean_cases[idx] = clean_np
                names[idx] = str(file_name)
                raw_payload[f"{model_name}_case{idx}_pred"] = pred_np
                raw_payload[f"case{idx}_noisy"] = noisy_np
                raw_payload[f"case{idx}_gt"] = clean_np
        del net, extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for idx in sorted(noisy_cases):
        fig, axes = plt.subplots(len(MODELS), 4, figsize=(11.8, 2.45 * len(MODELS)))
        for row, model_name in enumerate(MODELS):
            pred = model_preds.get(model_name, {}).get(idx)
            if pred is None:
                for ax in axes[row]:
                    ax.axis("off")
                continue
            err = np.mean(np.abs(cube_to_hwc(pred) - cube_to_hwc(clean_cases[idx])), axis=-1)
            panels = [
                ("Noisy input", cube_rgb(noisy_cases[idx]), None),
                ("Prediction", cube_rgb(pred), None),
                ("Ground truth", cube_rgb(clean_cases[idx]), None),
                ("Abs. error", norm01(err), "magma"),
            ]
            for col, (title, arr, cmap) in enumerate(panels):
                ax = axes[row, col]
                if cmap:
                    ax.imshow(arr, cmap=cmap)
                else:
                    ax.imshow(arr)
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(MODEL_LABELS[model_name], fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"WDC Case5 denoising prediction case {idx}: {names.get(idx, '')}", y=1.01)
        files.extend(save_fig(fig, out / f"case{idx:02d}_all_models_prediction"))

    if raw_payload:
        ensure(out)
        np.savez_compressed(out / "wdc_case5_ablation_prediction_cases.npz", **raw_payload)
        files.append(str(out / "wdc_case5_ablation_prediction_cases.npz"))
    return files, "WDC Case5 denoising predictions exported from each model best.pt"


def write_readme(run_dir: Path, files: Sequence[str], den_note: str) -> None:
    lines = [
        "# v3wa_kl003 Final Ablation Prediction Visualization Cases",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Source result root: `{ROOT}`",
        "",
        "This pack uses the final paper ablation table branches: Base, Full, w/o Ground, w/o Metadata, w/o Frequency.",
        "",
        "Figure types:",
        "- HanChuan classification: confusion matrices and per-class accuracy, because the classifier did not save whole-scene prediction maps.",
        "- HTD-AVIRIS detection: saved score maps and high-score masks.",
        "- WDC Case5 denoising: fresh prediction export from each branch `best.pt` on shared Case5 samples.",
        "- Urban4 unmixing: abundance maps and learned endmember spectra from `unmixing_result.mat`.",
        "",
        f"Denoising export note: {den_note}",
        "",
        "Important files:",
        "- `weight_manifest.json` / `weight_manifest.md`: pretraining and downstream weight paths.",
        "- `ablation_metric_table.csv`: numeric values used by the visualization.",
        "- `manifest.json`: generated figure file list.",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"generated_files": list(files), "source_root": str(ROOT), "denoising_note": den_note}, handle, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_root / run_id
    ensure(run_dir)

    all_files: List[str] = []
    make_weight_manifest(run_dir)
    write_metric_table(run_dir)
    all_files.extend(make_summary_figures(run_dir))
    all_files.extend(make_classification_figures(run_dir))
    all_files.extend(make_detection_figures(run_dir))
    all_files.extend(make_unmixing_figures(run_dir))
    den_files, den_note = draw_denoising_predictions(run_dir)
    all_files.extend(den_files)
    write_readme(run_dir, all_files, den_note)

    ensure(args.out_root)
    (args.out_root / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
