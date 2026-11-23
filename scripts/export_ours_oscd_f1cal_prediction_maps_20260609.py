#!/usr/bin/env python3
"""Export Ours OSCD f1cal scene prediction maps for HyperSIGMA-style assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


AWARE_ROOT = Path("/mnt/Data/SpectralData/Aware_open")
sys.path.insert(0, str(AWARE_ROOT / "scripts"))

import finetune_oscd_change_detection as oscd  # noqa: E402


DEFAULT_RESULT = (
    AWARE_ROOT
    / "result/ours_whu_oscd_improve_20260520/results"
    / "oscd_f1cal_posw68_focal015_dice008_ls001_scan_tta_lr3e6_28k_bs4acc8_f192_h256_seed3729"
    / "ours_v3wa_kl003_full450k"
)
DEFAULT_OUTPUT = (
    AWARE_ROOT
    / "result/hypersigma_style_single_prediction_maps_official_palette_corrected_rgb_20260515"
    / "run_20260609_oscd_f1cal/changedetection"
)
SCENE_TO_DIR = {
    "chongqing": "OSCD_chongqing",
    "lasvegas": "OSCD_lasvegas",
    "montpellier": "OSCD_montpellier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Ours OSCD f1cal prediction maps.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenes", default="chongqing,lasvegas,montpellier")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-component-size", type=int, default=None)
    parser.add_argument("--eval-tta", action="store_true", default=None)
    return parser.parse_args()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def save_binary(mask: np.ndarray, path: Path) -> None:
    mask = np.asarray(mask) > 0
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[...] = [8, 12, 20]
    rgb[mask] = [242, 193, 78]
    ensure(path.parent)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_probability(prob: np.ndarray, path: Path) -> None:
    z = norm01(prob)
    r = np.clip(255.0 * z, 0, 255)
    g = np.clip(255.0 * (1.0 - np.abs(z - 0.55) / 0.55), 0, 255)
    b = np.clip(255.0 * (1.0 - z), 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    ensure(path.parent)
    Image.fromarray(rgb, mode="RGB").save(path)


def confusion_rgb(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = np.asarray(label) > 0
    pr = np.asarray(pred) > 0
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[(~gt) & (~pr)] = [16, 24, 32]   # TN
    rgb[gt & pr] = [44, 160, 44]        # TP
    rgb[(~gt) & pr] = [214, 39, 40]     # FP
    rgb[gt & (~pr)] = [31, 119, 180]    # FN
    return rgb


def save_confusion(label: np.ndarray, pred: np.ndarray, png_path: Path, pdf_path: Path | None = None) -> None:
    rgb = confusion_rgb(label, pred)
    ensure(png_path.parent)
    image = Image.fromarray(rgb, mode="RGB")
    image.save(png_path)
    if pdf_path is not None:
        image.save(pdf_path)


def build_model(cfg: dict[str, Any], result_dir: Path, checkpoint_name: str, device: torch.device) -> oscd.AwareOSCDFusion:
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
    payload = torch.load(result_dir / checkpoint_name, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_scene(
    model: oscd.AwareOSCDFusion,
    item: dict[str, Any],
    device: torch.device,
    crop_size: int,
    precision: str,
    *,
    threshold: float,
    min_component_size: int,
    eval_tta: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = item["pre"].unsqueeze(0).to(device)
    post = item["post"].unsqueeze(0).to(device)
    label = item["label"].numpy().astype(np.uint8)
    pre_patches, coords = oscd.crop_tensor(pre, crop_size)
    post_patches, _ = oscd.crop_tensor(post, crop_size)
    use_amp = precision == "bf16" and device.type == "cuda"
    preds = [
        oscd.forward_eval_patch(model, pp, qq, use_amp=use_amp, eval_tta=eval_tta).float()
        for pp, qq in zip(pre_patches, post_patches)
    ]
    logits = oscd.combine_patches(preds, coords, label.shape, crop_size)
    prob = torch.softmax(logits, dim=1)[0, 1].detach().float().cpu().numpy()
    pred = (prob >= float(threshold)).astype(np.uint8)
    pred = oscd.apply_binary_postprocess(pred, min_component_size=int(min_component_size), close_iterations=0).astype(np.uint8)
    return pred, prob.astype(np.float32), label


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to export OSCD maps on CPU.")
    cfg = read_json(args.result_dir / "config.json")
    metrics = read_json(args.result_dir / "test_metrics.json").get("metrics", {})
    threshold = float(args.threshold if args.threshold is not None else metrics.get("threshold", 0.5))
    min_component_size = int(
        args.min_component_size
        if args.min_component_size is not None
        else metrics.get("postprocess_min_component_size", 0)
    )
    eval_tta = bool(metrics.get("eval_tta", True) if args.eval_tta is None else args.eval_tta)

    data_root = Path(cfg["data_root"])
    scenes = [x.strip() for x in str(args.scenes).split(",") if x.strip()]
    dataset = oscd.OSCDDataset(data_root, scenes, crop_size=int(cfg.get("crop_size", 128)), max_samples=None, split="test")

    device = torch.device("cuda")
    model = build_model(cfg, args.result_dir, args.checkpoint_name, device)
    rows = []
    for idx, scene in enumerate(scenes):
        item = dataset[idx]
        pred, prob, label = predict_scene(
            model,
            item,
            device,
            int(cfg.get("crop_size", 128)),
            str(cfg.get("precision", "bf16")),
            threshold=threshold,
            min_component_size=min_component_size,
            eval_tta=eval_tta,
        )
        scene_dir = ensure(args.output_root / SCENE_TO_DIR.get(scene, f"OSCD_{scene}"))
        save_binary(label, scene_dir / "groundtruth_binary.png")
        save_binary(pred, scene_dir / "ours_prediction_binary.png")
        save_probability(prob, scene_dir / "ours_probability_map.png")
        save_confusion(label, pred, scene_dir / "ours_error_map.png")
        save_confusion(label, pred, scene_dir / "ours_TN_TP_FP_FN_map.png", scene_dir / "ours_TN_TP_FP_FN_map.pdf")
        np.savez_compressed(scene_dir / "raw_arrays.npz", label=label, pred=pred, prob=prob)
        row = {
            "scene": scene,
            "dir": str(scene_dir),
            "height": int(label.shape[0]),
            "width": int(label.shape[1]),
            "threshold": threshold,
            "min_component_size": min_component_size,
            "eval_tta": eval_tta,
        }
        rows.append(row)
        print(json.dumps({"event": "scene_exported", **row}), flush=True)
    manifest = {
        "result_dir": str(args.result_dir),
        "checkpoint": str(args.result_dir / args.checkpoint_name),
        "metrics_json": str(args.result_dir / "test_metrics.json"),
        "rows": rows,
    }
    ensure(args.output_root).joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_root": str(args.output_root), "rows": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
