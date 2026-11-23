#!/usr/bin/env python3
"""Export dense, pixel-aligned AeroRIT and WHU-H2SR panels on common tiles.

The appendix previously cropped sparse center-pixel evaluation maps.  This exporter
loads the validation-selected dense segmentation checkpoints and runs their decoder
heads over the exact tiles used by the local small-model controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


AWARE = Path("/mnt/Data/SpectralData/Aware_open")
SGPT = Path("/mnt/Data/SpectralData/SpectralGPT_Aware_Comparison_20260516")
EARTH = Path("/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608")
SMALL = Path("/mnt/Data/SpectralData/small_model_benchmark_20260616")

RESULTS = {
    ("ours", "aerorit"): AWARE
    / "result/specaware_comparison_v1_20260510/aerorit_ours_upeernet_lr5e5_e100_crop128_acc16_20260511",
    ("spectralgpt", "aerorit"): SGPT
    / "results/specaware_segmentation_20260516/results/aerorit/spectralgpt_plus_seg",
    ("spectralearth", "aerorit"): EARTH
    / "results/repro_seed20260615/semantic_segmentation/aerorit/specvit_small_mae",
    ("ours", "whu"): AWARE
    / "result/ours_whu_oscd_improve_20260520/results/whu_target82_seed20260672_exp70init_crop288_dec384_lovasz020_siou032_focal010_boost3x15_8x22_lr9e6_e180/ours_v3wa_kl003_full450k",
    ("spectralgpt", "whu"): SGPT
    / "results/specaware_segmentation_20260516/results/whu/spectralgpt_target74_seed20260673_crop320_f128h384_lovasz025_siou040_focal012_boost3x18_8x28_lr1e5_e180",
    ("spectralearth", "whu"): EARTH
    / "results/hypersigma_specaware_protocol_20260614/semantic_segmentation/whu_h2sr/specvit_small_mae",
    ("ours", "qingpu"): AWARE
    / "result/specaware_comparison_v1_20260510/qingpu_ours_upeernet_lr5e5_e100_crop128_acc16_20260511",
    ("spectralgpt", "qingpu"): SGPT
    / "results/specaware_segmentation_20260516/results/qingpu/spectralgpt_plus_seg",
    ("spectralearth", "qingpu"): EARTH
    / "results/hypersigma_specaware_protocol_20260614/semantic_segmentation/qingpu_hsi/specvit_small_mae",
}

AERORIT_COORDS = np.asarray(
    [
        [512, 3840],
        [0, 1920],
        [896, 2304],
        [1280, 3072],
        [384, 256],
        [896, 3328],
    ],
    dtype=np.int64,
)
WHU_STEMS = ["36_1_0", "183_2_1", "33_1_13", "177_5_4", "183_8_1"]
QINGPU_COORDS = np.asarray(
    [[768, 4864], [1792, 19968], [0, 9728], [1792, 4608], [2560, 3584]],
    dtype=np.int64,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def load_ours(result: Path, device: torch.device):
    sys.path.insert(0, str(AWARE / "scripts"))
    sys.path.insert(0, str(AWARE))
    import finetune_specaware_segmentation as seg

    config = read_json(result / "config.json")
    raw = config["args"]
    bands = int(config["image_info"]["bands"])
    dataset = str(config["dataset"])
    if dataset == "aerorit":
        waves = config["image_info"].get("wavelengths", [])
        wavelengths = (
            torch.tensor(waves, dtype=torch.float32)
            if len(waves) == bands
            else torch.linspace(400.0, 897.0, bands)
        )
        sensor, level, gsd, classes = "others", "L1", 0.4, len(seg.AERORIT_CLASS_NAMES)
    elif dataset == "whu":
        wavelengths = torch.linspace(400.0, 1000.0, bands)
        sensor, level, gsd, classes = "others", "surface", 1.0, len(seg.WHU_H2SR_CLASS_NAMES)
    elif dataset == "qingpu":
        wavelengths = torch.linspace(400.0, 1000.0, bands)
        sensor, level, gsd, classes = "eo1", "L1", 0.75, len(seg.QINGPU_CLASS_NAMES)
    else:
        raise ValueError(dataset)

    metadata = seg.build_downstream_metadata(
        num_bands=bands,
        sensor=sensor,
        level=level,
        gsd=gsd,
        wavelengths_nm=wavelengths,
        fwhm_nm=None,
    )
    extractor = seg.AwareFeatureExtractor(
        checkpoint_path=Path(raw["checkpoint"]),
        prototype_bank_eo1=Path(raw["prototype_bank_eo1"]),
        prototype_bank_gf5=Path(raw["prototype_bank_gf5"]),
        metadata=metadata,
        freeze_encoder=bool(raw.get("freeze_encoder", False)),
        device=device,
    )
    model = seg.AwareUPerNetSeg(
        extractor,
        classes,
        input_channels=bands,
        decoder_channels=int(raw.get("decoder_channels", 256)),
        segmentation_channels=int(raw.get("segmentation_channels", 64)),
        raw_shortcut=bool(raw.get("raw_shortcut", False)),
        raw_shortcut_channels=int(raw.get("raw_shortcut_channels", 128)),
        head_channels=int(raw.get("head_channels", 320)),
        head_dropout=float(raw.get("head_dropout", 0.0)),
    ).to(device)
    payload = checkpoint_payload(result / "best_val_miou.pt", device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return seg, model, config


def load_spectral_model(family: str, result: Path, device: torch.device):
    for path in (SGPT / "scripts", AWARE / "scripts", AWARE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if family == "spectralearth":
        sys.path.insert(0, str(EARTH / "scripts"))
        import export_spectralearth_pixel_predictions_20260615 as wrapper

        wrapper.configure_legacy_adapter()
        exporter = wrapper.legacy
    else:
        import export_spectralgpt_pixel_predictions_20260529 as exporter

    args = exporter.namespace_from_seg_config(result)
    # Older accepted runs predate several later CLI options.  Recreate the
    # defaults used by their original training scripts before rebuilding the
    # loader; this changes no learned weights or normalization statistics.
    defaults = {
        "class_weight_power": 0.5,
        "class_boosts": "",
        "dice_loss_weight": 0.0,
        "focal_loss_weight": 0.0,
        "focal_gamma": 1.5,
        "label_smoothing": 0.0,
        "soft_iou_loss_weight": 0.0,
        "lovasz_loss_weight": 0.0,
        "split_mode": "random_per_class",
        "aerorit_band_subset": "full",
        "spatial_block_size": 1024,
        "spatial_train_fraction": 0.6,
        "spatial_val_fraction": 0.2,
        "train_area_size": 1024,
        "train_area_count": 16,
        "center_only_labels": False,
        "stats_from_train_only": False,
        "whu_train_fraction": 0.2,
        "whu_val_fraction": 0.1,
        "eval_every": 1,
        "final_qingpu_sliding_eval": False,
        "full_scene_stride": 128,
        "auto_resume": True,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    data = exporter.sgseg.make_loaders(args, device)
    model = exporter.build_seg_model(args, data, device)
    return exporter.sgseg, model, data, vars(args)


def aerorit_ours_inputs(seg, config: dict[str, Any]):
    raw = config["args"]
    data_dir = Path(raw["data_dir"])
    refl = data_dir / "full_reflectance/2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine-001_"
    hdr = data_dir / "headers/2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
    if not hdr.exists():
        hdr = data_dir / "headers_extracted/2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
    image, info = seg.open_envi_image(refl, hdr)
    image, _, _ = seg.apply_aerorit_band_subset(image, info, str(raw.get("aerorit_band_subset", "full")))
    stats = np.load(Path(raw["output_dir"]) / f"band_stats_{int(raw.get('stats_samples', 80000))}_seed{int(raw['seed'])}.npz")
    return image, stats["mean"].astype(np.float32), np.maximum(stats["std"].astype(np.float32), 1.0e-6)


def qingpu_ours_inputs(seg, config: dict[str, Any]):
    raw = config["args"]
    data_dir = Path(raw["data_dir"])
    image, _ = seg.open_envi_image(data_dir / "Qingpu_HSI_image.dat", data_dir / "Qingpu_HSI_image.hdr")
    stats = np.load(Path(raw["output_dir"]) / f"band_stats_{int(raw.get('stats_samples', 0))}_seed{int(raw['seed'])}.npz")
    return image, stats["mean"].astype(np.float32), np.maximum(stats["std"].astype(np.float32), 1.0e-6)


def whu_paths(seg, data_dir: Path) -> dict[str, tuple[Path, Path]]:
    base_seg = getattr(seg, "seg", seg)
    pairs = base_seg.build_whu_pairs(data_dir)
    return {image.stem: (image, label) for image, label in pairs}


def read_whu_image(seg, path: Path) -> np.ndarray:
    base_seg = getattr(seg, "seg", seg)
    return base_seg._read_tiff_array(path)


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    top = max((array.shape[0] - size) // 2, 0)
    left = max((array.shape[1] - size) // 2, 0)
    return np.asarray(array[top : top + size, left : left + size])


@torch.no_grad()
def export(family: str, dataset: str, output_root: Path, device: torch.device) -> None:
    result = RESULTS[(family, dataset)]
    output_dir = output_root / dataset / family
    output_dir.mkdir(parents=True, exist_ok=True)

    if family == "ours":
        seg, model, config = load_ours(result, device)
        raw_args = config["args"]
        if dataset == "aerorit":
            image, mean, std = aerorit_ours_inputs(seg, config)
            patches = [image.read_patch(int(top), int(left), 128) for top, left in AERORIT_COORDS]
            names = [f"tile_{i:02d}_y{int(top)}_x{int(left)}" for i, (top, left) in enumerate(AERORIT_COORDS)]
        elif dataset == "whu":
            pair_map = whu_paths(seg, Path(raw_args["data_dir"]))
            stats = np.load(
                result / f"tile_band_stats_{int(raw_args.get('stats_samples', 90000))}_seed{int(raw_args['seed'])}.npz"
            )
            mean = stats["mean"].astype(np.float32)
            std = np.maximum(stats["std"].astype(np.float32), 1.0e-6)
            patches = [center_crop(read_whu_image(seg, pair_map[name][0]), 256).transpose(2, 0, 1) for name in WHU_STEMS]
            names = WHU_STEMS
        else:
            image, mean, std = qingpu_ours_inputs(seg, config)
            patches = []
            names = [f"tile_{i:02d}_y{int(top)}_x{int(left)}" for i, (top, left) in enumerate(QINGPU_COORDS)]
    else:
        seg, model, data, raw_args = load_spectral_model(family, result, device)
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.maximum(np.asarray(data["std"], dtype=np.float32), 1.0e-6)
        if dataset == "aerorit":
            image = data["image"]
            patches = [image.read_patch(int(top), int(left), 128) for top, left in AERORIT_COORDS]
            names = [f"tile_{i:02d}_y{int(top)}_x{int(left)}" for i, (top, left) in enumerate(AERORIT_COORDS)]
        elif dataset == "whu":
            pair_map = whu_paths(seg, Path(raw_args["data_dir"]))
            patches = [center_crop(read_whu_image(seg, pair_map[name][0]), 256).transpose(2, 0, 1) for name in WHU_STEMS]
            names = WHU_STEMS
        else:
            image = data["image"]
            patches = []
            names = [f"tile_{i:02d}_y{int(top)}_x{int(left)}" for i, (top, left) in enumerate(QINGPU_COORDS)]

    use_amp = device.type == "cuda" and str(raw_args.get("precision", "bf16")) == "bf16"
    if dataset == "qingpu":
        predictions = []
        for tile_index, (top, left) in enumerate(QINGPU_COORDS):
            assembled = np.zeros((256, 256), dtype=np.uint8)
            for dy in (0, 128):
                for dx in (0, 128):
                    chw = np.asarray(image.read_patch(int(top + dy), int(left + dx), 128), dtype=np.float32)
                    if chw.shape[0] != mean.size and chw.shape[-1] == mean.size:
                        chw = chw.transpose(2, 0, 1)
                    normalized = (chw - mean[:, None, None]) / std[:, None, None]
                    x = torch.from_numpy(normalized.copy()).float().unsqueeze(0).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        logits = model(x)
                    assembled[dy : dy + 128, dx : dx + 128] = (
                        logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8) + 1
                    )
            predictions.append(assembled)
            np.save(output_dir / f"{tile_index:02d}_{names[tile_index]}_prediction.npy", assembled)
            print(json.dumps({"family": family, "dataset": dataset, "tile": tile_index, "shape": list(assembled.shape)}), flush=True)
        np.savez_compressed(
            output_dir / "dense_common_tile_predictions.npz",
            predictions=np.stack(predictions),
            tile_names=np.asarray(names, dtype=str),
            coordinates_yx=QINGPU_COORDS,
            checkpoint=np.asarray(str(result / "best_val_miou.pt")),
            source_result=np.asarray(str(result)),
            export_mode=np.asarray("dense_decoder_common_tiles_2x2_native_crops"),
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "family": family,
                    "dataset": dataset,
                    "tiles": names,
                    "coordinates_yx": QINGPU_COORDS.tolist(),
                    "checkpoint": str(result / "best_val_miou.pt"),
                    "export_mode": "dense_decoder_common_tiles_2x2_native_crops",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    tensors = []
    for patch in patches:
        chw = np.asarray(patch, dtype=np.float32)
        if chw.ndim != 3:
            raise ValueError(chw.shape)
        if chw.shape[0] != mean.size and chw.shape[-1] == mean.size:
            chw = chw.transpose(2, 0, 1)
        if chw.shape[0] != mean.size:
            raise ValueError(f"band mismatch: patch={chw.shape}, mean={mean.shape}")
        normalized = (chw - mean[:, None, None]) / std[:, None, None]
        tensors.append(torch.from_numpy(normalized.copy()).float())

    predictions = []
    for index, tensor in enumerate(tensors):
        x = tensor.unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8) + 1
        predictions.append(pred)
        np.save(output_dir / f"{index:02d}_{names[index]}_prediction.npy", pred)
        print(json.dumps({"family": family, "dataset": dataset, "tile": index, "shape": list(pred.shape)}), flush=True)

    np.savez_compressed(
        output_dir / "dense_common_tile_predictions.npz",
        predictions=np.stack(predictions),
        tile_names=np.asarray(names, dtype=str),
        checkpoint=np.asarray(str(result / "best_val_miou.pt")),
        source_result=np.asarray(str(result)),
        export_mode=np.asarray("dense_decoder_common_tiles"),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "family": family,
                "dataset": dataset,
                "tiles": names,
                "checkpoint": str(result / "best_val_miou.pt"),
                "export_mode": "dense_decoder_common_tiles",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ours", "spectralgpt", "spectralearth"), required=True)
    parser.add_argument("--dataset", choices=("aerorit", "whu", "qingpu"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export(args.family, args.dataset, args.output_root, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
