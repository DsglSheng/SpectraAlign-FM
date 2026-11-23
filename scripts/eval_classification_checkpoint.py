from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aware_open.downstream import (
    AwareFeatureExtractor,
    build_downstream_metadata,
    build_segmentation_scene,
    resolve_scene_config,
    seed_everything,
)
from scripts.finetune_classification import AwareSSFusionSeg, evaluate_merged_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved downstream classification checkpoint on the test split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved fine-tuned checkpoint, e.g. best_val_oa.pt")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload or "args" not in payload:
        raise ValueError(f"{args.checkpoint} does not look like a downstream classification checkpoint.")

    train_args = payload["args"]
    seed_everything(int(train_args["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_config = resolve_scene_config(str(train_args["dataset"]))
    scene = build_segmentation_scene(
        data_root=Path(train_args["data_root"]),
        config=scene_config,
        img_size=train_args.get("img_size"),
        overlap_size=train_args.get("overlap_size"),
        train_num=train_args.get("train_num"),
        val_num=train_args.get("val_num"),
        batch_size=int(train_args.get("batch_size") or 1),
        num_workers=int(train_args.get("num_workers") or 0),
        augment=False,
        normalization=str(train_args.get("normalization", "minmax")),
        train_repeat_factor=int(train_args.get("train_repeat_factor") or 1),
        train_spatial_jitter=int(train_args.get("train_spatial_jitter") or 0),
        centered_training=train_args.get("centered_training"),
        official_val_ratio=train_args.get("official_val_ratio"),
    )

    resolved_gsd = scene_config.gsd_m if train_args.get("gsd") is None else float(train_args["gsd"])
    channels = int(scene["channels"])
    metadata = build_downstream_metadata(
        num_bands=channels,
        sensor=scene_config.sensor,
        level=scene_config.level,
        gsd=resolved_gsd,
        wavelengths_nm=scene_config.wavelengths if int(scene_config.wavelengths.numel()) == channels else None,
        fwhm_nm=scene_config.fwhm if int(scene_config.fwhm.numel()) == channels else None,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=Path(train_args["checkpoint"]),
        prototype_bank_eo1=Path(train_args["prototype_bank_eo1"]),
        prototype_bank_gf5=Path(train_args["prototype_bank_gf5"]),
        metadata=metadata,
        freeze_encoder=bool(train_args.get("freeze_encoder", False)),
        device=device,
    )
    model = AwareSSFusionSeg(
        extractor,
        int(scene["class_count"]),
        feature_layers=int(train_args.get("feature_layers", 4)),
        fusion_channels=int(train_args.get("fusion_channels", 128)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)

    metrics = evaluate_merged_scene(
        model=model,
        loader=scene["test_loader"],
        device=device,
        class_count=int(scene["class_count"]),
        precision=str(train_args.get("precision", "bf16")),
        class_names=scene_config.class_names,
        starts=scene["test_starts"],
        height=int(scene["height"]),
        width=int(scene["width"]),
        padded_height=int(scene["padded_height"]),
        padded_width=int(scene["padded_width"]),
        img_size=int(scene["img_size"]),
        overlap_size=int(scene["overlap_size"]),
        label_map=scene["test_gt"],
        eval_tta=bool(train_args.get("eval_tta", False)),
    )
    result = {
        "selection_epoch": int(payload.get("epoch", -1)),
        "checkpoint": str(args.checkpoint),
        "test": metrics,
    }
    print(json.dumps(result, ensure_ascii=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
