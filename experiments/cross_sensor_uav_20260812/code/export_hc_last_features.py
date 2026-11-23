from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat


# Downstream runs used the frozen legacy-11 Aware package.  Loading the
# compatibility checkpoint with the newer 13-sensor package would incorrectly
# change the sensor-embedding shape, so feature export must use this exact code.
AWARE_CODE = Path(__file__).resolve().parents[3]
AWARE_SCRIPT = AWARE_CODE / "scripts"
SPEC_SCRIPTS = Path("/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/scripts")
SE_SCRIPTS = Path("/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/scripts")
SGPT_SCRIPTS = Path("/mnt/Data/SpectralData/SpectralGPT_Aware_Comparison_20260516/scripts")
for path in (AWARE_CODE, AWARE_SCRIPT, SPEC_SCRIPTS, SE_SCRIPTS, SGPT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aware_open.downstream import (  # noqa: E402
    AwareFeatureExtractor,
    build_downstream_metadata,
    build_segmentation_scene,
    resolve_scene_config,
    seed_everything,
)
from finetune_classification import AwareSSFusionSeg  # noqa: E402


ROIS = (
    {"name": "mixed_fields_and_built_objects", "row": 416, "col": 64, "size": 128},
    {"name": "fields_roads_water", "row": 896, "col": 64, "size": 128},
    {"name": "field_boundaries_and_roofs", "row": 64, "col": 64, "size": 128},
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def namespace_from_config(path: Path) -> SimpleNamespace:
    cfg = load_json(path)
    for key in ("checkpoint", "data_root", "output_dir", "specaware_repo", "spectralgpt_repo"):
        if key in cfg:
            cfg[key] = Path(cfg[key])
    return SimpleNamespace(**cfg)


def create_scene(*, pca_components: int | None = None) -> dict[str, Any]:
    config = resolve_scene_config("hc")
    return build_segmentation_scene(
        data_root=Path("/mnt/Data/SpectralData/Aware_open/data/classification/hanchuan"),
        config=config,
        split_mode="number",
        img_size=128,
        overlap_size=32,
        train_num=50,
        val_num=20,
        train_ratio=None,
        val_ratio=None,
        batch_size=1,
        num_workers=0,
        augment=False,
        normalization="mean_std",
        pca_components=pca_components,
        pca_whiten=bool(pca_components),
        train_repeat_factor=1,
        train_spatial_jitter=0,
        centered_training=True,
        official_val_ratio=None,
    )


def roi_tensor(cube: np.ndarray, device: torch.device) -> torch.Tensor:
    patches = []
    for roi in ROIS:
        r, c, size = int(roi["row"]), int(roi["col"]), int(roi["size"])
        patch = cube[r : r + size, c : c + size]
        if patch.shape[:2] != (size, size):
            raise ValueError(f"ROI outside cube: {roi}")
        patches.append(torch.from_numpy(np.moveaxis(patch.astype(np.float32, copy=False), -1, 0)))
    return torch.stack(patches).to(device=device)


def to_norm_map(feature: torch.Tensor, size: int = 128) -> torch.Tensor:
    if feature.ndim == 3:
        count = int(feature.shape[1])
        side = int(round(math.sqrt(count)))
        if side * side != count:
            raise ValueError(f"Cannot reshape {count} tokens to a square map")
        feature = feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[-1], side, side)
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW or BND feature, got {tuple(feature.shape)}")
    feature = feature.float()
    # LayerNorm makes the raw channel norm nearly constant for ViT tokens.
    # Subtracting the spatial mean vector yields a comparable measure of how
    # strongly each location deviates from the scene-level representation.
    centered = feature - feature.mean(dim=(-2, -1), keepdim=True)
    norm = torch.linalg.vector_norm(centered, ord=2, dim=1, keepdim=True)
    return F.interpolate(norm, size=(size, size), mode="bilinear", align_corners=False)[:, 0]


def load_ours(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Any]:
    run = Path(os.environ.get("HC_OURS_RUN", "/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/downstream_finetune_202607/results/aware_hsi_dz1_525k_fromscratch_v1/classification_hc"))
    cfg = load_json(run / "config.json")
    config = resolve_scene_config("hc")
    channels = int(scene["channels"])
    metadata = build_downstream_metadata(
        num_bands=channels,
        sensor=config.sensor,
        level=config.level,
        gsd=config.gsd_m,
        wavelengths_nm=config.wavelengths if int(config.wavelengths.numel()) == channels else None,
        fwhm_nm=config.fwhm if int(config.fwhm.numel()) == channels else None,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=Path(cfg["checkpoint"]),
        prototype_bank_eo1=Path(cfg["prototype_bank_eo1"]),
        prototype_bank_gf5=Path(cfg["prototype_bank_gf5"]),
        metadata=metadata,
        freeze_encoder=False,
        device=device,
    )
    model = AwareSSFusionSeg(extractor, int(scene["class_count"])).to(device)
    ckpt = torch.load(run / "best_val_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    def extract(x: torch.Tensor) -> torch.Tensor:
        features = model.extractor.forward_features(x)
        return features["patch_tokens"]

    return model, extract


def load_specaware(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Any]:
    from finetune_specaware_hypersigma import (  # type: ignore
        SpecAwareClassifier,
        build_backbone,
        classification_meta,
    )

    run = Path(os.environ.get("HC_SPECAWARE_RUN", "/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/results/hypersigma_downstream_20260517/results/classification_hc/specaware_base_hypersigma_config"))
    args = namespace_from_config(run / "config.json")
    config = resolve_scene_config("hc")
    wavelengths, fwhm = classification_meta(config, int(scene["channels"]))
    backbone = build_backbone(
        args,
        in_channels=int(scene["channels"]),
        wavelengths=wavelengths,
        fwhm=fwhm,
        sensor=config.sensor,
        level=config.level,
        gsd_m=config.gsd_m,
        img_size=128,
    )
    model = SpecAwareClassifier(backbone, int(scene["class_count"]), float(args.dropout)).to(device)
    ckpt = torch.load(run / "best_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    def extract(x: torch.Tensor) -> torch.Tensor:
        maps, _ = model.backbone(x)
        return maps[-1]

    return model, extract


def load_spectralearth(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Any]:
    from spectralearth_adapter import (  # type: ignore
        SpectralEarthClassifier,
        import_spectralearth,
        prepare_spectral_input,
    )

    run = Path(os.environ.get("HC_SPECTRALEARTH_RUN", "/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/results/hypersigma_specaware_protocol_20260614/classification/hc/specvit_small_mae"))
    args = namespace_from_config(run / "config.json")
    model_cls = import_spectralearth(args.spectralgpt_repo)
    model = SpectralEarthClassifier(
        model_cls,
        img_size=128,
        num_classes=int(scene["class_count"]),
        drop_path=float(args.drop_path),
        dropout=float(args.dropout),
    ).to(device)
    ckpt = torch.load(run / "best_val_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    def extract(x: torch.Tensor) -> torch.Tensor:
        x = prepare_spectral_input(x)
        return model.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]

    return model, extract


def load_hypersigma(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Any]:
    from finetune_hypersigma_hc_transfer import HyperSIGMADenseAdapter, HyperSIGMAExtractor

    run = Path(os.environ.get("HC_HYPERSIGMA_RUN", "/mnt/Data/SpectralData/cross_sensor_uav_20260812/results/hypersigma_hc_seed3704"))
    metadata = SimpleNamespace(wavelengths_nm=list(range(int(scene["channels"]))))
    extractor = HyperSIGMAExtractor(
        Path("/mnt/Data/SpectralData/HyperSIGMA-main/Pretrain/spat-vit-base-ultra-checkpoint-1599.pth"),
        Path("."), Path("."), metadata, False, device,
    )
    model = HyperSIGMADenseAdapter(extractor, int(scene["class_count"])).to(device)
    ckpt = torch.load(run / "best_val_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    def extract(x: torch.Tensor) -> torch.Tensor:
        _, features = model(x, return_features=True)
        return features["last_map"]

    return model, extract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ours", "spectralearth", "specaware", "hypersigma"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(3704)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca = 30 if args.method == "hypersigma" else None
    scene = create_scene(pca_components=pca)
    model, extractor = {
        "ours": load_ours,
        "specaware": load_specaware,
        "spectralearth": load_spectralearth,
        "hypersigma": load_hypersigma,
    }[args.method](device, scene)
    model.eval()
    x = roi_tensor(scene["cube"], device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        feature = extractor(x)
    norm = to_norm_map(feature).cpu().numpy().astype(np.float32)
    if not np.isfinite(norm).all():
        raise FloatingPointError("Non-finite feature norm")

    raw_root = Path("/mnt/Data/SpectralData/Aware_open/data/classification/hanchuan")
    raw = loadmat(raw_root / "WHU_Hi_HanChuan.mat")["WHU_Hi_HanChuan"].astype(np.float32)
    labels = loadmat(raw_root / "WHU_Hi_HanChuan_gt.mat")["WHU_Hi_HanChuan_gt"].astype(np.int16)
    raw_rois = np.stack([raw[r["row"] : r["row"] + r["size"], r["col"] : r["col"] + r["size"]] for r in ROIS])
    label_rois = np.stack([labels[r["row"] : r["row"] + r["size"], r["col"] : r["col"] + r["size"]] for r in ROIS])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        method=args.method,
        feature_norm=norm,
        raw_hsi=raw_rois,
        labels=label_rois,
        rois=np.asarray([json.dumps(r) for r in ROIS]),
        feature_shape=np.asarray(feature.shape),
    )
    print(json.dumps({"method": args.method, "output": str(args.output), "feature_shape": list(feature.shape), "norm_min": float(norm.min()), "norm_max": float(norm.max())}))


if __name__ == "__main__":
    main()
