from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, uniform_filter


PROJECT = Path("/mnt/Data/SpectralData/pu_feature_visualization_20260813")
CROSS = Path("/mnt/Data/SpectralData/cross_sensor_uav_20260812")
AWARE = Path("/mnt/Data/SpectralData/Aware_open")
SPEC = Path("/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/scripts")
SE = Path("/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/scripts")
for p in (AWARE, AWARE / "scripts", CROSS / "code", CROSS / "upstream_snapshot", SPEC, SE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aware_open.downstream import (  # noqa: E402
    AwareFeatureExtractor,
    build_downstream_metadata,
    build_segmentation_scene,
    resolve_scene_config,
    seed_everything,
)
from finetune_classification import AwareSSFusionSeg  # noqa: E402


CLASS_NAMES = (
    "Asphalt", "Meadows", "Gravel", "Trees", "Metal sheets",
    "Bare soil", "Bitumen", "Paving bricks", "Shadows",
)
METHODS = ("spectralearth", "hypersigma", "specaware", "ours")
DATA_ROOT = Path("/mnt/Data/SpectralData/HyperSIGMA-main/ImageClassification/data")
OURS_RUN = Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/downstream_finetune_202607/targeted_classification_20260715/results/pu_balanced_lr3e5_e70_seed20260715")
SPECAWARE_RUN = Path("/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/results/hypersigma_downstream_20260517/results/classification_pu/specaware_base_hypersigma_config")
SE_RUN = Path("/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/results/hypersigma_specaware_protocol_20260614/classification/pu/specvit_small_mae")
HYPERSIGMA_RUN = PROJECT / "runs" / "hypersigma_pu_seed3704"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ns(path: Path) -> SimpleNamespace:
    d = load_json(path)
    for key in ("checkpoint", "data_root", "output_dir", "specaware_repo", "spectralgpt_repo"):
        if key in d:
            d[key] = Path(d[key])
    return SimpleNamespace(**d)


def create_scene(pca_components: int | None = None) -> dict[str, Any]:
    config = resolve_scene_config("pu")
    return build_segmentation_scene(
        data_root=DATA_ROOT,
        config=config,
        split_mode="number",
        img_size=128,
        overlap_size=32,
        train_num=20,
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


def load_raw() -> tuple[np.ndarray, np.ndarray]:
    cube_path = next(DATA_ROOT.rglob("PaviaU.mat"))
    gt_path = next(DATA_ROOT.rglob("PaviaU_gt.mat"))
    cube_data = loadmat(cube_path)
    gt_data = loadmat(gt_path)
    cube = next(v for k, v in cube_data.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3)
    labels = next(v for k, v in gt_data.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 2)
    return cube.astype(np.float32), labels.astype(np.int16)


def crop_center(array: np.ndarray, row: int, col: int, size: int = 128) -> np.ndarray:
    half = size // 2
    return array[row - half : row + half, col - half : col + half]


def generate_candidates(output: Path, per_class: int = 16) -> None:
    _, labels = load_raw()
    scene = create_scene()
    test_gt = scene["test_gt"]
    half = 64
    interior = np.zeros_like(labels, dtype=bool)
    interior[half : labels.shape[0] - half, half : labels.shape[1] - half] = True
    records: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES, 1):
        mask = labels == class_id
        distance = distance_transform_edt(mask)
        purity = uniform_filter(mask.astype(np.float32), size=41, mode="nearest")
        valid = (test_gt == class_id) & interior & (distance >= 3.0)
        rr, cc = np.where(valid)
        merit = distance[rr, cc] + 20.0 * purity[rr, cc]
        chosen: list[tuple[int, int]] = []
        for idx in np.argsort(merit)[::-1]:
            r, c = int(rr[idx]), int(cc[idx])
            if any((r - a) ** 2 + (c - b) ** 2 < 36 ** 2 for a, b in chosen):
                continue
            chosen.append((r, c))
            records.append({
                "id": f"c{class_id:02d}_r{r:04d}_c{c:04d}",
                "class_id": class_id,
                "class_name": class_name,
                "row": r,
                "col": c,
                "distance": float(distance[r, c]),
                "local_purity_41": float(purity[r, c]),
            })
            if len(chosen) >= per_class:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"crop_size": 128, "candidates": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "count": len(records)}))


def to_scores(logits: Any, class_count: int = 9) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim == 2:
        return logits.float()
    if logits.ndim == 3:
        if logits.shape[-1] != class_count:
            logits = logits.transpose(1, 2)
        n = int(logits.shape[1])
        side = int(round(math.sqrt(n)))
        if side * side == n:
            maps = logits.reshape(logits.shape[0], side, side, class_count)
            radius = max(1, side // 32)
            mid = side // 2
            return maps[:, mid - radius : mid + radius + 1, mid - radius : mid + radius + 1].mean((1, 2)).float()
        return logits[:, n // 2].float()
    if logits.ndim == 4:
        maps = logits.permute(0, 2, 3, 1) if logits.shape[1] == class_count else logits
        h, w = maps.shape[1:3]
        return maps[:, h // 2, w // 2].float()
    raise ValueError(tuple(logits.shape))


def to_feature_map(feature: torch.Tensor, size: int = 128) -> torch.Tensor:
    if feature.ndim == 3:
        n = int(feature.shape[1])
        side = int(round(math.sqrt(n)))
        if side * side != n:
            # Drop one or more prefix tokens when present.
            for prefix in range(1, min(8, n)):
                side = int(round(math.sqrt(n - prefix)))
                if side * side == n - prefix:
                    feature = feature[:, prefix:]
                    n = n - prefix
                    break
        side = int(round(math.sqrt(n)))
        feature = feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[-1], side, side)
    if feature.ndim != 4:
        raise ValueError(tuple(feature.shape))
    feature = feature.float()
    centered = feature - feature.mean(dim=(-2, -1), keepdim=True)
    response = torch.linalg.vector_norm(centered, ord=2, dim=1, keepdim=True)
    return F.interpolate(response, size=(size, size), mode="bicubic", align_corners=False)[:, 0]


def load_ours(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    cfg = load_json(OURS_RUN / "config.json")
    config = resolve_scene_config("pu")
    metadata = build_downstream_metadata(
        num_bands=int(scene["channels"]), sensor=config.sensor, level=config.level, gsd=config.gsd_m,
        wavelengths_nm=config.wavelengths, fwhm_nm=config.fwhm,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=Path(cfg["checkpoint"]),
        prototype_bank_eo1=Path(cfg["prototype_bank_eo1"]),
        prototype_bank_gf5=Path(cfg["prototype_bank_gf5"]),
        metadata=metadata, freeze_encoder=False, device=device,
    )
    model = AwareSSFusionSeg(
        extractor, int(scene["class_count"]),
        feature_layers=int(cfg.get("feature_layers", 4)),
        fusion_channels=int(cfg.get("fusion_channels", 128)),
    ).to(device)
    checkpoint = torch.load(OURS_RUN / "best_val_loss.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, lambda x: model.extractor.forward_features(x)["patch_tokens"]


def load_specaware(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    from finetune_specaware_hypersigma import SpecAwareClassifier, build_backbone, classification_meta
    args = ns(SPECAWARE_RUN / "config.json")
    config = resolve_scene_config("pu")
    wavelengths, fwhm = classification_meta(config, int(scene["channels"]))
    backbone = build_backbone(
        args, in_channels=int(scene["channels"]), wavelengths=wavelengths, fwhm=fwhm,
        sensor=config.sensor, level=config.level, gsd_m=config.gsd_m, img_size=128,
    )
    model = SpecAwareClassifier(backbone, int(scene["class_count"]), float(args.dropout)).to(device)
    checkpoint = torch.load(SPECAWARE_RUN / "best_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    def extract(x: torch.Tensor) -> torch.Tensor:
        maps, _ = model.backbone(x)
        return maps[-1]
    return model, extract


def load_spectralearth(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    from spectralearth_adapter import SpectralEarthClassifier, import_spectralearth, prepare_spectral_input
    args = ns(SE_RUN / "config.json")
    model_cls = import_spectralearth(args.spectralgpt_repo)
    model = SpectralEarthClassifier(
        model_cls, img_size=128, num_classes=int(scene["class_count"]),
        drop_path=float(args.drop_path), dropout=float(args.dropout),
    ).to(device)
    checkpoint = torch.load(SE_RUN / "best_val_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, lambda x: model.backbone.get_intermediate_layers(prepare_spectral_input(x), n=1, reshape=True, norm=True)[0]


def load_hypersigma(device: torch.device, scene: dict[str, Any]) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    from finetune_hypersigma_hc_transfer import HyperSIGMADenseAdapter, HyperSIGMAExtractor
    metadata = SimpleNamespace(wavelengths_nm=list(range(int(scene["channels"]))))
    extractor = HyperSIGMAExtractor(
        Path("/mnt/Data/SpectralData/HyperSIGMA-main/Pretrain/spat-vit-base-ultra-checkpoint-1599.pth"),
        Path("."), Path("."), metadata, False, device,
    )
    model = HyperSIGMADenseAdapter(extractor, int(scene["class_count"])).to(device)
    checkpoint = torch.load(HYPERSIGMA_RUN / "best_val_oa.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    def extract(x: torch.Tensor) -> torch.Tensor:
        _, features = model(x, return_features=True)
        return features["last_map"]
    return model, extract


@torch.inference_mode()
def export_method(method: str, candidates_path: Path, output: Path, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = create_scene(pca_components=30 if method == "hypersigma" else None)
    model, extractor = {
        "ours": load_ours, "specaware": load_specaware,
        "spectralearth": load_spectralearth, "hypersigma": load_hypersigma,
    }[method](device, scene)
    model.eval()
    candidates = load_json(candidates_path)["candidates"]
    predictions, confidence, responses = [], [], []
    for start in range(0, len(candidates), batch_size):
        group = candidates[start : start + batch_size]
        patches = np.stack([crop_center(scene["cube"], int(v["row"]), int(v["col"])) for v in group])
        x = torch.from_numpy(np.moveaxis(patches, -1, 1).astype(np.float32)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = to_scores(model(x), int(scene["class_count"]))
            feature = extractor(x)
        prob = logits.softmax(-1).float().cpu().numpy()
        fmap = to_feature_map(feature).cpu().numpy().astype(np.float32)
        predictions.extend((prob.argmax(-1) + 1).tolist())
        confidence.extend(prob.max(-1).tolist())
        responses.extend(fmap)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, method=method, ids=np.asarray([v["id"] for v in candidates]),
        predictions=np.asarray(predictions, dtype=np.int16), confidence=np.asarray(confidence, dtype=np.float32),
        response=np.asarray(responses, dtype=np.float32), feature_definition="centered channel L2 norm",
    )
    print(json.dumps({"method": method, "output": str(output), "count": len(candidates)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidates", "export"), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    seed_everything(3704)
    if args.mode == "candidates":
        generate_candidates(args.output)
    else:
        export_method(args.method, args.candidates, args.output, args.batch_size)


if __name__ == "__main__":
    main()
