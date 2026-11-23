from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


AWARE_ROOT = Path("/mnt/Data/SpectralData/Aware_open")
SPECAWARE_ROOT = Path("/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517")
SPECTRALGPT_ROOT = Path("/mnt/Data/SpectralData/SpectralGPT_Aware_Comparison_20260516")
SPECTRALEARTH_ROOT = Path("/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense sliding-window detection visualization export.")
    parser.add_argument("--model", choices=("spectralgpt", "specaware", "spectralearth"), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_config(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def import_detection_scene() -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(AWARE_ROOT))
    sys.path.insert(0, str(AWARE_ROOT / "scripts"))
    from aware_open.downstream.detection_scene import (
        build_detection_scene,
        detection_default_wavelengths,
        resolve_detection_scene_config,
    )

    return build_detection_scene, detection_default_wavelengths, resolve_detection_scene_config


def build_scene(config: dict[str, Any]) -> dict[str, Any]:
    build_detection_scene, _, resolve_detection_scene_config = import_detection_scene()
    scene_config = resolve_detection_scene_config(config["task"], config["dataset"])
    return build_detection_scene(
        data_root=Path(config["data_root"]),
        config=scene_config,
        img_size=int(config["img_size"]),
        overlap_size=config.get("overlap_size"),
        batch_size=1,
        num_workers=0,
        augment=False,
        normalization=config.get("normalization") or "mean_std",
    )


def checkpoint_path(result_dir: Path) -> Path:
    for name in ("best_auc.pt", "latest.pt"):
        candidate = result_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No detection checkpoint in {result_dir}")


def load_state(model: torch.nn.Module, result_dir: Path) -> Path:
    path = checkpoint_path(result_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    return path


def build_spectralgpt(config: dict[str, Any], result_dir: Path) -> tuple[torch.nn.Module, Path]:
    import finetune_spectralgpt_downstream_binary as module

    models = module.import_spectralgpt(Path(config["spectralgpt_repo"]))
    model = module.SpectralGPTBinary(models, img_size=int(config["img_size"]))
    module.load_spectralgpt_checkpoint(model.model, Path(config["checkpoint"]))
    return model, load_state(model, result_dir)


def build_spectralearth(config: dict[str, Any], result_dir: Path) -> tuple[torch.nn.Module, Path]:
    sys.path.insert(0, str(SPECTRALEARTH_ROOT / "scripts"))
    import finetune_spectralearth_downstream_binary as wrapper

    module = wrapper.base
    models = module.import_spectralgpt(Path(config["spectralgpt_repo"]))
    model = module.SpectralGPTBinary(models, img_size=int(config["img_size"]))
    module.load_spectralgpt_checkpoint(model.model, Path(config["checkpoint"]))
    return model, load_state(model, result_dir)


def build_specaware(
    config: dict[str, Any], scene: dict[str, Any], result_dir: Path
) -> tuple[torch.nn.Module, Path]:
    os.environ.setdefault(
        "SPECAWARE_SENTENCE_MODEL",
        str(SPECAWARE_ROOT / "pretrained" / "sentence-transformers_all-MiniLM-L6-v2"),
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(SPECTRALGPT_ROOT / "scripts"))
    sys.path.insert(0, str(SPECAWARE_ROOT / "scripts"))
    import finetune_specaware_hypersigma as module

    _, detection_default_wavelengths, _ = import_detection_scene()
    scene_config = scene["config"]
    channels = int(scene["channels"])
    wavelengths = (
        scene_config.wavelengths
        if int(scene_config.wavelengths.numel()) == channels
        else detection_default_wavelengths(scene_config.task, scene_config.dataset, channels)
    )
    namespace = SimpleNamespace(**config)
    namespace.specaware_repo = Path(config["specaware_repo"])
    namespace.checkpoint = Path(config["checkpoint"])
    backbone = module.build_backbone(
        namespace,
        in_channels=channels,
        wavelengths=wavelengths,
        fwhm=module.fwhm_from_wavelengths(wavelengths),
        sensor=scene_config.sensor,
        level=scene_config.level,
        gsd_m=scene_config.gsd_m,
        img_size=int(scene["img_size"]),
    )
    model = module.SpecAwareBinary(backbone)
    return model, load_state(model, result_dir)


def build_model(
    name: str, config: dict[str, Any], scene: dict[str, Any], result_dir: Path
) -> tuple[torch.nn.Module, Path]:
    if name == "spectralgpt":
        return build_spectralgpt(config, result_dir)
    if name == "specaware":
        return build_specaware(config, scene, result_dir)
    if name == "spectralearth":
        return build_spectralearth(config, result_dir)
    raise ValueError(name)


def grid_coordinates(length: int, stride: int) -> np.ndarray:
    return np.unique(np.append(np.arange(0, length, stride), length - 1)).astype(np.int64)


def extract_batch(
    padded: np.ndarray,
    coordinates: list[tuple[int, int]],
    patch_size: int,
) -> torch.Tensor:
    patches = [padded[row : row + patch_size, col : col + patch_size] for row, col in coordinates]
    array = np.stack(patches, axis=0).transpose(0, 3, 1, 2).astype(np.float32, copy=False)
    return torch.from_numpy(array)


@torch.no_grad()
def dense_scores(
    model: torch.nn.Module,
    cube: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    precision: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = cube.shape[:2]
    rows = grid_coordinates(height, stride)
    cols = grid_coordinates(width, stride)
    coordinates = [(int(row), int(col)) for row in rows for col in cols]
    pad_before = patch_size // 2
    pad_after = patch_size - pad_before - 1
    padded = np.pad(
        cube,
        ((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
        mode="reflect",
    )
    model.eval().to(device)
    values: list[np.ndarray] = []
    for start in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[start : start + batch_size]
        image = extract_batch(padded, batch_coords, patch_size).to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            probability = torch.softmax(model(image), dim=1)[:, 1]
        values.append(probability.float().cpu().numpy())
    grid = np.concatenate(values).reshape(len(rows), len(cols)).astype(np.float32)
    full = F.interpolate(
        torch.from_numpy(grid)[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )[0, 0].numpy().astype(np.float32)
    return full, rows, cols


def top_fraction_mask(score: np.ndarray, fraction: float = 0.005) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValueError("Score map contains non-finite values")
    if float(flat.max() - flat.min()) < 1.0e-8:
        return np.zeros(score.shape, dtype=np.uint8)
    count = max(1, int(round(flat.size * fraction)))
    indices = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.shape, dtype=np.uint8)
    mask[indices] = 1
    return mask.reshape(score.shape)


def main() -> int:
    args = parse_args()
    started = time.time()
    config = load_config(args.result_dir)
    scene = build_scene(config)
    model, checkpoint = build_model(args.model, config, scene, args.result_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score, rows, cols = dense_scores(
        model,
        np.asarray(scene["cube"], dtype=np.float32),
        patch_size=int(scene["img_size"]),
        stride=max(1, int(args.stride)),
        batch_size=max(1, int(args.batch_size)),
        precision=args.precision,
        device=device,
    )
    binary = top_fraction_mask(score)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dense_detection_visual.npz",
        score_map=score,
        prediction_binary=binary,
        label=np.asarray(scene["labels"], dtype=np.uint8),
        grid_rows=rows,
        grid_cols=cols,
        visual_stride=np.asarray([int(args.stride)], dtype=np.int16),
    )
    manifest = {
        "model": args.model,
        "dataset": config["dataset"],
        "task": config["task"],
        "source_result_dir": str(args.result_dir),
        "checkpoint": str(checkpoint),
        "scene_shape": list(score.shape),
        "patch_size": int(scene["img_size"]),
        "visual_stride": int(args.stride),
        "grid_shape": [int(len(rows)), int(len(cols))],
        "binary_policy": "top 0.5% of visualization-only dense score map; empty for constant scores",
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "elapsed_sec": float(time.time() - started),
        "table_metrics_recomputed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(manifest), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
