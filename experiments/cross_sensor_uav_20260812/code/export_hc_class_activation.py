from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, uniform_filter


ROOT = Path("/mnt/Data/SpectralData/cross_sensor_uav_20260812")
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
from export_hc_last_features import (  # noqa: E402
    create_scene,
    load_hypersigma,
    load_ours,
    load_specaware,
    load_spectralearth,
)


CLASS_NAMES = (
    "Strawberry", "Cowpea", "Soybean", "Sorghum", "Water spinach",
    "Watermelon", "Greens", "Trees", "Grass", "Red roof", "Gray roof",
    "Plastic", "Bare soil", "Road", "Bright object", "Water",
)
PREFERRED_CLASSES = (1, 10, 13, 16, 6, 8, 14)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def raw_data() -> tuple[np.ndarray, np.ndarray]:
    root = Path("/mnt/Data/SpectralData/Aware_open/data/classification/hanchuan")
    cube = loadmat(root / "WHU_Hi_HanChuan.mat")["WHU_Hi_HanChuan"].astype(np.float32)
    labels = loadmat(root / "WHU_Hi_HanChuan_gt.mat")["WHU_Hi_HanChuan_gt"].astype(np.int16)
    return cube, labels


def centered_crop(array: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    half = size // 2
    pad = ((half, half), (half, half)) + ((0, 0),) * max(array.ndim - 2, 0)
    padded = np.pad(array, pad, mode="reflect")
    return padded[row : row + size, col : col + size]


def generate_candidates(output: Path, crop_size: int, per_class: int) -> None:
    _, labels = raw_data()
    scene = create_scene()
    test_gt = scene["test_gt"]
    half = crop_size // 2
    interior = np.zeros_like(labels, dtype=bool)
    interior[half : labels.shape[0] - half, half : labels.shape[1] - half] = True
    candidates: list[dict[str, Any]] = []
    for class_id in range(1, len(CLASS_NAMES) + 1):
        mask = labels == class_id
        distance = distance_transform_edt(mask)
        purity = uniform_filter(mask.astype(np.float32), size=33, mode="nearest")
        valid = (test_gt == class_id) & (distance >= 5.0) & (purity >= 0.50) & interior
        rows, cols = np.where(valid)
        if rows.size == 0:
            continue
        merit = distance[rows, cols] + 12.0 * purity[rows, cols]
        order = np.argsort(merit)[::-1]
        chosen: list[tuple[int, int]] = []
        for idx in order:
            r, c = int(rows[idx]), int(cols[idx])
            if any((r - rr) ** 2 + (c - cc) ** 2 < 48 ** 2 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            candidates.append({
                "id": f"c{class_id:02d}_r{r:04d}_c{c:04d}",
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id - 1],
                "row": r,
                "col": c,
                "distance": float(distance[r, c]),
                "local_purity_33": float(purity[r, c]),
                "preferred": class_id in PREFERRED_CLASSES,
            })
            if len(chosen) >= per_class:
                break
    save_json(output, {"crop_size": crop_size, "candidates": candidates})
    print(json.dumps({"output": str(output), "count": len(candidates)}))


def class_scores(logits: Any, class_count: int) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not torch.is_tensor(logits):
        raise TypeError(type(logits))
    if logits.ndim == 2:
        return logits.float()
    if logits.ndim == 3:
        if logits.shape[-1] == class_count:
            spatial = logits
        elif logits.shape[1] == class_count:
            spatial = logits.transpose(1, 2)
        else:
            raise ValueError(f"Cannot locate class dimension in {tuple(logits.shape)}")
        count = int(spatial.shape[1])
        side = int(round(math.sqrt(count)))
        if side * side == count:
            maps = spatial.reshape(spatial.shape[0], side, side, class_count)
            radius = max(1, side // 32)
            mid = side // 2
            return maps[:, mid - radius : mid + radius + 1, mid - radius : mid + radius + 1].mean((1, 2)).float()
        return spatial[:, count // 2].float()
    if logits.ndim == 4:
        if logits.shape[1] == class_count:
            maps = logits.permute(0, 2, 3, 1)
        elif logits.shape[-1] == class_count:
            maps = logits
        else:
            raise ValueError(f"Cannot locate class dimension in {tuple(logits.shape)}")
        h, w = int(maps.shape[1]), int(maps.shape[2])
        rh, rw = max(1, h // 32), max(1, w // 32)
        return maps[:, h // 2 - rh : h // 2 + rh + 1, w // 2 - rw : w // 2 + rw + 1].mean((1, 2)).float()
    raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}")


def load_method(method: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    scene = create_scene(pca_components=30 if method == "hypersigma" else None)
    loader = {
        "ours": load_ours,
        "spectralearth": load_spectralearth,
        "specaware": load_specaware,
        "hypersigma": load_hypersigma,
    }[method]
    model, _ = loader(device, scene)
    model.eval()
    return model, scene


@torch.inference_mode()
def score_candidates(method: str, candidates_path: Path, output: Path, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scene = load_method(method, device)
    spec = json.loads(candidates_path.read_text(encoding="utf-8"))
    size = int(spec["crop_size"])
    candidates = spec["candidates"]
    records: list[dict[str, Any]] = []
    for start in range(0, len(candidates), batch_size):
        group = candidates[start : start + batch_size]
        x = np.stack([centered_crop(scene["cube"], int(v["row"]), int(v["col"]), size) for v in group])
        x = torch.from_numpy(np.moveaxis(x, -1, 1).astype(np.float32)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scores = class_scores(model(x), int(scene["class_count"]))
        probabilities = scores.softmax(dim=-1).cpu().numpy()
        for candidate, prob in zip(group, probabilities):
            target = int(candidate["class_id"]) - 1
            records.append({
                "id": candidate["id"],
                "target": target + 1,
                "prediction": int(prob.argmax()) + 1,
                "target_probability": float(prob[target]),
                "max_probability": float(prob.max()),
                "correct": int(prob.argmax()) == target,
            })
    save_json(output, {"method": method, "records": records})
    print(json.dumps({"method": method, "output": str(output), "count": len(records)}))


@torch.inference_mode()
def export_cam(
    method: str,
    selected_path: Path,
    output: Path,
    window: int,
    stride: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scene = load_method(method, device)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    size = int(selected["crop_size"])
    maps, probabilities, predictions = [], [], []
    for item in selected["samples"]:
        patch = centered_crop(scene["cube"], int(item["row"]), int(item["col"]), size)
        base = torch.from_numpy(np.moveaxis(patch, -1, 0).astype(np.float32)).unsqueeze(0).to(device)
        target = int(item["class_id"]) - 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            clean_scores = class_scores(model(base), int(scene["class_count"]))
        clean_prob = clean_scores.softmax(-1)[0]
        probabilities.append(float(clean_prob[target]))
        predictions.append(int(clean_prob.argmax()) + 1)
        positions = [(r, c) for r in range(0, size - window + 1, stride) for c in range(0, size - window + 1, stride)]
        heat = np.zeros((size, size), dtype=np.float32)
        weight = np.zeros((size, size), dtype=np.float32)
        for start in range(0, len(positions), batch_size):
            group = positions[start : start + batch_size]
            masked = base.repeat(len(group), 1, 1, 1)
            for idx, (r, c) in enumerate(group):
                masked[idx, :, r : r + window, c : c + window] = 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                masked_prob = class_scores(model(masked), int(scene["class_count"])).softmax(-1)[:, target]
            # Probability drop has the same [0, 1] interpretation for every
            # heterogeneous classifier; raw-logit drops are not cross-model
            # comparable because each head can learn a different logit scale.
            drops = torch.relu(clean_prob[target].float() - masked_prob.float()).cpu().numpy()
            for value, (r, c) in zip(drops, group):
                heat[r : r + window, c : c + window] += float(value)
                weight[r : r + window, c : c + window] += 1.0
        maps.append(heat / np.maximum(weight, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        method=method,
        attribution=np.stack(maps),
        target_probability=np.asarray(probabilities, dtype=np.float32),
        prediction=np.asarray(predictions, dtype=np.int16),
        class_names=np.asarray(CLASS_NAMES),
        window=np.int16(window),
        stride=np.int16(stride),
    )
    print(json.dumps({"method": method, "output": str(output), "samples": len(maps)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidates", "score", "cam"), required=True)
    parser.add_argument("--method", choices=("ours", "spectralearth", "specaware", "hypersigma"))
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "candidates":
        generate_candidates(args.output, args.crop_size, args.per_class)
    elif args.mode == "score":
        score_candidates(args.method, args.candidates, args.output, args.batch_size)
    else:
        export_cam(args.method, args.selected, args.output, args.window, args.stride, args.batch_size)


if __name__ == "__main__":
    main()
