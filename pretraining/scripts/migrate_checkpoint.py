#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand an Aware checkpoint for appended DZ1 sensor tokens.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-sensor-count", type=int, default=13)
    parser.add_argument("--others-index", type=int, default=10)
    parser.add_argument("--keep-optimizer", action="store_true")
    parser.add_argument("--prototype-bank-dz1-vnir", type=Path, default=None)
    parser.add_argument("--prototype-bank-dz1-swir", type=Path, default=None)
    return parser.parse_args()


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def find_sensor_embedding(state: Dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor]:
    matches = [(key, value) for key, value in state.items() if key.endswith("sample_encoder.sensor_embed.weight")]
    if len(matches) != 1:
        raise ValueError("Expected one sensor embedding tensor, found {}".format([key for key, _ in matches]))
    return matches[0]


def add_prototype_buffers(state: Dict[str, torch.Tensor], sensor: str, path: Path) -> list[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("sensor", "")).lower() != sensor:
        raise ValueError("{} is not a {} prototype bank.".format(path, sensor))
    centers = torch.as_tensor(payload["centers"]).float()
    wavelengths = torch.as_tensor(payload["wavelengths_nm"]).float()
    if centers.ndim != 2 or wavelengths.ndim != 1 or centers.shape[1] != wavelengths.shape[0]:
        raise ValueError("Invalid prototype shapes in {}.".format(path))
    safe_sensor = sensor.replace("-", "_")
    center_key = "extra_{}_bank_centers".format(safe_sensor)
    wavelength_key = "extra_{}_bank_wavelengths".format(safe_sensor)
    if center_key in state or wavelength_key in state:
        raise ValueError("Checkpoint already contains {} prototype buffers.".format(sensor))
    state[center_key] = F.normalize(centers, dim=-1)
    state[wavelength_key] = wavelengths
    return [center_key, wavelength_key]


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("Output must differ from input; legacy checkpoints are immutable.")
    payload: Dict[str, Any] = torch.load(args.input, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model/state_dict mapping.")
    key, old = find_sensor_embedding(state)
    old_rows, width = old.shape
    if old_rows > args.target_sensor_count:
        raise ValueError("Checkpoint already has more sensor rows than requested.")
    unchanged_before = {name: tensor_digest(value) for name, value in state.items() if name != key and torch.is_tensor(value)}
    if old_rows < args.target_sensor_count:
        expanded = old.new_empty(args.target_sensor_count, width)
        expanded[:old_rows].copy_(old)
        source_index = min(args.others_index, old_rows - 1)
        expanded[old_rows:].copy_(old[source_index].view(1, -1).expand(args.target_sensor_count - old_rows, -1))
        state[key] = expanded
    prototype_paths = {
        "dz1_vnir": args.prototype_bank_dz1_vnir,
        "dz1_swir": args.prototype_bank_dz1_swir,
    }
    if (prototype_paths["dz1_vnir"] is None) != (prototype_paths["dz1_swir"] is None):
        raise ValueError("Provide both DZ1 prototype banks or neither.")
    added_buffers: list[str] = []
    for sensor, path in prototype_paths.items():
        if path is not None:
            added_buffers.extend(add_prototype_buffers(state, sensor, path))
    if not args.keep_optimizer:
        payload.pop("optimizer", None)
        payload.pop("scheduler", None)
        payload["optimizer_reset_required"] = True
    payload["dz1_migration"] = {
        "source_checkpoint": str(args.input.resolve()),
        "sensor_embedding_key": key,
        "old_sensor_count": old_rows,
        "new_sensor_count": args.target_sensor_count,
        "new_rows_initialized_from": min(args.others_index, old_rows - 1),
        "added_prototype_buffers": added_buffers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(args.output))

    unchanged_after = {name: tensor_digest(value) for name, value in state.items() if name != key and torch.is_tensor(value)}
    changed_unexpectedly = sorted(name for name in unchanged_before if unchanged_before[name] != unchanged_after.get(name))
    report = {
        **payload["dz1_migration"],
        "input_bytes": args.input.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "optimizer_kept": args.keep_optimizer,
        "unchanged_tensor_count": len(unchanged_before),
        "unexpected_changed_tensors": changed_unexpectedly,
        "added_prototype_buffers": added_buffers,
        "success": not changed_unexpectedly,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
