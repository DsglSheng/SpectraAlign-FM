#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DROP_KEYS = {
    "extra_dz1_swir_bank_centers",
    "extra_dz1_swir_bank_wavelengths",
    "extra_dz1_vnir_bank_centers",
    "extra_dz1_vnir_bank_wavelengths",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_from(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "state_dict", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an 11-sensor downstream-compatible view of a DZ1 checkpoint.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--legacy-sensor-count", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        report = {
            "output": str(args.output),
            "output_sha256": sha256(args.output),
            "reused": True,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return 0

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(payload)!r}")
    model = state_dict_from(payload)
    if not isinstance(model, dict):
        raise TypeError("Could not locate model state dict.")

    removed: list[str] = []
    for key in sorted(DROP_KEYS):
        if key in model:
            model.pop(key)
            removed.append(key)

    sensor_key = "sample_encoder.sensor_embed.weight"
    original_shape = None
    new_shape = None
    if sensor_key in model:
        tensor = model[sensor_key]
        original_shape = list(tensor.shape)
        if tensor.ndim != 2 or tensor.shape[0] < args.legacy_sensor_count:
            raise ValueError(f"Unexpected {sensor_key} shape: {tuple(tensor.shape)}")
        if tensor.shape[0] != args.legacy_sensor_count:
            model[sensor_key] = tensor[: args.legacy_sensor_count].clone()
        new_shape = list(model[sensor_key].shape)

    payload.setdefault("downstream_compatibility", {})
    payload["downstream_compatibility"] = {
        "source_checkpoint": str(args.input),
        "source_sha256": sha256(args.input),
        "legacy_sensor_count": args.legacy_sensor_count,
        "removed_keys": removed,
        "sensor_embed_original_shape": original_shape,
        "sensor_embed_new_shape": new_shape,
        "note": (
            "Compatibility checkpoint for old downstream code. Shared weights are preserved; "
            "DZ1-only sensor rows and prototype buffers are omitted because old downstream datasets "
            "do not use DZ1 sensor ids."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        **payload["downstream_compatibility"],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
