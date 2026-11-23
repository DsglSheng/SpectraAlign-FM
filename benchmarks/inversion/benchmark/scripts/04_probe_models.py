from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from model_adapters import build_model


TRAIN_SCRIPT = Path(__file__).with_name("03_train_regression.py")
SPEC = importlib.util.spec_from_file_location("train_regression", TRAIN_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
TRAIN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_MODULE)
DEFAULTS = TRAIN_MODULE.DEFAULTS
fwhm_from_wavelengths = TRAIN_MODULE.fwhm_from_wavelengths


def main() -> int:
    parser = argparse.ArgumentParser(description="Load every inversion backbone and run a small forward pass.")
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["aware", "hypersigma", "spectralgpt", "spectralearth", "specaware"],
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.dataset_npz, allow_pickle=True) as data:
        spectra = np.asarray(data["spectra"], dtype=np.float32)
        wavelengths = np.asarray(data["wavelengths_nm"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        train_idx = np.asarray(data["train_idx"], dtype=np.int64)
    train = spectra[train_idx]
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train.std(axis=0, dtype=np.float64).astype(np.float32), 1.0e-6)
    batch = torch.from_numpy((spectra[train_idx[:2]] - mean) / std).to(device)
    paths = {name: value for name, value in DEFAULTS.items()}
    wavelengths_tensor = torch.from_numpy(wavelengths)
    fwhm_tensor = torch.from_numpy(fwhm_from_wavelengths(wavelengths))

    for model_name in args.models:
        model, info = build_model(
            model_name,
            wavelengths_nm=wavelengths_tensor,
            fwhm_nm=fwhm_tensor,
            hidden_dim=256,
            dropout=0.1,
            device=device,
            paths=paths,
            sensor=str(metadata.get("sensor", "field")),
            level=str(metadata.get("level", "L2")),
            gsd=float(metadata.get("gsd", 1.0)),
        )
        model.to(device).eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch)
        print(json.dumps({"model": model_name, "output_shape": list(output.shape), "info": info}, default=str))
        del model, output
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
