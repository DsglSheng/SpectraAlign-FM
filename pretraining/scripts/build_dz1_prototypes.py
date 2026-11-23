#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from aware_open.data.pretrain import GroundSpectralDataset
from aware_open.dz1_profiles import build_dz1_swir_profile, build_dz1_vnir_profile
from aware_open.spectral_prototypes import (
    PrototypeBank,
    build_prototype_bank,
    resample_spectrum_via_auxiliary_grid,
    save_prototype_bank,
    shape_normalize_spectra,
    summarize_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DZ01 VNIR/SWIR ground-spectral prototype banks.")
    parser.add_argument("--ground-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prototype-count", type=int, default=256)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--aux-grid-step-nm", type=float, default=1.0)
    parser.add_argument("--max-interpolation-gap-factor", type=float, default=4.0)
    parser.add_argument("--vnir-max-source-step-nm", type=float, default=2.0)
    parser.add_argument("--swir-max-source-step-nm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def boxcar_weights(grid: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    values = grid.float().view(1, -1)
    return ((values >= lower.float().view(-1, 1)) & (values <= upper.float().view(-1, 1))).float()


def collect(
    dataset: GroundSpectralDataset,
    *,
    auxiliary: torch.Tensor,
    weights: torch.Tensor,
    min_coverage: float,
    max_source_step_nm: float,
    max_gap_factor: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    accepted: List[torch.Tensor] = []
    coverage_values: List[float] = []
    source_steps: List[float] = []
    rejected = {"low_coverage": 0, "coarse_step": 0, "invalid_source": 0, "zero_signal": 0}
    for index in range(len(dataset)):
        sample = dataset[index]
        resampled, coverage, source_step = resample_spectrum_via_auxiliary_grid(
            sample.values.float(),
            sample.wavelengths.float(),
            auxiliary_wavelengths_nm=auxiliary,
            target_srf_weights=weights,
            valid_mask=None if sample.valid_mask is None else sample.valid_mask.bool(),
            max_interpolation_gap_factor=max_gap_factor,
        )
        if not math.isfinite(source_step):
            rejected["invalid_source"] += 1
            continue
        source_steps.append(source_step)
        if source_step > max_source_step_nm:
            rejected["coarse_step"] += 1
            continue
        coverage_ratio = float(coverage.mean().item())
        if coverage_ratio < min_coverage:
            rejected["low_coverage"] += 1
            continue
        # The balanced v3 source deliberately extends VNIR-only water spectra
        # with exact zeros to 2500 nm.  Those samples remain meaningful to a
        # full-range EO1/GF5 vector, but become an all-zero vector for a
        # SWIR-only sensor and have no defined cosine direction.
        if float(torch.linalg.vector_norm(resampled).item()) <= 1.0e-8:
            rejected["zero_signal"] += 1
            continue
        accepted.append(shape_normalize_spectra(resampled))
        coverage_values.append(coverage_ratio)
    if not accepted:
        raise ValueError("No ground spectra passed the DZ1 profile filters.")
    return torch.stack(accepted), {
        "total_samples": len(dataset),
        "accepted_samples": len(accepted),
        "rejected_low_coverage": rejected["low_coverage"],
        "rejected_coarse_step": rejected["coarse_step"],
        "rejected_invalid_source": rejected["invalid_source"],
        "rejected_zero_signal": rejected["zero_signal"],
        "coverage_mean": sum(coverage_values) / len(coverage_values),
        "coverage_min": min(coverage_values),
        "coverage_max": max(coverage_values),
        "source_step_p50": float(torch.tensor(source_steps).median().item()) if source_steps else None,
    }


def build_one(args: argparse.Namespace, dataset: GroundSpectralDataset, sensor: str) -> Tuple[PrototypeBank, Dict[str, Any]]:
    profile = build_dz1_vnir_profile() if sensor == "dz1_vnir" else build_dz1_swir_profile()
    lower = torch.from_numpy(profile.wavelength_min_nm)
    upper = torch.from_numpy(profile.wavelength_max_nm)
    center = torch.from_numpy(profile.wavelengths_nm)
    bandwidth = torch.from_numpy(profile.effective_bandwidth_nm)
    start = math.floor(float(lower.min().item()))
    end = math.ceil(float(upper.max().item()))
    auxiliary = torch.arange(start, end + args.aux_grid_step_nm, args.aux_grid_step_nm, dtype=torch.float32)
    weights = boxcar_weights(auxiliary, lower, upper)
    max_step = args.vnir_max_source_step_nm if sensor == "dz1_vnir" else args.swir_max_source_step_nm
    spectra, stats = collect(
        dataset,
        auxiliary=auxiliary,
        weights=weights,
        min_coverage=args.min_coverage,
        max_source_step_nm=max_step,
        max_gap_factor=args.max_interpolation_gap_factor,
    )
    bank = build_prototype_bank(
        sensor=sensor,
        resampled_spectra=spectra,
        target_wavelengths_nm=center,
        target_fwhm_nm=bandwidth,
        num_clusters=args.prototype_count,
        num_iters=args.kmeans_iters,
        batch_size=args.kmeans_batch_size,
        min_coverage=args.min_coverage,
        seed=args.seed,
    )
    output_path = args.output_dir / ("prototype_bank_" + sensor + ".pt")
    save_prototype_bank(
        output_path,
        bank,
        extra_metadata={
            **stats,
            "response_model": profile.response_model,
            "bandwidth_source": profile.bandwidth_source,
            "native_gsd_m": profile.native_gsd_m,
            "effective_gsd_m": profile.effective_gsd_m,
            "max_source_step_nm": max_step,
        },
    )
    return bank, {**summarize_bank(bank), **stats, "path": str(output_path)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GroundSpectralDataset(args.ground_root, normalization="none")
    summary: Dict[str, Any] = {
        "ground_root": str(args.ground_root.resolve()),
        "ground_sample_count": len(dataset),
        "prototype_count": args.prototype_count,
        "min_coverage": args.min_coverage,
        "response_model": "boxcar_measured_span",
        "seed": args.seed,
        "banks": {},
    }
    for sensor in ("dz1_vnir", "dz1_swir"):
        _, stats = build_one(args, dataset, sensor)
        summary["banks"][sensor] = stats
    destination = args.output_dir / "prototype_build_summary.json"
    temporary = destination.with_suffix(".json.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temporary), str(destination))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
