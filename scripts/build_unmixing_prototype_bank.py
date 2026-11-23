from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aware_open.data.pretrain import GroundSpectralDataset
from aware_open.downstream import resolve_unmixing_scene_config
from aware_open.spectral_prototypes import (
    PrototypeBank,
    build_auxiliary_wavelength_grid,
    build_prototype_bank,
    gaussian_srf_matrix,
    resample_spectrum_via_auxiliary_grid,
    save_prototype_bank,
    shape_normalize_spectra,
    summarize_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a downstream-specific ground prototype bank for unmixing.")
    parser.add_argument("--ground-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="Urban4")
    parser.add_argument("--prototype-count", type=int, default=256)
    parser.add_argument("--prototype-min-coverage", type=float, default=0.95)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--aux-grid-step-nm", type=float, default=1.0)
    parser.add_argument("--max-interpolation-gap-factor", type=float, default=4.0)
    parser.add_argument("--max-source-step-nm", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_resampled_spectra(
    dataset: GroundSpectralDataset,
    *,
    auxiliary_wavelengths_nm: torch.Tensor,
    target_srf_weights: torch.Tensor,
    target_wavelengths_nm: torch.Tensor,
    min_coverage: float,
    max_source_step_nm: float,
    max_interpolation_gap_factor: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    accepted: list[torch.Tensor] = []
    coverage_values: list[float] = []
    source_step_values: list[float] = []
    accepted_source_step_values: list[float] = []
    rejected_low_coverage = 0
    rejected_coarse_step = 0
    rejected_invalid_source = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        valid_mask = None if sample.valid_mask is None else sample.valid_mask.bool()
        resampled, coverage, source_step_nm = resample_spectrum_via_auxiliary_grid(
            sample.values.float(),
            sample.wavelengths.float(),
            auxiliary_wavelengths_nm=auxiliary_wavelengths_nm,
            target_srf_weights=target_srf_weights,
            valid_mask=valid_mask,
            max_interpolation_gap_factor=max_interpolation_gap_factor,
        )
        if not math.isfinite(source_step_nm):
            rejected_invalid_source += 1
            continue
        source_step_values.append(source_step_nm)
        if source_step_nm > max_source_step_nm:
            rejected_coarse_step += 1
            continue
        coverage_ratio = float(coverage.mean().item())
        if coverage_ratio < min_coverage:
            rejected_low_coverage += 1
            continue
        accepted.append(shape_normalize_spectra(resampled))
        coverage_values.append(coverage_ratio)
        accepted_source_step_values.append(source_step_nm)
    if not accepted:
        raise ValueError("No ground spectra satisfied the configured coverage threshold.")
    return torch.stack(accepted, dim=0), {
        "total_samples": int(len(dataset)),
        "accepted_samples": int(len(accepted)),
        "rejected_low_coverage": int(rejected_low_coverage),
        "rejected_coarse_step": int(rejected_coarse_step),
        "rejected_invalid_source": int(rejected_invalid_source),
        "coverage_mean": float(sum(coverage_values) / len(coverage_values)),
        "coverage_min": float(min(coverage_values)),
        "coverage_max": float(max(coverage_values)),
        "source_step_p50": float(torch.tensor(source_step_values).median().item()) if source_step_values else None,
        "accepted_source_step_p50": float(torch.tensor(accepted_source_step_values).median().item())
        if accepted_source_step_values
        else None,
        "accepted_source_step_max": float(max(accepted_source_step_values)) if accepted_source_step_values else None,
        "target_band_count": int(target_wavelengths_nm.shape[0]),
        "aux_grid_size": int(auxiliary_wavelengths_nm.shape[0]),
    }


def build_for_scene(
    dataset: GroundSpectralDataset,
    *,
    dataset_name: str,
    output_path: Path,
    prototype_count: int,
    min_coverage: float,
    kmeans_iters: int,
    kmeans_batch_size: int,
    auxiliary_grid_step_nm: float,
    max_interpolation_gap_factor: float,
    max_source_step_nm: float,
    seed: int,
) -> tuple[PrototypeBank, dict[str, object]]:
    scene_config = resolve_unmixing_scene_config(dataset_name)
    target_wavelengths_nm = scene_config.wavelengths
    target_fwhm_nm = scene_config.fwhm
    auxiliary_wavelengths_nm = build_auxiliary_wavelength_grid(
        target_wavelengths_nm,
        target_fwhm_nm,
        step_nm=auxiliary_grid_step_nm,
    )
    target_srf_weights = gaussian_srf_matrix(
        auxiliary_wavelengths_nm,
        target_wavelengths_nm,
        target_fwhm_nm,
    )
    spectra, scene_stats = collect_resampled_spectra(
        dataset,
        auxiliary_wavelengths_nm=auxiliary_wavelengths_nm,
        target_srf_weights=target_srf_weights,
        target_wavelengths_nm=target_wavelengths_nm,
        min_coverage=min_coverage,
        max_source_step_nm=max_source_step_nm,
        max_interpolation_gap_factor=max_interpolation_gap_factor,
    )
    bank = build_prototype_bank(
        sensor=str(dataset_name).lower(),
        resampled_spectra=spectra,
        target_wavelengths_nm=target_wavelengths_nm,
        target_fwhm_nm=target_fwhm_nm,
        num_clusters=prototype_count,
        num_iters=kmeans_iters,
        batch_size=kmeans_batch_size,
        min_coverage=min_coverage,
        seed=seed,
    )
    save_prototype_bank(
        output_path,
        bank,
        extra_metadata={
            **scene_stats,
            "dataset": scene_config.dataset,
            "scene_sensor": scene_config.sensor,
            "scene_level": scene_config.level,
            "scene_gsd_m": float(scene_config.gsd_m),
            "scene_metadata_note": scene_config.metadata_note,
            "aux_grid_step_nm": float(auxiliary_grid_step_nm),
            "max_interpolation_gap_factor": float(max_interpolation_gap_factor),
            "max_source_step_nm": float(max_source_step_nm),
        },
    )
    return bank, scene_stats


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = GroundSpectralDataset(args.ground_root, normalization="none")
    scene_config = resolve_unmixing_scene_config(args.dataset)
    bank, bank_stats = build_for_scene(
        dataset,
        dataset_name=args.dataset,
        output_path=args.output_path,
        prototype_count=args.prototype_count,
        min_coverage=args.prototype_min_coverage,
        kmeans_iters=args.kmeans_iters,
        kmeans_batch_size=args.kmeans_batch_size,
        auxiliary_grid_step_nm=args.aux_grid_step_nm,
        max_interpolation_gap_factor=args.max_interpolation_gap_factor,
        max_source_step_nm=args.max_source_step_nm,
        seed=args.seed,
    )
    summary = {
        "dataset": scene_config.dataset,
        "scene_sensor": scene_config.sensor,
        "scene_level": scene_config.level,
        "scene_gsd_m": float(scene_config.gsd_m),
        "prototype_count": int(args.prototype_count),
        "prototype_min_coverage": float(args.prototype_min_coverage),
        "kmeans_iters": int(args.kmeans_iters),
        "kmeans_batch_size": int(args.kmeans_batch_size),
        "aux_grid_step_nm": float(args.aux_grid_step_nm),
        "max_interpolation_gap_factor": float(args.max_interpolation_gap_factor),
        "max_source_step_nm": float(args.max_source_step_nm),
        "seed": int(args.seed),
        "bank": {
            **summarize_bank(bank),
            **bank_stats,
            "path": str(args.output_path),
        },
    }
    summary_path = args.output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
