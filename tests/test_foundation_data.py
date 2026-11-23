from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from aware_open.data.pretrain import GroundSpectralDataset
from aware_open.sigma450k_profiles import build_gf5_ahsi_vnir150_profile
from aware_open.spectral_prototypes import (
    build_auxiliary_wavelength_grid,
    gaussian_srf_matrix,
    load_prototype_bank,
)
from scripts.build_ground_prototypes import build_for_sensor, collect_resampled_spectra


class FoundationDataTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base = Path(__file__).resolve().parents[2] / ".tmp-tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / uuid.uuid4().hex
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_ground_dataset_treats_zero_quality_code_as_good(self) -> None:
        root = self.make_workspace_tempdir() / "ground" / "native" / "library"
        root.mkdir(parents=True)
        (root / "sample.json").write_text(json.dumps({"measurement_date": "2024-04-10"}), encoding="utf-8")
        np.savez(
            root / "sample.npz",
            reflectance=np.linspace(0.1, 0.9, 4, dtype=np.float32),
            wavelengths=np.linspace(400.0, 700.0, 4, dtype=np.float32),
            quality_mask=np.asarray([0, 1, 0, 1], dtype=np.uint8),
            instrument_fwhm=np.full(4, 5.0, dtype=np.float32),
        )

        dataset = GroundSpectralDataset(root.parents[1], normalization="none")
        sample = dataset[0]

        self.assertEqual(sample.valid_mask.tolist(), [True, False, True, False])
        self.assertEqual(sample.fwhm.shape[0], 4)
        self.assertEqual(sample.metadata.sensor, "field")
        self.assertEqual(sample.metadata.level, "field")

    def test_build_for_sensor_writes_frozen_eo1_and_gf5_banks(self) -> None:
        ground_root = self.make_workspace_tempdir() / "ground" / "native"
        output_root = self.make_workspace_tempdir() / "banks"
        for index in range(6):
            sample_root = ground_root / f"library_{index:02d}"
            sample_root.mkdir(parents=True)
            (sample_root / f"sample_{index:02d}.json").write_text(json.dumps({}), encoding="utf-8")
            wavelengths = np.linspace(380.0, 2500.0, 2501, dtype=np.float32)
            reflectance = np.sin(wavelengths / 120.0 + index).astype(np.float32) + index * 0.01
            np.savez(
                sample_root / f"sample_{index:02d}.npz",
                reflectance=reflectance,
                wavelengths=wavelengths,
                quality_mask=np.zeros(2501, dtype=np.uint8),
                instrument_fwhm=np.full(2501, 3.0, dtype=np.float32),
            )

        dataset = GroundSpectralDataset(ground_root.parent, normalization="none")

        eo1_path = output_root / "prototype_bank_eo1.pt"
        gf5_path = output_root / "prototype_bank_gf5.pt"
        eo1_bank = build_for_sensor(
            dataset,
            sensor="eo1",
            output_path=eo1_path,
            prototype_count=2,
            min_coverage=0.95,
            kmeans_iters=3,
            kmeans_batch_size=16,
            auxiliary_grid_step_nm=1.0,
            max_interpolation_gap_factor=4.0,
            max_source_step_nm=5.0,
            seed=7,
        )
        gf5_bank = build_for_sensor(
            dataset,
            sensor="gf5",
            output_path=gf5_path,
            prototype_count=2,
            min_coverage=0.95,
            kmeans_iters=3,
            kmeans_batch_size=16,
            auxiliary_grid_step_nm=1.0,
            max_interpolation_gap_factor=4.0,
            max_source_step_nm=2.0,
            seed=7,
        )

        self.assertTrue(eo1_path.exists())
        self.assertTrue(gf5_path.exists())
        self.assertEqual(eo1_bank[0].centers.shape, (2, 175))
        self.assertEqual(gf5_bank[0].centers.shape, (2, 150))
        self.assertEqual(load_prototype_bank(eo1_path).centers.shape, (2, 175))
        self.assertEqual(load_prototype_bank(gf5_path).centers.shape, (2, 150))

    def test_collect_resampled_spectra_filters_coarse_sampling(self) -> None:
        ground_root = self.make_workspace_tempdir() / "ground" / "native"
        fine_root = ground_root / "fine"
        coarse_root = ground_root / "coarse"
        fine_root.mkdir(parents=True)
        coarse_root.mkdir(parents=True)

        fine_wavelengths = np.linspace(380.0, 2500.0, 2501, dtype=np.float32)
        coarse_wavelengths = np.linspace(380.0, 2500.0, 200, dtype=np.float32)
        for index in range(2):
            (fine_root / f"fine_{index}.json").write_text(json.dumps({}), encoding="utf-8")
            np.savez(
                fine_root / f"fine_{index}.npz",
                reflectance=np.cos(fine_wavelengths / 140.0 + index).astype(np.float32),
                wavelengths=fine_wavelengths,
                quality_mask=np.zeros(fine_wavelengths.shape[0], dtype=np.uint8),
            )
        (coarse_root / "coarse.json").write_text(json.dumps({}), encoding="utf-8")
        np.savez(
            coarse_root / "coarse.npz",
            reflectance=np.cos(coarse_wavelengths / 140.0).astype(np.float32),
            wavelengths=coarse_wavelengths,
            quality_mask=np.zeros(coarse_wavelengths.shape[0], dtype=np.uint8),
        )

        dataset = GroundSpectralDataset(ground_root.parent, normalization="none")
        profile = build_gf5_ahsi_vnir150_profile()
        target_wavelengths = torch.from_numpy(profile.wavelengths_nm.astype(np.float32))
        target_fwhm = torch.from_numpy(profile.fwhm_nm.astype(np.float32))
        auxiliary_grid = build_auxiliary_wavelength_grid(target_wavelengths, target_fwhm, step_nm=1.0)
        target_srf_weights = gaussian_srf_matrix(auxiliary_grid, target_wavelengths, target_fwhm)

        spectra, stats = collect_resampled_spectra(
            dataset,
            auxiliary_wavelengths_nm=auxiliary_grid,
            target_srf_weights=target_srf_weights,
            target_wavelengths_nm=target_wavelengths,
            min_coverage=0.95,
            max_source_step_nm=2.0,
            max_interpolation_gap_factor=4.0,
        )

        self.assertEqual(spectra.shape[0], 2)
        self.assertEqual(stats["accepted_samples"], 2)
        self.assertEqual(stats["rejected_coarse_step"], 1)
        self.assertLessEqual(stats["accepted_source_step_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
