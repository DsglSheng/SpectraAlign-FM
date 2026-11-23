from __future__ import annotations

import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from aware_open.data_audit import audit_400k, audit_sigma450k
from aware_open.recovery_400k import (
    apply_gsd_to_patch_metadata,
    build_recovery_manifest,
    extract_gsd_from_netcdf,
)
from aware_open.sigma450k_profiles import resolve_sigma450k_profile

try:
    import netCDF4 as nc
except ImportError:  # pragma: no cover - optional dependency
    nc = None

try:
    from scipy.io import savemat
except ImportError:  # pragma: no cover - optional dependency
    savemat = None


class DataToolsTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base = Path(__file__).resolve().parents[2] / ".tmp-tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / uuid.uuid4().hex
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_build_recovery_manifest_matches_source_file_stem(self) -> None:
        root = self.make_workspace_tempdir()
        patches_root = root / "400k" / "patches"
        scene_dir = patches_root / "AV320240420t194811_000_L1B_RDN_cbeae6f8_RDN"
        scene_dir.mkdir(parents=True)
        metadata_path = scene_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "source_file": "AV320240420t194811_000_L1B_RDN_cbeae6f8_RDN.nc",
                    "wavelength": [1.0],
                    "fwhm": [2.0],
                    "gsd": None,
                }
            ),
            encoding="utf-8",
        )

        flights_csv = root / "selected_flights.csv"
        with flights_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["flight_id", "sensor", "level", "sensor_level", "download_url"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "flight_id": "AV320240420t194811_000_L1B_RDN_cbeae6f8_RDN",
                    "sensor": "AVIRIS-3",
                    "level": "L1",
                    "sensor_level": "AVIRIS-3_L1",
                    "download_url": "",
                }
            )

        entries = build_recovery_manifest(patches_root, flights_csv)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].source_key, "AV320240420t194811_000_L1B_RDN_cbeae6f8_RDN")
        self.assertTrue(entries[0].download_url.endswith(".nc"))
        self.assertEqual(entries[0].expected_raw_name, "AV320240420t194811_000_L1B_RDN_cbeae6f8_RDN.nc")

    def test_apply_gsd_to_patch_metadata_records_provenance(self) -> None:
        root = self.make_workspace_tempdir()
        metadata_path = root / "metadata.json"
        metadata_path.write_text(json.dumps({"source_file": "scene.nc", "gsd": None}), encoding="utf-8")

        apply_gsd_to_patch_metadata(metadata_path, gsd=2.5, scene_path=Path("/tmp/scene.nc"))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["gsd"], 2.5)
        self.assertEqual(payload["gsd_source"], "raw_georeferencing")
        self.assertEqual(payload["gsd_scene_file"], "scene.nc")

    def test_audit_400k_counts_missing_gsd(self) -> None:
        root = self.make_workspace_tempdir() / "400k" / "patches" / "scene_a"
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "source_file": "AV320240420t194811_000_L1B_RDN_cbeae6f8_RDN.nc",
                    "wavelength": [1.0],
                    "fwhm": [2.0],
                    "gsd": None,
                }
            ),
            encoding="utf-8",
        )

        report = audit_400k(root.parents[1])
        self.assertEqual(report["scene_dirs"], 1)
        self.assertEqual(report["missing_gsd"], 1)
        self.assertEqual(report["sensor_counts"]["AVIRIS-3"], 1)

    def test_audit_sigma_detects_numeric_gf5_names(self) -> None:
        sigma_root = self.make_workspace_tempdir() / "sigma450k"
        eo1 = sigma_root / "EO1-part1"
        gf5 = sigma_root / "GF5-part1"
        eo1.mkdir(parents=True)
        gf5.mkdir(parents=True)
        (eo1 / "EO1H0010262012256110K9_1.mat").write_bytes(b"mat")
        (gf5 / "10000.mat").write_bytes(b"mat")

        report = audit_sigma450k(sigma_root)
        self.assertEqual(report["mat_files"], 2)
        self.assertEqual(report["sidecar_files"], 0)
        self.assertEqual(report["eo1_scene_id_like_files"], 1)
        self.assertEqual(report["gf5_numeric_only_files"], 1)

    def test_resolve_sigma450k_profile_matches_known_partition_layouts(self) -> None:
        eo1 = resolve_sigma450k_profile("EO1-part1", 175, file_stem="EO1H0010262012256110K9_1")
        gf5 = resolve_sigma450k_profile("GF5-part1", 150, file_stem="10000")

        self.assertEqual(eo1.profile_id, "eo1_hyperion_good175")
        self.assertEqual(eo1.sensor, "eo1")
        self.assertEqual(len(eo1.wavelengths_nm), 175)
        self.assertEqual(len(eo1.fwhm_nm), 175)
        self.assertEqual(eo1.gsd_m, 30.0)

        self.assertEqual(gf5.profile_id, "gf5_ahsi_vnir150")
        self.assertEqual(gf5.sensor, "gf5")
        self.assertEqual(len(gf5.wavelengths_nm), 150)
        self.assertEqual(len(gf5.fwhm_nm), 150)
        self.assertEqual(gf5.gsd_m, 30.0)

    @unittest.skipIf(savemat is None, "scipy is required for the sigma450k audit profile test.")
    def test_audit_sigma_resolves_profiles_from_real_mat_shapes(self) -> None:
        sigma_root = self.make_workspace_tempdir() / "sigma450k"
        eo1 = sigma_root / "EO1-part1"
        gf5 = sigma_root / "GF5-part1"
        eo1.mkdir(parents=True)
        gf5.mkdir(parents=True)

        savemat(eo1 / "EO1H0010262012256110K9_1.mat", {"img": np.zeros((175, 4, 4), dtype=np.float32)})
        savemat(gf5 / "10000.mat", {"img": np.zeros((150, 4, 4), dtype=np.float32)})

        report = audit_sigma450k(sigma_root)

        self.assertEqual(report["resolved_profiles"]["EO1-part1"], "eo1_hyperion_good175")
        self.assertEqual(report["resolved_profiles"]["GF5-part1"], "gf5_ahsi_vnir150")
        self.assertFalse(report["unresolved_partitions"])

    @unittest.skipIf(nc is None, "netCDF4 is required for the NetCDF GSD extraction test.")
    def test_extract_gsd_from_netcdf_coordinate_spacing(self) -> None:
        root = self.make_workspace_tempdir()
        scene_path = root / "scene.nc"
        dataset = nc.Dataset(scene_path, "w")
        try:
            dataset.createDimension("x", 4)
            dataset.createDimension("y", 3)
            x_var = dataset.createVariable("x", "f4", ("x",))
            y_var = dataset.createVariable("y", "f4", ("y",))
            x_var[:] = [500000.0, 500003.0, 500006.0, 500009.0]
            y_var[:] = [4100000.0, 4100003.0, 4100006.0]
        finally:
            dataset.close()

        self.assertAlmostEqual(extract_gsd_from_netcdf(scene_path), 3.0, places=5)
