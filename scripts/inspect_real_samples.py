from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

from aware_open.sigma450k_profiles import resolve_sigma450k_profile_from_path


DATA_ROOT = Path("/mnt/Data/SpectralData")


def inspect_ground() -> None:
    base = DATA_ROOT / "ground" / "native"
    json_path = sorted(base.rglob("*.json"))[0]
    npz_path = json_path.with_suffix(".npz")
    metadata = json.loads(json_path.read_text())
    arrays = np.load(npz_path)

    print("[ground]")
    print("json_path", json_path)
    print("json_keys", sorted(metadata.keys()))
    for key in arrays.files:
        array = arrays[key]
        print("npz", key, array.shape, array.dtype)


def inspect_400k() -> None:
    patch_root = sorted(path for path in (DATA_ROOT / "400k" / "patches").glob("*") if path.is_dir())[0]
    metadata_path = patch_root / "metadata.json"
    patch_path = sorted(patch_root.glob("*.npz"))[0]
    metadata = json.loads(metadata_path.read_text())
    arrays = np.load(patch_path)

    print("[400k]")
    print("patch_root", patch_root)
    print("metadata_keys", sorted(metadata.keys()))
    for key in arrays.files:
        array = arrays[key]
        print("npz", key, array.shape, array.dtype)


def inspect_sigma() -> None:
    mat_path = sorted((DATA_ROOT / "sigma450k").rglob("*.mat"))[0]
    print("[sigma450k]")
    print("mat_path", mat_path)
    print("whosmat", whosmat(mat_path))
    content = loadmat(mat_path)
    for key, value in content.items():
        if key.startswith("__"):
            continue
        print("mat", key, getattr(value, "shape", None), getattr(value, "dtype", None))
        if key == "img" and getattr(value, "shape", None) is not None and len(value.shape) == 3:
            profile = resolve_sigma450k_profile_from_path(mat_path, int(value.shape[0]))
            print("resolved_profile", profile.profile_id)
            print("resolved_sensor", profile.sensor_name)
            print("resolved_gsd_m", profile.gsd_m)
            print("resolved_wavelength_count", int(profile.wavelengths_nm.shape[0]))
            print("resolved_fwhm_count", None if profile.fwhm_nm is None else int(profile.fwhm_nm.shape[0]))


def main() -> None:
    inspect_ground()
    inspect_400k()
    inspect_sigma()


if __name__ == "__main__":
    main()
