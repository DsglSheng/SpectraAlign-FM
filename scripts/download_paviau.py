from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
import scipy.io as sio


DATA_SOURCES = {
    "PaviaU.mat": [
        "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "https://www.ehu.eus/ccwintco/uploads/2/26/PaviaU.mat",
    ],
    "PaviaU_gt.mat": [
        "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "https://www.ehu.eus/ccwintco/uploads/3/3b/PaviaU_gt.mat",
    ],
}

EXPECTED_KEYS = {
    "PaviaU.mat": "paviaU",
    "PaviaU_gt.mat": "paviaU_gt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mat(path: Path, expected_key: str) -> tuple[tuple[int, ...], str]:
    payload = sio.loadmat(path)
    if expected_key not in payload:
        keys = sorted(key for key in payload.keys() if not key.startswith("__"))
        raise ValueError(f"{path.name} is missing key {expected_key!r}. Available keys: {keys}")
    array = payload[expected_key]
    return tuple(int(dim) for dim in array.shape), str(array.dtype)


def stream_download(url: str, target: Path, timeout: float) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_with_fallbacks(filename: str, output_dir: Path, timeout: float, force: bool) -> None:
    target = output_dir / filename
    expected_key = EXPECTED_KEYS[filename]
    if target.exists() and not force:
        shape, dtype = validate_mat(target, expected_key)
        print(
            f"[skip] {filename} already exists, key={expected_key}, shape={shape}, dtype={dtype}, sha256={sha256_file(target)}"
        )
        return

    last_error: Exception | None = None
    tmp_path = target.with_suffix(target.suffix + ".part")
    for url in DATA_SOURCES[filename]:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"[download] {filename} <- {url}")
            stream_download(url, tmp_path, timeout=timeout)
            shape, dtype = validate_mat(tmp_path, expected_key)
            tmp_path.replace(target)
            print(
                f"[ok] {filename}: key={expected_key}, shape={shape}, dtype={dtype}, sha256={sha256_file(target)}"
            )
            return
        except Exception as exc:  # pragma: no cover - surface exact network or mat errors to the user
            last_error = exc
            print(f"[failed] {filename} <- {url}: {exc}", file=sys.stderr)
    raise RuntimeError(f"Failed to download {filename} from all configured sources.") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Pavia University hyperspectral dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive PaviaU.mat and PaviaU_gt.mat.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("PaviaU.mat", "PaviaU_gt.mat"):
        download_with_fallbacks(filename, args.output_dir, timeout=float(args.timeout), force=bool(args.force))


if __name__ == "__main__":
    main()
