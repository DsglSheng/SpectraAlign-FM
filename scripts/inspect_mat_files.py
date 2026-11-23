from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for path in sorted(args.root.rglob("*.mat")):
        print(f"--- {path} {path.stat().st_size}")
        try:
            payload = sio.loadmat(path)
        except Exception as exc:
            print(f"ERR {type(exc).__name__}: {exc}")
            continue
        for key, value in payload.items():
            if key.startswith("__"):
                continue
            array = np.asarray(value)
            if array.size:
                print(f"{key}: shape={array.shape} dtype={array.dtype} min={np.nanmin(array)} max={np.nanmax(array)}")
                if array.ndim <= 2 and array.size <= 1_000_000:
                    values, counts = np.unique(array, return_counts=True)
                    print(f"{key}: counts={dict(zip(values.tolist(), counts.tolist()))}")
            else:
                print(f"{key}: shape={array.shape} dtype={array.dtype}")


if __name__ == "__main__":
    main()
