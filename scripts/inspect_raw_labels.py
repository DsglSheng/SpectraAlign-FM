from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        for dtype in (np.int32, np.uint32, np.uint8):
            values = np.fromfile(path, dtype=dtype)
            unique, counts = np.unique(values, return_counts=True)
            summary = dict(zip(unique[:20].tolist(), counts[:20].tolist()))
            print(f"{path} dtype={dtype.__name__} n={values.size} counts_head={summary}")


if __name__ == "__main__":
    main()
