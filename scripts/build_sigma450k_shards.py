from __future__ import annotations

import argparse
import json
from pathlib import Path

from aware_open.data.pretrain import build_sigma450k_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack sigma450k MAT files into contiguous shard files for training.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_sigma450k_shards(
        args.source_root,
        args.output_root,
        samples_per_shard=args.samples_per_shard,
        overwrite=args.overwrite,
    )
    payload = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        **summary,
    }
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
