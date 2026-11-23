from __future__ import annotations

import argparse
import json
from pathlib import Path

from aware_open.data_audit import audit_real_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit real hyperspectral dataset mirrors.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/Data/SpectralData"),
        help="Root directory containing ground, 400k, and sigma450k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_real_datasets(args.data_root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nWrote audit report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
