from __future__ import annotations

import argparse
from pathlib import Path

from aware_open.recovery_400k import (
    backfill_manifest_entries,
    build_recovery_manifest,
    download_manifest_entries,
    load_manifest_csv,
    write_manifest_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict 400k GSD recovery and metadata backfill.")
    parser.add_argument(
        "--patches-root",
        type=Path,
        default=Path("/mnt/Data/SpectralData/400k/patches"),
        help="400k patch directory containing one subdirectory per source scene",
    )
    parser.add_argument(
        "--flights-csv",
        type=Path,
        default=Path("/mnt/Data/SpectralData/getdata/metadata/flight_metadata.csv"),
        help="Official AVIRIS flight metadata CSV covering the current 400k patch sources",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/Data/SpectralData/400k/raw_data"),
        help="Directory holding recovered raw scenes",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/400k_missing_gsd_manifest.csv"),
        help="Manifest CSV written by the manifest step",
    )
    parser.add_argument(
        "--getdata-root",
        type=Path,
        default=Path("/mnt/Data/SpectralData/getdata"),
        help="Root of the official getdata pipeline used for downloading",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("/mnt/Data/SpectralData/400k/raw_data/download_progress_missing_gsd.json"),
        help="Progress file for official resumable downloads",
    )
    parser.add_argument("--username", default="", help="Earthdata username override")
    parser.add_argument("--password", default="", help="Earthdata password override")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of scenes processed for validation")
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Chunk size passed through to the official resumable downloader",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Retry count passed through to the official resumable downloader",
    )
    parser.add_argument(
        "command",
        choices=("manifest", "download", "backfill", "all"),
        help="Recovery step to run",
    )
    return parser.parse_args()


def _slice_entries(entries, limit: int):
    if limit > 0:
        return entries[:limit]
    return entries


def main() -> int:
    args = parse_args()

    if args.command in {"manifest", "all"}:
        entries = build_recovery_manifest(args.patches_root, args.flights_csv)
        entries = _slice_entries(entries, args.limit)
        write_manifest_csv(entries, args.manifest)
        print(f"Manifest entries: {len(entries)}")
        print(f"Manifest path: {args.manifest}")
        if args.command == "manifest":
            return 0
    else:
        entries = load_manifest_csv(args.manifest)
        entries = _slice_entries(entries, args.limit)

    if args.command in {"download", "all"}:
        stats = download_manifest_entries(
            entries,
            getdata_root=args.getdata_root,
            raw_root=args.raw_root,
            progress_file=args.progress_file,
            username=args.username,
            password=args.password,
            chunk_size_mb=args.chunk_size_mb,
            retry_count=args.retry_count,
        )
        print(f"Downloaded: {stats['downloaded']}")
        print(f"Skipped existing: {stats['skipped_existing']}")
        if args.command == "download":
            return 0

    stats = backfill_manifest_entries(entries, args.raw_root)
    print(f"Backfilled metadata: {stats['updated']}")
    print(f"Missing raw scenes: {stats['missing_raw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
