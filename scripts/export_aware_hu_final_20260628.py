#!/usr/bin/env python3
"""Export the Table 2 Aware-HSI Houston branch without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path("/mnt/Data/SpectralData/Aware_open")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import make_hypersigma_style_single_assets_unified_20260628 as exporter  # noqa: E402


RESULT_DIR = (
    PROJECT
    / "result/latest200e_remaining_improve_20260531/results"
    / "hu_latest200e_inv_f192_lr4e5_e240_rep6_seed20260531/ours_v3wa_kl003_full450k"
)
OUTPUT_DIR = PROJECT / "result/classification_visual_unified_20260628/Aware-HSI"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exporter.CLASSIFICATION_RUNS = {"HU": RESULT_DIR}
    exporter.CLASSIFICATION_CKPT["HU"] = "latest_classification.pt"
    rows = exporter.export_classification_maps(OUTPUT_DIR, OUTPUT_DIR)
    (OUTPUT_DIR / "manifest_hu.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
