#!/usr/bin/env python3
"""Export the Table 2 Aware-HSI Indian Pines branch without retraining."""

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
    / "result/ip_hypersigma_seg_proto_improve_20260520/results"
    / "ip_seg_proto_eo1_w015_seed3704_tta/ours_v3wa_kl003_full450k"
)
OUTPUT_DIR = PROJECT / "result/classification_visual_unified_20260628/Aware-HSI"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exporter.CLASSIFICATION_RUNS = {"IP": RESULT_DIR}
    exporter.CLASSIFICATION_CKPT["IP"] = "best_val_loss.pt"
    rows = exporter.export_classification_maps(OUTPUT_DIR, OUTPUT_DIR)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
