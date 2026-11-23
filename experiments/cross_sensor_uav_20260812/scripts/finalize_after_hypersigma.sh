#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/Data/SpectralData/cross_sensor_uav_20260812
PY=/home/user/miniconda3/envs/aware/bin/python
RESULTS=(
  "${ROOT}/results/hypersigma_hc_seed3704/test_metrics.json"
)

while true; do
  ready=1
  for result in "${RESULTS[@]}"; do
    [[ -s "${result}" ]] || ready=0
  done
  [[ "${ready}" -eq 1 ]] && break
  sleep 60
done

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/mnt/Data/SpectralData/Aware_open:/mnt/Data/SpectralData/Aware_open/scripts:${ROOT}/code:${ROOT}/upstream_snapshot"
export HC_HYPERSIGMA_RUN="${ROOT}/results/hypersigma_hc_seed3704"
export SPECAWARE_SENTENCE_MODEL=/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/pretrained/sentence-transformers_all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for method in hypersigma; do
  "${PY}" "${ROOT}/code/export_hc_last_features.py" \
    --method "${method}" \
    --output "${ROOT}/features/${method}.npz" \
    > "${ROOT}/logs/feature_${method}.log" 2>&1
done

"${PY}" "${ROOT}/code/summarize_transfer_metrics.py" \
  --spec "${ROOT}/configs/transfer_metrics_sources.json" \
  --output "${ROOT}/reports/transfer_metrics.json" \
  > "${ROOT}/logs/summarize_transfer.log" 2>&1

"${PY}" "${ROOT}/code/plot_hc_feature_comparison.py" \
  --input-dir "${ROOT}/features" \
  --output "${ROOT}/reports/Figure_HanChuan_feature_comparison.png" \
  > "${ROOT}/logs/plot_features.log" 2>&1

"${PY}" - "${ROOT}/reports/finalize_status.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "metrics": str(path.parent / "transfer_metrics.json"),
    "figure_png": str(path.parent / "Figure_HanChuan_feature_comparison.png"),
    "figure_pdf": str(path.parent / "Figure_HanChuan_feature_comparison.pdf"),
}, indent=2) + "\n", encoding="utf-8")
PY
