#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/Data/SpectralData/cross_sensor_uav_20260812
PY=/home/user/miniconda3/envs/aware/bin/python
export PYTHONPATH="/mnt/Data/SpectralData/Aware_open:/mnt/Data/SpectralData/Aware_open/scripts:${ROOT}/code:${ROOT}/upstream_snapshot"
export SPECAWARE_SENTENCE_MODEL=/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/pretrained/sentence-transformers_all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
OUT=${ROOT}/class_activation
mkdir -p "${OUT}/scores" "${OUT}/cams" "${ROOT}/logs"

"${PY}" "${ROOT}/code/export_hc_class_activation.py" \
  --mode candidates --output "${OUT}/candidates.json" --per-class 12 \
  > "${ROOT}/logs/class_activation_candidates.log" 2>&1

run_score() {
  local gpu=$1 method=$2 batch=$3
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${ROOT}/code/export_hc_class_activation.py" \
    --mode score --method "${method}" --candidates "${OUT}/candidates.json" \
    --output "${OUT}/scores/scores_${method}.json" --batch-size "${batch}" \
    > "${ROOT}/logs/class_activation_score_${method}.log" 2>&1
}
run_score 0 ours 8 & p0=$!
run_score 0 spectralearth 8 & p1=$!
run_score 1 hypersigma 2 & p2=$!
run_score 2 specaware 8 & p3=$!
wait ${p0} ${p1} ${p2} ${p3}

"${PY}" "${ROOT}/code/select_hc_activation_samples.py" \
  --candidates "${OUT}/candidates.json" --scores-dir "${OUT}/scores" \
  --output "${OUT}/selected_samples.json" --count 4 \
  > "${ROOT}/logs/class_activation_select.log" 2>&1

run_cam() {
  local gpu=$1 method=$2 batch=$3
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${ROOT}/code/export_hc_class_activation.py" \
    --mode cam --method "${method}" --selected "${OUT}/selected_samples.json" \
    --output "${OUT}/cams/cam_${method}.npz" --window 24 --stride 8 --batch-size "${batch}" \
    > "${ROOT}/logs/class_activation_cam_${method}.log" 2>&1
}
run_cam 0 ours 8 & p0=$!
run_cam 0 spectralearth 8 & p1=$!
run_cam 1 hypersigma 2 & p2=$!
run_cam 2 specaware 8 & p3=$!
wait ${p0} ${p1} ${p2} ${p3}

"${PY}" "${ROOT}/code/plot_hc_class_activation.py" \
  --selected "${OUT}/selected_samples.json" --input-dir "${OUT}/cams" \
  --output "${ROOT}/reports/Figure_HanChuan_class_activation_comparison.png" \
  > "${ROOT}/logs/class_activation_plot.log" 2>&1

"${PY}" - "${OUT}/status.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({
  "status":"complete", "completed_utc":datetime.now(timezone.utc).isoformat(),
  "selected":str(p.parent/"selected_samples.json"),
  "figure_png":str(p.parents[1]/"reports/Figure_HanChuan_class_activation_comparison.png"),
  "figure_pdf":str(p.parents[1]/"reports/Figure_HanChuan_class_activation_comparison.pdf"),
}, indent=2)+"\n", encoding="utf-8")
PY
