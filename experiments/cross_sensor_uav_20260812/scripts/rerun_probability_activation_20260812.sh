#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/Data/SpectralData/cross_sensor_uav_20260812
PY=/home/user/miniconda3/envs/aware/bin/python
export PYTHONPATH="/mnt/Data/SpectralData/Aware_open:/mnt/Data/SpectralData/Aware_open/scripts:${ROOT}/code:${ROOT}/upstream_snapshot"
export SPECAWARE_SENTENCE_MODEL=/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/pretrained/sentence-transformers_all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
run_cam() {
  CUDA_VISIBLE_DEVICES=$1 "${PY}" "${ROOT}/code/export_hc_class_activation.py" \
    --mode cam --method "$2" --selected "${ROOT}/class_activation/selected_samples.json" \
    --output "${ROOT}/class_activation/cams/cam_$2.npz" --window 24 --stride 8 --batch-size "$3" \
    > "${ROOT}/logs/class_activation_prob_$2.log" 2>&1
}
run_cam 0 ours 8 & a=$!
run_cam 0 spectralearth 8 & b=$!
run_cam 1 hypersigma 2 & c=$!
run_cam 2 specaware 8 & d=$!
wait ${a} ${b} ${c} ${d}
"${PY}" "${ROOT}/code/plot_hc_class_activation.py" \
  --selected "${ROOT}/class_activation/selected_samples.json" \
  --input-dir "${ROOT}/class_activation/cams" \
  --output "${ROOT}/reports/Figure_HanChuan_class_activation_comparison.png" \
  > "${ROOT}/logs/class_activation_prob_plot.log" 2>&1
echo complete > "${ROOT}/class_activation/status_prob.txt"
