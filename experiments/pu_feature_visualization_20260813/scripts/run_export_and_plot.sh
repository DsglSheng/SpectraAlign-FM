#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/Data/SpectralData/pu_feature_visualization_20260813
PY=/home/user/miniconda3/envs/aware/bin/python
export PYTHONPATH="/mnt/Data/SpectralData/Aware_open:/mnt/Data/SpectralData/Aware_open/scripts:/mnt/Data/SpectralData/cross_sensor_uav_20260812/code:/mnt/Data/SpectralData/cross_sensor_uav_20260812/upstream_snapshot:/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/scripts:/mnt/Data/SpectralData/SpectralEarth_Aware_Comparison_20260608/scripts:${ROOT}/code"
export SPECAWARE_SENTENCE_MODEL=/mnt/Data/SpectralData/SpecAware_HyperSigma_Comparison_20260517/pretrained/sentence-transformers_all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "${ROOT}/features" "${ROOT}/reports" "${ROOT}/logs"
"${PY}" "${ROOT}/code/export_pu_features.py" --mode candidates --output "${ROOT}/features/candidates.json" > "${ROOT}/logs/candidates.log" 2>&1
run_export(){
  CUDA_VISIBLE_DEVICES=$1 "${PY}" "${ROOT}/code/export_pu_features.py" --mode export --method "$2" \
    --candidates "${ROOT}/features/candidates.json" --output "${ROOT}/features/$2.npz" --batch-size "$3" \
    > "${ROOT}/logs/export_$2.log" 2>&1
}
run_export 0 ours 4 & a=$!
run_export 0 spectralearth 4 & b=$!
run_export 1 hypersigma 2 & c=$!
run_export 2 specaware 4 & d=$!
wait $a $b $c $d
"${PY}" "${ROOT}/code/select_and_plot_pu_features.py" \
  --candidates "${ROOT}/features/candidates.json" --features-dir "${ROOT}/features" \
  --output "${ROOT}/reports/Figure_PU_cross_model_features.png" > "${ROOT}/logs/plot.log" 2>&1
echo complete > "${ROOT}/reports/status.txt"
