#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PY=${PY:-python}
DATA_ROOT=${SPECTRAALIGN_DATA_ROOT:-${REPO_ROOT}/data}
INPUT=${INPUT:-${DATA_ROOT}/raw_dz1}
LOG=${ROOT}/logs/prepare_full.log
classification_args=()
if [ -n "${CLASSIFICATION:-}" ]; then
  classification_args=(--classification-csv "${CLASSIFICATION}")
fi

mkdir -p "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PY}" scripts/prepare_dz1_dataset.py all \
  --input-root "${INPUT}" \
  --output-root "${DATA_ROOT}/dz1_build" \
  "${classification_args[@]}" \
  --cloud-cover-max 20 \
  --saturation-fraction-max 0.005 \
  --robust-scale-min 1e-6 \
  --samples-per-shard 256 \
  --vnir-mode context128 2>&1 | tee -a "${LOG}"
