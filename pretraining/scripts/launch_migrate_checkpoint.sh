#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT=${SPECTRAALIGN_DATA_ROOT:-${REPO_ROOT}/data}
PY=${PY:-python}
INPUT=${INPUT:-${DATA_ROOT}/checkpoints/neutral_warmup.pt}
OUTPUT=${OUTPUT:-${ROOT}/artifacts/checkpoints/neutral_epoch0045_dz1.pt}
REPORT=${REPORT:-${ROOT}/reports/checkpoint_migration.json}
DZPROTO=${DZPROTO:-${DATA_ROOT}/prototype_banks}

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${REPORT}")" "${ROOT}/logs"
cd "${ROOT}"
exec "${PY}" scripts/migrate_checkpoint.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --report "${REPORT}" \
  --target-sensor-count 13 \
  --others-index 10 \
  --prototype-bank-dz1-vnir "${DZPROTO}/prototype_bank_dz1_vnir.pt" \
  --prototype-bank-dz1-swir "${DZPROTO}/prototype_bank_dz1_swir.pt"
