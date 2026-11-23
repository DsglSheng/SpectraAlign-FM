#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT=${SPECTRAALIGN_DATA_ROOT:-${REPO_ROOT}/data}
LEGACY=${HYPERGLOBAL_SHARDS:-${DATA_ROOT}/sigma450k_shards}
DZ1_SHARDS=${DZ1_SHARDS:-${DATA_ROOT}/dz1_build/data/dz1_shards}
GROUND_ROOT=${GROUND_ROOT:-${DATA_ROOT}/ground}
PROTO_DIR=${DATA_ROOT}/prototype_banks
PY=${PY:-python}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PY}" scripts/build_joint_manifest.py \
  --hyperglobal-root "${LEGACY}" \
  --dz1-shard-root "${DZ1_SHARDS}" \
  --output-manifest "${DATA_ROOT}/manifests/hyperglobal_dz1_core.jsonl" \
  --subset-manifest "${DATA_ROOT}/manifests/hyperglobal_dz1_100k.jsonl" \
  --subset-count 100000 \
  --sampling-temperature 0.5 \
  --summary "${DATA_ROOT}/reports/joint_manifest_summary.json"

if [ ! -f "${PROTO_DIR}/prototype_bank_dz1_vnir.pt" ] || \
   [ ! -f "${PROTO_DIR}/prototype_bank_dz1_swir.pt" ]; then
  "${PY}" scripts/build_dz1_prototypes.py \
    --ground-root "${GROUND_ROOT}" \
    --output-dir "${PROTO_DIR}" \
    --prototype-count 256 \
    --seed 20260705
fi

"${PY}" scripts/smoke_test_joint.py \
  --legacy-root "${LEGACY}" \
  --joint-root "${DZ1_SHARDS}" \
  --joint-manifest "${DATA_ROOT}/manifests/hyperglobal_dz1_core.jsonl" \
  --report "${DATA_ROOT}/reports/mixed_forward_smoke.json" \
  --device "${SMOKE_DEVICE:-cpu}"
