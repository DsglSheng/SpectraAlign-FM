#!/usr/bin/env bash
set -euo pipefail
if [ "${CONFIRM_RUN:-0}" != "1" ]; then
  echo 'Refusing to launch Stage D without CONFIRM_RUN=1.' >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT=${SPECTRAALIGN_DATA_ROOT:-${REPO_ROOT}/data}
PY=${PY:-python}
INIT=${INIT:-${DATA_ROOT}/checkpoints/neutral_warmup.pt}
LEGACY=${HYPERGLOBAL_SHARDS:-${DATA_ROOT}/sigma450k_shards}
PROTO=${DATA_ROOT}/prototype_banks
DZPROTO=${DATA_ROOT}/prototype_banks
JOINT_MANIFEST=${DATA_ROOT}/manifests/hyperglobal_dz1_core.jsonl
MAX_STEPS=${MAX_STEPS:-154535}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

common=(
  --init-checkpoint "${INIT}" --reset-optimizer
  --sigma-shard-root "${LEGACY}"
  --prototype-bank-eo1 "${PROTO}/prototype_bank_eo1.pt"
  --prototype-bank-gf5 "${PROTO}/prototype_bank_gf5.pt"
  --epochs 999 --max-steps "${MAX_STEPS}"
  --batch-size 112 --num-workers 2 --prefetch-factor 1 --precision bf16
  --sensor-sampling-temperature 0.5 --space-crop-size 64 --space-crop-mode center
  --metadata-ablation full --lr 2e-5 --min-lr 3e-6 --warmup-steps 1000 --warmup-steps-loss 0
  --proto-kl-weight 0.03 --proto-balance-weight 0.005
  --reconstruction-low-weight 0.10 --reconstruction-mid-weight 0.20 --reconstruction-full-weight 0.70
  --original-loss-weight 0.55 --sam-loss-weight 0.22 --charbonnier-loss-weight 0.13
  --derivative-loss-weight 0.06 --band-dropout-loss-weight 0.04
  --save-every-epochs 5 --probe-every-epochs 5 --seed 20260705
)

"${PY}" -m torch.distributed.run --nproc_per_node=4 scripts/train_foundation_dz1.py \
  "${common[@]}" \
  --manifest-path "${LEGACY}/manifest.jsonl" \
  --output-dir "${ROOT}/runs/fair_pair/hyperglobal_control"

"${PY}" -m torch.distributed.run --nproc_per_node=4 scripts/train_foundation_dz1.py \
  "${common[@]}" \
  --manifest-path "${JOINT_MANIFEST}" \
  --prototype-bank-dz1-vnir "${DZPROTO}/prototype_bank_dz1_vnir.pt" \
  --prototype-bank-dz1-swir "${DZPROTO}/prototype_bank_dz1_swir.pt" \
  --output-dir "${ROOT}/runs/fair_pair/hyperglobal_plus_dz1"
