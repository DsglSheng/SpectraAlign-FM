#!/usr/bin/env bash
set -euo pipefail
if [ "${CONFIRM_RUN:-0}" != "1" ]; then
  echo 'Refusing to launch Stage C without CONFIRM_RUN=1.' >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT=${SPECTRAALIGN_DATA_ROOT:-${REPO_ROOT}/data}
PY=${PY:-python}
INIT=${INIT:-${DATA_ROOT}/checkpoints/neutral_warmup.pt}
MANIFEST=${DATA_ROOT}/manifests/hyperglobal_dz1_100k.jsonl
LEGACY=${HYPERGLOBAL_SHARDS:-${DATA_ROOT}/sigma450k_shards}
PROTO=${DATA_ROOT}/prototype_banks
DZPROTO=${DATA_ROOT}/prototype_banks
MAX_STEPS=${MAX_STEPS:-20000}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

run_one() {
  local name=$1 metadata=$2
  "${PY}" -m torch.distributed.run --nproc_per_node=4 scripts/train_foundation_dz1.py \
    --init-checkpoint "${INIT}" --reset-optimizer \
    --sigma-shard-root "${LEGACY}" --manifest-path "${MANIFEST}" \
    --prototype-bank-eo1 "${PROTO}/prototype_bank_eo1.pt" \
    --prototype-bank-gf5 "${PROTO}/prototype_bank_gf5.pt" \
    --prototype-bank-dz1-vnir "${DZPROTO}/prototype_bank_dz1_vnir.pt" \
    --prototype-bank-dz1-swir "${DZPROTO}/prototype_bank_dz1_swir.pt" \
    --output-dir "${ROOT}/runs/metadata_ablation/${name}" \
    --epochs 999 --max-steps "${MAX_STEPS}" --batch-size 112 \
    --num-workers 2 --prefetch-factor 1 --precision bf16 \
    --sensor-sampling-temperature 0.5 --space-crop-size 64 --space-crop-mode center \
    --metadata-ablation "${metadata}" --find-unused-parameters \
    --lr 2e-5 --min-lr 3e-6 --warmup-steps 1000 --warmup-steps-loss 0 \
    --proto-kl-weight 0.03 --proto-balance-weight 0.005 \
    --reconstruction-low-weight 0.10 --reconstruction-mid-weight 0.20 --reconstruction-full-weight 0.70 \
    --original-loss-weight 0.55 --sam-loss-weight 0.22 --charbonnier-loss-weight 0.13 \
    --derivative-loss-weight 0.06 --band-dropout-loss-weight 0.04 \
    --save-every-epochs 5 --probe-every-epochs 5 --seed 20260705
}

run_one full_metadata full
run_one no_sample_metadata no_sample
run_one no_band_metadata no_band
run_one no_metadata none
