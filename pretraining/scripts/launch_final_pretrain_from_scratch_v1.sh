#!/usr/bin/env bash
set -euo pipefail

if [ "${CONFIRM_FINAL_PRETRAIN:-0}" != "1" ]; then
  echo 'Refusing to launch the final run without CONFIRM_FINAL_PRETRAIN=1.' >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PY=${PY:-python}
export SPECTRAALIGN_DATA_ROOT="${SPECTRAALIGN_DATA_ROOT:-${REPO_ROOT}/data}"
CONFIG=${ROOT}/configs/final_pretrain_from_scratch_v1.json
RUN_ROOT=${ROOT}/runs/final_pretrain_hyperglobal_dz1_525k_fromscratch_v1
STAGE_A=${RUN_ROOT}/stage_a_neutral_joint10k
STAGE_B=${RUN_ROOT}/stage_b_full_joint154535
DZ1_SHARDS=${DZ1_SHARDS:-${SPECTRAALIGN_DATA_ROOT}/dz1_build/data/dz1_shards}
STAGE_A_STEPS=10035
STAGE_B_STEPS=154535

mkdir -p "${RUN_ROOT}/logs" "${STAGE_A}" "${STAGE_B}"
cd "${ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

wait_for_idle_gpus() {
  if [ "${WAIT_FOR_GPUS:-1}" != "1" ]; then return 0; fi
  local gpu_ids=${GPU_IDS:-${CUDA_VISIBLE_DEVICES}}
  local memory_limit=${GPU_MEM_LIMIT_MB:-1000}
  while true; do
    local busy=0
    mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    IFS=',' read -ra requested <<< "${gpu_ids}"
    for gpu in "${requested[@]}"; do
      used=$(printf '%s\n' "${rows[@]}" | awk -F',' -v id="${gpu}" '$1+0==id+0 {gsub(/ /,"",$2); print $2+0}')
      if [ -z "${used}" ] || [ "${used}" -ge "${memory_limit}" ]; then busy=1; fi
    done
    if [ "${busy}" -eq 0 ]; then return 0; fi
    echo "{\"status\":\"waiting_for_gpus\",\"gpu_ids\":\"${gpu_ids}\",\"memory_limit_mb\":${memory_limit}}"
    sleep 300
  done
}

"${PY}" scripts/preflight_final_pretrain.py \
  --config "${CONFIG}" \
  --report "${RUN_ROOT}/preflight.json" >/dev/null
mapfile -t training_inputs < <("${PY}" - "${RUN_ROOT}/preflight.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(report["manifest"])
for sensor in ("eo1", "gf5", "dz1_vnir", "dz1_swir"):
    print(report["prototype_banks"][sensor]["path"])
PY
)
wait_for_idle_gpus

latest_checkpoint() {
  local output=$1
  "${PY}" - "${output}" <<'PY'
from pathlib import Path
import re, sys
root=Path(sys.argv[1])/"checkpoints"
items=[]
for path in root.glob("epoch_*.pt") if root.exists() else []:
    match=re.fullmatch(r"epoch_(\d+)\.pt",path.name)
    if match: items.append((int(match.group(1)),path))
print(max(items)[1] if items else "")
PY
}

checkpoint_step() {
  "${PY}" - "$1" <<'PY'
import sys, torch
payload=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
print(int(payload.get("step",0)))
PY
}

write_completion() {
  local stage=$1 checkpoint=$2 target=$3 destination=$4
  "${PY}" - "${stage}" "${checkpoint}" "${target}" "${destination}" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
stage,checkpoint,target,destination=sys.argv[1],Path(sys.argv[2]),int(sys.argv[3]),Path(sys.argv[4])
payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
step=int(payload.get("step",0))
if step != target: raise SystemExit("checkpoint step {} != target {}".format(step,target))
h=hashlib.sha256()
with checkpoint.open("rb") as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
destination.write_text(json.dumps({"stage":stage,"step":step,"target_step":target,"checkpoint":str(checkpoint),"checkpoint_sha256":h.hexdigest(),"complete":True},indent=2)+"\n")
PY
}

COMMON=(
  --data-root "${SPECTRAALIGN_DATA_ROOT}"
  --sigma-shard-root "${DZ1_SHARDS}"
  --manifest-path "${training_inputs[0]}"
  --prototype-bank-eo1 "${training_inputs[1]}"
  --prototype-bank-gf5 "${training_inputs[2]}"
  --prototype-bank-dz1-vnir "${training_inputs[3]}"
  --prototype-bank-dz1-swir "${training_inputs[4]}"
  --epochs 999
  --batch-size 112 --num-workers 2 --prefetch-factor 1
  --probe-samples 512 --probe-batch-size 64 --probe-workers 0
  --progress-refresh-seconds 30 --grad-accum-steps 1 --grad-clip 1.0
  --precision bf16 --find-unused-parameters --seed 20260705
  --sensor-sampling-temperature 1.0
  --space-crop-size 64 --space-crop-mode center --space-normalization mean_std
  --weight-decay 0.05 --beta1 0.9 --beta2 0.95
  --model-dim 768 --band-dim 192 --context-dim 256 --spectral-dim 96 --spatial-dim 96
  --num-routing-prototypes 12 --prototype-heads 8 --prototype-depth 2
  --local-patch-size 8 --token-encoder-depth 12 --token-encoder-heads 12 --drop-rate 0.1
  --mask-ratio 0.75 --decoder-dim 384 --decoder-depth 2 --decoder-heads 6
  --decoder-dropout 0.0 --tdfa-depth 2 --frequency-groups 5
  --frequency-cutoff-low 0.15 --frequency-cutoff-mid 0.40
  --prototype-temperature-teacher 0.07 --prototype-temperature-student 0.07
  --warmup-steps-loss 0 --band-dropout-ratio 0.15 --spectral-loss-eps 1e-3
)

stage_a_latest=$(latest_checkpoint "${STAGE_A}")
stage_a_step=0
if [ -n "${stage_a_latest}" ]; then stage_a_step=$(checkpoint_step "${stage_a_latest}"); fi
if [ "${stage_a_step}" -lt "${STAGE_A_STEPS}" ]; then
  resume=()
  if [ -n "${stage_a_latest}" ]; then resume=(--resume "${stage_a_latest}"); fi
  "${PY}" -m torch.distributed.run --nproc_per_node=4 scripts/train_foundation_dz1.py \
    "${COMMON[@]}" "${resume[@]}" \
    --output-dir "${STAGE_A}" --max-steps "${STAGE_A_STEPS}" \
    --save-every-epochs 1 --probe-every-epochs 1 \
    --lr 3e-5 --min-lr 3e-6 --warmup-steps 1000 \
    --metadata-ablation none \
    --reconstruction-low-weight 0 --reconstruction-mid-weight 0 --reconstruction-full-weight 1 \
    --proto-kl-weight 0 --proto-balance-weight 0 \
    --original-loss-weight 1 --sam-loss-weight 0 --charbonnier-loss-weight 0 \
    --derivative-loss-weight 0 --band-dropout-loss-weight 0 \
    2>&1 | tee -a "${RUN_ROOT}/logs/stage_a.log"
  stage_a_latest=$(latest_checkpoint "${STAGE_A}")
fi
write_completion stage_a_neutral "${stage_a_latest}" "${STAGE_A_STEPS}" "${STAGE_A}/stage_completion.json"

stage_b_latest=$(latest_checkpoint "${STAGE_B}")
stage_b_step=0
if [ -n "${stage_b_latest}" ]; then stage_b_step=$(checkpoint_step "${stage_b_latest}"); fi
if [ "${stage_b_step}" -lt "${STAGE_B_STEPS}" ]; then
  start=(--init-checkpoint "${stage_a_latest}" --reset-optimizer)
  if [ -n "${stage_b_latest}" ]; then start=(--resume "${stage_b_latest}"); fi
  "${PY}" -m torch.distributed.run --nproc_per_node=4 scripts/train_foundation_dz1.py \
    "${COMMON[@]}" "${start[@]}" \
    --output-dir "${STAGE_B}" --max-steps "${STAGE_B_STEPS}" \
    --save-every-epochs 2 --probe-every-epochs 2 \
    --lr 2e-5 --min-lr 3e-6 --warmup-steps 1000 \
    --metadata-ablation full \
    --reconstruction-low-weight 0.10 --reconstruction-mid-weight 0.20 --reconstruction-full-weight 0.70 \
    --proto-kl-weight 0.03 --proto-balance-weight 0.005 \
    --original-loss-weight 0.55 --sam-loss-weight 0.22 --charbonnier-loss-weight 0.13 \
    --derivative-loss-weight 0.06 --band-dropout-loss-weight 0.04 \
    2>&1 | tee -a "${RUN_ROOT}/logs/stage_b.log"
  stage_b_latest=$(latest_checkpoint "${STAGE_B}")
fi
write_completion stage_b_full "${stage_b_latest}" "${STAGE_B_STEPS}" "${STAGE_B}/stage_completion.json"

echo '{"status":"complete","run":"final_pretrain_hyperglobal_dz1_525k_fromscratch_v1"}'
