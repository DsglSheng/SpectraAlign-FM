#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUEUE_CONFIG="${PROJECT_ROOT}/configs/gpu_queues_20260708.json"
TASK_CONFIG="${PROJECT_ROOT}/configs/downstream_tasks_v1.json"
RUNNER="${SCRIPT_DIR}/run_task_from_config.py"
PY="${PY:-/home/user/miniconda3/envs/aware/bin/python}"
LOG_DIR="${PROJECT_ROOT}/logs"
STATUS_DIR="${PROJECT_ROOT}/queue_status"
PRETRAIN_STAGE="/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/runs/final_pretrain_hyperglobal_dz1_525k_fromscratch_v1/stage_b_full_joint154535"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

json_log() {
  local file="$1"
  shift
  "${PY}" - "$file" "$@" <<'PY'
from pathlib import Path
import datetime
import json
import sys

path = Path(sys.argv[1])
pairs = sys.argv[2:]
payload = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
for item in pairs:
    key, _, value = item.partition("=")
    payload[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
PY
}

screen_exists() {
  local name="$1"
  screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"
}

pretrain_complete() {
  "${PY}" - "${PRETRAIN_STAGE}" <<'PY'
from pathlib import Path
import json
import subprocess
import sys

stage = Path(sys.argv[1])
completion_path = stage / "stage_completion.json"
if completion_path.exists():
    try:
        payload = json.loads(completion_path.read_text(errors="ignore"))
        if payload.get("complete") is True:
            raise SystemExit(0)
    except Exception:
        pass
screen_out = subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
if "dz1_final_pretrain_fromscratch_v1" not in screen_out:
    log_path = stage / "train_log.jsonl"
    if log_path.exists():
        latest = None
        for line in log_path.read_text(errors="ignore").splitlines():
            idx = line.find("{")
            if idx >= 0:
                try:
                    latest = json.loads(line[idx:])
                except Exception:
                    pass
        if latest and int(latest.get("epoch", 0)) >= int(latest.get("epochs", 10**9)):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

select_checkpoint() {
  "${PY}" - "${PRETRAIN_STAGE}" <<'PY'
from pathlib import Path
import json
import sys

stage = Path(sys.argv[1])
checkpoints = stage / "checkpoints"
candidates = [checkpoints / "best_total.pt"]
completion_path = stage / "stage_completion.json"
if completion_path.exists():
    try:
        payload = json.loads(completion_path.read_text(errors="ignore"))
        checkpoint = payload.get("checkpoint")
        if checkpoint:
            candidates.append(Path(checkpoint))
    except Exception:
        pass
candidates.extend([checkpoints / "epoch_0132.pt", checkpoints / "latest.pt"])
for path in candidates:
    if path.exists():
        print(path)
        raise SystemExit(0)
raise SystemExit("no usable checkpoint found")
PY
}

prepare_compatible_checkpoint() {
  local raw_checkpoint="${1:?raw checkpoint required}"
  local compat_dir="${PROJECT_ROOT}/checkpoints"
  local compat_checkpoint="${compat_dir}/$(basename "${raw_checkpoint}" .pt)_downstream_legacy11.pt"
  local compat_report="${compat_dir}/$(basename "${raw_checkpoint}" .pt)_downstream_legacy11.json"
  mkdir -p "${compat_dir}"
  "${PY}" "${SCRIPT_DIR}/prepare_downstream_compatible_checkpoint.py" \
    --input "${raw_checkpoint}" \
    --output "${compat_checkpoint}" \
    --report "${compat_report}" >&2
  printf '%s\n' "${compat_checkpoint}"
}

wait_for_pretrain() {
  local status_file="${STATUS_DIR}/pretrain_wait.jsonl"
  while true; do
    if pretrain_complete; then
      local ckpt
      ckpt="$(select_checkpoint)"
      json_log "${status_file}" event=pretrain_ready checkpoint="${ckpt}" stage="${PRETRAIN_STAGE}" >&2
      printf '%s\n' "${ckpt}"
      return 0
    fi
    json_log "${status_file}" event=wait_pretrain stage="${PRETRAIN_STAGE}" >&2
    sleep "${PRETRAIN_WAIT_SECONDS:-300}"
  done
}

task_done() {
  local task="$1"
  local result_root
  result_root="$("${PY}" - "${TASK_CONFIG}" "${task}" <<'PY'
from pathlib import Path
import json
import os
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
task_id = sys.argv[2]
project = Path(os.environ.get("DZ1_DOWNSTREAM_ROOT", cfg["project_root"]))
model = os.environ.get("DZ1_DOWNSTREAM_MODEL_NAME", cfg["model_name"])
print(project / "results" / model / task_id)
PY
)"
  [[ -d "${result_root}" ]] && find "${result_root}" -name test_metrics.json -type f | grep -q .
}

queue_tasks_for_gpu() {
  local gpu="$1"
  "${PY}" - "${QUEUE_CONFIG}" "${gpu}" <<'PY'
import json
import sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
for task in cfg["queues"][str(sys.argv[2])]:
    print(task)
PY
}

run_gpu_queue() {
  local gpu="${1:?gpu required}"
  local checkpoint="${2:?checkpoint required}"
  local queue_log="${LOG_DIR}/gpu${gpu}_queue_20260708.log"
  local status_file="${STATUS_DIR}/gpu${gpu}.jsonl"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export AWARE_DZ1_CKPT="${checkpoint}"
  cd "${PROJECT_ROOT}"
  json_log "${status_file}" event=queue_start gpu="${gpu}" checkpoint="${checkpoint}" >> "${queue_log}"
  while IFS= read -r task; do
    [[ -n "${task}" ]] || continue
    if task_done "${task}"; then
      json_log "${status_file}" event=task_skip_done gpu="${gpu}" task="${task}" >> "${queue_log}"
      continue
    fi
    json_log "${status_file}" event=task_start gpu="${gpu}" task="${task}" checkpoint="${checkpoint}" >> "${queue_log}"
    set +e
    "${PY}" "${RUNNER}" --task "${task}" 2>&1 | tee -a "${queue_log}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -eq 0 ]]; then
      json_log "${status_file}" event=task_done gpu="${gpu}" task="${task}" status="${status}" >> "${queue_log}"
    else
      json_log "${status_file}" event=task_failed_continue gpu="${gpu}" task="${task}" status="${status}" >> "${queue_log}"
    fi
  done < <(queue_tasks_for_gpu "${gpu}")
  json_log "${status_file}" event=queue_done gpu="${gpu}" checkpoint="${checkpoint}" >> "${queue_log}"
}

submit_queues() {
  local raw_checkpoint checkpoint
  raw_checkpoint="$(wait_for_pretrain)"
  checkpoint="$(prepare_compatible_checkpoint "${raw_checkpoint}")"
  local master_status="${STATUS_DIR}/submit.jsonl"
  json_log "${master_status}" event=submit_queues checkpoint="${checkpoint}" raw_checkpoint="${raw_checkpoint}" config="${QUEUE_CONFIG}"
  for gpu in 0 1 2 3; do
    local screen_name="dz1_ft_queue_gpu${gpu}_20260708"
    if screen_exists "${screen_name}"; then
      json_log "${master_status}" event=skip_existing_queue_screen gpu="${gpu}" screen="${screen_name}"
      continue
    fi
    json_log "${master_status}" event=launch_queue gpu="${gpu}" screen="${screen_name}" checkpoint="${checkpoint}"
    screen -dmS "${screen_name}" bash -lc "
      set -euo pipefail
      '${BASH_SOURCE[0]}' run-gpu-queue '${gpu}' '${checkpoint}'
    "
  done
  screen -ls | grep -E 'dz1_ft_queue_gpu[0-3]_20260708' || true
}

case "${1:-submit}" in
  submit)
    submit_queues
    ;;
  run-gpu-queue)
    run_gpu_queue "${2:?gpu required}" "${3:?checkpoint required}"
    ;;
  plan)
    "${PY}" - "${QUEUE_CONFIG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(cfg, ensure_ascii=False, indent=2))
PY
    ;;
  checkpoint)
    if pretrain_complete; then select_checkpoint; else echo "PRETRAIN_NOT_COMPLETE"; exit 1; fi
    ;;
  *)
    echo "Usage: $0 {submit|run-gpu-queue GPU CHECKPOINT|plan|checkpoint}" >&2
    exit 2
    ;;
esac
