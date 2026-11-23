#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: launch_task.sh TASK_ID [GPU_ID] [--foreground] [extra run_task_from_config.py args...]}"
GPU="${2:-}"
MODE="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-python}"
RUNNER="${SCRIPT_DIR}/run_task_from_config.py"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ -z "${GPU}" ]]; then
  GPU="$("${PY}" - "${PROJECT_ROOT}/configs/downstream_tasks_v1.json" "${TASK}" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1], encoding="utf-8"))
for task in cfg["tasks"]:
    if task["id"] == sys.argv[2]:
        print(task.get("default_gpu", 0))
        raise SystemExit(0)
raise SystemExit(f"unknown task {sys.argv[2]}")
PY
)"
fi

SCREEN_NAME="dz1_ft_${TASK}"
LOG_FILE="${LOG_DIR}/${TASK}.log"
shift || true
shift || true
if [[ "${MODE}" == "--foreground" ]]; then
  shift || true
  export CUDA_VISIBLE_DEVICES="${GPU}"
  cd "${PROJECT_ROOT}"
  exec "${PY}" "${RUNNER}" --task "${TASK}" "$@" 2>&1 | tee -a "${LOG_FILE}"
fi

if screen -ls 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "{\"event\":\"skip_existing_screen\",\"screen\":\"${SCREEN_NAME}\",\"task\":\"${TASK}\"}"
  exit 0
fi

echo "{\"event\":\"launch_screen\",\"screen\":\"${SCREEN_NAME}\",\"task\":\"${TASK}\",\"gpu\":\"${GPU}\",\"log\":\"${LOG_FILE}\"}"
screen -dmS "${SCREEN_NAME}" bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES='${GPU}'
  cd '${PROJECT_ROOT}'
  '${PY}' '${RUNNER}' --task '${TASK}' \"\$@\" 2>&1 | tee -a '${LOG_FILE}'
" bash "$@"
