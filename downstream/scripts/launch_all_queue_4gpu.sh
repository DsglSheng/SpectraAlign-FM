#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
CONFIG="${SCRIPT_DIR}/../configs/downstream_tasks_v1.json"

mapfile -t TASKS < <("${PY}" - "${CONFIG}" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1], encoding="utf-8"))
for task in cfg["tasks"]:
    print(f"{task['id']} {task.get('default_gpu', 0)}")
PY
)

for row in "${TASKS[@]}"; do
  task="${row%% *}"
  gpu="${row##* }"
  bash "${SCRIPT_DIR}/launch_task.sh" "${task}" "${gpu}"
done

screen -ls | grep -E 'dz1_ft_' || true
