#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-/home/user/miniconda3/envs/aware/bin/python}"
CONFIG="${PROJECT_ROOT}/configs/downstream_tasks_v1.json"

"${PY}" - "${CONFIG}" "${PROJECT_ROOT}" <<'PY'
from pathlib import Path
import datetime
import json
import subprocess
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[2])
results_root = root / "results" / cfg["model_name"]
screen_out = subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
tasks = []
for task in cfg["tasks"]:
    out = results_root / task["id"]
    metrics = out / "test_metrics.json"
    if not metrics.exists():
        nested = sorted(out.glob("**/test_metrics.json"))
        metrics = nested[0] if nested else metrics
    tasks.append({
        "id": task["id"],
        "family": task["family"],
        "screen_running": f"dz1_ft_{task['id']}" in screen_out,
        "done": metrics.exists(),
        "metrics": str(metrics) if metrics.exists() else None,
    })
payload = {
    "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "project_root": str(root),
    "total_tasks": len(tasks),
    "done_tasks": sum(1 for task in tasks if task["done"]),
    "running_tasks": [task["id"] for task in tasks if task["screen_running"]],
    "tasks": tasks,
}
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
