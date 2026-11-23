#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "downstream_tasks_v1.json"
TASK_DIR = ROOT / "scripts" / "tasks"


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    for task in cfg["tasks"]:
        path = TASK_DIR / f"{task['id']}.sh"
        gpu = task.get("default_gpu", 0)
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
            f"exec \"${{DIR}}/launch_task.sh\" \"{task['id']}\" \"${{1:-{gpu}}}\" \"${{@:2}}\"\n",
            encoding="utf-8",
        )
    readme = TASK_DIR / "README.md"
    readme.write_text(
        "# Per-dataset launch wrappers\n\n"
        "Each `*.sh` file starts one dataset in a detached `screen` session by default.\n"
        "Pass a GPU id as the first argument, for example `bash classification_hc.sh 2`.\n"
        "Pass `--foreground` as the second argument to run without screen.\n",
        encoding="utf-8",
    )
    print(json.dumps({"generated": len(cfg["tasks"]), "task_dir": str(TASK_DIR)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
