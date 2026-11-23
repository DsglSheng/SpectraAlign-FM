#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "downstream_tasks_v1.json"


SKIP_CONFIG_KEYS = {
    "implementation_note",
    "protocol_note",
    "image_info",
    "gt_info",
    "class_names",
    "split_counts",
    "class_weights",
    "baselines",
    "init_model_info",
}

DROP_OLD_INIT_KEYS = {"init_model", "resume_from"}

BOOL_FALSE_FLAGS = {
    "train_drop_last": "--no-train-drop-last",
    "eval_drop_last": "--no-eval-drop-last",
}


def load_config(path: Path) -> dict[str, Any]:
    roots = {
        "REPO_ROOT": str(REPO_ROOT),
        "DOWNSTREAM_ROOT": str(PROJECT_ROOT),
        "SPECTRAALIGN_DATA_ROOT": str(
            Path(os.environ.get("SPECTRAALIGN_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
        ),
    }

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            return Template(value).safe_substitute(roots)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(json.loads(path.read_text(encoding="utf-8")))


def load_registry() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


def tasks_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["id"]: task for task in registry["tasks"]}


def resolve_checkpoint(registry: dict[str, Any], explicit: str | None, allow_missing: bool) -> Path:
    candidates: list[Path] = []
    for value in (
        explicit,
        os.environ.get("AWARE_DZ1_CKPT"),
        registry.get("default_checkpoint"),
        registry.get("fallback_checkpoint"),
    ):
        if value:
            candidates.append(Path(value))
    for path in candidates:
        if path.exists():
            return path
    if allow_missing:
        return candidates[0]
    raise FileNotFoundError(
        "No DZ1 final-pretrain checkpoint is available yet. "
        "Set AWARE_DZ1_CKPT or wait for Stage B completion. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def output_dir_for(registry: dict[str, Any], task: dict[str, Any]) -> Path:
    project = Path(os.environ.get("DZ1_DOWNSTREAM_ROOT", registry["project_root"]))
    model_name = os.environ.get("DZ1_DOWNSTREAM_MODEL_NAME", registry["model_name"])
    return project / "results" / model_name / task["id"]


def script_env(registry: dict[str, Any]) -> dict[str, str]:
    aware_root = Path(os.environ.get("AWARE_ROOT", registry["aware_root"]))
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(aware_root), str(aware_root / "scripts")]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    return env


def load_old_args(config_path: Path) -> dict[str, Any]:
    payload = load_config(config_path)
    if isinstance(payload.get("args"), dict):
        return dict(payload["args"])
    return dict(payload)


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_arg(argv: list[str], key: str, value: Any) -> None:
    if key in SKIP_CONFIG_KEYS or value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag_name(key))
        elif key in BOOL_FALSE_FLAGS:
            argv.append(BOOL_FALSE_FLAGS[key])
        return
    if isinstance(value, (list, tuple)):
        argv.append(flag_name(key))
        argv.extend(str(item) for item in value)
        return
    if isinstance(value, dict):
        return
    argv.extend([flag_name(key), str(value)])


def build_config_cli(
    registry: dict[str, Any],
    task: dict[str, Any],
    checkpoint: Path,
    *,
    allow_old_init: bool,
) -> list[str]:
    old_config = Path(task["old_config"])
    cfg = load_old_args(old_config)
    if "aware_checkpoint" in cfg:
        cfg["aware_checkpoint"] = str(checkpoint)
    if "checkpoint" in cfg:
        cfg["checkpoint"] = str(checkpoint)
    if "prototype_bank_eo1" in cfg:
        cfg["prototype_bank_eo1"] = os.environ.get("PROTO_EO1", registry["prototype_bank_eo1"])
    if "prototype_bank_gf5" in cfg:
        cfg["prototype_bank_gf5"] = os.environ.get("PROTO_GF5", registry["prototype_bank_gf5"])
    cfg["output_dir"] = str(output_dir_for(registry, task))
    if not allow_old_init:
        for key in DROP_OLD_INIT_KEYS:
            cfg.pop(key, None)
    argv = [sys.executable, task["script"]]
    for key in sorted(cfg):
        append_arg(argv, key, cfg[key])
    return argv


def inversion_root() -> Path:
    return REPO_ROOT / "benchmarks" / "inversion" / "benchmark"


def build_extract_features_cmd(registry: dict[str, Any], dataset_npz: str, output_npz: Path, checkpoint: Path) -> list[str]:
    root = inversion_root()
    return [
        sys.executable,
        str(root / "scripts/16_extract_aware_multiview_features.py"),
        "--dataset-npz",
        dataset_npz,
        "--output-npz",
        str(output_npz),
        "--batch-size",
        "32",
        "--num-workers",
        "2",
        "--precision",
        "bf16",
        "--aware-checkpoint",
        str(checkpoint),
        "--prototype-bank-eo1",
        os.environ.get("PROTO_EO1", registry["prototype_bank_eo1"]),
        "--prototype-bank-gf5",
        os.environ.get("PROTO_GF5", registry["prototype_bank_gf5"]),
    ]


def build_inversion_commands(registry: dict[str, Any], task: dict[str, Any], checkpoint: Path) -> list[list[str]]:
    root = inversion_root()
    out = output_dir_for(registry, task)
    runner = task["runner"]
    if runner == "inversion_multiview":
        features = out / "aware_multiview_features.npz"
        return [
            build_extract_features_cmd(registry, task["dataset_npz"], features, checkpoint),
            [
                sys.executable,
                str(root / "scripts/17_train_aware_multiview_ensemble.py"),
                "--dataset-npz",
                task["dataset_npz"],
                "--features-npz",
                str(features),
                "--output-dir",
                str(out),
                "--epochs",
                "320",
                "--patience",
                "45",
                "--batch-size",
                "64",
                "--num-workers",
                "2",
                "--precision",
                "bf16",
            ],
        ]
    if runner == "inversion_domain_specialist":
        branch = task["branch"]
        aware_features = out / "aware_multiview_features.npz"
        domain_features = out / "domain_features.npz"
        return [
            build_extract_features_cmd(registry, task["dataset_npz"], aware_features, checkpoint),
            [
                sys.executable,
                str(root / "scripts/19_prepare_domain_features.py"),
                "--dataset",
                branch,
                "--dataset-npz",
                task["dataset_npz"],
                "--output-npz",
                str(domain_features),
            ],
            [
                sys.executable,
                str(root / "scripts/20_train_aware_domain_specialist.py"),
                "--branch",
                branch,
                "--dataset-npz",
                task["dataset_npz"],
                "--aware-features-npz",
                str(aware_features),
                "--domain-features-npz",
                str(domain_features),
                "--output-dir",
                str(out),
                "--epochs",
                "360",
                "--patience",
                "55",
                "--batch-size",
                "64",
                "--num-workers",
                "2",
                "--precision",
                "bf16",
            ],
        ]
    if runner == "inversion_true_fullft":
        seeds = [str(seed) for seed in task["seeds"]]
        return [
            [
                sys.executable,
                str(root / "scripts/26_train_aware_true_fullft.py"),
                "--mode",
                task["mode"],
                "--dataset-npz",
                task["dataset_npz"],
                "--domain-features-npz",
                task["domain_features_npz"],
                "--groups-npz",
                task["groups_npz"],
                "--raw-root",
                task["raw_root"],
                "--output-dir",
                str(out),
                "--batch-size",
                "16",
                "--num-workers",
                "2",
                "--head-lr",
                str(task["head_lr"]),
                "--adapter-lr",
                str(task["adapter_lr"]),
                "--backbone-lr",
                str(task["backbone_lr"]),
                "--decoder-lr",
                str(task["decoder_lr"]),
                "--weight-decay",
                "0.015",
                "--dropout",
                str(task["dropout"]),
                "--precision",
                "bf16",
                "--seeds",
                *seeds,
                "--checkpoint",
                str(checkpoint),
                "--prototype-bank-eo1",
                os.environ.get("PROTO_EO1", registry["prototype_bank_eo1"]),
                "--prototype-bank-gf5",
                os.environ.get("PROTO_GF5", registry["prototype_bank_gf5"]),
            ]
        ]
    raise ValueError(f"Unknown inversion runner: {runner}")


def build_commands(
    registry: dict[str, Any],
    task: dict[str, Any],
    checkpoint: Path,
    *,
    allow_old_init: bool,
) -> list[list[str]]:
    if task["runner"] == "config_cli":
        return [build_config_cli(registry, task, checkpoint, allow_old_init=allow_old_init)]
    return build_inversion_commands(registry, task, checkpoint)


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def check_registry(registry: dict[str, Any], *, allow_missing_checkpoint: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        ckpt = resolve_checkpoint(registry, None, allow_missing_checkpoint)
    except Exception as exc:  # noqa: BLE001
        ckpt = None
        errors.append(str(exc))
    for task in registry["tasks"]:
        if task["runner"] == "config_cli":
            for key in ("script", "old_config"):
                path = Path(task[key])
                if not path.exists():
                    errors.append(f"{task['id']}: missing {key}: {path}")
        for key in ("old_result", "dataset_npz", "domain_features_npz", "groups_npz", "raw_root"):
            if key in task and not Path(str(task[key])).exists():
                warnings.append(f"{task['id']}: missing optional/reference {key}: {task[key]}")
    return {
        "task_count": len(registry["tasks"]),
        "checkpoint": str(ckpt) if ckpt else None,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prepared DZ1 downstream fine-tuning task.")
    parser.add_argument("--task", help="Task id to run.")
    parser.add_argument("--list", action="store_true", help="List prepared task ids.")
    parser.add_argument("--check", action="store_true", help="Validate scripts/config paths without training.")
    parser.add_argument("--checkpoint", help="Override DZ1 pre-trained checkpoint.")
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    parser.add_argument("--allow-old-init", action="store_true", help="Allow old downstream init_model weights; diagnostic only.")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = load_registry()
    task_map = tasks_by_id(registry)
    if args.list:
        print(json.dumps([
            {
                "id": task["id"],
                "family": task["family"],
                "display_name": task["display_name"],
                "default_gpu": task["default_gpu"],
                "runner": task["runner"],
            }
            for task in registry["tasks"]
        ], ensure_ascii=False, indent=2))
        return 0
    if args.check:
        report = check_registry(registry, allow_missing_checkpoint=args.allow_missing_checkpoint)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if not args.task:
        parser.error("--task is required unless --list or --check is used")
    if args.task not in task_map:
        raise KeyError(f"Unknown task {args.task!r}. Use --list.")
    checkpoint = resolve_checkpoint(
        registry,
        args.checkpoint,
        allow_missing=args.allow_missing_checkpoint or args.print_command or args.dry_run,
    )
    task = task_map[args.task]
    out = output_dir_for(registry, task)
    commands = build_commands(registry, task, checkpoint, allow_old_init=args.allow_old_init)
    if args.print_command or args.dry_run:
        print(json.dumps({
            "task": task["id"],
            "display_name": task["display_name"],
            "output_dir": str(out),
            "checkpoint": str(checkpoint),
            "commands": [shell_join(command) for command in commands],
            "allow_old_init": args.allow_old_init,
        }, ensure_ascii=False, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    env = script_env(registry)
    print(json.dumps({
        "event": "start_task",
        "task": task["id"],
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "allow_old_init": args.allow_old_init,
    }, ensure_ascii=False), flush=True)
    for command in commands:
        print("[cmd] " + shell_join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    print(json.dumps({"event": "done_task", "task": task["id"], "output_dir": str(out)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
