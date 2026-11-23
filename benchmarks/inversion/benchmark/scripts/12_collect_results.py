from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED = "seed20260624"
DATASETS = (
    "greenhyperspectra_leaf_nitrogen",
    "clay_lithium_kaolinite_abundance",
    "karly_soil_moisture_256",
    "gloria_chlorophyll_256",
)
MODELS = ("spectralearth", "specaware", "spectralgpt", "aware", "hypersigma")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    summary: dict[str, Any] = {"datasets": {}, "complete": True}
    lines = ["# Inversion results status", ""]
    for dataset in DATASETS:
        dataset_rows: dict[str, Any] = {}
        lines.extend((f"## {dataset}", "", "| Model | Val RMSE | Test RMSE | MAE | R2 |", "|---|---:|---:|---:|---:|"))
        for model in MODELS:
            path = ROOT / "results" / dataset / model / SEED / "test_metrics.json"
            payload = load_json(path)
            if payload is None:
                if dataset in ("greenhyperspectra_leaf_nitrogen", "clay_lithium_kaolinite_abundance"):
                    continue
                summary["complete"] = False
                lines.append(f"| {model} | pending | pending | pending | pending |")
                continue
            test = payload["test"]
            dataset_rows[model] = payload
            lines.append(
                f"| {model} | {payload['best_val_rmse']:.8g} | {test['rmse']:.8g} | "
                f"{test['mae']:.8g} | {test['r2']:.6f} |"
            )

        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in ("fusion_ft", "fusion_frozen"):
            path = ROOT / "results_aware_fusion" / dataset / candidate / SEED / "test_metrics.json"
            payload = load_json(path)
            if payload is None:
                summary["complete"] = False
                continue
            candidates.append((float(payload["best_val_rmse"]), candidate, payload))
        if candidates:
            _, selected_name, selected_payload = min(candidates, key=lambda item: item[0])
            test = selected_payload["test"]
            dataset_rows["aware_fusion_selected_by_val"] = {
                "candidate": selected_name,
                **selected_payload,
            }
            lines.append(
                f"| Aware fusion ({selected_name}, val-selected) | "
                f"{selected_payload['best_val_rmse']:.8g} | {test['rmse']:.8g} | "
                f"{test['mae']:.8g} | {test['r2']:.6f} |"
            )
        else:
            lines.append("| Aware fusion (val-selected) | pending | pending | pending | pending |")
        lines.append("")
        summary["datasets"][dataset] = dataset_rows

    output_dir = ROOT / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inversion_results_20260624.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "inversion_results_20260624.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": summary["complete"], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
