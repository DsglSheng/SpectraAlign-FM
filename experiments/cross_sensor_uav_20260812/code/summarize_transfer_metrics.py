from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def mean_iou(confusion: list[list[int]] | np.ndarray) -> tuple[float, list[float]]:
    cm = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    valid = union > 0
    per_class = np.full(tp.shape, np.nan, dtype=np.float64)
    per_class[valid] = tp[valid] / union[valid]
    return float(np.nanmean(per_class)), per_class.tolist()


def selected_metrics(path: Path, selector: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "confusion_matrix" in payload:
        return payload, "validation-selected checkpoint reported by source run"
    if selector not in payload:
        raise KeyError(f"{selector!r} not found in {path}")
    selected = payload[selector]
    if "test" not in selected:
        raise KeyError(f"No test metrics below {selector!r} in {path}")
    return selected["test"], f"{selector} at epoch {selected.get('epoch', 'unknown')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True, help="JSON list of method/path/selector entries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.spec.read_text(encoding="utf-8"))
    rows = []
    for spec in specs:
        metrics, selection = selected_metrics(Path(spec["path"]), spec.get("selector", "best_val_oa"))
        miou, per_class_iou = mean_iou(metrics["confusion_matrix"])
        rows.append(
            {
                "method": spec["method"],
                "oa": float(metrics["oa"]),
                "aa": float(metrics["aa"]),
                "kappa": float(metrics["kappa"]),
                "miou": miou,
                "per_class_accuracy": metrics.get("per_class_accuracy"),
                "per_class_iou": per_class_iou,
                "selection": selection,
                "source": str(spec["path"]),
            }
        )
    output = {
        "dataset": "WHU-Hi HanChuan",
        "transfer": "satellite-pretrained foundation model -> UAV hyperspectral scene",
        "protocol": {
            "seed": 3704,
            "train_per_class": 50,
            "validation_per_class": 20,
            "input_size": 128,
            "epochs": 80,
            "normalization": "mean_std",
            "selection": "validation metric only; test labels never used for checkpoint selection",
        },
        "metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
