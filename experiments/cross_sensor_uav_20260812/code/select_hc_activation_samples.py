from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("spectralearth", "hypersigma", "specaware", "ours")
PREFERRED = (1, 10, 13, 16, 6, 8, 14)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    spec = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = {v["id"]: v for v in spec["candidates"]}
    score_sets = {
        method: {v["id"]: v for v in json.loads((args.scores_dir / f"scores_{method}.json").read_text(encoding="utf-8"))["records"]}
        for method in METHODS
    }
    eligible = []
    for sample_id, item in candidates.items():
        rows = [score_sets[m][sample_id] for m in METHODS]
        if not all(v["correct"] for v in rows):
            continue
        entry = dict(item)
        entry["method_scores"] = {m: score_sets[m][sample_id] for m in METHODS}
        entry["min_target_probability"] = min(v["target_probability"] for v in rows)
        eligible.append(entry)
    chosen = []
    for class_id in PREFERRED + tuple(range(1, 17)):
        if any(v["class_id"] == class_id for v in chosen):
            continue
        pool = [v for v in eligible if v["class_id"] == class_id]
        if not pool:
            continue
        pool.sort(key=lambda v: (v["min_target_probability"], v["distance"], v["local_purity_33"]), reverse=True)
        chosen.append(pool[0])
        if len(chosen) >= args.count:
            break
    if len(chosen) < args.count:
        raise RuntimeError(f"Only {len(chosen)} jointly correct classes available")
    out = {"crop_size": int(spec["crop_size"]), "selection_rule": "jointly correct; one sample per class; preferred semantic classes; then max minimum target probability", "samples": chosen}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": [(v["class_name"], v["row"], v["col"]) for v in chosen]}))


if __name__ == "__main__":
    main()
