"""合并 V18.2 JSONL，并以 validation GT p95 定义可实现阈值。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--event-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); val, test = read_rows(args.val), read_rows(args.test)
    event_group = {int(row["event_index"]): row["group"]
                   for row in json.loads(args.event_audit.read_text())}
    thresholds = {}
    for mode in ("r", "rd", "rdu"):
        values = [row["final_rmse_cm"] for row in val if row["target"] == "gt"
                  and row["init"] == "current_repeat" and row["loss_mode"] == mode]
        if not values: raise ValueError(f"validation 缺少 {mode} 的 GT current_repeat")
        thresholds[mode] = float(np.quantile(values, .95))
    by_sample = defaultdict(dict)
    for row in test:
        if row["target"] == "generated":
            by_sample[(row["sample_index"], row["loss_mode"])][row["init"]] = row
    result = {"val_rows": len(val), "test_rows": len(test), "threshold_p95_cm": thresholds,
              "groups": {}}
    for mode in ("r", "rd", "rdu"):
        result["groups"][mode] = {}
        for stage_name in ("overall", "formation", "transition", "maintenance"):
            for hand_group in ("all", "single_hand_clean", "bilateral"):
                counts = {"directly_realizable": 0, "basin_rescued": 0, "persistent_gap": 0}
                component_values = defaultdict(list); total = 0
                for (sample_index, sample_mode), rows in by_sample.items():
                    if sample_mode != mode or set(rows) != {"current_repeat", "gt_future", "multistart"}: continue
                    current = rows["current_repeat"]
                    if stage_name != "overall" and current["stage"] != stage_name: continue
                    if hand_group != "all" and event_group[current["event_index"]] != hand_group: continue
                    total += 1; threshold = thresholds[mode]
                    if current["final_rmse_cm"] <= threshold: category = "directly_realizable"
                    elif min(rows["gt_future"]["final_rmse_cm"], rows["multistart"]["final_rmse_cm"]) <= threshold:
                        category = "basin_rescued"
                    else: category = "persistent_gap"
                    counts[category] += 1
                    for init, row in rows.items():
                        for metric in ("final_rmse_cm", "r_rmse_cm", "d_rmse_cm", "u_rmse_cm", "contact_f1"):
                            component_values[f"{init}_{metric}"].append(row[metric])
                key = f"{stage_name}/{hand_group}"
                result["groups"][mode][key] = {
                    "samples": total, "counts": counts,
                    "rates": {name: value / max(total, 1) for name, value in counts.items()},
                    "means": {name: float(np.mean(values)) for name, values in component_values.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
