"""从 V18.1 checkpoint 的 validation 指标提取非支配候选。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def dominates(left: dict, right: dict) -> bool:
    # RMSE 与 contact calibration 越低越好，stable success 越高越好。
    a = (left["rmse"], left["contact_error"], -left["stable"])
    b = (right["rmse"], right["contact_error"], -right["stable"])
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); rows = []
    for path in sorted(args.directory.glob("epoch_*.pt")):
        checkpoint = torch.load(path, map_location="cpu")
        metrics = checkpoint["val"]; overall = metrics["subsets"]["overall"]
        prediction = overall["prediction_terminal_contact"]
        target = overall["gt_terminal_contact"]
        calibration = {key: abs(prediction[key] - target[key])
                       for key in ("mean", "median", "p10", "p90")}
        rows.append({"path": str(path), "epoch": checkpoint["epoch"],
                     "rmse": overall["single_rmse_cm"],
                     "stable": overall["diffusion_stable_rate"],
                     "contact_error": sum(calibration.values()) / len(calibration),
                     "contact_calibration": calibration})
    pareto = [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]
    result = {"all": rows, "pareto": pareto}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
