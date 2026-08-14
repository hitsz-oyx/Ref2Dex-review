from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset import build_dataset
from src.task.PointWorldWAM.pointworld_forward import GRABPointWorldForward
from src.task.PointWorldWAM.train import evaluate_conditions


ROOT = Path(__file__).resolve().parents[5]
NAMES = (
    "pretrained_two_seq",
    "scratch_two_seq",
    "pretrained_single_seq",
    "scratch_single_seq",
)


def read_evaluations(directory: Path):
    rows = {}
    for path in sorted(directory.glob("*.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_gt_ade_m" in value:
                rows[int(value["step"])] = float(value["eval_gt_ade_m"])
    return sorted(rows.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="统一复算 PointWorldWAM long-overfit 矩阵")
    parser.add_argument(
        "--experiment-root", default="output/exp/pointworld_wam_forward_long"
    )
    parser.add_argument(
        "--output", default="output/research/pointworld_wam/long_overfit"
    )
    args = parser.parse_args()
    experiment_root = ROOT / args.experiment_root
    report = {}
    figure, axis = plt.subplots(figsize=(9, 5))
    for name in NAMES:
        directory = experiment_root / name
        cfg = load_config(str(directory / "config.yaml"))
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        device = torch.device(cfg.train.device)
        dataset = build_dataset(cfg.data)
        loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size), shuffle=False)
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        model = GRABPointWorldForward(cfg.model)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        conditions = evaluate_conditions(model, loader, device)
        evaluations = read_evaluations(directory)
        gt_ade = conditions["gt"]["ade_m"]
        static_ade = conditions["static"]["ade_m"]
        shuffled_ade = conditions["shuffled"]["ade_m"]
        recent = evaluations[-4:]
        report[name] = {
            "initialization": "pretrained" if cfg.model.pretrained else "scratch",
            "max_sequences": cfg.data.max_sequences,
            "windows": len(dataset),
            "trained_steps": max(step for step, _ in evaluations),
            "best_step": int(checkpoint["step"]),
            "best_selection_ade_m": min(value for _, value in evaluations),
            "recent_eval_ade_m": recent,
            "recent_range_m": max(value for _, value in recent) - min(value for _, value in recent),
            "conditions": conditions,
            "gt_ade_better_than_static_pct": 100.0 * (static_ade - gt_ade) / static_ade,
            "gt_ade_better_than_shuffled_pct": 100.0 * (shuffled_ade - gt_ade) / shuffled_ade,
        }
        axis.plot(
            [step for step, _ in evaluations],
            [1000.0 * value for _, value in evaluations],
            marker="o",
            markersize=3,
            label=name,
        )
        del model, checkpoint
        torch.cuda.empty_cache()

    axis.set_xlabel("training step")
    axis.set_ylabel("GT-hand ADE (mm)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    figure.savefig(output / "convergence.png", dpi=180)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
