from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset import build_dataset
from src.task.PointWorldWAM.pointworld_forward import GRABPointWorldForward
from src.task.PointWorldWAM.train import evaluate_conditions, initialize_model


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评估 PointWorldWAM E0-E3")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_forward_overfit.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="output/research/pointworld_wam/forward_smoke")
    args = parser.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.train.device)
    dataset = build_dataset(cfg.data)
    loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size), shuffle=False, num_workers=0)

    initial, load_report = initialize_model(cfg.model)
    initial_metrics = evaluate_conditions(initial.to(device), loader, device)
    del initial
    torch.cuda.empty_cache()

    checkpoint_path = Path(args.checkpoint or Path(cfg.train.output_dir) / "best.pt")
    trained = GRABPointWorldForward(cfg.model).to(device)
    trained.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    trained_metrics = evaluate_conditions(trained, loader, device)

    gt_ade = trained_metrics["gt"]["ade_m"]
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "windows": len(dataset),
        "load_report": load_report,
        "pretrained_init": initial_metrics,
        "trained_best": trained_metrics,
        "ade_improvement_vs_init_pct": 100.0 * (initial_metrics["gt"]["ade_m"] - gt_ade)
        / initial_metrics["gt"]["ade_m"],
        "gt_ade_better_than_static_pct": 100.0
        * (trained_metrics["static"]["ade_m"] - gt_ade)
        / trained_metrics["static"]["ade_m"],
        "gt_ade_better_than_shuffled_pct": 100.0
        * (trained_metrics["shuffled"]["ade_m"] - gt_ade)
        / trained_metrics["shuffled"]["ade_m"],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
