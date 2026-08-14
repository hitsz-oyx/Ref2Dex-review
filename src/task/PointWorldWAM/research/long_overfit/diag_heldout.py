from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset import GRABWindowDataset
from src.task.PointWorldWAM.pointworld_forward import GRABPointWorldForward
from src.task.PointWorldWAM.train import evaluate_conditions


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 F0 在未训练 GRAB sequence 上做 action sensitivity")
    parser.add_argument(
        "--experiment",
        default="output/exp/pointworld_wam_forward_long/pretrained_two_seq",
    )
    parser.add_argument("--sequence-offset", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument(
        "--output", default="output/research/pointworld_wam/long_overfit/heldout.json"
    )
    args = parser.parse_args()
    experiment = ROOT / args.experiment
    cfg = load_config(str(experiment / "config.yaml"))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    data_args = vars(cfg.data).copy()
    data_args["sequence_offset"] = args.sequence_offset
    data_args["max_sequences"] = args.max_sequences
    dataset = GRABWindowDataset(**data_args)
    train_args = vars(cfg.data).copy()
    train_args.setdefault("sequence_offset", 0)
    train_dataset = GRABWindowDataset(**train_args)
    loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size), shuffle=False)
    device = torch.device(cfg.train.device)
    checkpoint = torch.load(experiment / "best.pt", map_location="cpu", weights_only=False)
    model = GRABPointWorldForward(cfg.model)
    model.load_state_dict(checkpoint["model"])
    conditions = evaluate_conditions(model.to(device), loader, device)
    gt = conditions["gt"]["ade_m"]
    static = conditions["static"]["ade_m"]
    shuffled = conditions["shuffled"]["ade_m"]
    report = {
        "train_sequences": sorted(
            {ref.shared_path.parent.relative_to(Path(cfg.data.root)).as_posix() for ref in train_dataset.windows}
        ),
        "heldout_sequences": sorted(
            {ref.shared_path.parent.relative_to(Path(cfg.data.root)).as_posix() for ref in dataset.windows}
        ),
        "windows": len(dataset),
        "conditions": conditions,
        "gt_ade_better_than_static_pct": 100.0 * (static - gt) / static,
        "gt_ade_better_than_shuffled_pct": 100.0 * (shuffled - gt) / shuffled,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
