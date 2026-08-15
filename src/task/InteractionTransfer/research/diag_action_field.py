"""诊断 hand point flow 中局部差异相对整体运动的比例。"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.task.InteractionTransfer.dataset import GRABOneStepDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequence", action="append", required=True)
    args = parser.parse_args()
    dataset = GRABOneStepDataset(args.root, args.sequence, max_transitions=64)
    ratios = []
    for index in range(len(dataset)):
        flow = dataset[index]["hand_flow"]
        centered = flow - flow.mean(0, keepdim=True)
        ratios.append(float(centered.norm(dim=-1).mean() / flow.norm(dim=-1).mean().clamp_min(1e-8)))
    print({"sequences": args.sequence, "transitions": len(dataset),
           "action_local_ratio_mean": sum(ratios) / len(ratios),
           "action_local_ratio_min": min(ratios), "action_local_ratio_max": max(ratios)})


if __name__ == "__main__":
    main()
