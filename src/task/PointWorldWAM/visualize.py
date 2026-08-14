from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import load_config
from .dataset import build_dataset
from .pointworld_forward import GRABPointWorldForward


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 PointWorldWAM forward 轨迹")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_forward_overfit.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="output/research/pointworld_wam/forward_smoke.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg.train.output_dir) / "best.pt")
    dataset = build_dataset(cfg.data)
    sample = dataset[args.index]
    device = torch.device(cfg.train.device)
    model = GRABPointWorldForward(cfg.model).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(
            sample["object_points"][None].to(device),
            sample["object_normals"][None].to(device),
            sample["hand_points"][None].to(device),
            sample["hand_normals"][None].to(device),
        )["object_tracks"][0].cpu().numpy()
    target = sample["object_points"].numpy()
    hand = sample["hand_points"].numpy()
    fig = plt.figure(figsize=(15, 5))
    for column, frame in enumerate((0, 5, 10), start=1):
        ax = fig.add_subplot(1, 3, column, projection="3d")
        ax.scatter(*hand[frame, ::4].T, s=2, c="tab:orange", label="hand")
        ax.scatter(*target[frame, ::8].T, s=3, c="tab:blue", label="GT")
        ax.scatter(*pred[frame, ::8].T, s=3, c="tab:red", marker="x", label="pred")
        ax.set_title(f"frame {frame}")
        ax.set_box_aspect((1, 1, 1))
    fig.axes[0].legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    np.savez(output.with_suffix(".npz"), gt=target, pred=pred, hand=hand)
    print(output)


if __name__ == "__main__":
    main()
