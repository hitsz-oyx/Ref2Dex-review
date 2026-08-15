from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset_wam import build_wam_dataset
from src.task.PointWorldWAM.train import move_batch
from src.task.PointWorldWAM.wam_v1 import WAMV1


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    cfg = load_config(str(ROOT / "src/task/PointWorldWAM/configs/grab_wam_v1.yaml"))
    dataset = build_wam_dataset(cfg.data)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    device = torch.device(cfg.train.device)
    batch = move_batch(batch, device)
    model = WAMV1(cfg.model, dataset.action_statistics()).to(device)
    checkpoint = torch.load(
        Path(cfg.train.output_dir) / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 2000)
    initial = {
        side: torch.randn((1, 30), generator=generator, device=device)
        for side in ("left", "right")
    }
    with torch.no_grad():
        action = model.sample(
            batch,
            initial["left"],
            initial["right"],
            cfg.eval.fm_inference_steps,
        )
        prediction = {
            side: model.mano.points_from_action(
                side,
                batch[f"{side}_global_orient"],
                batch[f"{side}_hand_pose"],
                batch[f"{side}_transl"],
                batch[f"{side}_betas"],
                action[f"{side}_action"],
            )[0].cpu().numpy()
            for side in ("left", "right")
        }
    figure = plt.figure(figsize=(12, 5))
    for column, side in enumerate(("left", "right"), start=1):
        axis = figure.add_subplot(1, 2, column, projection="3d")
        current = batch[f"{side}_hand_points"][0].cpu().numpy()
        target = batch[f"{side}_next_hand_points"][0].cpu().numpy()
        axis.scatter(*current[::8].T, s=3, label="current")
        axis.scatter(*target[::8].T, s=3, label="GT next")
        axis.scatter(*prediction[side][::8].T, s=3, marker="x", label="generated")
        axis.set_title(side)
        axis.set_box_aspect((1, 1, 1))
    figure.axes[0].legend()
    output = ROOT / "output/research/pointworld_wam/wam_v1/generation.png"
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    np.savez(
        output.with_suffix(".npz"),
        left_prediction=prediction["left"],
        right_prediction=prediction["right"],
    )


if __name__ == "__main__":
    main()
