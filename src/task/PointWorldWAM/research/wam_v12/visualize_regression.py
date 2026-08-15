from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset_wam_chunk import build_wam_chunk_dataset
from src.task.PointWorldWAM.train import move_batch
from src.task.PointWorldWAM.train_wam_v2_regression import regression_inputs
from src.task.PointWorldWAM.wam_v2 import ChunkJointWAM


ROOT = Path(__file__).resolve().parents[5]


def load_model(cfg, dataset, mode: str, device: torch.device):
    model = ChunkJointWAM(cfg.model, dataset.statistics(), dataset.subjects).to(device)
    checkpoint = torch.load(
        Path(cfg.train.output_dir) / mode / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main() -> None:
    cfg = load_config(
        str(ROOT / "src/task/PointWorldWAM/configs/grab_wam_v12_debug.yaml")
    )
    dataset = build_wam_chunk_dataset(cfg.data)
    batch = next(iter(DataLoader(Subset(dataset, [len(dataset) // 2]), batch_size=1)))
    device = torch.device(cfg.train.device)
    batch = move_batch(batch, device)
    inverse_model = load_model(cfg, dataset, "inverse", device)
    forward_model = load_model(cfg, dataset, "forward", device)
    with torch.no_grad():
        inverse_args = regression_inputs(inverse_model, batch, "inverse")[1:]
        inverse = inverse_model(batch, *inverse_args)
        forward_args = regression_inputs(forward_model, batch, "forward")[1:]
        forward = forward_model(batch, *forward_args)
        predicted_hands = {
            side: inverse_model.hand_points_from_normalized_action(
                batch, side, inverse[f"{side}_action_velocity"]
            )[0].cpu().numpy()
            for side in ("left", "right")
        }
        predicted_object = (
            batch["object_points"][:, None]
            + forward_model.denormalize_world(forward["world_velocity"])
        )[0].cpu().numpy()
    gt_object = (
        batch["object_points"][:, None] + batch["world_chunk"]
    )[0].cpu().numpy()
    figure = plt.figure(figsize=(15, 5))
    for column, frame in enumerate((0, 4, 9), start=1):
        axis = figure.add_subplot(1, 3, column, projection="3d")
        axis.scatter(*gt_object[frame, ::8].T, s=3, label="GT object")
        axis.scatter(
            *predicted_object[frame, ::8].T,
            s=3,
            marker="x",
            label="pred object",
        )
        for side, color in (("left", "tab:green"), ("right", "tab:orange")):
            target = batch[f"{side}_hand_chunk"][0, frame].cpu().numpy()
            axis.scatter(*target[::12].T, s=3, color=color, alpha=0.25)
            axis.scatter(
                *predicted_hands[side][frame, ::12].T,
                s=3,
                marker="x",
                color=color,
                label=f"pred {side}" if frame == 0 else None,
            )
        axis.set_title(f"future frame {frame + 1}")
        axis.set_box_aspect((1, 1, 1))
    figure.axes[0].legend(fontsize=8)
    figure.tight_layout()
    output = ROOT / "output/research/pointworld_wam/wam_v12"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "regression_generation.png", dpi=180)
    np.savez(
        output / "regression_generation.npz",
        gt_object=gt_object,
        predicted_object=predicted_object,
        left_prediction=predicted_hands["left"],
        right_prediction=predicted_hands["right"],
    )


if __name__ == "__main__":
    main()
