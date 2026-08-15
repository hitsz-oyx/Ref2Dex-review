from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset_wam_chunk import build_wam_chunk_dataset
from src.task.PointWorldWAM.train import move_batch
from src.task.PointWorldWAM.train_wam_v2 import sample_forward, sample_inverse
from src.task.PointWorldWAM.wam_v2 import ChunkJointWAM


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    cfg = load_config(str(ROOT / "src/task/PointWorldWAM/configs/grab_wam_v2_chunk.yaml"))
    dataset = build_wam_chunk_dataset(cfg.data)
    batch = next(iter(DataLoader(Subset(dataset, [len(dataset) // 2]), batch_size=1)))
    device = torch.device(cfg.train.device)
    batch = move_batch(batch, device)
    model = ChunkJointWAM(cfg.model, dataset.statistics(), dataset.subjects).to(device)
    checkpoint = torch.load(
        Path(cfg.train.output_dir) / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    target = model.normalized_targets(batch)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 100000)
    initial = {
        key: torch.randn(
            value.shape,
            device=device,
            dtype=value.dtype,
            generator=generator,
        )
        for key, value in target.items()
    }
    with torch.no_grad():
        inverse = sample_inverse(
            model,
            batch,
            initial["left"],
            initial["right"],
            cfg.flow_matching.inference_steps,
        )
        world = sample_forward(
            model,
            batch,
            target,
            initial["world"],
            cfg.flow_matching.inference_steps,
        )
        predicted_hands = {
            side: model.hand_points_from_normalized_action(batch, side, inverse[side])[
                0
            ].cpu().numpy()
            for side in ("left", "right")
        }
        predicted_object = (
            batch["object_points"][:, None] + model.denormalize_world(world)
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
            label="generated object",
        )
        for side, color in (("left", "tab:green"), ("right", "tab:orange")):
            target_hand = batch[f"{side}_hand_chunk"][0, frame].cpu().numpy()
            axis.scatter(
                *target_hand[::12].T, s=3, color=color, alpha=0.35
            )
            axis.scatter(
                *predicted_hands[side][frame, ::12].T,
                s=3,
                marker="x",
                color=color,
                label=f"generated {side}" if frame == 0 else None,
            )
        axis.set_title(f"future frame {frame + 1}")
        axis.set_box_aspect((1, 1, 1))
    figure.axes[0].legend(fontsize=8)
    figure.tight_layout()
    output = ROOT / "output/research/pointworld_wam/wam_v2"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "chunk_generation.png", dpi=180)
    np.savez(
        output / "chunk_generation.npz",
        gt_object=gt_object,
        predicted_object=predicted_object,
        left_prediction=predicted_hands["left"],
        right_prediction=predicted_hands["right"],
    )


if __name__ == "__main__":
    main()
