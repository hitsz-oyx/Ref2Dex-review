"""Visualize the current, GT-future, and predicted-future hand point clouds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DISPLAY", "localhost:10.0")

import numpy as np
import open3d as o3d
import torch

from src.base import build_runner_from_checkpoint

from .dataset import Stage5CpDataset


def _cloud(points: np.ndarray, color: list[float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(color)
    return cloud


def _save_snapshot(
    path: Path,
    current: np.ndarray,
    future_gt: np.ndarray,
    future_pred: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 8), dpi=180)
    axes = figure.add_subplot(111, projection="3d")
    stride = 3
    axes.scatter(*current[::stride].T, s=1.5, c="#60656d", label="Current hand")
    axes.scatter(*future_gt[::stride].T, s=1.5, c="#20b84b", label="GT future hand")
    axes.scatter(*future_pred[::stride].T, s=1.5, c="#ef3b2c", label="Cp→Cm→Fm prediction")
    axes.set_title("Cp human closure")
    axes.set_axis_off()
    axes.legend(loc="upper right", markerscale=4)
    all_points = np.concatenate([current, future_gt, future_pred], axis=0)
    center = all_points.mean(axis=0)
    radius = max(np.ptp(all_points, axis=0).max() / 2.0, 1e-4)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Cp hand-flow prediction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--save", default=None, help="Optional static PNG output path.")
    args = parser.parse_args()

    runner = build_runner_from_checkpoint(args.checkpoint, mode="eval", device=args.device, build_data=False)
    runner.setup_inference(args.checkpoint)
    dataset = Stage5CpDataset(
        args.input,
        num_obj_points=runner.cfg.meta.num_obj_points,
        num_hand_points=runner.cfg.meta.num_hand_points,
        base_seed=runner.cfg.train.seed,
    )
    sample = dataset[args.pair]
    batch = {key: value.unsqueeze(0).to(runner.device) for key, value in sample.items() if torch.is_tensor(value)}
    with torch.no_grad():
        prediction = runner.inference(runner.model, batch)

    flow_key = "pred_hand_flow" if runner.cfg.meta.stage == "closed_loop" else "pred_hand_flow_teacher"
    predicted_flow = prediction[flow_key][0].detach().cpu().numpy()
    current_hand = sample["hand_points"].numpy()
    gt_future = current_hand + sample["hand_flow"].numpy()
    predicted_future = current_hand + predicted_flow
    epe_mm = np.linalg.norm(predicted_flow - sample["hand_flow"].numpy(), axis=-1).mean() * 1000
    print(f"pair={args.pair} hand EPE={epe_mm:.3f} mm")

    if args.save:
        output = Path(args.save)
        _save_snapshot(output, current_hand, gt_future, predicted_future)
        print(f"saved snapshot: {output}")
    if not args.check_only:
        o3d.visualization.draw_geometries(
            [_cloud(current_hand, [0.35, 0.35, 0.35]), _cloud(gt_future, [0.1, 0.9, 0.2]), _cloud(predicted_future, [0.95, 0.15, 0.1])],
            window_name="Cp closure: current / GT / prediction",
            width=1280,
            height=800,
        )


if __name__ == "__main__":
    main()
