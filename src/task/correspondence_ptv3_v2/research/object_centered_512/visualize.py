"""Visualize 512-point random object sampling in the object coordinate frame.

The Stage 3 object points are stored in ``hand_root``.  This diagnostic first
maps them to world with ``hand_root_pose`` and then to the object frame with the
inverse ``obj_root_pose_world``.  The resulting object points should agree with
``obj_points_canonical`` for rigid-canonical samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def _to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _to_object(points_hand_root: np.ndarray, hand_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
    world = _to_world(points_hand_root, hand_pose)
    inv_obj = np.linalg.inv(obj_pose)
    return _to_world(world, inv_obj)


def _set_equal_3d(ax, points: np.ndarray, hand: np.ndarray) -> None:
    all_points = np.concatenate([points, hand], axis=0)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) / 2.0) * 1.08
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")


def _nn_distance(points: np.ndarray) -> tuple[float, float]:
    distance = cKDTree(points).query(points, k=2, workers=1)[0][:, 1]
    return float(np.mean(distance)), float(np.median(distance))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    frame = int(args.frame)
    full_hand_root = np.asarray(data["obj_points"][frame], dtype=np.float64)
    hand_hand_root = np.asarray(data["hand_points"][frame], dtype=np.float64)
    hand_pose = np.asarray(data["hand_root_pose"][frame], dtype=np.float64)
    obj_pose = np.asarray(data["obj_root_pose_world"][frame], dtype=np.float64)
    full_object = _to_object(full_hand_root, hand_pose, obj_pose)
    hand_object = _to_object(hand_hand_root, hand_pose, obj_pose)

    if args.num_points > len(full_object):
        raise ValueError(f"num-points={args.num_points} exceeds pool size {len(full_object)}")
    rng = np.random.default_rng(args.seed)
    selected_idx = rng.choice(len(full_object), size=args.num_points, replace=False)
    sampled = full_object[selected_idx]

    canonical_rmse = None
    if "obj_points_canonical" in data.files:
        canonical = np.asarray(data["obj_points_canonical"], dtype=np.float64)
        if canonical.shape == full_object.shape:
            canonical_rmse = float(np.sqrt(np.mean((full_object - canonical) ** 2)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 6), dpi=180)
    views = [("Object frame: full pool + 512 random points", full_object, sampled, 1),
             ("Object frame: 512 points only", sampled, sampled, 2)]
    for title, background, foreground, subplot in views:
        ax = fig.add_subplot(1, 2, subplot, projection="3d")
        if subplot == 1:
            ax.scatter(background[:, 0], background[:, 1], background[:, 2], s=1.0,
                       c="#b8c0cc", alpha=0.18, depthshade=False, label="full pool (4096)")
            ax.scatter(foreground[:, 0], foreground[:, 1], foreground[:, 2], s=8.0,
                       c="#e63946", alpha=0.85, depthshade=False, label="random sample (512)")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.scatter(foreground[:, 0], foreground[:, 1], foreground[:, 2], s=9.0,
                       c="#e63946", alpha=0.9, depthshade=False, label="random sample (512)")
        ax.scatter(hand_object[:, 0], hand_object[:, 1], hand_object[:, 2], s=1.2,
                   c="#2563eb", alpha=0.28, depthshade=False, label="hand (1538)")
        ax.set_title(title, fontsize=10)
        _set_equal_3d(ax, background, hand_object)
        ax.view_init(elev=22, azim=-58)

    nn_full = _nn_distance(full_object)
    nn_sampled = _nn_distance(sampled)
    subtitle = (
        f"sample={args.npz.name}, frame={frame}, seed={args.seed}; "
        f"NN mean full/sample={nn_full[0]*1000:.2f}/{nn_sampled[0]*1000:.2f} mm; "
        f"median={nn_full[1]*1000:.2f}/{nn_sampled[1]*1000:.2f} mm"
    )
    if canonical_rmse is not None:
        subtitle += f"; object-frame vs canonical RMSE={canonical_rmse:.2e} m"
    fig.suptitle(subtitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved={args.output}")
    print(f"pool={len(full_object)} sample={len(sampled)} nn_full_mean_m={nn_full[0]:.8g} nn_sample_mean_m={nn_sampled[0]:.8g}")
    if canonical_rmse is not None:
        print(f"object_frame_canonical_rmse_m={canonical_rmse:.8g}")


if __name__ == "__main__":
    main()
