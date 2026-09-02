"""Cross-hand GRAB-action to Inspire-F1 runtime rollout.

GRAB hand-flow is used only to produce frozen Cm action tokens.  The Inspire
state is initialized independently and then fed back through the current
object_pose_t point-flow decoder; no Inspire ground-truth state is used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.utils import import_from_path
from src.task.Cm.dataset.object_v2 import _MmapSequenceDataset
from src.task.CmDecoder.dataset import _load_hrdex_io, _robot_hand_mesh, _rotate_to_frame, _to_frame
from src.task.CmDecoder.grab_retarget import _pose_from_approach
from src.task.CmDecoder.q_optimizer import DifferentiableInspireHand, optimize_q_from_hand_points, rotvec_to_matrix


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAB_ROOT = ROOT / "data/processed_data/cm_object_v2_surface512_object_pose_20260830"
DEFAULT_CHECKPOINT = ROOT / "outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt"


def _local_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _world_to_frame(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points - pose[:3, 3]) @ pose[:3, :3]


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _first_within_surface_distance(
    hand_world: np.ndarray,
    object_world: np.ndarray,
    *,
    threshold_m: float,
) -> tuple[int, float]:
    """Find the first frame with any hand/object surface pair within threshold."""
    if threshold_m <= 0.0:
        raise ValueError("proximity threshold must be positive")
    for frame in range(min(len(hand_world), len(object_world))):
        object_tree = cKDTree(np.asarray(object_world[frame], dtype=np.float64))
        nearest = object_tree.query(np.asarray(hand_world[frame], dtype=np.float64), k=1)[0]
        distance_m = float(np.min(nearest))
        if distance_m <= threshold_m:
            return frame, distance_m
    raise ValueError(f"No frame satisfies the {threshold_m * 1000.0:.1f} mm hand/object proximity threshold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="s1/scissors_offhand_1")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--grab-root", type=Path, default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument(
        "--start-frame", type=int, default=None,
        help="Explicit GRAB start frame; by default use the first frame within --proximity-threshold.",
    )
    parser.add_argument("--proximity-threshold", type=float, default=0.05,
                        help="Surface proximity threshold in metres used for automatic start selection.")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--approach-distance", type=float, default=0.12)
    parser.add_argument("--q-fit-steps", type=int, default=40)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("output/research/grab_runtime_rollout_cmdecoder_best_20260902.npz"))
    parser.add_argument("--figure", type=Path, default=Path("output/research/grab_runtime_rollout_cmdecoder_best_20260902.png"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    payload = load_checkpoint(args.decoder_checkpoint.resolve(), map_location="cpu")
    cfg = task_config_from_dict(payload["config"])
    if str(getattr(cfg.meta, "coordinate_frame", "")) != "object_pose_t":
        raise ValueError("Current cross-hand runtime requires checkpoint coordinate_frame=object_pose_t")
    model_cfg = copy.copy(cfg.model); model_cfg.meta = cfg.meta
    model = import_from_path(model_cfg.class_path)(model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    hand_model = DifferentiableInspireHand(cfg.meta.robot_urdf, num_hand_points=1538,
                                           sample_seed=int(cfg.meta.sample_seed), device=device)

    sequence = args.grab_root / args.sequence
    dataset = _MmapSequenceDataset(
        [sequence], num_obj_points=512, num_hand_points=1538,
        fixed_stride=int(args.stride), min_stride=int(args.stride), max_stride=int(args.stride),
        active_only=False, coordinate_frame="object_pose_t",
    )
    shared = sequence / "shared"; side = sequence / args.side
    obj_pose = _load_array(shared / "obj_pose_world.npy")
    obj_world = _load_array(shared / "obj_points_world.npy")
    hand_world = _load_array(side / "hand_points_world.npy")
    hand_normals_world = _load_array(side / "hand_normals_world.npy")
    hand_pose = _load_array(side / "hand_root_pose_world.npy")
    if args.start_frame is None:
        selected_start, selected_distance = _first_within_surface_distance(
            hand_world, obj_world, threshold_m=float(args.proximity_threshold)
        )
        args.start_frame = selected_start
        print(
            f"selected first proximity frame={args.start_frame} "
            f"surface_distance_mm={selected_distance * 1000.0:.3f} "
            f"threshold_mm={float(args.proximity_threshold) * 1000.0:.3f}"
        )
    if args.start_frame < 0 or args.start_frame + args.frames * args.stride >= len(obj_pose):
        raise ValueError("start-frame/frames/stride exceed GRAB trajectory length")

    center = np.asarray(obj_world[args.start_frame].mean(axis=0), np.float32)
    robot_wrist = np.asarray(hand_pose[args.start_frame], np.float32)
    robot_pose = _pose_from_approach(center, robot_wrist[:3, 3], float(args.approach_distance))
    q = ((hand_model.q_lower + hand_model.q_upper) * 0.5).detach().cpu().numpy().astype(np.float32)
    # Keep the object-facing orientation, but translate the neutral Inspire
    # sampled hand so its centroid starts at the GRAB hand centroid.  This
    # removes the previous 12 cm approach-pose offset without pretending the
    # two embodiments have identical wrist frames or morphology.
    neutral_hand_local = hand_model.points(torch.from_numpy(q)[None].to(device))[0].detach().cpu().numpy()
    neutral_hand_centroid_world = _local_to_world(neutral_hand_local, robot_pose).mean(axis=0)
    grab_hand_centroid_world = np.asarray(hand_world[args.start_frame], dtype=np.float32).mean(axis=0)
    robot_pose[:3, 3] += grab_hand_centroid_world - neutral_hand_centroid_world
    initial_robot_world = _local_to_world(neutral_hand_local, robot_pose).astype(np.float32)
    initial_sample = dataset[args.start_frame]
    initial_object_world = np.asarray(
        obj_world[args.start_frame, initial_sample["selected_obj_idx"].numpy()], dtype=np.float32
    )
    initial_robot_obj_distance_mm = float(
        np.linalg.norm(initial_robot_world.mean(axis=0) - initial_object_world.mean(axis=0)) * 1000.0
    )
    io = _load_hrdex_io(Path(cfg.meta.dataset_root).expanduser().resolve().parent)
    urdf = io.parse_urdf(Path(cfg.meta.robot_urdf).expanduser().resolve())
    mesh_cache: dict = {}

    robot_world_all, grab_world_all, object_world_all, q_all, robot_pose_all = [], [], [], [], []
    for i in range(int(args.frames)):
        frame = int(args.start_frame + i * args.stride)
        sample = dataset[frame]
        human = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
        with torch.no_grad():
            _, _, cm_tokens = model._frozen_features(human)

        q_tensor = torch.from_numpy(q)[None].to(device)
        robot_points_local = hand_model.points(q_tensor)
        robot_normals_local = hand_model.normals(q_tensor)
        pose = np.asarray(obj_pose[frame], dtype=np.float32)
        robot_world = _local_to_world(robot_points_local[0].detach().cpu().numpy(), robot_pose)
        robot_normals_world = robot_normals_local[0].detach().cpu().numpy() @ robot_pose[:3, :3].T
        batch = {
            "obj_points": sample["obj_points"][None].to(device),
            "obj_normals": sample["obj_normals"][None].to(device),
            "obj_valid_mask": sample["obj_valid_mask"][None].to(device),
            "hand_points": torch.from_numpy(_world_to_frame(robot_world, pose))[None].to(device),
            "hand_normals": torch.from_numpy(_rotate_to_frame(robot_normals_world, pose))[None].to(device),
            "hand_flow": sample["hand_flow"][None].to(device),
            "cm_tokens": cm_tokens,
        }
        with torch.no_grad():
            prediction = model(batch)
            target_object = batch["hand_points"] + prediction["pred_hand_flow"]
        target_world = _local_to_world(target_object[0].detach().cpu().numpy(), pose)
        target_local = _world_to_frame(target_world, robot_pose)
        fit = optimize_q_from_hand_points(
            hand_model, q_t=q_tensor, current_hand_points=robot_points_local,
            target_hand_points=torch.from_numpy(target_local)[None].to(device),
            steps=int(args.q_fit_steps), lr=float(getattr(cfg.meta, "q_fit_lr", 0.05)),
            prior_weight=float(getattr(cfg.meta, "q_fit_prior_weight", 1e-4)),
            wrist_prior_weight=float(getattr(cfg.meta, "wrist_fit_prior_weight", 1e-6)),
        )
        q = fit["q"][0].detach().cpu().numpy().astype(np.float32)
        relative = np.eye(4, dtype=np.float32)
        relative[:3, :3] = rotvec_to_matrix(fit["wrist_delta_rotvec"])[0].detach().cpu().numpy()
        relative[:3, 3] = fit["wrist_delta_translation"][0].detach().cpu().numpy()
        robot_pose = robot_pose @ relative
        q_all.append(q.copy()); robot_pose_all.append(robot_pose.copy())
        robot_world_all.append(_local_to_world(hand_model.points(torch.from_numpy(q)[None].to(device))[0].detach().cpu().numpy(), robot_pose).astype(np.float32))
        grab_world_all.append(np.asarray(hand_world[frame], np.float32))
        selected = sample["selected_obj_idx"].numpy()
        object_world_all.append(np.asarray(obj_world[frame, selected], np.float32))

    robot_world_all = np.stack(robot_world_all); grab_world_all = np.stack(grab_world_all); object_world_all = np.stack(object_world_all)
    robot_centroid = robot_world_all.mean(axis=1); grab_centroid = grab_world_all.mean(axis=1); obj_centroid = object_world_all.mean(axis=1)
    robot_obj_dist = np.linalg.norm(robot_centroid - obj_centroid, axis=1) * 1000.0
    grab_obj_dist = np.linalg.norm(grab_centroid - obj_centroid, axis=1) * 1000.0
    robot_step = np.linalg.norm(np.diff(robot_centroid, axis=0), axis=1) * 1000.0
    grab_step = np.linalg.norm(np.diff(grab_centroid, axis=0), axis=1) * 1000.0
    args.output.parent.mkdir(parents=True, exist_ok=True); args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, robot_hand_world=robot_world_all, grab_hand_world=grab_world_all, object_points_world=object_world_all, q=np.stack(q_all), robot_wrist_world=np.stack(robot_pose_all), robot_obj_distance_mm=robot_obj_dist, grab_obj_distance_mm=grab_obj_dist, robot_step_mm=np.concatenate([[0.0], robot_step]), grab_step_mm=np.concatenate([[0.0], grab_step]), initial_robot_hand_world=initial_robot_world, initial_robot_obj_distance_mm=np.float32(initial_robot_obj_distance_mm), fps=np.float32(30.0 / args.stride), sequence=args.sequence, side=args.side, start_frame=np.int32(args.start_frame), stride=np.int32(args.stride), checkpoint=str(args.decoder_checkpoint.resolve()))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*object_world_all[0].T, s=2, c="gray", label="object")
    ax.scatter(*grab_world_all[0][::8].T, s=3, c="tab:blue", label="GRAB source")
    ax.scatter(*robot_world_all[0][::8].T, s=3, c="tab:orange", label="Inspire initial")
    ax.scatter(*robot_world_all[-1][::8].T, s=3, c="tab:red", label="Inspire predicted")
    ax.set_title(f"GRAB → Inspire F1, stride={args.stride}, {args.sequence}, start={args.start_frame}"); ax.legend(); fig.tight_layout(); fig.savefig(args.figure, dpi=160); plt.close(fig)
    print(f"sequence={args.sequence} side={args.side} start_frame={args.start_frame} stride={args.stride} frames={len(robot_world_all)}")
    print(f"aligned_initial_robot_object_distance_mm={initial_robot_obj_distance_mm:.3f}")
    print(f"robot_object_distance_mm/initial_final={robot_obj_dist.mean():.3f}/{robot_obj_dist[0]:.3f}/{robot_obj_dist[-1]:.3f}")
    print(f"grab_object_distance_mm/initial_final={grab_obj_dist.mean():.3f}/{grab_obj_dist[0]:.3f}/{grab_obj_dist[-1]:.3f}")
    print(f"robot_centroid_step_mm_mean/cumulative={robot_step.mean():.3f}/{robot_step.sum():.3f}")
    print(f"grab_centroid_step_mm_mean/cumulative={grab_step.mean():.3f}/{grab_step.sum():.3f}")
    print(f"saved trajectory: {args.output}"); print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
