"""30 Hz GRAB-right-hand to Inspire F1 retargeting smoke test.

The script uses GRAB human geometry only to produce frozen Cm tokens.  The
point decoder then receives an Inspire F1 hand initialized at a deterministic
object-relative approach pose and predicts a robot hand flow.  It writes a
small NPZ trajectory and a static 3-D PNG for inspection.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.utils import import_from_path
from src.task.Cm.dataset_object_v2 import _MmapSequenceDataset
from src.task.CmDecoder.q_optimizer import DifferentiableInspireHand, optimize_q_from_hand_points


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_URDF = ROOT / "dataset" / "HRDexDB" / "assets" / "robots" / "xarm_inspire_f1_right.urdf"


def _pose_from_approach(center: np.ndarray, wrist: np.ndarray, distance: float) -> np.ndarray:
    direction = wrist - center
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    position = center + distance * direction
    z = (center - position) / max(float(np.linalg.norm(center - position)), 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(up, z))) > 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x = np.cross(up, z); x /= max(float(np.linalg.norm(x)), 1e-8)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = position
    return pose


def _world_to_frame(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points - pose[:3, 3]) @ pose[:3, :3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="s1/scissors_offhand_1")
    parser.add_argument("--grab-root", type=Path, default=Path("data/processed_data/cm_object_v2_subject_template_20260820/grab"))
    parser.add_argument("--decoder-checkpoint", type=Path, default=Path("outputs/cmdecoder/cm_decoder_20260823_000423/checkpoints/best.pt"))
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--approach-distance", type=float, default=0.12)
    parser.add_argument("--q-fit-steps", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("output/research/grab_retarget_30hz.npz"))
    parser.add_argument("--figure", type=Path, default=Path("output/research/grab_retarget_30hz.png"))
    args = parser.parse_args()
    device = torch.device(args.device)

    payload = load_checkpoint(args.decoder_checkpoint.resolve(), map_location="cpu")
    cfg = task_config_from_dict(payload["config"])
    model_cfg = copy.copy(cfg.model); model_cfg.meta = cfg.meta
    model = import_from_path(model_cfg.class_path)(model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    hand_model = DifferentiableInspireHand(cfg.meta.robot_urdf, num_hand_points=1538,
                                           sample_seed=int(cfg.meta.sample_seed), device=device)

    sequence = args.grab_root / args.sequence
    dataset = _MmapSequenceDataset([sequence], num_obj_points=512, num_hand_points=1538,
                                   fixed_stride=1, min_stride=1, max_stride=1,
                                   active_only=False)
    shared = sequence / "shared"; side = sequence / "right"
    obj_world = np.load(shared / "obj_points_world.npy", mmap_mode="r")
    obj_normals_world = np.load(shared / "obj_normals_world.npy", mmap_mode="r")
    hand_world = np.load(side / "hand_points_world.npy", mmap_mode="r")
    hand_pose = np.load(side / "hand_root_pose_world.npy", mmap_mode="r")
    start_frame = int(args.start_frame)
    if start_frame < 0 or start_frame >= len(dataset):
        raise ValueError(f"start-frame must be in [0, {len(dataset) - 1}], got {start_frame}")
    max_frames = min(int(args.frames), len(dataset) - start_frame)
    center = np.asarray(obj_world[start_frame].mean(axis=0), np.float32)
    robot_pose = _pose_from_approach(
        center, np.asarray(hand_pose[start_frame, :3, 3]), float(args.approach_distance)
    )
    q = ((hand_model.q_lower + hand_model.q_upper) * 0.5).detach().cpu().numpy().astype(np.float32)

    robot_world_all, grab_world_all, object_world_all, q_all = [], [], [], []
    for i in range(max_frames):
            frame = start_frame + i
            sample = dataset[frame]
            selected = sample["selected_obj_idx"].numpy()
            human = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
            # Human GRAB hand flow is used only for the frozen action token.
            with torch.no_grad():
                _, _, cm_tokens = model._frozen_features(human)
            q_tensor = torch.from_numpy(q)[None].to(device)
            robot_points = hand_model.points(q_tensor)
            robot_normals = hand_model.normals(q_tensor)
            obj_local = _world_to_frame(np.asarray(obj_world[frame, selected]), robot_pose)
            obj_n_local = np.asarray(obj_normals_world[frame, selected]) @ robot_pose[:3, :3]
            batch = {
                "obj_points": torch.from_numpy(obj_local)[None].to(device),
                "obj_normals": torch.from_numpy(obj_n_local.astype(np.float32))[None].to(device),
                "obj_valid_mask": torch.ones((1, 512), dtype=torch.bool, device=device),
                "hand_points": robot_points,
                "hand_normals": robot_normals,
                "cm_tokens": cm_tokens,
            }
            with torch.no_grad():
                output = model(batch)
            target = robot_points + output["pred_hand_flow"]
            fit = optimize_q_from_hand_points(hand_model, q_t=q_tensor,
                                              current_hand_points=robot_points,
                                              target_hand_points=target,
                                              steps=int(args.q_fit_steps), lr=float(cfg.meta.q_fit_lr),
                                              prior_weight=float(cfg.meta.q_fit_prior_weight),
                                              wrist_prior_weight=float(cfg.meta.wrist_fit_prior_weight))
            q = fit["q"][0].detach().cpu().numpy().astype(np.float32)
            wrist_t = fit["wrist_delta_translation"][0].detach().cpu().numpy()
            wrist_r = fit["wrist_delta_rotvec"][0].detach().cpu().numpy()
            from scipy.spatial.transform import Rotation
            relative = np.eye(4, dtype=np.float32)
            relative[:3, :3] = Rotation.from_rotvec(wrist_r).as_matrix().astype(np.float32)
            relative[:3, 3] = wrist_t.astype(np.float32)
            robot_pose = robot_pose @ relative
            robot_world = (hand_model.points(torch.from_numpy(q)[None].to(device))[0].detach().cpu().numpy() @ robot_pose[:3, :3].T + robot_pose[:3, 3])
            robot_world_all.append(robot_world.astype(np.float32))
            grab_world_all.append(np.asarray(hand_world[frame], np.float32))
            object_world_all.append(np.asarray(obj_world[frame, selected], np.float32))
            q_all.append(q)

    args.output.parent.mkdir(parents=True, exist_ok=True); args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, robot_hand_world=np.stack(robot_world_all), grab_hand_world=np.stack(grab_world_all), object_points_world=np.stack(object_world_all), q=np.stack(q_all), fps=np.float32(30.0), sequence=args.sequence, start_frame=np.int32(start_frame))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*object_world_all[0].T, s=2, c="gray", label="object")
    ax.scatter(*grab_world_all[0][::8].T, s=3, c="tab:blue", label="GRAB right hand")
    ax.scatter(*robot_world_all[0][::8].T, s=3, c="tab:orange", label="Inspire F1 initial")
    ax.scatter(*robot_world_all[-1][::8].T, s=3, c="tab:red", label="Inspire F1 predicted")
    ax.set_title(f"GRAB → Inspire F1 retargeting, 30 Hz, {args.sequence}, start={start_frame}"); ax.legend(); fig.tight_layout(); fig.savefig(args.figure, dpi=160); plt.close(fig)
    print(f"saved trajectory: {args.output}"); print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
