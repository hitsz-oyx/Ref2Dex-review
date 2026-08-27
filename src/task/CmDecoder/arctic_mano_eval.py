"""Evaluate a GRAB-trained MANO point decoder on an ARCTIC trajectory.

The evaluation deliberately keeps the task small and shape-matched: the
frozen mixed C=64 Cm produces action tokens from ARCTIC, while the decoder was
trained only on GRAB.  Teacher-forced one-step predictions and a short
autoregressive rollout are both exported, together with a static diagnostic
figure.  The rollout uses the recorded ARCTIC wrist poses to move predicted
points between consecutive wrist frames; it does not claim full hand-pose
retargeting.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.utils import import_from_path
from src.task.Cm.dataset_object_v2 import _MmapSequenceDataset


def _as_batch(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }


def _local_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _world_to_local(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points - pose[:3, 3]) @ pose[:3, :3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder-checkpoint", type=Path, required=True,
        help="GRAB-trained CmPointFlowModel checkpoint",
    )
    parser.add_argument(
        "--arctic-root", type=Path,
        default=Path("data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/arctic"),
    )
    parser.add_argument("--sequence", default="s01/box_use_01")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("output/research/arctic_mano_cm64_eval.npz"))
    parser.add_argument("--figure", type=Path, default=Path("output/research/arctic_mano_cm64_eval.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = load_checkpoint(args.decoder_checkpoint.resolve(), map_location="cpu")
    cfg = task_config_from_dict(payload["config"])
    model_cfg = copy.copy(cfg.model)
    model_cfg.meta = cfg.meta
    model = import_from_path(model_cfg.class_path)(model_cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)

    sequence = (args.arctic_root / args.sequence).resolve()
    if not (sequence / "shared" / "meta.json").is_file():
        raise FileNotFoundError(f"ARCTIC object-v2 sequence not found: {sequence}")
    dataset = _MmapSequenceDataset(
        [sequence], num_obj_points=512, num_hand_points=1538,
        fixed_stride=1, min_stride=1, max_stride=1, active_only=True,
    )
    selected_rows = [
        index for index, (path, side, _frame) in enumerate(dataset.rows)
        if path.resolve() == sequence and side == args.side
    ]
    if not selected_rows:
        raise ValueError(f"No active {args.side} rows in {sequence}")
    start = int(args.start_frame)
    if start < 0 or start >= len(selected_rows):
        raise ValueError(f"start-frame must be in [0, {len(selected_rows) - 1}]")
    rows = selected_rows[start:start + int(args.frames)]
    if not rows:
        raise ValueError("No evaluation rows selected")

    side_dir = sequence / args.side
    hand_world = np.load(side_dir / "hand_points_world.npy", mmap_mode="r")
    hand_pose = np.load(side_dir / "hand_root_pose_world.npy", mmap_mode="r")
    raw_frame = np.load(side_dir / "raw_frame_id.npy", mmap_mode="r")

    tf_epe_mm, tf_zero_mm = [], []
    tf_pred_flow, gt_flow, tf_pred_next = [], [], []
    rollout_epe_mm, rollout_points, gt_points = [], [], []
    object_points = []
    raw_pairs = []
    target_local = None
    with torch.inference_mode():
        for row_index in rows:
            sample = dataset[row_index]
            current_frame_index = int(dataset.rows[row_index][2])
            next_frame_index = current_frame_index + 1
            batch = _as_batch(sample, device)
            prediction = model(batch)
            pred_flow = prediction["pred_hand_flow"][0].cpu().numpy().astype(np.float32)
            gt = sample["hand_flow"].numpy().astype(np.float32)
            current = sample["hand_points"].numpy().astype(np.float32)
            future = current + gt
            tf_pred_flow.append(pred_flow)
            gt_flow.append(gt)
            tf_pred_next.append(current + pred_flow)
            tf_epe_mm.append(float(np.linalg.norm(pred_flow - gt, axis=-1).mean() * 1000.0))
            tf_zero_mm.append(float(np.linalg.norm(gt, axis=-1).mean() * 1000.0))

            if target_local is None:
                target_local = current.copy()
            rollout_batch = dict(batch)
            rollout_batch["hand_points"] = torch.from_numpy(target_local)[None].to(device)
            # Keep the recorded current normals as a shape-matched context;
            # predicted normals are not part of this simplified MANO probe.
            rollout_prediction = model(rollout_batch)
            rollout_flow = rollout_prediction["pred_hand_flow"][0].cpu().numpy().astype(np.float32)
            predicted_next_current = target_local + rollout_flow
            pose_current = np.asarray(hand_pose[current_frame_index], dtype=np.float32)
            next_id = int(sample["next_raw_frame_id"])
            pose_next = np.asarray(hand_pose[next_frame_index], dtype=np.float32)
            target_next = _world_to_local(_local_to_world(predicted_next_current, pose_current), pose_next)
            gt_next = np.asarray(_world_to_local(hand_world[next_frame_index], pose_next), dtype=np.float32)
            rollout_points.append(target_next)
            gt_points.append(gt_next)
            rollout_epe_mm.append(float(np.linalg.norm(target_next - gt_next, axis=-1).mean() * 1000.0))
            object_points.append(sample["obj_points"].numpy().astype(np.float32))
            raw_pairs.append((int(sample["raw_frame_id"]), next_id))
            target_local = target_next

    tf_epe_mm = np.asarray(tf_epe_mm, dtype=np.float32)
    tf_zero_mm = np.asarray(tf_zero_mm, dtype=np.float32)
    rollout_epe_mm = np.asarray(rollout_epe_mm, dtype=np.float32)
    rollout_points = np.stack(rollout_points)
    gt_points = np.stack(gt_points)
    object_points = np.stack(object_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tf_pred_flow=np.stack(tf_pred_flow),
        gt_flow=np.stack(gt_flow),
        tf_pred_next=np.stack(tf_pred_next),
        rollout_points=rollout_points,
        gt_points=gt_points,
        object_points=object_points,
        tf_epe_mm=tf_epe_mm,
        tf_zero_flow_epe_mm=tf_zero_mm,
        rollout_epe_mm=rollout_epe_mm,
        raw_pairs=np.asarray(raw_pairs, dtype=np.int64),
        sequence=args.sequence,
        side=args.side,
        start_frame=np.int32(start),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(tf_epe_mm, label="teacher-forced EPE")
    axes[0, 0].plot(tf_zero_mm, label="zero-flow", alpha=0.7)
    axes[0, 0].set_title("ARCTIC teacher-forced one-step")
    axes[0, 0].set_ylabel("EPE (mm)"); axes[0, 0].legend()
    axes[0, 1].plot(rollout_epe_mm, color="tab:red")
    axes[0, 1].set_title("ARCTIC autoregressive rollout")
    axes[0, 1].set_ylabel("EPE to ARCTIC GT (mm)")
    obj = object_points[0]
    fig.delaxes(axes[1, 0])
    axes[1, 0] = fig.add_subplot(2, 2, 3, projection="3d")
    axes[1, 0].scatter(*obj.T, s=2, c="0.65", label="object")
    axes[1, 0].scatter(*gt_points[-1][::8].T, s=3, c="tab:green", label="ARCTIC GT")
    axes[1, 0].scatter(*rollout_points[-1][::8].T, s=3, c="tab:red", label="rollout")
    axes[1, 0].set_title("Final local-frame hand")
    axes[1, 0].legend()
    gt_centroid = gt_points.mean(axis=1)
    pred_centroid = rollout_points.mean(axis=1)
    axes[1, 1].plot(np.linalg.norm(gt_centroid - gt_centroid[0], axis=1), label="GT")
    axes[1, 1].plot(np.linalg.norm(pred_centroid - gt_centroid[0], axis=1), label="rollout")
    axes[1, 1].set_title("Hand centroid displacement")
    axes[1, 1].set_xlabel("step"); axes[1, 1].legend()
    fig.suptitle(f"GRAB-trained CmDecoder on ARCTIC MANO: {args.sequence}/{args.side}")
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(f"teacher_forced_epe_mm={tf_epe_mm.mean():.4f} zero_flow_mm={tf_zero_mm.mean():.4f}")
    print(f"rollout_epe_mm={rollout_epe_mm.mean():.4f} final_rollout_epe_mm={rollout_epe_mm[-1]:.4f}")
    print(f"saved trajectory: {args.output}")
    print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
