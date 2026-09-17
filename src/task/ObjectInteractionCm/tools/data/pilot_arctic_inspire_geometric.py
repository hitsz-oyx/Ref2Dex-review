#!/usr/bin/env python3
"""Run a small ARCTIC MANO -> Inspire geometric retargeting pilot.

The pilot intentionally writes no cache.  It reconstructs each hand from the
ARCTIC MANO parameters, sends the five fingertip targets to the official
``dex-retargeting`` position optimizer, and reports per-side position error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[5]
DEX_RETARGET_ROOT = Path("/home/wbcd/workspace/dex/retarget/third_party/dex-retargeting")
# Official SMPL-X/MANO vertex_ids.py mapping (thumb, index, middle, ring,
# pinky).  These are not the full-body SMPL-X indices used by Dexplore's GRAB
# converter.
TIP_VERTEX_IDS = np.asarray([744, 320, 443, 554, 671], dtype=np.int64)
TIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


def _retargeter(side: str):
    from dex_retargeting.retargeting_config import RetargetingConfig

    RetargetingConfig.set_default_urdf_dir(DEX_RETARGET_ROOT / "assets" / "robots" / "hands")
    config_path = DEX_RETARGET_ROOT / "dex_retargeting" / "configs" / "offline" / f"inspire_hand_{side}.yml"
    return RetargetingConfig.load_from_file(config_path).build()


def _mano_landmarks(adapter, fields: dict, side: str, device: str) -> np.ndarray:
    prefix = f"{side}_mano_"
    model = adapter.mano_r if side == "right" else adapter.mano_l
    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(fields[prefix + name], dtype=torch.float32, device=device)

    global_orient = tensor("global_orient")
    hand_pose = tensor("pose")
    transl = tensor("transl")
    betas = tensor("betas")
    if betas.ndim == 1:
        betas = betas.expand(global_orient.shape[0], -1)
    with torch.no_grad():
        result = model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            transl=transl,
        )
    # ARCTIC MANO order is wrist + 15 joints + five fingertip vertices.  The
    # optimizer only consumes the final five points, but keeping all 21 here
    # makes the adapter contract explicit for the later Dexplore wrapper.
    joints = result.joints.detach().cpu().numpy().astype(np.float32)
    vertices = result.vertices.detach().cpu().numpy().astype(np.float32)
    tips = vertices[:, TIP_VERTEX_IDS]
    landmarks = np.concatenate((joints, tips), axis=1)
    if landmarks.shape[1:] != (21, 3) or not np.isfinite(landmarks).all():
        raise ValueError(f"Invalid {side} ARCTIC landmark output: {landmarks.shape}")
    return landmarks


def _run_side(retargeting, landmarks: np.ndarray) -> tuple[np.ndarray, float, float]:
    target = landmarks[:, 16:21]
    qposes = []
    errors = []
    robot = retargeting.optimizer.robot
    target_indices = retargeting.optimizer.target_link_indices
    for target_pos in target:
        qpos = retargeting.retarget(target_pos)
        qposes.append(qpos)
        robot.compute_forward_kinematics(qpos)
        predicted = np.asarray([robot.get_link_pose(i)[:3, 3] for i in target_indices], dtype=np.float32)
        errors.append(np.linalg.norm(predicted - target_pos, axis=-1))
    values = np.asarray(errors, dtype=np.float32)
    return np.asarray(qposes, dtype=np.float32), float(values.mean()), float(values.max())


def run_pilot(
    sequence: str | Path,
    *,
    max_frames: int = 8,
    device: str = "cpu",
    right_x_rotation_deg: float = 0.0,
) -> dict:
    # Import after the caller has configured REF2DEX_ARCTIC_ROOT, because the
    # adapter resolves its raw/model roots at module import time.
    from process.ARCTIC.raw import ArcticRawAdapter, RAW_SEQS_DIR

    path = Path(sequence)
    if not path.is_absolute():
        path = Path(RAW_SEQS_DIR) / path
    if path.suffix != ".npy" or not path.name.endswith(".mano.npy"):
        raise ValueError(f"Expected an ARCTIC .mano.npy file, got {path}")
    adapter = ArcticRawAdapter(num_obj_points=64, device=device, max_frames=max_frames, nn_batch_size=4, mano_batch_size=16)
    fields = adapter.process_sequence(str(path))
    results = {}
    for side in ("left", "right"):
        retargeting = _retargeter(side)
        landmarks = _mano_landmarks(adapter, fields, side, device)
        if side == "right" and float(right_x_rotation_deg) != 0.0:
            rotation = Rotation.from_euler("x", float(right_x_rotation_deg), degrees=True).as_matrix().astype(np.float32)
            landmarks = landmarks @ rotation.T
        qpos, mean_error, max_error = _run_side(retargeting, landmarks)
        results[side] = {
            "frames": int(len(qpos)),
            "qpos_shape": list(qpos.shape),
            "mean_tip_error_mm": mean_error * 1000.0,
            "max_tip_error_mm": max_error * 1000.0,
            "finite": bool(np.isfinite(qpos).all()),
        }
    return {"sequence": str(path), "tip_names": list(TIP_NAMES), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", nargs="?", default="s07/scissors_use_01.mano.npy")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--right-x-rotation-deg", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run_pilot(
        args.sequence,
        max_frames=args.max_frames,
        device=args.device,
        right_x_rotation_deg=args.right_x_rotation_deg,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
