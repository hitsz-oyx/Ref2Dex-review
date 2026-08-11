"""比较 raw MANO 与 Dataset 动态物体系路径产生的 Y-Teacher V1。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.eval_interaction_compression import make_dataset
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.inverse_mano import build_mano, face_centers
from src.task.InteractionDynamics.inverse_mano_y import points_to_frames
from src.task.InteractionDynamics.uni3d import deterministic_fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v16_object_contact_diverse_overfit.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = make_dataset(args.config)
    hand_path, current = dataset.sample_location(args.sample_index)
    sample = dataset[args.sample_index]
    with np.load(hand_path.parent / "shared.npz", allow_pickle=False) as shared:
        raw_file = str(shared["source_raw_file"].item())
        raw_ids = np.asarray(shared["raw_frame_id"][current:current + 9], np.int64)
        object_poses = torch.from_numpy(np.asarray(
            shared["obj_root_pose_world"][current:current + 9], np.float32)).to(device)
    with np.load(hand_path, allow_pickle=False) as hand_cache:
        side = str(hand_cache["side"].item())
    mano, _, params = build_mano(
        args.grab_root / raw_file, args.grab_root, side, args.mano_path, device)

    def parameter(name: str, width: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(params[name][raw_ids], np.float32).reshape(9, width)).to(device)

    betas_np = np.asarray(params["betas"], np.float32)
    betas_np = betas_np[raw_ids] if betas_np.ndim > 1 else np.broadcast_to(
        betas_np, (9, betas_np.shape[-1])).copy()
    with torch.no_grad():
        output = mano(global_orient=parameter("global_orient", 3),
                      hand_pose=parameter("hand_pose", 24),
                      betas=torch.from_numpy(betas_np).to(device),
                      transl=parameter("transl", 3))
        faces = torch.from_numpy(np.asarray(mano.faces, np.int64)).to(device)
        raw_surface = points_to_frames(face_centers(output.vertices, faces), object_poses)
        dataset_surface = sample["action_hand_points_object_sequence"].to(device)
        object_points = sample["world_obj_points_object"].to(device)
        anchors = object_points[deterministic_fps(object_points[None], 128)[0]]
        raw_y = build_interaction_y(raw_surface, anchors, args.tau_m)
        dataset_y = build_interaction_y(dataset_surface, anchors, args.tau_m)
    result = {"surface_max_abs_m": float((raw_surface - dataset_surface).abs().max())}
    for name in ["relative_geometry", "relative_distance", "relative_motion"]:
        result[f"{name}_max_abs_m"] = float((raw_y[name] - dataset_y[name]).abs().max())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if max(result.values()) >= 1e-5:
        raise SystemExit("Y Teacher parity 未达到 1e-5 m")


if __name__ == "__main__":
    main()
