"""按解析 Y 的 GT 接触覆盖率筛选 V16.1 controlled chunks。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.base import load_config
from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.uni3d import deterministic_fps


def transform_to_object(points_world: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "tni,tij->tnj", points_world - poses[:, None, :3, 3], poses[:, :3, :3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="src/task/InteractionDynamics/configs/grab_v16_object_contact_diverse_overfit.yaml")
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--sigma-m", type=float, default=.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.data
    dataset = InteractionDynamicsDataset(
        data_cfg.train_path, dominant_hand_manifest=data_cfg.dominant_hand_manifest,
        hand_side=data_cfg.hand_side, num_effect_points=cfg.meta.num_effect_points,
        chunk_len=cfg.meta.chunk_len, temporal_stride=cfg.meta.temporal_stride,
        base_seed=cfg.train.seed, max_samples=data_cfg.max_train_samples,
        max_samples_per_sequence=getattr(data_cfg, "max_samples_per_sequence", None),
        min_object_effect_norm=data_cfg.min_object_effect_norm)

    rows = []
    for index in range(len(dataset)):
        hand_path, current = dataset.sample_location(index)
        with np.load(hand_path.parent / "shared.npz", allow_pickle=False) as shared, \
             np.load(hand_path, allow_pickle=False) as hand:
            poses = torch.from_numpy(np.asarray(
                shared["obj_root_pose_world"][current:current + 9], np.float32))
            object_world = torch.from_numpy(np.asarray(
                shared["obj_points_world"][current], np.float32))
            hand_world = torch.from_numpy(np.asarray(
                hand["hand_points_world"][current:current + 9], np.float32))
            seq_id = str(shared["seq_id"].item())
            raw_frame = int(shared["raw_frame_id"][current])
        object_canonical = transform_to_object(object_world[None], poses[:1])[0]
        anchors = object_canonical[deterministic_fps(object_canonical[None], 128)[0]]
        hand_object = transform_to_object(hand_world, poses)
        y = build_interaction_y(hand_object, anchors, args.tau_m)
        contact = torch.exp(
            -y["relative_distance"].square() / (2 * args.sigma_m ** 2)) > .5
        rows.append((float(contact.float().mean()), index, seq_id, raw_frame))

    for active, index, seq_id, raw_frame in sorted(rows, reverse=True):
        print(f"sample={index:02d} active={100 * active:6.2f}% "
              f"raw_frame={raw_frame:06d} seq={seq_id}")


if __name__ == "__main__":
    main()
