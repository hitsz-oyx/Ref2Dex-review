"""量化 V16 controlled set 的目标多样性及当前几何捷径强度。"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.base import load_config
from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.runner import object_contact_target
from src.task.InteractionDynamics.uni3d import gather_points, patchify


def binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction > .5
    truth = target > .5
    tp = (pred & truth).sum().float()
    precision = tp / pred.sum().clamp_min(1)
    recall = tp / truth.sum().clamp_min(1)
    return float(2 * precision * recall / (precision + recall).clamp_min(1e-8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg.data
    dataset = InteractionDynamicsDataset(
        data.train_path, dominant_hand_manifest=data.dominant_hand_manifest,
        hand_side=data.hand_side, num_effect_points=cfg.meta.num_effect_points,
        chunk_len=cfg.meta.chunk_len, temporal_stride=cfg.meta.temporal_stride,
        base_seed=cfg.train.seed, max_samples=data.max_train_samples,
        max_samples_per_sequence=getattr(data, "max_samples_per_sequence", None),
        min_object_effect_norm=data.min_object_effect_norm)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    _, _, obj_knn = patchify(batch["world_obj_points_object"], 64, 32)
    target = object_contact_target(batch, {"obj_knn_idx": obj_knn}, cfg.meta.contact_sigma_m)
    rolled = target.roll(1, 0)

    current_hand = gather_points(
        batch["world_hand_points_object"], batch["action_patch_knn_idx"]).mean(2)
    current_obj = gather_points(batch["world_obj_points_object"], obj_knn).mean(2)
    distance = torch.cdist(current_obj.float(), current_hand.float()).amin(-1)
    current_geometry = torch.exp(-distance.square() / (2 * cfg.meta.contact_sigma_m ** 2))
    mean_target = target.mean(0, keepdim=True).expand_as(target)
    zero_mse = float(target.square().mean())
    result = {
        "samples": len(dataset),
        "target_active_fraction": float((target > .5).float().mean()),
        "zero_mse": zero_mse,
        "cross_sample_target_mse": float(F.mse_loss(rolled, target)),
        "cross_sample_target_f1": binary_f1(rolled, target),
        "mean_target_mse": float(F.mse_loss(mean_target, target)),
        "mean_target_f1": binary_f1(mean_target, target),
        "current_geometry_mse": float(F.mse_loss(current_geometry, target)),
        "current_geometry_relative_improvement": float(
            1 - F.mse_loss(current_geometry, target) / target.square().mean()),
        "current_geometry_f1": binary_f1(current_geometry, target),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
