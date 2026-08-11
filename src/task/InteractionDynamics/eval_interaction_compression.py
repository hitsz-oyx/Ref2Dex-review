"""在 sequence-disjoint GRAB chunks 上评估 Y-Teacher V1 的 PCA 压缩基线。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.base import load_config
from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.uni3d import deterministic_fps


def pack_y(y: dict[str, torch.Tensor]) -> torch.Tensor:
    """按 cm 尺度 flatten 冻结的 r[9]、d[9]、u[8]。"""
    return torch.cat([
        (100 * y["relative_geometry"]).flatten(),
        (100 * y["relative_distance"]).flatten(),
        (100 * y["relative_motion"]).flatten(),
    ])


def unpack_y(vector: torch.Tensor, anchors: int = 128) -> dict[str, torch.Tensor]:
    r_size = 9 * anchors * 3
    d_size = 9 * anchors
    return {
        "relative_geometry": vector[..., :r_size].reshape(*vector.shape[:-1], 9, anchors, 3),
        "relative_distance": vector[..., r_size:r_size + d_size].reshape(
            *vector.shape[:-1], 9, anchors),
        "relative_motion": vector[..., r_size + d_size:].reshape(
            *vector.shape[:-1], 8, anchors, 3),
    }


def make_dataset(config_path: str) -> InteractionDynamicsDataset:
    cfg = load_config(config_path)
    data = cfg.data
    return InteractionDynamicsDataset(
        data.train_path, dominant_hand_manifest=data.dominant_hand_manifest,
        hand_side=data.hand_side, num_effect_points=cfg.meta.num_effect_points,
        chunk_len=cfg.meta.chunk_len, temporal_stride=cfg.meta.temporal_stride,
        base_seed=cfg.train.seed, min_object_effect_norm=data.min_object_effect_norm)


def sample_y(dataset: InteractionDynamicsDataset, index: int,
             device: torch.device, tau_m: float) -> torch.Tensor:
    sample = dataset[index]
    hand0 = torch.as_tensor(sample["world_hand_points_object"], device=device)
    displacement = torch.as_tensor(sample["action_hand_disp_chunk_object"], device=device)
    hand = torch.cat([hand0[None], hand0[None] + displacement], dim=0)
    obj = torch.as_tensor(sample["world_obj_points_object"], device=device)
    anchors = obj[deterministic_fps(obj[None], 128)[0]]
    return pack_y(build_interaction_y(hand, anchors, tau_m)).cpu()


def sequence_disjoint_indices(dataset: InteractionDynamicsDataset, seed: int,
                              max_train: int, max_eval: int) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        path, _ = dataset.sample_location(index)
        groups.setdefault(str(path.parent), []).append(index)
    names = sorted(groups)
    np.random.default_rng(seed).shuffle(names)
    boundary = max(1, int(.8 * len(names)))
    train = [i for name in names[:boundary] for i in groups[name]][:max_train]
    evaluate = [i for name in names[boundary:] for i in groups[name]][:max_eval]
    if len(train) <= 1024:
        raise ValueError("PCA-1024 至少需要 1025 个 sequence-disjoint 训练 chunks")
    return train, evaluate


def reconstruction_metrics(prediction: torch.Tensor,
                           target: torch.Tensor) -> dict[str, float]:
    pred, gt = unpack_y(prediction), unpack_y(target)
    result = {}
    for short, name in [("r", "relative_geometry"), ("d", "relative_distance"),
                        ("u", "relative_motion")]:
        result[f"{short}_rmse_cm"] = float(
            torch.nn.functional.mse_loss(pred[name], gt[name]).sqrt())
    pred_contact = torch.exp(-pred["relative_distance"].square() / 2) > .5
    gt_contact = torch.exp(-gt["relative_distance"].square() / 2) > .5
    tp = (pred_contact & gt_contact).sum().float()
    precision = tp / pred_contact.sum().clamp_min(1)
    recall = tp / gt_contact.sum().clamp_min(1)
    result["contact_f1"] = float(
        2 * precision * recall / (precision + recall).clamp_min(1e-8))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--max-train", type=int, default=1280)
    parser.add_argument("--max-eval", type=int, default=256)
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = make_dataset(args.config)
    train_indices, eval_indices = sequence_disjoint_indices(
        dataset, args.seed, args.max_train, args.max_eval)
    train = torch.stack([sample_y(dataset, i, device, args.tau_m) for i in train_indices])
    evaluate = torch.stack([sample_y(dataset, i, device, args.tau_m) for i in eval_indices])
    mean = train.mean(0)
    centered = (train - mean).to(device)
    _, _, basis = torch.pca_lowrank(centered, q=max(args.dims), center=False, niter=4)
    eval_device = evaluate.to(device)
    rows = []
    for dim in args.dims:
        projection = basis[:, :dim]
        reconstruction = (eval_device - mean.to(device)) @ projection @ projection.T + mean.to(device)
        row = {"representation": "PCA", "dim": dim,
               **reconstruction_metrics(reconstruction.cpu(), evaluate)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "basis": basis.cpu(), "metrics": rows,
                "train_indices": train_indices, "eval_indices": eval_indices}, args.output)


if __name__ == "__main__":
    main()
