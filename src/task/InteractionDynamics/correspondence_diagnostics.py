"""诊断 V5 固定 correspondence、累计 target 与静态近场 mask。"""
from __future__ import annotations

import argparse
import json

import torch

from src.base import build_runner_from_checkpoint
from src.task.InteractionDynamics.runner import InteractionDynamicsRunner, patch_motion_target
from src.task.InteractionDynamics.uni3d import gather_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--edge-radius-cm", type=float, default=5.0)
    return parser.parse_args()


def _gather_time_patches(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather [B,K,O,D] by per-step [B,K,H] object-patch indices."""
    batch, steps, hand_patches = index.shape
    return values[
        torch.arange(batch, device=values.device)[:, None, None],
        torch.arange(steps, device=values.device)[None, :, None],
        index,
    ].reshape(batch, steps, hand_patches, values.shape[-1])


def build_correspondence_targets(batch: dict[str, torch.Tensor],
                                 prediction: dict[str, torch.Tensor],
                                 motion_scale: float = 100.0) -> dict[str, torch.Tensor]:
    hand_center = gather_points(batch["world_hand_points_object"].float(),
                                prediction["hand_knn_idx"]).mean(2)
    obj_center = prediction["obj_patch_centers_object"].float()
    hand_cumulative = patch_motion_target(batch["hand_disp_chunk_object_gt"].float(),
                                          prediction["hand_knn_idx"], motion_scale)
    obj_cumulative = patch_motion_target(batch["obj_disp_chunk_gt"].float(),
                                         prediction["obj_knn_idx"], motion_scale)
    future_obj_normal = torch.stack([
        gather_points(batch["obj_normals_chunk_object_gt"][:, step].float(),
                      prediction["obj_knn_idx"]).mean(2)
        for step in range(obj_cumulative.shape[1])
    ], 1)
    future_obj_normal = torch.nn.functional.normalize(future_obj_normal, dim=-1)

    hand_position = hand_center[:, None] * motion_scale + hand_cumulative
    obj_position = obj_center[:, None] * motion_scale + obj_cumulative
    dynamic_nearest = torch.stack([
        torch.cdist(hand_position[:, step].float(), obj_position[:, step].float()).argmin(-1)
        for step in range(hand_position.shape[1])
    ], 1)
    initial_nearest = torch.cdist(hand_center.float(), obj_center.float()).argmin(-1)
    batch_index = torch.arange(hand_center.shape[0], device=hand_center.device)[:, None]
    initial_obj = obj_center[batch_index, initial_nearest]
    initial_distance = (hand_center - initial_obj).norm(dim=-1) * motion_scale
    dynamic_obj_position = _gather_time_patches(obj_position, dynamic_nearest)
    dynamic_distance = (hand_position - dynamic_obj_position).norm(dim=-1)

    hand_increment = torch.diff(hand_cumulative, dim=1,
                                prepend=torch.zeros_like(hand_cumulative[:, :1]))
    obj_increment = torch.diff(obj_cumulative, dim=1,
                               prepend=torch.zeros_like(obj_cumulative[:, :1]))
    fixed_index = initial_nearest[:, None].expand(-1, obj_increment.shape[1], -1)
    fixed_relative_increment = hand_increment - _gather_time_patches(obj_increment, fixed_index)
    dynamic_relative_increment = hand_increment - _gather_time_patches(obj_increment, dynamic_nearest)
    fixed_relative_cumulative = hand_cumulative - _gather_time_patches(obj_cumulative, fixed_index)
    fixed_normal = _gather_time_patches(future_obj_normal, fixed_index)
    dynamic_normal = _gather_time_patches(future_obj_normal, dynamic_nearest)

    def split(relative: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        normal_value = (relative * normal).sum(-1, keepdim=True)
        tangent = relative - normal_value * normal
        return torch.cat([normal_value, tangent], -1)

    return {
        "initial_nearest": initial_nearest,
        "dynamic_nearest": dynamic_nearest,
        "initial_distance_cm": initial_distance,
        "dynamic_distance_cm": dynamic_distance,
        "fixed_cumulative": split(fixed_relative_cumulative, fixed_normal),
        "fixed_increment": split(fixed_relative_increment, fixed_normal),
        "dynamic_increment": split(dynamic_relative_increment, dynamic_normal),
    }


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten().cpu()
    q = torch.quantile(values, torch.tensor([.5, .9, .99]))
    return {"mean": float(values.mean()), "rms": float(values.square().mean().sqrt()),
            "q50": float(q[0]), "q90": float(q[1]), "q99": float(q[2]),
            "max": float(values.max()), "q99_over_q50": float(q[2] / q[0].clamp_min(1e-8))}


def diagnose(runner: InteractionDynamicsRunner, split: str, max_batches: int,
             edge_radius_cm: float) -> dict[str, object]:
    loader = runner.train_loader if split == "train" else (
        runner.val_loaders["val/"] if split == "val" else runner.test_loaders["test/"])
    change_counts, fixed_matches = [], []
    current_near_all, active_all, enter_all, leave_all = [], [], [], []
    magnitudes: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {mask: [] for mask in ("all", "current_near", "chunk_active")}
        for name in ("fixed_cumulative", "fixed_increment", "dynamic_increment")}
    runner.eval_mode()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = runner.prepare_batch(raw_batch)
        with runner.eval_context():
            prediction = runner.model(batch)
        target = build_correspondence_targets(batch, prediction, float(runner.cfg.meta.motion_scale))
        dynamic = target["dynamic_nearest"]
        initial = target["initial_nearest"]
        index_trajectory = torch.cat([initial[:, None], dynamic], 1)
        change_counts.append((index_trajectory[:, 1:] != index_trajectory[:, :-1]).sum(1).cpu())
        fixed_matches.append((dynamic == initial[:, None]).float().cpu())
        current_near = target["initial_distance_cm"] < edge_radius_cm
        future_near = target["dynamic_distance_cm"] < edge_radius_cm
        chunk_active = current_near | future_near.any(1)
        enter = ~current_near & future_near.any(1)
        leave = current_near & ~future_near[:, -1]
        for destination, value in ((current_near_all, current_near), (active_all, chunk_active),
                                   (enter_all, enter), (leave_all, leave)):
            destination.append(value.cpu())
        masks = {"all": torch.ones_like(current_near), "current_near": current_near,
                 "chunk_active": chunk_active}
        for name in magnitudes:
            # Four supervised components use one scalar magnitude so cumulative
            # and step-wise targets remain directly comparable.
            magnitude = target[name].square().mean(-1).sqrt()
            for mask_name, mask in masks.items():
                magnitudes[name][mask_name].append(magnitude[mask[:, None].expand_as(magnitude)].cpu())

    changes = torch.cat(change_counts).float()
    matches = torch.cat(fixed_matches)
    current_near = torch.cat(current_near_all)
    active = torch.cat(active_all)
    enter = torch.cat(enter_all)
    leave = torch.cat(leave_all)
    result: dict[str, object] = {
        "edge_radius_cm": edge_radius_cm,
        "correspondence/fixed_match_ratio": float(matches.mean()),
        "correspondence/edge_changed_ratio": float((changes > 0).float().mean()),
        "correspondence/mean_change_count": float(changes.mean()),
        "correspondence/change_count_quantiles": {
            "q50": float(torch.quantile(changes, .5)), "q90": float(torch.quantile(changes, .9)),
            "q99": float(torch.quantile(changes, .99)), "max": float(changes.max())},
        "mask/current_near_ratio": float(current_near.float().mean()),
        "mask/chunk_active_ratio": float(active.float().mean()),
        "mask/missed_entry_ratio_all_edges": float(enter.float().mean()),
        "mask/missed_entry_ratio_active_edges": float(enter.sum() / active.sum().clamp_min(1)),
        "mask/current_near_final_leave_ratio": float(leave.sum() / current_near.sum().clamp_min(1)),
    }
    for name, by_mask in magnitudes.items():
        for mask_name, chunks in by_mask.items():
            result[f"target/{name}/{mask_name}"] = _distribution(torch.cat(chunks))
    return result


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=args.config, mode="eval", device=args.device)
    if not isinstance(runner, InteractionDynamicsRunner):
        raise ValueError("correspondence_diagnostics.py 只支持 InteractionDynamicsRunner checkpoint")
    print(json.dumps(diagnose(runner, args.split, args.max_batches, args.edge_radius_cm),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
