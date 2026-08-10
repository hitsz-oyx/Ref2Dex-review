"""统计 hand-object 局部相对轨迹是否比 raw patch flow 更具辨识力。"""
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
    parser.add_argument("--contact-threshold-cm", type=float, default=2.0)
    return parser.parse_args()


def relative_patch_targets(batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor],
                           motion_scale: float = 100.0) -> dict[str, torch.Tensor]:
    """构造每个 hand patch 到当前最近 object patch 的 8 步相对轨迹。"""
    hand_center = gather_points(batch["world_hand_points_object"].float(),
                                prediction["hand_knn_idx"]).mean(2)
    obj_center = prediction["obj_patch_centers_object"].float()
    obj_normal = gather_points(batch["world_obj_normals_object"].float(),
                               prediction["obj_knn_idx"]).mean(2)
    obj_normal = torch.nn.functional.normalize(obj_normal, dim=-1)
    nearest = torch.cdist(hand_center, obj_center).argmin(-1)
    batch_index = torch.arange(hand_center.shape[0], device=hand_center.device)[:, None]
    paired_obj_center = obj_center[batch_index, nearest]
    paired_obj_normal = obj_normal[batch_index, nearest]

    hand_motion = patch_motion_target(batch["hand_disp_chunk_object_gt"].float(),
                                      prediction["hand_knn_idx"], motion_scale)
    obj_motion_all = patch_motion_target(batch["obj_disp_chunk_gt"].float(),
                                         prediction["obj_knn_idx"], motion_scale)
    obj_motion = obj_motion_all[
        torch.arange(obj_motion_all.shape[0], device=obj_motion_all.device)[:, None, None],
        torch.arange(obj_motion_all.shape[1], device=obj_motion_all.device)[None, :, None],
        nearest[:, None, :],
    ]
    relative_motion = hand_motion - obj_motion
    normal_motion = (relative_motion * paired_obj_normal[:, None]).sum(-1)
    tangent_motion = relative_motion - normal_motion[..., None] * paired_obj_normal[:, None]
    initial_relative_cm = (hand_center - paired_obj_center) * motion_scale
    relative_position_cm = initial_relative_cm[:, None] + relative_motion
    distance_cm = relative_position_cm.norm(dim=-1)
    initial_distance_cm = initial_relative_cm.norm(dim=-1)
    return {"initial_distance_cm": initial_distance_cm,
            "distance_cm": distance_cm, "distance_change_cm": distance_cm - initial_distance_cm[:, None],
            "normal_motion_cm": normal_motion, "tangent_motion_cm": tangent_motion,
            "nearest_obj_patch": nearest}


def _energy(values: torch.Tensor) -> tuple[float, float]:
    local = values - values.mean(2, keepdim=True)
    return float(values.square().sum()), float(local.square().sum())


def diagnose(runner: InteractionDynamicsRunner, split: str, max_batches: int,
             contact_threshold_cm: float) -> dict[str, object]:
    loader = runner.train_loader if split == "train" else (
        runner.val_loaders["val/"] if split == "val" else runner.test_loaders["test/"])
    samples: dict[str, list[torch.Tensor]] = {key: [] for key in
        ("distance_cm", "distance_change_cm", "normal_motion_cm", "tangent_speed_cm")}
    total_energy = {key: [0.0, 0.0] for key in
                    ("distance_change", "normal_motion", "tangent_motion")}
    transitions = {"far_to_far": 0, "far_to_contact": 0,
                   "contact_to_contact": 0, "contact_to_far": 0}
    runner.eval_mode()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = runner.prepare_batch(raw_batch)
        with runner.eval_context():
            prediction = runner.model(batch)
        target = relative_patch_targets(batch, prediction, float(runner.cfg.meta.motion_scale))
        tangent_speed = target["tangent_motion_cm"].norm(dim=-1)
        for key, value in (("distance_cm", target["distance_cm"]),
                           ("distance_change_cm", target["distance_change_cm"]),
                           ("normal_motion_cm", target["normal_motion_cm"]),
                           ("tangent_speed_cm", tangent_speed)):
            samples[key].append(value.detach().cpu().flatten())
        for key, value in (("distance_change", target["distance_change_cm"]),
                           ("normal_motion", target["normal_motion_cm"]),
                           ("tangent_motion", target["tangent_motion_cm"])):
            total, local = _energy(value)
            total_energy[key][0] += total
            total_energy[key][1] += local
        contact = target["distance_cm"] < contact_threshold_cm
        initial = (target["initial_distance_cm"] < contact_threshold_cm)[:, None]
        previous = torch.cat([initial, contact[:, :-1]], dim=1)
        transitions["far_to_far"] += int((~previous & ~contact).sum())
        transitions["far_to_contact"] += int((~previous & contact).sum())
        transitions["contact_to_contact"] += int((previous & contact).sum())
        transitions["contact_to_far"] += int((previous & ~contact).sum())

    quantiles = torch.tensor([0.0, .1, .25, .5, .75, .9, .99, 1.0])
    result: dict[str, object] = {}
    for key, chunks in samples.items():
        values = torch.cat(chunks)
        result[f"{key}/mean"] = float(values.mean())
        result[f"{key}/quantiles"] = {str(float(q)): float(v) for q, v in
                                       zip(quantiles, torch.quantile(values, quantiles))}
    for key, (total, local) in total_energy.items():
        result[f"{key}/local_energy_ratio"] = local / max(total, 1e-12)
    transition_total = sum(transitions.values())
    result["contact/threshold_cm"] = contact_threshold_cm
    result["contact/transitions"] = transitions
    result["contact/change_ratio"] = (
        transitions["far_to_contact"] + transitions["contact_to_far"]) / max(transition_total, 1)
    result["contact/occupancy"] = (
        transitions["far_to_contact"] + transitions["contact_to_contact"]) / max(transition_total, 1)
    return result


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=args.config, mode="eval", device=args.device)
    if not isinstance(runner, InteractionDynamicsRunner):
        raise ValueError("relative_diagnostics.py 只支持 InteractionDynamicsRunner checkpoint")
    print(json.dumps(diagnose(runner, args.split, args.max_batches, args.contact_threshold_cm),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
