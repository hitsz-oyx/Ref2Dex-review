"""训练并评估 V16.6 active + next-effect interaction diffusion。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.eval_interaction_compression import make_dataset
from src.task.InteractionDynamics.goal_interaction_diffusion import (
    GoalInteractionDiffusion, estimate_active_interval, extract_goal,
    meaningful_motion_mask, sample_goal_ddim, stratified_frames)
from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future
from src.task.InteractionDynamics.train_state_interaction_diffusion import future_metrics
from src.task.InteractionDynamics.uni3d import gather_points, patchify


def controlled_dataset(config: str, sequence_index: int, samples: int, horizon: int,
                       translation_threshold_cm: float, rotation_threshold_deg: float,
                       motion_window: int) -> tuple[InteractionDynamicsDataset, list[int], dict]:
    manifest_dataset = make_dataset(config)
    paths = sorted({path for path, _ in manifest_dataset._samples})
    path = paths[sequence_index]
    dataset = InteractionDynamicsDataset(manifest_dataset.data_root, file_list=[path])
    data = dataset._load(path)
    poses = torch.from_numpy(np.asarray(data["obj_root_pose_world"], np.float32))
    meaningful = meaningful_motion_mask(
        poses, motion_window, translation_threshold_cm, rotation_threshold_deg).numpy()
    active_start, active_end, distance = estimate_active_interval(data, meaningful)
    # 底层 V1 Dataset 仍固定物化 8-step chunk，即使 V16.6 只消费 H=4。
    frames = stratified_frames(len(poses), active_start, active_end, max(horizon, 8), samples)
    frame_to_index = {frame: index for index, (_, frame) in enumerate(dataset._samples)}
    indices = [frame_to_index[frame] for frame in frames]
    return dataset, indices, {"path": str(path), "frames": frames, "active_start": active_start,
                              "active_end": active_end, "distance_cm": distance,
                              "object_poses": poses}


def materialize(dataset, indices, metadata, device, horizon, goal_segment, tau_m,
                motion_window, translation_threshold_cm, rotation_threshold_deg):
    rows = {key: [] for key in ["anchors_cm", "object_patches", "state", "goal", "future"]}
    onsets = []
    poses = metadata["object_poses"].to(device)
    for index in indices:
        sample = dataset[index]
        _, current = dataset.sample_location(index)
        active = metadata["active_start"] <= current <= metadata["active_end"]
        goal, onset = extract_goal(poses, current, active, goal_segment, motion_window,
                                   translation_threshold_cm, rotation_threshold_deg)
        hand = sample["action_hand_points_object_sequence"][:horizon + 1].to(device)
        points, normals = sample["world_obj_points_object"].to(device), sample["world_obj_normals_object"].to(device)
        anchors, _, knn = patchify(points[None], 128, 32)
        patches = torch.cat([100 * (gather_points(points[None], knn) - anchors[:, :, None]),
                             gather_points(normals[None], knn)], -1)[0]
        y = build_interaction_y(hand, anchors[0], tau_m)
        y = {key: 100 * value for key, value in y.items()
             if key in {"relative_geometry", "relative_distance", "relative_motion"}}
        state, future = pack_state_future(y, horizon)
        for key, value in [("anchors_cm", 100 * anchors[0]), ("object_patches", patches),
                           ("state", state), ("goal", goal), ("future", future)]:
            rows[key].append(value.cpu())
        onsets.append(onset)
    result = {key: torch.stack(value) for key, value in rows.items()}
    result["onset"] = torch.tensor(onsets)
    return result


def persistence_future(state: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.cat([torch.cat([torch.zeros_like(state[..., :3]), state], -1)
                      for _ in range(horizon)], -1)


@torch.no_grad()
def generate(model, data, stats, steps, batch_size, device, state_mode="correct", goal_mode="correct"):
    predictions = []
    active_donors = torch.nonzero(data["goal"][:, 0] > .5).flatten()
    for start in range(0, len(data["future"]), batch_size):
        batch = {key: value[start:start + batch_size].to(device)
                 for key, value in data.items() if key != "onset"}
        state = batch["state"] if state_mode == "correct" else (
            batch["state"].roll(1, 0) if state_mode == "shuffle" else torch.zeros_like(batch["state"]))
        goal = batch["goal"].clone()
        if goal_mode == "motion_shuffle":
            active = goal[:, 0] > .5
            goal[active, 1:] = goal[active, 1:].roll(1, 0)
        elif goal_mode == "motion_zero":
            goal[goal[:, 0] > .5, 1:] = 0
        elif goal_mode == "active_zero":
            goal[goal[:, 0] > .5] = 0
        elif goal_mode == "active_flip":
            inactive = goal[:, 0] < .5
            if inactive.any() and len(active_donors):
                donor = data["goal"][active_donors[(start // batch_size) % len(active_donors)]].to(device)
                goal[inactive] = donor
        elif goal_mode != "correct":
            raise ValueError(goal_mode)
        generator = torch.Generator(device=device).manual_seed(20260812 + start)
        normalized = sample_goal_ddim(
            model, (state - stats["state_mean"]) / stats["state_std"], batch["anchors_cm"],
            batch["object_patches"], (goal - stats["goal_mean"]) / stats["goal_std"],
            steps, generator=generator)
        predictions.append((normalized * stats["future_std"] + stats["future_mean"]).cpu())
    return torch.cat(predictions)


def subset_metrics(prediction, target, masks, horizon):
    return {name: ({"count": int(mask.sum()), **future_metrics(prediction[mask], target[mask], horizon)}
                   if mask.any() else {"count": 0}) for name, mask in masks.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--goal-segment", type=int, default=4)
    parser.add_argument("--motion-window", type=int, default=3)
    parser.add_argument("--translation-threshold-cm", type=float, default=.2)
    parser.add_argument("--rotation-threshold-deg", type=float, default=1.)
    parser.add_argument("--dynamic-threshold-cm", type=float, default=.2)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset, indices, metadata = controlled_dataset(
        args.config, args.sequence_index, args.samples, args.horizon,
        args.translation_threshold_cm, args.rotation_threshold_deg, args.motion_window)
    data = materialize(dataset, indices, metadata, device, args.horizon, args.goal_segment,
                       args.tau_m, args.motion_window, args.translation_threshold_cm,
                       args.rotation_threshold_deg)
    stats = {}
    for key, dimensions in [("state", (0, 1)), ("future", (0, 1)), ("goal", (0,))]:
        stats[f"{key}_mean"] = data[key].mean(dimensions, keepdim=True).to(device)
        stats[f"{key}_std"] = data[key].std(dimensions, keepdim=True).clamp_min(.01 if key == "goal" else .05).to(device)
    model = GoalInteractionDiffusion(args.horizon, args.goal_segment, args.dim, 8, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    _, alpha_bar = cosine_schedule(args.diffusion_steps, device)
    index_generator = torch.Generator().manual_seed(args.seed)
    history = []
    model.train()
    for step in range(1, args.train_steps + 1):
        index = torch.randint(len(data["future"]), (args.batch_size,), generator=index_generator)
        batch = {key: value[index].to(device) for key, value in data.items() if key != "onset"}
        clean = (batch["future"] - stats["future_mean"]) / stats["future_std"]
        state = (batch["state"] - stats["state_mean"]) / stats["state_std"]
        goal = (batch["goal"] - stats["goal_mean"]) / stats["goal_std"]
        timestep = torch.randint(args.diffusion_steps, (args.batch_size,), device=device)
        noise = torch.randn_like(clean)
        scale = alpha_bar[timestep][:, None, None]
        noisy = scale.sqrt() * clean + (1 - scale).sqrt() * noise
        permutation = torch.stack([torch.randperm(128, device=device) for _ in range(args.batch_size)])
        def gather(value):
            shape = (*permutation.shape, *((1,) * (value.ndim - 2)))
            return value.gather(1, permutation.reshape(shape).expand_as(value))
        prediction = model(gather(noisy), gather(state), gather(batch["anchors_cm"]),
                           gather(batch["object_patches"]), goal, timestep)
        loss = torch.nn.functional.mse_loss(prediction, gather(noise))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "noise_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    persistence = persistence_future(data["state"], args.horizon)
    change = (data["future"] - persistence).square().mean((1, 2)).sqrt()
    masks = {"overall": torch.ones(len(change), dtype=torch.bool),
             "dynamic": change >= args.dynamic_threshold_cm,
             "static": change < args.dynamic_threshold_cm,
             "active": data["goal"][:, 0] > .5,
             "inactive": data["goal"][:, 0] < .5}
    metrics = {"persistence": subset_metrics(persistence, data["future"], masks, args.horizon)}
    model.eval()
    predictions = {}
    for mode in ["correct", "motion_shuffle", "motion_zero", "active_zero", "active_flip"]:
        predictions[mode] = generate(model, data, stats, args.diffusion_steps,
                                     args.batch_size, device, goal_mode=mode)
        metrics[mode] = subset_metrics(predictions[mode], data["future"], masks, args.horizon)
    state_shuffle = generate(model, data, stats, args.diffusion_steps, args.batch_size,
                             device, state_mode="shuffle")
    metrics["state_shuffle"] = subset_metrics(state_shuffle, data["future"], masks, args.horizon)
    inactive = masks["inactive"]
    metrics["active_flip"]["generated_change_from_correct_inactive_cm"] = (
        float((predictions["active_flip"][inactive] - predictions["correct"][inactive]).square().mean().sqrt())
        if inactive.any() else 0.)
    summary = {"sequence": metadata["path"], "frames": metadata["frames"],
               "active_interval": [metadata["active_start"], metadata["active_end"]],
               "active_count": int(masks["active"].sum()), "inactive_count": int(masks["inactive"].sum()),
               "change_cm": {"min": float(change.min()), "median": float(change.median()),
                             "max": float(change.max())}, "metrics": metrics}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "indices": indices, "history": history,
                "horizon": args.horizon, "goal_segment": args.goal_segment,
                "dim": args.dim, "layers": args.layers, **summary,
                **{key: value.cpu() for key, value in stats.items()}}, args.output)


if __name__ == "__main__":
    main()
