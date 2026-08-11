"""训练和干预评估 V16.5 state-conditioned interaction diffusion。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch3d.transforms import matrix_to_axis_angle

from src.task.InteractionDynamics.eval_interaction_compression import make_dataset
from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.state_interaction_diffusion import (
    StateInteractionDiffusion, pack_state_future, sample_state_ddim)
from src.task.InteractionDynamics.train_interaction_diffusion import controlled_indices
from src.task.InteractionDynamics.uni3d import gather_points, patchify


def materialize(dataset, indices: list[int], device: torch.device, horizon: int,
                tau_m: float, patch_size: int = 32) -> dict[str, torch.Tensor]:
    rows = {key: [] for key in ["anchors_cm", "object_patches", "state", "effect", "future"]}
    for index in indices:
        sample = dataset[index]
        hand = sample["action_hand_points_object_sequence"][:horizon + 1].to(device)
        points = sample["world_obj_points_object"].to(device)
        normals = sample["world_obj_normals_object"].to(device)
        anchors, _, knn = patchify(points[None], 128, patch_size)
        local = gather_points(points[None], knn) - anchors[:, :, None]
        patches = torch.cat([100 * local, gather_points(normals[None], knn)], -1)[0]
        y = build_interaction_y(hand, anchors[0], tau_m)
        y_cm = {key: 100 * value for key, value in y.items()
                if key in {"relative_geometry", "relative_distance", "relative_motion"}}
        state, future = pack_state_future(y_cm, horizon)
        endpoint = torch.eye(4, device=device)
        for increment in sample["obj_increment_pose_gt"][:horizon].to(device):
            endpoint = endpoint @ increment
        effect = torch.cat([100 * endpoint[:3, 3], matrix_to_axis_angle(endpoint[:3, :3])])
        for key, value in [("anchors_cm", 100 * anchors[0]), ("object_patches", patches),
                           ("state", state), ("effect", effect), ("future", future)]:
            rows[key].append(value.cpu())
    return {key: torch.stack(value) for key, value in rows.items()}


def future_metrics(prediction: torch.Tensor, target: torch.Tensor,
                   horizon: int) -> dict[str, float]:
    shaped_p = prediction.reshape(*prediction.shape[:-1], horizon, 7)
    shaped_t = target.reshape(*target.shape[:-1], horizon, 7)
    rmse = lambda a, b: float(torch.nn.functional.mse_loss(a, b).sqrt())
    pred_contact = torch.exp(-shaped_p[..., 6].square() / 2) > .5
    gt_contact = torch.exp(-shaped_t[..., 6].square() / 2) > .5
    tp = (pred_contact & gt_contact).sum().float()
    precision = tp / pred_contact.sum().clamp_min(1)
    recall = tp / gt_contact.sum().clamp_min(1)
    return {"u_rmse_cm": rmse(shaped_p[..., :3], shaped_t[..., :3]),
            "r_rmse_cm": rmse(shaped_p[..., 3:6], shaped_t[..., 3:6]),
            "d_rmse_cm": rmse(shaped_p[..., 6], shaped_t[..., 6]),
            "contact_precision": float(precision), "contact_recall": float(recall),
            "contact_f1": float(2 * precision * recall / (precision + recall).clamp_min(1e-8))}


@torch.no_grad()
def generate(model, data, stats, steps, batch_size, device, state_mode, effect_mode):
    predictions = []
    for start in range(0, len(data["future"]), batch_size):
        batch = {key: value[start:start + batch_size].to(device) for key, value in data.items()}
        state = batch["state"] if state_mode == "correct" else (
            batch["state"].roll(1, 0) if state_mode == "shuffle" else torch.zeros_like(batch["state"]))
        effect = batch["effect"] if effect_mode == "correct" else (
            batch["effect"].roll(1, 0) if effect_mode == "shuffle" else torch.zeros_like(batch["effect"]))
        generator = torch.Generator(device=device).manual_seed(20260811 + start)
        normalized = sample_state_ddim(
            model, (state - stats["state_mean"]) / stats["state_std"],
            batch["anchors_cm"], batch["object_patches"],
            (effect - stats["effect_mean"]) / stats["effect_std"], steps, generator=generator)
        predictions.append((normalized * stats["future_std"] + stats["future_mean"]).cpu())
    return torch.cat(predictions)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = make_dataset(args.config)
    indices = controlled_indices(dataset, args.samples, args.sequence_index)
    data = materialize(dataset, indices, device, args.horizon, args.tau_m)
    stats = {}
    for key in ["state", "future"]:
        stats[f"{key}_mean"] = data[key].mean((0, 1), keepdim=True).to(device)
        stats[f"{key}_std"] = data[key].std((0, 1), keepdim=True).clamp_min(.05).to(device)
    stats["effect_mean"] = data["effect"].mean(0, keepdim=True).to(device)
    stats["effect_std"] = data["effect"].std(0, keepdim=True).clamp_min(.01).to(device)
    model = StateInteractionDiffusion(args.horizon, args.dim, 8, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    _, alpha_bar = cosine_schedule(args.diffusion_steps, device)
    index_generator = torch.Generator().manual_seed(args.seed)
    history = []
    model.train()
    for step in range(1, args.train_steps + 1):
        index = torch.randint(len(data["future"]), (args.batch_size,), generator=index_generator)
        batch = {key: value[index].to(device) for key, value in data.items()}
        clean = (batch["future"] - stats["future_mean"]) / stats["future_std"]
        state = (batch["state"] - stats["state_mean"]) / stats["state_std"]
        effect = (batch["effect"] - stats["effect_mean"]) / stats["effect_std"]
        timestep = torch.randint(args.diffusion_steps, (args.batch_size,), device=device)
        noise = torch.randn_like(clean)
        scale = alpha_bar[timestep][:, None, None]
        noisy = scale.sqrt() * clean + (1 - scale).sqrt() * noise
        permutation = torch.stack([torch.randperm(128, device=device) for _ in range(args.batch_size)])
        def gather(value):
            index_shape = (*permutation.shape, *((1,) * (value.ndim - 2)))
            return value.gather(1, permutation.reshape(index_shape).expand_as(value))
        prediction = model(gather(noisy), gather(state), gather(batch["anchors_cm"]),
                           gather(batch["object_patches"]), effect, timestep)
        loss = torch.nn.functional.mse_loss(prediction, gather(noise))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "noise_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    model.eval()
    metrics = {}
    for intervention, state_mode, effect_mode in [
            ("correct", "correct", "correct"), ("state_shuffle", "shuffle", "correct"),
            ("state_zero", "zero", "correct"), ("effect_shuffle", "correct", "shuffle"),
            ("effect_zero", "correct", "zero")]:
        prediction = generate(model, data, stats, args.diffusion_steps, args.batch_size,
                              device, state_mode, effect_mode)
        metrics[intervention] = future_metrics(prediction, data["future"], args.horizon)
        print(json.dumps({"intervention": intervention, **metrics[intervention]}, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "indices": indices, "horizon": args.horizon,
                "dim": args.dim, "layers": args.layers, "history": history, "metrics": metrics,
                **{key: value.cpu() for key, value in stats.items()}}, args.output)


if __name__ == "__main__":
    main()
