"""训练并干预评估 V16.4 Object-Effect conditional Y diffusion。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from src.task.InteractionDynamics.eval_interaction_compression import make_dataset
from src.task.InteractionDynamics.interaction_diffusion import (
    InteractionFieldDiffusion, cosine_schedule, pack_anchor_y, sample_ddpm)
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.uni3d import deterministic_fps


def controlled_indices(dataset, count: int, sequence_index: int = 0) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        path, _ = dataset.sample_location(index)
        groups.setdefault(str(path.parent), []).append(index)
    candidates = [values for _, values in sorted(groups.items()) if len(values) >= count]
    if sequence_index >= len(candidates):
        raise ValueError(f"只有 {len(candidates)} 条 sequence 至少包含 {count} 个 chunks")
    values = candidates[sequence_index]
    positions = np.linspace(0, len(values) - 1, count)
    return [values[int(round(position))] for position in positions]


def materialize(dataset, indices: list[int], device: torch.device,
                tau_m: float) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in ["anchors_cm", "anchor_normals", "effect", "y"]}
    for index in indices:
        sample = dataset[index]
        hand = sample["action_hand_points_object_sequence"].to(device)
        obj = sample["world_obj_points_object"].to(device)
        normals = sample["world_obj_normals_object"].to(device)
        anchor_index = deterministic_fps(obj[None], 128)[0]
        anchors = obj[anchor_index]
        y = build_interaction_y(hand, anchors, tau_m)
        increment = sample["obj_increment_pose_gt"]
        effect = torch.cat([
            100 * increment[:, :3, 3], matrix_to_axis_angle(increment[:, :3, :3])], -1)
        rows["anchors_cm"].append((100 * anchors).cpu())
        rows["anchor_normals"].append(normals[anchor_index].cpu())
        rows["effect"].append(effect.cpu())
        y_cm = {name: 100 * value for name, value in y.items()
                if name in {"relative_geometry", "relative_distance", "relative_motion"}}
        rows["y"].append(pack_anchor_y(y_cm).cpu())
    return {name: torch.stack(values) for name, values in rows.items()}


def field_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result = {
        "r_rmse_cm": float(torch.nn.functional.mse_loss(prediction[..., :27], target[..., :27]).sqrt()),
        "d_rmse_cm": float(torch.nn.functional.mse_loss(prediction[..., 27:36], target[..., 27:36]).sqrt()),
        "u_rmse_cm": float(torch.nn.functional.mse_loss(prediction[..., 36:], target[..., 36:]).sqrt()),
    }
    pred_contact = torch.exp(-prediction[..., 27:36].square() / 2) > .5
    gt_contact = torch.exp(-target[..., 27:36].square() / 2) > .5
    tp = (pred_contact & gt_contact).sum().float()
    precision = tp / pred_contact.sum().clamp_min(1)
    recall = tp / gt_contact.sum().clamp_min(1)
    result.update(contact_precision=float(precision), contact_recall=float(recall),
                  contact_f1=float(2 * precision * recall / (precision + recall).clamp_min(1e-8)))
    return result


@torch.no_grad()
def generate(model: InteractionFieldDiffusion, data: dict[str, torch.Tensor],
             y_mean: torch.Tensor, y_std: torch.Tensor,
             effect_mean: torch.Tensor, effect_std: torch.Tensor,
             diffusion_steps: int, batch_size: int, device: torch.device,
             effect_mode: str) -> torch.Tensor:
    predictions = []
    for start in range(0, len(data["y"]), batch_size):
        batch = {name: value[start:start + batch_size].to(device) for name, value in data.items()}
        if effect_mode == "correct":
            effect = batch["effect"]
        elif effect_mode == "zero":
            effect = torch.zeros_like(batch["effect"])
        elif effect_mode == "shuffle":
            effect = batch["effect"].roll(1, 0)
        else:
            raise ValueError(effect_mode)
        effect = (effect - effect_mean) / effect_std
        generator = torch.Generator(device=device).manual_seed(20260811 + start)
        normalized = sample_ddpm(
            model, batch["anchors_cm"], batch["anchor_normals"], effect,
            diffusion_steps, generator=generator)
        predictions.append((normalized * y_std + y_mean).cpu())
    return torch.cat(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sequence-index", type=int, default=0)
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = make_dataset(args.config)
    indices = controlled_indices(dataset, args.samples, args.sequence_index)
    data = materialize(dataset, indices, device, args.tau_m)
    y_mean = data["y"].mean((0, 1), keepdim=True).to(device)
    y_std = data["y"].std((0, 1), keepdim=True).clamp_min(.05).to(device)
    effect_mean = data["effect"].mean((0, 1), keepdim=True).to(device)
    effect_std = data["effect"].std((0, 1), keepdim=True).clamp_min(.01).to(device)
    model = InteractionFieldDiffusion(args.dim, 8, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    _, alpha_bar = cosine_schedule(args.diffusion_steps, device)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    model.train()
    for step in range(1, args.train_steps + 1):
        index = torch.randint(len(data["y"]), (args.batch_size,), generator=generator)
        batch = {name: value[index].to(device) for name, value in data.items()}
        clean = (batch["y"] - y_mean) / y_std
        timestep = torch.randint(args.diffusion_steps, (args.batch_size,), device=device)
        noise = torch.randn_like(clean)
        scale = alpha_bar[timestep][:, None, None]
        noisy = scale.sqrt() * clean + (1 - scale).sqrt() * noise
        permutation = torch.stack([torch.randperm(128, device=device) for _ in range(args.batch_size)])
        gather_y = permutation[..., None].expand(-1, -1, 60)
        gather_x = permutation[..., None].expand(-1, -1, 3)
        prediction = model(
            noisy.gather(1, gather_y), batch["anchors_cm"].gather(1, gather_x),
            batch["anchor_normals"].gather(1, gather_x),
            (batch["effect"] - effect_mean) / effect_std, timestep)
        loss = torch.nn.functional.mse_loss(prediction, noise.gather(1, gather_y))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "noise_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    model.eval()
    metrics = {}
    for mode in ["correct", "shuffle", "zero"]:
        prediction = generate(model, data, y_mean, y_std, effect_mean, effect_std,
                              args.diffusion_steps, args.batch_size, device, mode)
        metrics[mode] = field_metrics(prediction, data["y"])
        print(json.dumps({"effect_mode": mode, **metrics[mode]}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "dim": args.dim, "layers": args.layers,
                "indices": indices, "y_mean": y_mean.cpu(), "y_std": y_std.cpu(),
                "effect_mean": effect_mean.cpu(), "effect_std": effect_std.cpu(),
                "history": history, "metrics": metrics}, args.output)


if __name__ == "__main__":
    main()
