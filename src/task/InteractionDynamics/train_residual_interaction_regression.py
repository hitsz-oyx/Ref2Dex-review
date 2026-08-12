"""训练并评估 V16.7 clean residual interaction regression。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.InteractionDynamics.residual_interaction_regression import (
    ResidualInteractionRegressor, persistence_future, residual_target)
from src.task.InteractionDynamics.train_goal_interaction_diffusion import (
    controlled_dataset, materialize, subset_metrics)


def residual_statistics(residual: torch.Tensor, dynamic: torch.Tensor) -> dict:
    magnitude = residual.square().mean((1, 2)).sqrt()
    channels = residual.reshape(*residual.shape[:2], -1, 7)
    return {
        "magnitude_cm": {name: float(getattr(magnitude, name)())
                         for name in ["min", "median", "mean", "max"]},
        "dynamic_mean_cm": float(magnitude[dynamic].mean()),
        "static_mean_cm": float(magnitude[~dynamic].mean()),
        "channel_std_cm": {
            "u": float(channels[..., :3].std()),
            "r": float(channels[..., 3:6].std()),
            "d": float(channels[..., 6].std()),
        },
    }


def intervene_goal(raw_goal: torch.Tensor, mode: str) -> torch.Tensor:
    """在物理 Goal 空间做干预；调用方随后再用训练统计量标准化。"""
    result = raw_goal.clone()
    active = raw_goal[:, 0] > .5
    if mode == "correct":
        pass
    elif mode == "motion_zero":
        result[active, 1:] = 0
    elif mode == "motion_shuffle":
        result[active, 1:] = result[active, 1:].roll(1, 0)
    elif mode == "motion_reverse":
        result[active, 1:] *= -1
    elif mode == "active_zero":
        result[active] = 0
    else:
        raise ValueError(mode)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--sequence-index", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--goal-segment", type=int, default=4)
    parser.add_argument("--motion-window", type=int, default=3)
    parser.add_argument("--translation-threshold-cm", type=float, default=.2)
    parser.add_argument("--rotation-threshold-deg", type=float, default=1.)
    parser.add_argument("--dynamic-threshold-cm", type=float, default=.2)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset, indices, metadata = controlled_dataset(
        args.config, args.sequence_index, args.samples, args.horizon,
        args.translation_threshold_cm, args.rotation_threshold_deg, args.motion_window)
    data = materialize(dataset, indices, metadata, device, args.horizon, args.goal_segment,
                       args.tau_m, args.motion_window, args.translation_threshold_cm,
                       args.rotation_threshold_deg)
    persistence = persistence_future(data["state"], args.horizon)
    residual = residual_target(data["state"], data["future"], args.horizon)
    magnitude = residual.square().mean((1, 2)).sqrt()
    dynamic = magnitude >= args.dynamic_threshold_cm
    masks = {"overall": torch.ones(len(dynamic), dtype=torch.bool), "dynamic": dynamic,
             "static": ~dynamic, "active": data["goal"][:, 0] > .5,
             "inactive": data["goal"][:, 0] < .5,
             "dynamic_active": dynamic & (data["goal"][:, 0] > .5)}
    stats = {}
    for key, value, dims, floor in [
            ("state", data["state"], (0, 1), .05), ("goal", data["goal"], (0,), .01),
            ("residual", residual, (0, 1), .05)]:
        stats[f"{key}_mean"] = value.mean(dims, keepdim=True).to(device)
        stats[f"{key}_std"] = value.std(dims, keepdim=True).clamp_min(floor).to(device)
    print(json.dumps({"residual_statistics": residual_statistics(residual, dynamic)},
                     ensure_ascii=False), flush=True)
    normalized = {"state": (data["state"].to(device) - stats["state_mean"]) / stats["state_std"],
                  "goal": (data["goal"].to(device) - stats["goal_mean"]) / stats["goal_std"],
                  "residual": (residual.to(device) - stats["residual_mean"]) / stats["residual_std"],
                  "anchors": data["anchors_cm"].to(device),
                  "patches": data["object_patches"].to(device)}
    model = ResidualInteractionRegressor(
        args.horizon, args.goal_segment, args.dim, 8, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    model.train()
    for step in range(1, args.train_steps + 1):
        index = torch.randint(len(residual), (args.batch_size,), generator=generator)
        permutation = torch.stack([torch.randperm(128, device=device) for _ in index])
        def gather(value: torch.Tensor) -> torch.Tensor:
            chosen = value[index]
            shape = (*permutation.shape, *((1,) * (chosen.ndim - 2)))
            return chosen.gather(1, permutation.reshape(shape).expand_as(chosen))
        prediction = model(gather(normalized["state"]), gather(normalized["anchors"]),
                           gather(normalized["patches"]), normalized["goal"][index])
        loss = torch.nn.functional.mse_loss(prediction, gather(normalized["residual"]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "residual_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    model.eval()
    metrics = {"persistence": subset_metrics(persistence, data["future"], masks, args.horizon)}
    residual_norm = {}
    with torch.no_grad():
        for mode in ["correct", "motion_zero", "motion_shuffle", "motion_reverse", "active_zero"]:
            raw_goal = intervene_goal(data["goal"], mode).to(device)
            goal = (raw_goal - stats["goal_mean"]) / stats["goal_std"]
            prediction = model(normalized["state"], normalized["anchors"],
                               normalized["patches"], goal).cpu()
            prediction = prediction * stats["residual_std"].cpu() + stats["residual_mean"].cpu()
            future = persistence + prediction
            metrics[mode] = subset_metrics(future, data["future"], masks, args.horizon)
            residual_norm[mode] = {name: float(prediction[mask].square().mean().sqrt())
                                   for name, mask in masks.items() if mask.any()}
    summary = {"sequence": metadata["path"], "frames": metadata["frames"],
               "active_interval": [metadata["active_start"], metadata["active_end"]],
               "residual_statistics": residual_statistics(residual, dynamic),
               "metrics": metrics, "predicted_residual_rms_cm": residual_norm}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history, "indices": indices,
                "horizon": args.horizon, "goal_segment": args.goal_segment,
                "dim": args.dim, "layers": args.layers, **summary,
                **{key: value.cpu() for key, value in stats.items()}}, args.output)


if __name__ == "__main__":
    main()
