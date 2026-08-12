"""训练并评估 V16.8 v-prediction conditional residual diffusion。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
from src.task.InteractionDynamics.residual_interaction_diffusion import (
    ResidualInteractionDiffusion, make_v_target, recover_x0_noise, sample_residual_v)
from src.task.InteractionDynamics.residual_interaction_regression import (
    persistence_future, residual_target)
from src.task.InteractionDynamics.train_goal_interaction_diffusion import (
    controlled_dataset, materialize, subset_metrics)
from src.task.InteractionDynamics.train_residual_interaction_regression import intervene_goal
from src.task.InteractionDynamics.train_state_interaction_diffusion import future_metrics


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
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--indices-from", type=Path)
    parser.add_argument("--clean-baseline", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_goal(raw_goal: torch.Tensor, mode: str, stats: dict, device) -> torch.Tensor:
    value = intervene_goal(raw_goal, mode).to(device)
    return (value - stats["goal_mean"]) / stats["goal_std"]


@torch.no_grad()
def sample_mode(model, normalized, raw_goal, stats, mode, steps, initial_noise):
    return sample_residual_v(model, normalized["state"], normalized["anchors"],
                             normalized["patches"], normalized_goal(raw_goal, mode, stats,
                             normalized["state"].device), steps, initial_noise=initial_noise)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset, indices, metadata = controlled_dataset(
        args.config, args.sequence_index, args.samples, args.horizon,
        args.translation_threshold_cm, args.rotation_threshold_deg, args.motion_window)
    if args.indices_from:
        saved = torch.load(args.indices_from, map_location="cpu")
        if list(saved["indices"]) != list(indices):
            raise ValueError("V16.7 checkpoint indices differ from the reconstructed controlled set")
    data = materialize(dataset, indices, metadata, device, args.horizon, args.goal_segment,
                       args.tau_m, args.motion_window, args.translation_threshold_cm,
                       args.rotation_threshold_deg)
    persistence = persistence_future(data["state"], args.horizon)
    residual = residual_target(data["state"], data["future"], args.horizon)
    change = residual.square().mean((1, 2)).sqrt()
    dynamic, active = change >= args.dynamic_threshold_cm, data["goal"][:, 0] > .5
    masks = {"overall": torch.ones(len(dynamic), dtype=torch.bool), "dynamic": dynamic,
             "static": ~dynamic, "dynamic_active": dynamic & active}
    stats = {}
    for key, value, dims, floor in [
            ("state", data["state"], (0, 1), .05), ("goal", data["goal"], (0,), .01),
            ("residual", residual, (0, 1), .05)]:
        stats[f"{key}_mean"] = value.mean(dims, keepdim=True).to(device)
        stats[f"{key}_std"] = value.std(dims, keepdim=True).clamp_min(floor).to(device)
    normalized = {"state": (data["state"].to(device) - stats["state_mean"]) / stats["state_std"],
                  "goal": (data["goal"].to(device) - stats["goal_mean"]) / stats["goal_std"],
                  "residual": (residual.to(device) - stats["residual_mean"]) / stats["residual_std"],
                  "anchors": data["anchors_cm"].to(device),
                  "patches": data["object_patches"].to(device)}
    model = ResidualInteractionDiffusion(
        args.horizon, args.goal_segment, args.dim, 8, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    _, alpha_bar = cosine_schedule(args.diffusion_steps, device)
    cpu_generator = torch.Generator().manual_seed(args.seed)
    fixed_noise = torch.randn((4, 128, 7 * args.horizon), generator=cpu_generator).to(device)
    history = []
    model.train()
    for step in range(1, args.train_steps + 1):
        index = torch.randint(len(residual), (args.batch_size,), generator=cpu_generator)
        timestep = torch.randint(args.diffusion_steps, (args.batch_size,), device=device)
        noise = torch.randn_like(normalized["residual"][index])
        scale = alpha_bar[timestep][:, None, None]
        noisy, target_v = make_v_target(normalized["residual"][index], noise, scale)
        permutation = torch.stack([torch.randperm(128, device=device) for _ in index])
        def gather(value):
            shape = (*permutation.shape, *((1,) * (value.ndim - 2)))
            return value.gather(1, permutation.reshape(shape).expand_as(value))
        prediction = model(gather(noisy), gather(normalized["state"][index]),
                           gather(normalized["anchors"][index]),
                           gather(normalized["patches"][index]), normalized["goal"][index], timestep)
        loss = torch.nn.functional.mse_loss(prediction, gather(target_v))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "v_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.sample_every == 0:
            model.eval()
            sampled, _ = sample_residual_v(
                model, normalized["state"][:4], normalized["anchors"][:4],
                normalized["patches"][:4], normalized["goal"][:4], args.diffusion_steps,
                initial_noise=fixed_noise)
            sampled_cm = sampled * stats["residual_std"] + stats["residual_mean"]
            future = persistence[:4].to(device) + sampled_cm
            diag = future_metrics(future.cpu(), data["future"][:4], args.horizon)
            diag.update({"step": step, "sampled_residual_rmse_norm": float(
                (sampled - normalized["residual"][:4]).square().mean().sqrt())})
            print(json.dumps({"sampling_diagnostic": diag}), flush=True)
            model.train()
    model.eval()
    full_noise = torch.randn(normalized["residual"].shape,
                             generator=torch.Generator().manual_seed(args.seed)).to(device)
    metrics = {"persistence": subset_metrics(persistence, data["future"], masks, args.horizon)}
    traces, predictions = {}, {}
    for mode in ["correct", "motion_zero", "motion_shuffle", "motion_reverse", "active_zero"]:
        sampled, trace = sample_mode(model, normalized, data["goal"], stats, mode,
                                     args.diffusion_steps, full_noise)
        residual_cm = sampled.cpu() * stats["residual_std"].cpu() + stats["residual_mean"].cpu()
        predictions[mode] = persistence + residual_cm
        metrics[mode] = subset_metrics(predictions[mode], data["future"], masks, args.horizon)
        traces[mode] = trace
    _, stability = sample_residual_v(
        model, normalized["state"], normalized["anchors"], normalized["patches"],
        normalized["goal"], args.diffusion_steps, initial_noise=full_noise, trace_every=10)
    oracle = {}
    oracle_noise = torch.randn_like(normalized["residual"])
    for timestep_value in [10, 30, 50, 70, 90]:
        timestep = torch.full((len(residual),), timestep_value, device=device, dtype=torch.long)
        noisy, _ = make_v_target(normalized["residual"], oracle_noise,
                                 alpha_bar[timestep][:, None, None])
        velocity = model(noisy, normalized["state"], normalized["anchors"],
                         normalized["patches"], normalized["goal"], timestep)
        clean, _ = recover_x0_noise(noisy, velocity, alpha_bar[timestep][:, None, None])
        oracle[f"t{timestep_value}"] = float(
            (clean - normalized["residual"]).square().mean().sqrt())
    seed_metrics = []
    for seed in range(4):
        noise = torch.randn(normalized["residual"].shape,
                            generator=torch.Generator().manual_seed(args.seed + seed)).to(device)
        sampled, _ = sample_mode(model, normalized, data["goal"], stats, "correct",
                                 args.diffusion_steps, noise)
        future = persistence + (sampled.cpu() * stats["residual_std"].cpu()
                                + stats["residual_mean"].cpu())
        seed_metrics.append(future_metrics(future[dynamic], data["future"][dynamic], args.horizon))
    keys = ["u_rmse_cm", "r_rmse_cm", "d_rmse_cm", "contact_f1"]
    four_seed = {key: {"mean": float(torch.tensor([row[key] for row in seed_metrics]).mean()),
                       "std": float(torch.tensor([row[key] for row in seed_metrics]).std())}
                 for key in keys}
    clean_baseline = (torch.load(args.clean_baseline, map_location="cpu")["metrics"]["correct"]
                      if args.clean_baseline else None)
    summary = {"sequence": metadata["path"], "frames": metadata["frames"],
               "v16_7_deterministic": clean_baseline, "metrics": metrics,
               "oracle_denoising_rmse_norm": oracle,
               "sampling_stability": {"finite": True,
                    "max_xt_rms": max(row["xt_rms"] for row in stability),
                    "max_xt_abs": max(row["xt_abs_max"] for row in stability),
                    "max_x0_rms": max(row["x0_rms"] for row in stability),
                    "max_x0_abs": max(row["x0_abs_max"] for row in stability),
                    "trace": stability}, "four_seed_dynamic": four_seed}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history, "indices": indices,
                "horizon": args.horizon, "goal_segment": args.goal_segment,
                "dim": args.dim, "layers": args.layers, **summary,
                **{key: value.cpu() for key, value in stats.items()}}, args.output)


if __name__ == "__main__":
    main()
