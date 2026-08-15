from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import config_to_dict, load_config
from .dataset_wam_chunk import build_wam_chunk_dataset
from .train import move_batch
from .wam_v2 import ChunkJointWAM


def sample_mode(cfg, forced_mode=None) -> str:
    if forced_mode is not None:
        return forced_mode
    value = random.random()
    if value < cfg.inverse_prob:
        return "inverse"
    if value < cfg.inverse_prob + cfg.forward_prob:
        return "forward"
    return "joint"


def make_fm_state(model, batch, mode: str, generator=None):
    target = model.normalized_targets(batch)
    noise = {
        key: torch.randn(
            value.shape,
            device=value.device,
            dtype=value.dtype,
            generator=generator,
        )
        for key, value in target.items()
    }
    batch_size = len(target["world"])
    device, dtype = target["world"].device, target["world"].dtype
    tau_world = (
        torch.ones(batch_size, device=device, dtype=dtype)
        if mode == "inverse"
        else torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    )
    tau_action = (
        torch.ones(batch_size, device=device, dtype=dtype)
        if mode == "forward"
        else torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    )

    def interpolate(key, tau):
        shape = [len(tau)] + [1] * (target[key].ndim - 1)
        tau_view = tau.reshape(shape)
        return (1.0 - tau_view) * noise[key] + tau_view * target[key]

    noisy = {
        "world": target["world"]
        if mode == "inverse"
        else interpolate("world", tau_world),
        "left": target["left"]
        if mode == "forward"
        else interpolate("left", tau_action),
        "right": target["right"]
        if mode == "forward"
        else interpolate("right", tau_action),
    }
    return target, noise, noisy, tau_world, tau_action


def chunk_losses(model, batch, cfg, mode: str, generator=None):
    target, noise, noisy, tau_world, tau_action = make_fm_state(
        model, batch, mode, generator
    )
    prediction = model(
        batch,
        noisy["world"],
        noisy["left"],
        noisy["right"],
        tau_world,
        tau_action,
    )
    zero = prediction["world_velocity"].new_zeros(())
    world_loss = (
        F.mse_loss(prediction["world_velocity"], target["world"] - noise["world"])
        if mode in ("forward", "joint")
        else zero
    )
    action_loss = zero
    surface_loss = zero
    if mode in ("inverse", "joint"):
        action_loss = 0.5 * (
            F.mse_loss(
                prediction["left_action_velocity"], target["left"] - noise["left"]
            )
            + F.mse_loss(
                prediction["right_action_velocity"], target["right"] - noise["right"]
            )
        )
        tau = tau_action[:, None, None]
        surfaces = []
        for side in ("left", "right"):
            endpoint = noisy[side] + (1.0 - tau) * prediction[
                f"{side}_action_velocity"
            ]
            points = model.hand_points_from_normalized_action(batch, side, endpoint)
            surfaces.append(
                torch.linalg.vector_norm(
                    points - batch[f"{side}_hand_chunk"], dim=-1
                ).mean()
            )
        surface_loss = 0.5 * sum(surfaces)
    total = (
        cfg.world_weight * world_loss
        + cfg.action_weight * action_loss
        + cfg.surface_weight * surface_loss
    )
    return {
        "total": total,
        "world": world_loss,
        "action": action_loss,
        "surface": surface_loss,
    }


@torch.no_grad()
def sample_inverse(model, batch, initial_left, initial_right, steps):
    target = model.normalized_targets(batch)
    left, right = initial_left, initial_right
    tau_world = torch.ones(len(left), device=left.device, dtype=left.dtype)
    step_size = 1.0 / steps
    for index in range(steps):
        tau_action = torch.full_like(tau_world, index / steps)
        prediction = model(
            batch, target["world"], left, right, tau_world, tau_action
        )
        left = left + step_size * prediction["left_action_velocity"]
        right = right + step_size * prediction["right_action_velocity"]
    return {"left": left, "right": right}


@torch.no_grad()
def sample_forward(model, batch, action, initial_world, steps):
    world = initial_world
    tau_action = torch.ones(
        len(world), device=world.device, dtype=world.dtype
    )
    step_size = 1.0 / steps
    for index in range(steps):
        tau_world = torch.full_like(tau_action, index / steps)
        prediction = model(
            batch,
            world,
            action["left"],
            action["right"],
            tau_world,
            tau_action,
        )
        world = world + step_size * prediction["world_velocity"]
    return world


@torch.no_grad()
def sample_joint(model, batch, initial, steps):
    state = {key: value.clone() for key, value in initial.items()}
    step_size = 1.0 / steps
    for index in range(steps):
        tau = torch.full(
            (len(state["world"]),),
            index / steps,
            device=state["world"].device,
            dtype=state["world"].dtype,
        )
        prediction = model(
            batch, state["world"], state["left"], state["right"], tau, tau
        )
        state["world"] += step_size * prediction["world_velocity"]
        state["left"] += step_size * prediction["left_action_velocity"]
        state["right"] += step_size * prediction["right_action_velocity"]
    return state


@torch.no_grad()
def evaluate_generation(model, loader, device, steps, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    values = {
        key: []
        for key in (
            "inverse_hand_mm",
            "identity_hand_mm",
            "forward_world_mm",
            "zero_action_world_mm",
            "zero_flow_world_mm",
        )
    }
    joint_finite = True
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        target = model.normalized_targets(batch)
        initial = {
            key: torch.randn(
                value.shape,
                device=device,
                dtype=value.dtype,
                generator=generator,
            )
            for key, value in target.items()
        }
        inverse = sample_inverse(
            model, batch, initial["left"], initial["right"], steps
        )
        inverse_errors, identity_errors = [], []
        for side in ("left", "right"):
            generated_points = model.hand_points_from_normalized_action(
                batch, side, inverse[side]
            )
            inverse_errors.append(
                torch.linalg.vector_norm(
                    generated_points - batch[f"{side}_hand_chunk"], dim=-1
                ).mean(dim=(-1, -2))
            )
            identity_errors.append(
                torch.linalg.vector_norm(
                    batch[f"{side}_hand_points"][:, None]
                    - batch[f"{side}_hand_chunk"],
                    dim=-1,
                ).mean(dim=(-1, -2))
            )
        values["inverse_hand_mm"].append(500.0 * sum(inverse_errors))
        values["identity_hand_mm"].append(500.0 * sum(identity_errors))
        generated_world = sample_forward(model, batch, target, initial["world"], steps)
        identity_action = {
            side: model.normalize_action(
                side, torch.zeros_like(batch[f"{side}_action_chunk"])
            )
            for side in ("left", "right")
        }
        zero_action_world = sample_forward(
            model, batch, identity_action, initial["world"], steps
        )
        generated_metric = model.denormalize_world(generated_world)
        zero_action_metric = model.denormalize_world(zero_action_world)
        target_metric = batch["world_chunk"]
        values["forward_world_mm"].append(
            1000.0
            * torch.linalg.vector_norm(
                generated_metric - target_metric, dim=-1
            ).mean(dim=(-1, -2))
        )
        values["zero_action_world_mm"].append(
            1000.0
            * torch.linalg.vector_norm(
                zero_action_metric - target_metric, dim=-1
            ).mean(dim=(-1, -2))
        )
        values["zero_flow_world_mm"].append(
            1000.0
            * torch.linalg.vector_norm(target_metric, dim=-1).mean(dim=(-1, -2))
        )
        if batch_index == 0:
            joint = sample_joint(model, batch, initial, steps)
            joint_finite = all(bool(torch.isfinite(value).all()) for value in joint.values())
    result = {
        key: float(torch.cat(parts).mean()) for key, parts in values.items()
    }
    result["inverse_beats_identity"] = (
        result["inverse_hand_mm"] < result["identity_hand_mm"]
    )
    result["forward_beats_zero_flow"] = (
        result["forward_world_mm"] < result["zero_flow_world_mm"]
    )
    result["forward_uses_action"] = (
        result["forward_world_mm"] < result["zero_action_world_mm"]
    )
    result["joint_finite"] = joint_finite
    result["checkpoint_score"] = (
        result["inverse_hand_mm"] / result["identity_hand_mm"]
        + result["forward_world_mm"] / result["zero_flow_world_mm"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V1.1 Chunk Joint WAM")
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_wam_v2_chunk.yaml"
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mode", choices=("inverse", "forward", "joint"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    target_steps = args.steps or cfg.train.steps
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(args.device or cfg.train.device)
    dataset = build_wam_chunk_dataset(cfg.data)
    statistics = dataset.statistics()
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0
    )
    eval_indices = np.linspace(
        0, len(dataset) - 1, min(cfg.train.eval_max_windows, len(dataset)), dtype=int
    ).tolist()
    eval_loader = DataLoader(
        Subset(dataset, eval_indices), batch_size=1, shuffle=False, num_workers=0
    )
    model = ChunkJointWAM(cfg.model, statistics, dataset.subjects)
    load_report = model.load_pointworld_checkpoint(cfg.model.pointworld_checkpoint)
    start_step = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_checkpoint["model"])
        start_step = int(resume_checkpoint["step"])
        load_report["resume"] = str(Path(args.resume).resolve())
    model.to(device)
    groups = {"head": [], "new": [], "backbone": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("world_head", "left_action_head", "right_action_head")):
            groups["head"].append(parameter)
        elif not name.startswith("predictor_model.") or ".rpe." in name:
            groups["new"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["head"], "lr": cfg.train.lr_head},
            {"params": groups["new"], "lr": cfg.train.lr_new},
            {"params": groups["backbone"], "lr": cfg.train.lr_backbone},
        ],
        weight_decay=cfg.train.weight_decay,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
    output = Path(args.output_dir or cfg.train.output_dir)
    if args.mode is not None:
        output = output / args.mode
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if start_step and history_path.is_file()
        else []
    )
    history = [row for row in history if int(row["step"]) <= start_step]
    best_score = math.inf
    summary_path = output / "summary.json"
    if resume_checkpoint is not None:
        resume_evaluation = resume_checkpoint.get("evaluation", {})
        best_score = {
            "inverse": resume_evaluation.get("inverse_hand_mm", math.inf),
            "forward": resume_evaluation.get("forward_world_mm", math.inf),
            "joint": resume_evaluation.get("checkpoint_score", math.inf),
        }.get(args.mode, resume_evaluation.get("checkpoint_score", math.inf))
    iterator = iter(loader)
    if start_step >= target_steps:
        raise ValueError(f"resume step={start_step} 必须小于目标 steps={target_steps}")
    for step in range(start_step + 1, target_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        mode = sample_mode(cfg.flow_matching, args.mode if args.mode != "joint" else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = chunk_losses(model, batch, cfg.loss, mode)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"step={step} mode={mode} loss={losses}")
        losses["total"].backward()
        optimizer.step()
        row = {
            "step": step,
            "mode": mode,
            **{f"{key}_loss": float(value.detach()) for key, value in losses.items()},
        }
        if step % cfg.train.eval_every == 0 or step == target_steps:
            evaluation = evaluate_generation(
                model,
                eval_loader,
                device,
                cfg.flow_matching.inference_steps,
                cfg.seed + 100000,
            )
            row["evaluation"] = evaluation
            print(json.dumps({"step": step, "evaluation": evaluation}), flush=True)
            checkpoint_metric = {
                "inverse": evaluation["inverse_hand_mm"],
                "forward": evaluation["forward_world_mm"],
                "joint": evaluation["checkpoint_score"],
            }.get(args.mode, evaluation["checkpoint_score"])
            row["checkpoint_metric"] = checkpoint_metric
            checkpoint_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config_to_dict(cfg),
                "statistics": statistics,
                "subjects": sorted(dataset.subjects),
                "step": step,
                "evaluation": evaluation,
            }
            torch.save(checkpoint_state, output / "last.pt")
            if checkpoint_metric < best_score:
                best_score = checkpoint_metric
                torch.save(checkpoint_state, output / "best.pt")
        if step == 1 or step % cfg.train.log_every == 0:
            print(json.dumps(row), flush=True)
        history.append(row)
        if step % cfg.train.eval_every == 0 or step == target_steps:
            history_path.write_text(json.dumps(history, indent=2) + "\n")
            summary_path.write_text(
                json.dumps(
                    {
                        "windows": len(dataset),
                        "full_windows": dataset.full_window_count,
                        "best_score": best_score,
                        "load_report": load_report,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_evaluation = evaluate_generation(
        model,
        eval_loader,
        device,
        cfg.flow_matching.inference_steps,
        cfg.seed + 100000,
    )
    summary = {
        "windows": len(dataset),
        "full_windows": dataset.full_window_count,
        "best_step": int(checkpoint["step"]),
        "best_score": float(
            {
                "inverse": checkpoint["evaluation"]["inverse_hand_mm"],
                "forward": checkpoint["evaluation"]["forward_world_mm"],
                "joint": checkpoint["evaluation"]["checkpoint_score"],
            }.get(args.mode, checkpoint["evaluation"]["checkpoint_score"])
        ),
        "checkpoint_evaluation": checkpoint["evaluation"],
        "final_evaluation": final_evaluation,
        "load_report": load_report,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
