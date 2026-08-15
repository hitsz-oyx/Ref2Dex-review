from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import config_to_dict, load_config
from .dataset_wam_chunk import build_wam_chunk_dataset
from .train import move_batch
from .wam_v2 import ChunkJointWAM


def regression_inputs(model, batch, mode: str):
    target = model.normalized_targets(batch)
    batch_size = len(target["world"])
    device, dtype = target["world"].device, target["world"].dtype
    zero_world = model.normalize_world(torch.zeros_like(batch["world_chunk"]))
    zero_action = {
        side: model.normalize_action(
            side, torch.zeros_like(batch[f"{side}_action_chunk"])
        )
        for side in ("left", "right")
    }
    if mode == "inverse":
        return (
            target,
            target["world"],
            zero_action["left"],
            zero_action["right"],
            torch.ones(batch_size, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=dtype),
        )
    if mode == "forward":
        return (
            target,
            zero_world,
            target["left"],
            target["right"],
            torch.zeros(batch_size, device=device, dtype=dtype),
            torch.ones(batch_size, device=device, dtype=dtype),
        )
    raise ValueError(f"不支持的 regression mode: {mode}")


def regression_losses(model, batch, cfg, mode: str):
    target, world, left, right, tau_world, tau_action = regression_inputs(
        model, batch, mode
    )
    prediction = model(batch, world, left, right, tau_world, tau_action)
    zero = prediction["world_velocity"].new_zeros(())
    world_loss = zero
    action_loss = zero
    surface_loss = zero
    if mode == "forward":
        world_loss = F.mse_loss(prediction["world_velocity"], target["world"])
    else:
        action_loss = 0.5 * (
            F.mse_loss(prediction["left_action_velocity"], target["left"])
            + F.mse_loss(prediction["right_action_velocity"], target["right"])
        )
        surfaces = []
        for side in ("left", "right"):
            points = model.hand_points_from_normalized_action(
                batch, side, prediction[f"{side}_action_velocity"]
            )
            surfaces.append(
                torch.linalg.vector_norm(
                    points - batch[f"{side}_hand_chunk"], dim=-1
                ).mean()
            )
        surface_loss = 0.5 * sum(surfaces)
    total = world_loss + action_loss + cfg.surface_weight * surface_loss
    return {
        "total": total,
        "world": world_loss,
        "action": action_loss,
        "surface": surface_loss,
    }


@torch.no_grad()
def evaluate_regression(model, loader, device, mode: str):
    model.eval()
    values = {}
    for batch in loader:
        batch = move_batch(batch, device)
        target, world, left, right, tau_world, tau_action = regression_inputs(
            model, batch, mode
        )
        prediction = model(batch, world, left, right, tau_world, tau_action)
        if mode == "inverse":
            generated_errors, identity_errors = [], []
            for side in ("left", "right"):
                generated = model.hand_points_from_normalized_action(
                    batch, side, prediction[f"{side}_action_velocity"]
                )
                generated_errors.append(
                    torch.linalg.vector_norm(
                        generated - batch[f"{side}_hand_chunk"], dim=-1
                    ).mean(dim=(-1, -2))
                )
                identity_errors.append(
                    torch.linalg.vector_norm(
                        batch[f"{side}_hand_points"][:, None]
                        - batch[f"{side}_hand_chunk"],
                        dim=-1,
                    ).mean(dim=(-1, -2))
                )
            values.setdefault("inverse_hand_mm", []).append(
                500.0 * sum(generated_errors)
            )
            values.setdefault("identity_hand_mm", []).append(
                500.0 * sum(identity_errors)
            )
        else:
            generated = model.denormalize_world(prediction["world_velocity"])
            zero_action = {
                side: model.normalize_action(
                    side, torch.zeros_like(batch[f"{side}_action_chunk"])
                )
                for side in ("left", "right")
            }
            zero_condition = model(
                batch,
                world,
                zero_action["left"],
                zero_action["right"],
                tau_world,
                tau_action,
            )
            zero_action_world = model.denormalize_world(
                zero_condition["world_velocity"]
            )
            metric_target = batch["world_chunk"]
            values.setdefault("forward_world_mm", []).append(
                1000.0
                * torch.linalg.vector_norm(
                    generated - metric_target, dim=-1
                ).mean(dim=(-1, -2))
            )
            values.setdefault("zero_action_world_mm", []).append(
                1000.0
                * torch.linalg.vector_norm(
                    zero_action_world - metric_target, dim=-1
                ).mean(dim=(-1, -2))
            )
            values.setdefault("zero_flow_world_mm", []).append(
                1000.0
                * torch.linalg.vector_norm(metric_target, dim=-1).mean(dim=(-1, -2))
            )
    result = {key: float(torch.cat(parts).mean()) for key, parts in values.items()}
    result["finite"] = all(math.isfinite(value) for value in result.values())
    if mode == "inverse":
        result["beats_identity"] = (
            result["inverse_hand_mm"] < result["identity_hand_mm"]
        )
        result["passes_30mm"] = result["inverse_hand_mm"] < 30.0
        result["checkpoint_metric"] = result["inverse_hand_mm"]
    else:
        result["beats_zero_flow"] = (
            result["forward_world_mm"] < result["zero_flow_world_mm"]
        )
        result["uses_action"] = (
            result["forward_world_mm"] < result["zero_action_world_mm"]
        )
        result["passes_30mm"] = result["forward_world_mm"] < 30.0
        result["checkpoint_metric"] = result["forward_world_mm"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V1.2 deterministic chunk regression")
    parser.add_argument("--mode", choices=("inverse", "forward"), required=True)
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_wam_v12_debug.yaml"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=None)
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
    output = Path(cfg.train.output_dir) / args.mode
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if start_step and history_path.is_file()
        else []
    )
    history = [row for row in history if int(row["step"]) <= start_step]
    best_metric = (
        float(resume_checkpoint["evaluation"]["checkpoint_metric"])
        if resume_checkpoint is not None
        else math.inf
    )
    if start_step >= target_steps:
        raise ValueError(f"resume step={start_step} 必须小于目标 steps={target_steps}")
    iterator = iter(loader)
    for step in range(start_step + 1, target_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = regression_losses(model, batch, cfg.regression, args.mode)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"step={step} mode={args.mode} loss={losses}")
        losses["total"].backward()
        optimizer.step()
        row = {
            "step": step,
            **{f"{key}_loss": float(value.detach()) for key, value in losses.items()},
        }
        if step % cfg.train.eval_every == 0 or step == target_steps:
            evaluation = evaluate_regression(model, eval_loader, device, args.mode)
            row["evaluation"] = evaluation
            print(json.dumps({"step": step, "evaluation": evaluation}), flush=True)
            if evaluation["checkpoint_metric"] < best_metric:
                best_metric = evaluation["checkpoint_metric"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config_to_dict(cfg),
                        "statistics": statistics,
                        "subjects": sorted(dataset.subjects),
                        "mode": args.mode,
                        "step": step,
                        "evaluation": evaluation,
                    },
                    output / "best.pt",
                )
        if step == 1 or step % cfg.train.log_every == 0:
            print(json.dumps(row), flush=True)
        history.append(row)
        if step % cfg.train.eval_every == 0 or step == target_steps:
            history_path.write_text(json.dumps(history, indent=2) + "\n")
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_evaluation = evaluate_regression(model, eval_loader, device, args.mode)
    summary = {
        "mode": args.mode,
        "windows": len(dataset),
        "full_windows": dataset.full_window_count,
        "best_step": int(checkpoint["step"]),
        "checkpoint_evaluation": checkpoint["evaluation"],
        "final_evaluation": final_evaluation,
        "load_report": load_report,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
