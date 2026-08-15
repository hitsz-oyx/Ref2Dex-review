from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_to_dict, load_config
from .dataset_wam import build_wam_dataset
from .mano_action import axis_angle_to_matrix
from .pointworld_bimanual_forward import GRABPointWorldBimanualForward
from .train import move_batch
from .wam_v1 import WAMV1


def flow_matching_loss(model, batch, generator=None):
    target = model.normalized_target(batch)
    left_noise = torch.randn(
        target["left"].shape,
        device=target["left"].device,
        dtype=target["left"].dtype,
        generator=generator,
    )
    right_noise = torch.randn(
        target["right"].shape,
        device=target["right"].device,
        dtype=target["right"].dtype,
        generator=generator,
    )
    tau = torch.rand(
        len(left_noise), device=left_noise.device, dtype=left_noise.dtype, generator=generator
    )
    noisy_left = (1.0 - tau[:, None]) * left_noise + tau[:, None] * target["left"]
    noisy_right = (1.0 - tau[:, None]) * right_noise + tau[:, None] * target["right"]
    prediction = model(batch, noisy_left, noisy_right, tau)
    left_loss = torch.nn.functional.mse_loss(
        prediction["left_velocity"], target["left"] - left_noise
    )
    right_loss = torch.nn.functional.mse_loss(
        prediction["right_velocity"], target["right"] - right_noise
    )
    return left_loss + right_loss, left_loss, right_loss


@torch.no_grad()
def evaluate_fm(model, loader, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = torch.zeros(3, device=device)
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        losses = flow_matching_loss(model, batch, generator)
        totals += torch.stack(losses) * len(batch["object_points"])
        count += len(batch["object_points"])
    return {
        "fm_loss": float(totals[0] / count),
        "left_fm_loss": float(totals[1] / count),
        "right_fm_loss": float(totals[2] / count),
    }


def rotation_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = axis_angle_to_matrix(pred).transpose(-1, -2) @ axis_angle_to_matrix(target)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


@torch.no_grad()
def evaluate_generation(model, loader, device, steps, seed, forward_model=None):
    model.eval()
    if forward_model is not None:
        forward_model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = {side: {key: [] for key in ("translation_mm", "rotation_deg", "pose_rmse", "hand_mm")} for side in ("left", "right")}
    baselines = {
        side: {key: [] for key in ("identity_hand_mm", "mean_action_hand_mm", "random_hand_mm")}
        for side in ("left", "right")
    }
    effect_errors, gt_effect_errors = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        initial = {
            side: torch.randn(
                batch[f"{side}_action"].shape,
                device=device,
                dtype=batch[f"{side}_action"].dtype,
                generator=generator,
            )
            for side in ("left", "right")
        }
        generated = model.sample(batch, initial["left"], initial["right"], steps)
        generated_points = {}
        for side in ("left", "right"):
            normalized = generated[f"{side}_action"]
            action = model.mano.denormalize(side, normalized)
            target = batch[f"{side}_action"]
            generated_points[side] = model.mano.points_from_action(
                side,
                batch[f"{side}_global_orient"],
                batch[f"{side}_hand_pose"],
                batch[f"{side}_transl"],
                batch[f"{side}_betas"],
                normalized,
            )
            rows[side]["translation_mm"].append(
                1000.0 * torch.linalg.vector_norm(action[:, :3] - target[:, :3], dim=-1)
            )
            rows[side]["rotation_deg"].append(rotation_error(action[:, 3:6], target[:, 3:6]))
            rows[side]["pose_rmse"].append(
                (action[:, 6:] - target[:, 6:]).square().mean(-1).sqrt()
            )
            rows[side]["hand_mm"].append(
                1000.0
                * torch.linalg.vector_norm(
                    generated_points[side] - batch[f"{side}_next_hand_points"], dim=-1
                ).mean(-1)
            )
            baselines[side]["identity_hand_mm"].append(
                1000.0
                * torch.linalg.vector_norm(
                    batch[f"{side}_hand_points"] - batch[f"{side}_next_hand_points"],
                    dim=-1,
                ).mean(-1)
            )
            mean_points = model.mano.points_from_action(
                side,
                batch[f"{side}_global_orient"],
                batch[f"{side}_hand_pose"],
                batch[f"{side}_transl"],
                batch[f"{side}_betas"],
                torch.zeros_like(initial[side]),
            )
            random_points = model.mano.points_from_action(
                side,
                batch[f"{side}_global_orient"],
                batch[f"{side}_hand_pose"],
                batch[f"{side}_transl"],
                batch[f"{side}_betas"],
                initial[side],
            )
            for key, points in (
                ("mean_action_hand_mm", mean_points),
                ("random_hand_mm", random_points),
            ):
                baselines[side][key].append(
                    1000.0
                    * torch.linalg.vector_norm(
                        points - batch[f"{side}_next_hand_points"], dim=-1
                    ).mean(-1)
                )
        if forward_model is not None:
            def forward_with(left_points, right_points):
                return forward_model(
                    batch["object_points"],
                    batch["object_normals"],
                    batch["prev_object_flow"],
                    batch["left_hand_points"],
                    batch["left_hand_normals"],
                    left_points - batch["left_hand_points"],
                    batch["right_hand_points"],
                    batch["right_hand_normals"],
                    right_points - batch["right_hand_points"],
                )["object_flow"]
            pred_effect = forward_with(generated_points["left"], generated_points["right"])
            gt_effect = forward_with(
                batch["left_next_hand_points"], batch["right_next_hand_points"]
            )
            target_effect = batch["target_object_flow"]
            effect_errors.append(
                1000.0 * torch.linalg.vector_norm(pred_effect - target_effect, dim=-1).mean(-1)
            )
            gt_effect_errors.append(
                1000.0 * torch.linalg.vector_norm(gt_effect - target_effect, dim=-1).mean(-1)
            )
    result = {}
    for side in ("left", "right"):
        result[side] = {
            key: float(torch.cat(values).mean()) for key, values in rows[side].items()
        }
        result[side].update(
            {key: float(torch.cat(values).mean()) for key, values in baselines[side].items()}
        )
    result["bimanual_hand_mm"] = 0.5 * (
        result["left"]["hand_mm"] + result["right"]["hand_mm"]
    )
    if effect_errors:
        result["effect_consistency_mm"] = float(torch.cat(effect_errors).mean())
        result["gt_hand_forward_effect_mm"] = float(torch.cat(gt_effect_errors).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bimanual Point-Flow WAM V1")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_wam_v1.yaml")
    parser.add_argument("--resume", default=None, help="从 WAM task checkpoint 续训")
    args = parser.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.train.device)
    dataset = build_wam_dataset(cfg.data)
    statistics = dataset.action_statistics()
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = WAMV1(cfg.model, statistics)
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
        if name.startswith(("left_action_head", "right_action_head")):
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
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
    output = Path(cfg.train.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if start_step and history_path.is_file()
        else []
    )
    history = [row for row in history if int(row["step"]) <= start_step]
    summary_path = output / "summary.json"
    previous_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if start_step and summary_path.is_file()
        else {}
    )
    best_loss = float(previous_summary.get("best_fm", {}).get("fm_loss", math.inf))
    if start_step >= cfg.train.steps:
        raise ValueError(f"resume step={start_step} 必须小于目标 steps={cfg.train.steps}")
    iterator = iter(loader)
    for step in range(start_step + 1, cfg.train.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, left_loss, right_loss = flow_matching_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step={step} loss={float(loss)}")
        loss.backward()
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "left_loss": float(left_loss.detach()),
            "right_loss": float(right_loss.detach()),
        }
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            evaluation = evaluate_fm(model, eval_loader, device, cfg.seed + 1000)
            row.update({f"eval_{key}": value for key, value in evaluation.items()})
            print(json.dumps({"step": step, "evaluation": evaluation}), flush=True)
            if evaluation["fm_loss"] < best_loss:
                best_loss = evaluation["fm_loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config_to_dict(cfg),
                        "step": step,
                    },
                    output / "best.pt",
                )
        if step == 1 or step % cfg.train.log_every == 0:
            print(json.dumps(row), flush=True)
        history.append(row)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    forward_model = None
    forward_path = Path(cfg.eval.forward_checkpoint)
    if forward_path.is_file():
        forward_cfg = load_config(cfg.eval.forward_config)
        forward_model = GRABPointWorldBimanualForward(forward_cfg.model).to(device)
        forward_state = torch.load(forward_path, map_location="cpu", weights_only=False)
        forward_model.load_state_dict(forward_state["model"])
    generation = evaluate_generation(
        model,
        eval_loader,
        device,
        cfg.eval.fm_inference_steps,
        cfg.seed + 2000,
        forward_model,
    )
    summary = {
        "transitions": len(dataset),
        "best_step": int(checkpoint["step"]),
        "load_report": load_report,
        "best_fm": evaluate_fm(model, eval_loader, device, cfg.seed + 1000),
        "generation": generation,
    }
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
