from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_to_dict, load_config
from .dataset_one_step import build_one_step_dataset
from .pointworld_one_step import GRABPointWorldOneStep
from .train import move_batch, rotation_error_deg


def flow_loss(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred, target, beta=beta)


@torch.no_grad()
def flow_metrics(
    pred: torch.Tensor, target: torch.Tensor, object_points: torch.Tensor
) -> Dict[str, float]:
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    return {
        "point_error_m": float(error.mean()),
        "translation_error_m": float(
            torch.linalg.vector_norm(pred.mean(1) - target.mean(1), dim=-1).mean()
        ),
        "rotation_deg": float(
            rotation_error_deg(
                object_points,
                object_points + target,
                object_points + pred,
            ).mean()
        ),
    }


@torch.no_grad()
def evaluate_conditions(model, loader, device) -> Dict[str, Dict[str, float]]:
    collected = {name: [] for name in ("gt", "static_action", "shuffled_action")}
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        flow = batch["hand_flow"]
        if len(flow) < 2:
            raise ValueError("shuffled-action evaluation 要求 eval batch 至少有两个样本")
        variants = {
            "gt": flow,
            "static_action": torch.zeros_like(flow),
            "shuffled_action": torch.roll(flow, shifts=1, dims=0),
        }
        for name, action in variants.items():
            pred = model(
                batch["object_points"],
                batch["object_normals"],
                batch["hand_points"],
                batch["hand_normals"],
                action,
            )["object_flow"]
            collected[name].append(
                (pred.cpu(), batch["object_flow"].cpu(), batch["object_points"].cpu())
            )
    result = {}
    for name, parts in collected.items():
        pred = torch.cat([part[0] for part in parts])
        target = torch.cat([part[1] for part in parts])
        points = torch.cat([part[2] for part in parts])
        result[name] = flow_metrics(pred, target, points)
    target = torch.cat([part[1] for part in collected["gt"]])
    points = torch.cat([part[2] for part in collected["gt"]])
    result["zero_flow_baseline"] = flow_metrics(torch.zeros_like(target), target, points)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PointWorldWAM V0.2 local point-flow overfit")
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_one_step_v02a.yaml"
    )
    parser.add_argument("--resume", default=None, help="从 task checkpoint 续训到 config 总步数")
    args = parser.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.train.device)
    dataset = build_one_step_dataset(cfg.data)
    train_loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0
    )
    eval_loader = DataLoader(
        dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=0
    )
    model = GRABPointWorldOneStep(cfg.model)
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
        if name.startswith("one_step_dynamics_head"):
            groups["head"].append(parameter)
        elif name.startswith(("scene_adapter", "hand_adapter", "hand_type_emb")) or ".rpe." in name:
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
    best_error = float(
        previous_summary.get("best_conditions", {}).get("gt", {}).get(
            "point_error_m", math.inf
        )
    )
    if start_step >= cfg.train.steps:
        raise ValueError(f"resume step={start_step} 必须小于目标 steps={cfg.train.steps}")
    iterator = iter(train_loader)
    for step in range(start_step + 1, cfg.train.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(
            batch["object_points"],
            batch["object_normals"],
            batch["hand_points"],
            batch["hand_normals"],
            batch["hand_flow"],
        )["object_flow"]
        loss = flow_loss(pred, batch["object_flow"], cfg.loss.huber_delta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step={step} loss={float(loss)}")
        loss.backward()
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach())}
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            conditions = evaluate_conditions(model, eval_loader, device)
            error = conditions["gt"]["point_error_m"]
            record["eval_gt_point_error_m"] = error
            print(json.dumps({"step": step, "conditions": conditions}), flush=True)
            if error < best_error:
                best_error = error
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
            print(json.dumps(record), flush=True)
        history.append(record)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_to_dict(cfg),
            "step": cfg.train.steps,
        },
        output / "last.pt",
    )
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_conditions = evaluate_conditions(model, eval_loader, device)
    summary = {
        "transitions": len(dataset),
        "sequences": sorted({str(ref.shared_path.parent) for ref in dataset.transitions}),
        "gaps": sorted({ref.gap for ref in dataset.transitions}),
        "dt_s": sorted({float(dataset[index]["dt"]) for index in range(len(dataset))}),
        "load_report": load_report,
        "lr_group_parameter_counts": {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in groups.items()
        },
        "best_step": int(checkpoint["step"]),
        "best_conditions": best_conditions,
    }
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
