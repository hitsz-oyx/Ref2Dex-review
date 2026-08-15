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
from .pointworld_bimanual_forward import GRABPointWorldBimanualForward
from .train import move_batch
from .train_one_step import flow_loss, flow_metrics


def predict(model, batch, left_flow, right_flow):
    return model(
        batch["object_points"],
        batch["object_normals"],
        batch["prev_object_flow"],
        batch["left_hand_points"],
        batch["left_hand_normals"],
        left_flow,
        batch["right_hand_points"],
        batch["right_hand_normals"],
        right_flow,
    )["object_flow"]


@torch.no_grad()
def evaluate(model, loader, device):
    collected = {name: [] for name in ("gt", "static", "shuffled")}
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        left_flow = batch["left_next_hand_points"] - batch["left_hand_points"]
        right_flow = batch["right_next_hand_points"] - batch["right_hand_points"]
        if len(left_flow) < 2:
            raise ValueError("bimanual shuffled evaluation 要求 batch>=2")
        variants = {
            "gt": (left_flow, right_flow),
            "static": (torch.zeros_like(left_flow), torch.zeros_like(right_flow)),
            "shuffled": (
                torch.roll(left_flow, 1, 0),
                torch.roll(right_flow, 1, 0),
            ),
        }
        for name, flows in variants.items():
            pred = predict(model, batch, *flows)
            collected[name].append(
                (pred.cpu(), batch["target_object_flow"].cpu(), batch["object_points"].cpu())
            )
    result = {}
    for name, parts in collected.items():
        pred = torch.cat([part[0] for part in parts])
        target = torch.cat([part[1] for part in parts])
        points = torch.cat([part[2] for part in parts])
        result[name] = flow_metrics(pred, target, points)
    target = torch.cat([part[1] for part in collected["gt"]])
    points = torch.cat([part[2] for part in collected["gt"]])
    result["zero_flow"] = flow_metrics(torch.zeros_like(target), target, points)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 bimanual PointWorld forward overfit")
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_bimanual_forward.yaml"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.train.device)
    dataset = build_wam_dataset(cfg.data)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(
        dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=0
    )
    model = GRABPointWorldBimanualForward(cfg.model)
    load_report = model.load_pointworld_checkpoint(cfg.model.pointworld_checkpoint)
    model.to(device)
    groups = {"head": [], "new": [], "backbone": []}
    for name, parameter in model.named_parameters():
        if name.startswith("object_flow_head"):
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
    output = Path(cfg.train.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    history, best_error = [], math.inf
    for step in range(1, cfg.train.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        left_flow = batch["left_next_hand_points"] - batch["left_hand_points"]
        right_flow = batch["right_next_hand_points"] - batch["right_hand_points"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = predict(model, batch, left_flow, right_flow)
        loss = flow_loss(pred, batch["target_object_flow"], cfg.loss.huber_delta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step={step} loss={float(loss)}")
        loss.backward()
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach())}
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            conditions = evaluate(model, eval_loader, device)
            row["eval_gt_point_error_m"] = conditions["gt"]["point_error_m"]
            print(json.dumps({"step": step, "conditions": conditions}), flush=True)
            if row["eval_gt_point_error_m"] < best_error:
                best_error = row["eval_gt_point_error_m"]
                torch.save(
                    {"model": model.state_dict(), "config": config_to_dict(cfg), "step": step},
                    output / "best.pt",
                )
        if step == 1 or step % cfg.train.log_every == 0:
            print(json.dumps(row), flush=True)
        history.append(row)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    conditions = evaluate(model, eval_loader, device)
    summary = {
        "transitions": len(dataset),
        "best_step": int(checkpoint["step"]),
        "load_report": load_report,
        "conditions": conditions,
    }
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
