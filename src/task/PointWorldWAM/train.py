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
from .dataset import build_dataset
from .pointworld_forward import GRABPointWorldForward


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    error = torch.nn.functional.smooth_l1_loss(
        pred, target, beta=cfg.huber_delta, reduction="none"
    ).mean(dim=-1)
    displacement = torch.linalg.vector_norm(target - target[:, :1], dim=-1)
    weight = 1.0 + cfg.motion_weight * (displacement > cfg.motion_epsilon).float()
    return (error * weight).sum() / weight.sum()


def rotation_error_deg(source: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    def fit(points: torch.Tensor) -> torch.Tensor:
        covariance = (source - source.mean(1, keepdim=True)).transpose(1, 2) @ (
            points - points.mean(1, keepdim=True)
        )
        u, _, vh = torch.linalg.svd(covariance)
        correction = torch.ones((len(source), 3), device=source.device, dtype=source.dtype)
        correction[:, -1] = torch.sign(torch.det(u @ vh))
        return u @ torch.diag_embed(correction) @ vh

    gt_rotation = fit(target)
    pred_rotation = fit(pred)
    relative = pred_rotation.transpose(1, 2) @ gt_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


@torch.no_grad()
def metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    source = target[:, 0]
    return {
        "ade_m": float(error.mean()),
        "fde_m": float(error[:, -1].mean()),
        "translation_m": float(
            torch.linalg.vector_norm(pred[:, -1].mean(1) - target[:, -1].mean(1), dim=-1).mean()
        ),
        "rotation_deg": float(rotation_error_deg(source, target[:, -1], pred[:, -1]).mean()),
    }


def move_batch(batch, device: torch.device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate_gt_ade(model, loader, device: torch.device) -> float:
    model.eval()
    total_error = 0.0
    total_points = 0
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(
            batch["object_points"],
            batch["object_normals"],
            batch["hand_points"],
            batch["hand_normals"],
        )["object_tracks"]
        total_error += float(torch.linalg.vector_norm(pred - batch["object_points"], dim=-1).sum())
        total_points += pred.shape[0] * pred.shape[1] * pred.shape[2]
    return total_error / total_points


@torch.no_grad()
def evaluate_conditions(model, loader, device: torch.device) -> Dict[str, Dict[str, float]]:
    model.eval()
    collected = {name: [] for name in ("gt", "static", "shuffled")}
    for batch in loader:
        batch = move_batch(batch, device)
        hand = batch["hand_points"]
        normals = batch["hand_normals"]
        shuffled_hand = torch.roll(hand, shifts=1, dims=0) if len(hand) > 1 else torch.flip(hand, dims=[1])
        shuffled_normals = (
            torch.roll(normals, shifts=1, dims=0) if len(normals) > 1 else torch.flip(normals, dims=[1])
        )
        variants = {
            "gt": (hand, normals),
            "static": (hand[:, :1].expand_as(hand), normals[:, :1].expand_as(normals)),
            "shuffled": (shuffled_hand, shuffled_normals),
        }
        for name, (hand_variant, normal_variant) in variants.items():
            output = model(
                batch["object_points"], batch["object_normals"], hand_variant, normal_variant
            )["object_tracks"]
            collected[name].append((output.cpu(), batch["object_points"].cpu()))
    result = {}
    for name, parts in collected.items():
        pred = torch.cat([part[0] for part in parts])
        target = torch.cat([part[1] for part in parts])
        result[name] = metrics(pred, target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PointWorldWAM V0 forward overfit")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_forward_overfit.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.train.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    dataset = build_dataset(cfg.data)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size), shuffle=False, num_workers=0)
    model = GRABPointWorldForward(cfg.model)
    load_report = model.load_pointworld_checkpoint(cfg.model.pointworld_checkpoint)
    model.to(device)

    adapter_names = ("scene_adapter", "hand_adapter", "time_embed", "hand_type_emb")
    adapters, pretrained = [], []
    for name, parameter in model.named_parameters():
        (adapters if name.startswith(adapter_names) else pretrained).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapters, "lr": cfg.train.lr_adapter},
            {"params": pretrained, "lr": cfg.train.lr_pretrained},
        ],
        weight_decay=cfg.train.weight_decay,
    )
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    iterator = iter(loader)
    best_train_loss = math.inf
    best_eval_ade = math.inf
    for step in range(1, cfg.train.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(
            batch["object_points"], batch["object_normals"], batch["hand_points"], batch["hand_normals"]
        )["object_tracks"]
        loss = weighted_huber(pred, batch["object_points"], cfg.loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step={step} loss={float(loss)}")
        loss.backward()
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach())}
        history.append(record)
        best_train_loss = min(best_train_loss, record["loss"])
        if step == 1 or step % cfg.train.log_every == 0:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            eval_ade = evaluate_gt_ade(model, eval_loader, device)
            print(json.dumps({"step": step, "eval_gt_ade_m": eval_ade}), flush=True)
            if eval_ade < best_eval_ade:
                best_eval_ade = eval_ade
                torch.save(
                    {"model": model.state_dict(), "config": config_to_dict(cfg), "step": step},
                    output_dir / "best.pt",
                )

    torch.save(
        {"model": model.state_dict(), "config": config_to_dict(cfg), "step": cfg.train.steps},
        output_dir / "last.pt",
    )
    condition_metrics = evaluate_conditions(model, eval_loader, device)
    summary = {
        "windows": len(dataset),
        "sequences": sorted({ref.shared_path.parent.as_posix() for ref in dataset.windows}),
        "load_report": load_report,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "best_train_loss": best_train_loss,
        "best_eval_ade_m": best_eval_ade,
        "conditions": condition_metrics,
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
