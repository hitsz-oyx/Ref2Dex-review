"""V0.5 单模型子进程：一次加载 PTv3，完成训练与 held-out intervention。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import torch
from torch.utils.data import DataLoader

from src.task.InteractionTransfer.baselines.direct_edge import DirectEdge
from src.task.InteractionTransfer.dataset import GRABOneStepDataset
from src.task.InteractionTransfer.losses import flow_loss
from src.task.InteractionTransfer.metrics import flow_metrics
from src.task.InteractionTransfer.model import InteractionTransfer


def report_memory(tag):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
    print(f"[{tag}] RSS={rss:.1f} MiB | GPU allocated={allocated:.1f} MiB | GPU reserved={reserved:.1f} MiB", flush=True)


def train(model, dataset, device, steps):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.to(device).train()
    assert not model.static_encoder.training and not model.static_encoder.backbone.training
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    iterator, history = iter(loader), []
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                    batch["hand_normals"], batch["hand_flow"])
        loss = flow_loss(out["object_flow"], batch["object_flow"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    if len(dataset) < 2:
        raise ValueError("cross-sample evaluation requires at least 2 validation transitions")
    names = ("gt", "zero", "reverse", "cross_sample", "point_shuffle", "mean_flow")
    sums = {name: {"point_error_m": 0.0, "translation_error_m": 0.0, "rotation_error_deg": 0.0}
            for name in names}
    field_dist = {name: 0.0 for name in names if name != "gt"}
    action_dist = {name: 0.0 for name in names if name != "gt"}
    permutation = torch.roll(torch.arange(len(dataset)), shifts=1)
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
        gt = batch["hand_flow"]
        wrong = dataset[int(permutation[index])]["hand_flow"].unsqueeze(0).to(device)
        variants = {"gt": gt, "zero": torch.zeros_like(gt), "reverse": -gt,
                    "cross_sample": wrong,
                    "point_shuffle": gt[:, torch.randperm(gt.shape[1], device=device)],
                    "mean_flow": gt.mean(1, keepdim=True).expand_as(gt)}
        gt_field = None
        for name, action in variants.items():
            out = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                        batch["hand_normals"], action)
            metric = flow_metrics(out["object_flow"], batch["object_flow"], batch["object_points"])
            sums[name] = {key: sums[name][key] + value for key, value in metric.items()}
            if name == "gt":
                gt_field = out["object_field"].detach()
            else:
                field_dist[name] += float((gt_field - out["object_field"]).norm(dim=-1).mean())
                action_dist[name] += float((gt - action).norm(dim=-1).mean())
            del out
    n = len(dataset)
    return ({name: {key: value / n for key, value in metric.items()} for name, metric in sums.items()},
            {name: value / n for name, value in field_dist.items()},
            {name: value / n for name, value in action_dist.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("cm", "direct_edge"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-sequence", action="append", required=True)
    parser.add_argument("--val-sequence", action="append", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-transitions", type=int, default=16)
    parser.add_argument("--train-max-transitions", type=int)
    parser.add_argument("--val-max-transitions", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(13)
    report_memory("process_start")
    train_ds = GRABOneStepDataset(args.root, args.train_sequence,
                                  max_transitions=args.train_max_transitions or args.max_transitions)
    val_ds = GRABOneStepDataset(args.root, args.val_sequence,
                                max_transitions=args.val_max_transitions or args.max_transitions)
    report_memory("after_dataset")
    model = InteractionTransfer() if args.model == "cm" else DirectEdge()
    report_memory("after_model_init")
    loss = train(model, train_ds, torch.device(args.device), args.steps)
    report_memory("after_train")
    metrics, field, action = evaluate(model, val_ds, torch.device(args.device))
    report_memory("after_eval")
    result = {"model": args.model, "pid": os.getpid(), "train_sequences": args.train_sequence,
              "val_sequences": args.val_sequence, "train_transitions": len(train_ds),
              "val_transitions": len(val_ds), "initial_loss": loss[0], "final_loss": loss[-1],
              "eval": metrics, "field_distance": field, "action_distance_m": action}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
