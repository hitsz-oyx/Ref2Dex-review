"""V0.5 sequence-level benchmark: Cm、DirectEdge 与 causal action intervention。"""
from __future__ import annotations

import argparse
from pathlib import Path
import gc
import torch
from torch.utils.data import DataLoader

from src.task.InteractionTransfer.baselines.direct_edge import DirectEdge
from src.task.InteractionTransfer.dataset import GRABOneStepDataset
from src.task.InteractionTransfer.losses import flow_loss
from src.task.InteractionTransfer.metrics import flow_metrics
from src.task.InteractionTransfer.model import InteractionTransfer


def sequences(root):
    root = Path(root)
    return sorted(str(p.parent.relative_to(root)) for p in root.glob("*/*/shared.npz"))


def train(model, loader, device, steps):
    model.to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    iterator = iter(loader)
    history = []
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
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {name: {"point_error_m": 0.0, "translation_error_m": 0.0, "rotation_error_deg": 0.0}
            for name in ("gt", "zero", "reverse", "cross_sample", "point_shuffle", "mean_flow")}
    field_dist = {name: 0.0 for name in ("zero", "reverse", "cross_sample", "point_shuffle", "mean_flow")}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        gt = batch["hand_flow"]
        variants = {
            "gt": gt, "zero": torch.zeros_like(gt), "reverse": -gt,
            "cross_sample": torch.roll(gt, shifts=1, dims=0),
            "point_shuffle": gt[:, torch.randperm(gt.shape[1], device=device)],
            "mean_flow": gt.mean(1, keepdim=True).expand_as(gt),
        }
        gt_field = None
        for name, action in variants.items():
            out = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                        batch["hand_normals"], action)
            metric = flow_metrics(out["object_flow"], batch["object_flow"], batch["object_points"])
            sums[name] = {k: sums[name][k] + v for k, v in metric.items()}
            if name == "gt":
                gt_field = out["object_field"].detach()
            elif name in field_dist:
                field_dist[name] += float((gt_field - out["object_field"]).norm(dim=-1).mean())
            del out
        count += 1
    return ({name: {k: v / count for k, v in values.items()} for name, values in sums.items()},
            {k: v / count for k, v in field_dist.items()})


def action_field_ratio(dataset, device):
    values = []
    for index in range(min(len(dataset), 64)):
        flow = dataset[index]["hand_flow"].to(device)
        centered = flow - flow.mean(0, keepdim=True)
        values.append(float(centered.norm(dim=-1).mean() / flow.norm(dim=-1).mean().clamp_min(1e-8)))
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-transitions", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=2,
                        help="限制 sequence 数量以控制 GRAB cache 内存；至少 2")
    parser.add_argument("--sequence", action="append", help="显式指定 sequence；至少两个")
    parser.add_argument("--train-sequence", action="append")
    parser.add_argument("--val-sequence", action="append")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(13)
    device = torch.device(args.device)
    seq = args.sequence if args.sequence else sequences(args.root)[:args.max_sequences]
    if args.train_sequence or args.val_sequence:
        if not args.train_sequence or not args.val_sequence:
            raise ValueError("--train-sequence 与 --val-sequence 必须同时提供")
        train_seq, val_seq = args.train_sequence, args.val_sequence
    else:
        train_seq, val_seq = seq[:max(1, len(seq) - 1)], seq[-1:]
    if len(seq) < 2:
        raise ValueError("V0.5 sequence-level benchmark 至少需要两个 sequence")
    split = max(1, int(len(seq) * 0.8))
    # PTv3/cache forward 的显存峰值较高；batch=1 仍保持 sequence-level split，
    # cross-sample intervention 在 val loader 只有一个样本时退化为自身，脚本仍报告该项。
    train_ds = GRABOneStepDataset(args.root, train_seq, max_transitions=args.max_transitions)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    # 两个模型各自包含一份 PTv3；顺序运行避免同时占满 GPU 显存。
    cm = InteractionTransfer()
    cm_loss = train(cm, train_loader, device, args.steps)
    del train_loader, train_ds
    gc.collect()
    val_ds = GRABOneStepDataset(args.root, val_seq, max_transitions=args.max_transitions)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    cm_eval, cm_field = evaluate(cm, val_loader, device)
    del cm
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    direct = DirectEdge()
    del val_loader, val_ds
    gc.collect()
    train_ds = GRABOneStepDataset(args.root, train_seq, max_transitions=args.max_transitions)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    direct_loss = train(direct, train_loader, device, args.steps)
    del train_loader, train_ds
    gc.collect()
    val_ds = GRABOneStepDataset(args.root, val_seq, max_transitions=args.max_transitions)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    direct_eval, _ = evaluate(direct, val_loader, device)
    print({"train_sequences": train_seq, "val_sequences": val_seq,
           "train_transitions": len(train_ds), "val_transitions": len(val_ds),
           "action_local_ratio": action_field_ratio(val_ds, device),
           "cm_initial_loss": cm_loss[0], "cm_final_loss": cm_loss[-1],
           "direct_initial_loss": direct_loss[0], "direct_final_loss": direct_loss[-1],
           "cm_eval": cm_eval, "direct_edge_eval_gt": direct_eval["gt"],
           "cm_direct_edge_ratio": cm_eval["gt"]["point_error_m"] / max(direct_eval["gt"]["point_error_m"], 1e-12),
           "cm_field_distance": cm_field})


if __name__ == "__main__":
    main()
