"""V0.4: 同一真实 cache 上训练 Cm/direct 并完成 intervention 比较。"""
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from src.task.InteractionTransfer.baselines.direct_forward import DirectForward
from src.task.InteractionTransfer.dataset import GRABOneStepDataset
from src.task.InteractionTransfer.losses import flow_loss
from src.task.InteractionTransfer.metrics import flow_metrics
from src.task.InteractionTransfer.model import InteractionTransfer


def _batch(sample, device):
    return {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}


def _train(model, loader, device, steps):
    model.to(device).train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    it = iter(loader)
    losses = []
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        pred = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                     batch["hand_normals"], batch["hand_flow"])
        loss = flow_loss(pred["object_flow"] if isinstance(pred, dict) else pred, batch["object_flow"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def _intervention(model, sample, device):
    model.eval()
    b = _batch(sample, device)
    action = b["hand_flow"]
    perm = torch.randperm(action.shape[1], device=device)
    variants = {"gt": action, "zero": torch.zeros_like(action), "reverse": -action,
                "shuffle": action[:, perm]}
    results, fields = {}, {}
    for name, flow in variants.items():
        out = model(b["object_points"], b["object_normals"], b["hand_points"], b["hand_normals"], flow)
        pred = out["object_flow"] if isinstance(out, dict) else out
        results[name] = flow_metrics(pred, b["object_flow"])["point_error_m"]
        if isinstance(out, dict):
            fields[name] = out["object_field"]
    distances = {name: float((fields["gt"] - fields[name]).norm(dim=-1).mean())
                 for name in ("zero", "reverse", "shuffle")}
    return results, distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    device = torch.device(args.device)
    dataset = GRABOneStepDataset(args.root, [args.sequence], max_transitions=32)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    cm = InteractionTransfer().to(device)
    direct = DirectForward().to(device)
    cm_losses = _train(cm, loader, device, args.steps)
    direct_losses = _train(direct, loader, device, args.steps)
    cm_epe, field_dist = _intervention(cm, dataset[0], device)
    direct.eval()
    direct_epe = flow_metrics(direct(**{k: _batch(dataset[0], device)[k] for k in
                                        ("object_points", "object_normals", "hand_points", "hand_normals", "hand_flow")}),
                              _batch(dataset[0], device)["object_flow"])["point_error_m"]
    print({"cm_initial_loss": cm_losses[0], "cm_final_loss": cm_losses[-1],
           "direct_initial_loss": direct_losses[0], "direct_final_loss": direct_losses[-1],
           "cm_intervention_epe": cm_epe, "direct_epe_gt": direct_epe,
           "field_distance_gt_to_variant": field_dist,
           "cm_direct_ratio_gt": cm_epe["gt"] / max(direct_epe, 1e-12)})


if __name__ == "__main__":
    main()
