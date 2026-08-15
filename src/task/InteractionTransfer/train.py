from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from .dataset import GRABOneStepDataset
from .losses import flow_loss
from .model import InteractionTransfer


def train_one_step(model, loader, optimizer, device="cpu", steps=100):
    model.to(device).train()
    iterator = iter(loader)
    history = []
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                     batch["hand_normals"], batch["hand_flow"])["object_flow"]
        loss = flow_loss(pred, batch["object_flow"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def main():
    parser = argparse.ArgumentParser(description="InteractionTransfer V0 one-step training")
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    dataset = GRABOneStepDataset(args.root, args.sequence)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = InteractionTransfer()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    history = train_one_step(model, loader, optimizer, steps=args.steps)
    print({"steps": args.steps, "initial_loss": history[0], "final_loss": history[-1]})


if __name__ == "__main__":
    main()
