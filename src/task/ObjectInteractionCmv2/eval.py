"""Evaluate a V1.0 pilot checkpoint on a small OakInk2 subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import ObjectInteractionCmv2Model, object_interaction_loss
from .oakink2 import OakInk2RigidDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=8)
    args = p.parse_args()
    model = ObjectInteractionCmv2Model()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = OakInk2RigidDataset(args.data_root, max_files=args.max_files, max_frames=args.max_frames)
    values = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=2, shuffle=False):
            batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
            values.append({k: float(v) for k, v in object_interaction_loss(model(batch), batch).items()})
    result = {k: sum(x[k] for x in values) / len(values) for k in values[0]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
