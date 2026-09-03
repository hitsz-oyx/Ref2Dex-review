"""Extract object-side Cm tokens from an ObjectInteractionCm checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.base import load_checkpoint, load_config, task_config_from_dict
from .dataset import ObjectInteractionCmDataset
from .model import ObjectInteractionCmModel


def extract(checkpoint: str | Path, *, output: str | Path, device: str = "cpu", max_samples: int | None = None) -> Path:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    cfg = task_config_from_dict(payload["config"])
    dataset = ObjectInteractionCmDataset(
        cfg.data.root or cfg.data.train_path,
        num_obj_points=int(cfg.meta.num_obj_points),
        num_hand_points=int(cfg.meta.num_hand_points),
        min_stride=int(cfg.data.min_stride),
        max_stride=int(cfg.data.max_stride),
        active_only=bool(cfg.data.active_only),
        fixed_stride=int(cfg.data.min_stride),
    )
    model = ObjectInteractionCmModel(cfg.model).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    count = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    tokens, anchors = [], []
    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            batch = {key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value for key, value in sample.items()}
            result = model(batch)
            tokens.append(result["cm_tokens"][0].cpu().numpy())
            anchors.append(result["cm_anchor_pos"][0].cpu().numpy())
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cm_tokens=np.asarray(tokens), cm_anchor_pos=np.asarray(anchors))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ObjectInteractionCm tokens.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    extract(args.checkpoint, output=args.output, device=args.device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
