from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .dataset_wam_chunk import build_wam_chunk_dataset
from .train_wam_v2 import evaluate_generation
from .wam_v2 import ChunkJointWAM


def main() -> None:
    parser = argparse.ArgumentParser(description="复算 Chunk Joint WAM generation gates")
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_wam_v2_chunk.yaml"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.train.device)
    dataset = build_wam_chunk_dataset(cfg.data)
    evaluation_dataset = dataset
    if args.max_windows and args.max_windows < len(dataset):
        indices = np.linspace(
            0, len(dataset) - 1, args.max_windows, dtype=np.int64
        ).tolist()
        evaluation_dataset = Subset(dataset, indices)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    checkpoint_path = Path(args.checkpoint or Path(cfg.train.output_dir) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ChunkJointWAM(cfg.model, dataset.statistics(), dataset.subjects).to(device)
    model.load_state_dict(checkpoint["model"])
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "step": int(checkpoint["step"]),
        "windows": len(evaluation_dataset),
        "generation": evaluate_generation(
            model,
            loader,
            device,
            cfg.flow_matching.inference_steps,
            cfg.seed + 100000,
        ),
    }
    output = Path("output/research/pointworld_wam/wam_v2")
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"eval_{len(evaluation_dataset)}.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
