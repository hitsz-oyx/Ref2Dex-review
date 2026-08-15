from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset_wam_chunk import build_wam_chunk_dataset
from .train_wam_v2_regression import evaluate_regression
from .wam_v2 import ChunkJointWAM


def main() -> None:
    parser = argparse.ArgumentParser(description="复算 V1.2 deterministic gates")
    parser.add_argument("--mode", choices=("inverse", "forward"), required=True)
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_wam_v12_debug.yaml"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.train.device)
    dataset = build_wam_chunk_dataset(cfg.data)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    checkpoint_path = Path(cfg.train.output_dir) / args.mode / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ChunkJointWAM(cfg.model, dataset.statistics(), dataset.subjects).to(device)
    model.load_state_dict(checkpoint["model"])
    result = {
        "mode": args.mode,
        "step": int(checkpoint["step"]),
        "windows": len(dataset),
        "evaluation": evaluate_regression(model, loader, device, args.mode),
    }
    output = Path("output/research/pointworld_wam/wam_v12")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.mode}_eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
