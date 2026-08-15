from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset_wam import build_wam_dataset
from .pointworld_bimanual_forward import GRABPointWorldBimanualForward
from .train_wam_v1 import evaluate_fm, evaluate_generation
from .wam_v1 import WAMV1


def main() -> None:
    parser = argparse.ArgumentParser(description="复算 WAM V1 best generation")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_wam_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.train.device)
    dataset = build_wam_dataset(cfg.data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    checkpoint_path = Path(cfg.train.output_dir) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WAMV1(cfg.model, dataset.action_statistics()).to(device)
    model.load_state_dict(checkpoint["model"])
    forward_cfg = load_config(cfg.eval.forward_config)
    forward_model = GRABPointWorldBimanualForward(forward_cfg.model).to(device)
    forward_checkpoint = torch.load(
        cfg.eval.forward_checkpoint, map_location="cpu", weights_only=False
    )
    forward_model.load_state_dict(forward_checkpoint["model"])
    report = {
        "best_step": int(checkpoint["step"]),
        "fm": evaluate_fm(model, loader, device, cfg.seed + 1000),
        "generation": evaluate_generation(
            model,
            loader,
            device,
            cfg.eval.fm_inference_steps,
            cfg.seed + 2000,
            forward_model,
        ),
    }
    output = Path("output/research/pointworld_wam/wam_v1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
