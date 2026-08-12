"""固定 V16.9 checkpoint、数据与噪声比较正规 DDIM timestep skipping。"""
from __future__ import annotations

import argparse
import json

import torch

from src.task.InteractionDynamics.residual_interaction_diffusion import (
    ResidualInteractionDiffusion, sample_residual_v)
from src.task.InteractionDynamics.residual_interaction_regression import (
    persistence_future, residual_target)
from src.task.InteractionDynamics.train_goal_interaction_diffusion import controlled_dataset, materialize
from src.task.InteractionDynamics.train_state_interaction_diffusion import future_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu")
    dataset, indices, metadata = controlled_dataset(
        "src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml", 4, 32, 4, .2, 1., 3)
    if list(indices) != list(saved["indices"]):
        raise ValueError("checkpoint and controlled dataset indices differ")
    data = materialize(dataset, indices, metadata, device, 4, 4, .015, 3, .2, 1.)
    model = ResidualInteractionDiffusion(4, 4, saved["dim"], 8, saved["layers"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    state = (data["state"].to(device) - saved["state_mean"].to(device)) / saved["state_std"].to(device)
    goal = (data["goal"].to(device) - saved["goal_mean"].to(device)) / saved["goal_std"].to(device)
    anchors, patches = data["anchors_cm"].to(device), data["object_patches"].to(device)
    persistence = persistence_future(data["state"], 4)
    dynamic = residual_target(data["state"], data["future"], 4).square().mean((1, 2)).sqrt() >= .2
    noise = torch.randn((32, 128, 28), generator=torch.Generator().manual_seed(42)).to(device)
    result = {}
    for count in [100, 50, 25, 10]:
        normalized, _ = sample_residual_v(
            model, state, anchors, patches, goal, 100, initial_noise=noise,
            sampling_steps=count)
        residual = normalized.cpu() * saved["residual_std"] + saved["residual_mean"]
        future = persistence + residual
        result[str(count)] = future_metrics(future[dynamic], data["future"][dynamic], 4)
    print(json.dumps({"checkpoint": args.checkpoint, "dynamic": result},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
