"""比较 V16.6 deterministic clean regression 是否使用 motion Goal。"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from src.task.InteractionDynamics.goal_interaction_diffusion import GoalInteractionDiffusion
from src.task.InteractionDynamics.train_goal_interaction_diffusion import (
    controlled_dataset, materialize, persistence_future)
from src.task.InteractionDynamics.train_state_interaction_diffusion import future_metrics


class CleanGoalRegressor(nn.Module):
    def __init__(self, use_motion: bool, dim: int = 256, layers: int = 6) -> None:
        super().__init__()
        self.use_motion = use_motion
        self.backbone = GoalInteractionDiffusion(4, 4, dim, 8, layers)

    def forward(self, state, anchors, patches, goal):
        if not self.use_motion:
            goal = torch.cat([goal[:, :1], torch.zeros_like(goal[:, 1:])], -1)
        noisy = torch.zeros((*state.shape[:2], 28), device=state.device)
        timestep = torch.zeros(len(state), dtype=torch.long, device=state.device)
        return self.backbone(noisy, state, anchors, patches, goal, timestep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset, indices, metadata = controlled_dataset(
        "src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml", 4, 32, 4, .2, 1., 3)
    data = materialize(dataset, indices, metadata, device, 4, 4, .015, 3, .2, 1.)
    state_mean, state_std = data["state"].mean((0, 1), True), data["state"].std((0, 1), True).clamp_min(.05)
    future_mean, future_std = data["future"].mean((0, 1), True), data["future"].std((0, 1), True).clamp_min(.05)
    goal_mean, goal_std = data["goal"].mean(0, True), data["goal"].std(0, True).clamp_min(.01)
    state = ((data["state"] - state_mean) / state_std).to(device)
    future = ((data["future"] - future_mean) / future_std).to(device)
    anchors, patches = data["anchors_cm"].to(device), data["object_patches"].to(device)
    goal = ((data["goal"] - goal_mean) / goal_std).to(device)
    persistence = persistence_future(data["state"], 4)
    dynamic = (data["future"] - persistence).square().mean((1, 2)).sqrt() >= .2
    results = {}
    for name, use_motion in [("active_only", False), ("active_motion", True)]:
        torch.manual_seed(42)
        model = CleanGoalRegressor(use_motion).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(42)
        model.train()
        for _ in range(args.steps):
            index = torch.randint(len(future), (4,), generator=generator)
            prediction = model(state[index], anchors[index], patches[index], goal[index])
            loss = torch.nn.functional.mse_loss(prediction, future[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = (model(state, anchors, patches, goal).cpu() * future_std + future_mean)
            shuffled_goal = goal.clone()
            active = data["goal"][:, 0] > .5
            shuffled_goal[active, 1:] = shuffled_goal[active, 1:].roll(1, 0)
            shuffled = model(state, anchors, patches, shuffled_goal).cpu() * future_std + future_mean
        results[name] = {
            "overall": future_metrics(prediction, data["future"], 4),
            "dynamic": future_metrics(prediction[dynamic], data["future"][dynamic], 4),
            "motion_shuffle_dynamic": future_metrics(shuffled[dynamic], data["future"][dynamic], 4),
        }
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
