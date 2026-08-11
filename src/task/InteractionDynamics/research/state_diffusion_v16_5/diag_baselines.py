"""V16.5 controlled 数据的 persistence baseline 与条件多样性诊断。"""
from __future__ import annotations

import argparse
import json

import torch

from src.task.InteractionDynamics.eval_interaction_compression import make_dataset
from src.task.InteractionDynamics.train_interaction_diffusion import controlled_indices
from src.task.InteractionDynamics.train_state_interaction_diffusion import (
    future_metrics, materialize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset = make_dataset(args.config)
    indices = controlled_indices(dataset, args.samples, args.sequence_index)
    data = materialize(dataset, indices, torch.device(args.device), args.horizon, args.tau_m)
    state = data["state"]
    persistence = torch.cat([
        torch.cat([torch.zeros_like(state[..., :3]), state], -1)
        for _ in range(args.horizon)], -1)
    target = data["future"]
    print(json.dumps({"diagnostic": "persistence", **future_metrics(
        persistence, target, args.horizon)}, ensure_ascii=False))
    print(json.dumps({
        "diagnostic": "condition_diversity",
        "effect_std": data["effect"].std(0).tolist(),
        "effect_shuffle_rmse": float((data["effect"] - data["effect"].roll(1, 0)).square().mean().sqrt()),
        "state_shuffle_rmse": float((state - state.roll(1, 0)).square().mean().sqrt()),
        "future_shuffle_rmse": float((target - target.roll(1, 0)).square().mean().sqrt()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
