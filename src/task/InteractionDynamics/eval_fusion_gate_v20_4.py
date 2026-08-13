"""V20.4 learned velocity gate 的 split 指标、常数干预与行为分析。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.task.InteractionDynamics.eval_direct_h_v20_3 import behavior_metrics
from src.task.InteractionDynamics.fusion_gate_v20_4 import (
    FusionPredictionDataset, VelocityFusionGate, fuse_velocity, fusion_features,
)
from src.task.InteractionDynamics.train_h_realizer_v20_2 import error_metrics


def gate_statistics(gate: torch.Tensor, distance: torch.Tensor) -> dict:
    bins = (("d_lt_0.5", -float("inf"), .5), ("d_0.5_1", .5, 1.),
            ("d_1_2", 1., 2.), ("d_ge_2", 2., float("inf")))
    by_distance = {}
    for name, lower, upper in bins:
        mask = (distance >= lower) & (distance < upper)
        by_distance[name] = {"count": int(mask.sum()),
                             "mean": float(gate[..., 0][mask].mean()) if mask.any() else 0.0}
    return {"mean": float(gate.mean()), "std": float(gate.std()),
            "by_timestep": gate.mean((0, 1, 3)).tolist(), "by_free_distance": by_distance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(); device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = VelocityFusionGate(hidden=payload["args"]["hidden"]).to(device)
    model.load_state_dict(payload["model"]); model.eval()
    loader = DataLoader(FusionPredictionDataset(args.cache / f"{args.split}.pt"),
                        args.batch_size, shuffle=False, num_workers=0)
    free_rows, hand_rows, target_rows, current_rows, gate_rows = [], [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            gate = model(fusion_features(batch["y_free"], batch["y_h"]))
            for rows, value in ((free_rows, batch["y_free"]), (hand_rows, batch["y_h"]),
                                (target_rows, batch["future_y"]), (current_rows, batch["current_y"]),
                                (gate_rows, gate)):
                rows.append(value.cpu())
    free, hand, target, current, learned = map(torch.cat,
        (free_rows, hand_rows, target_rows, current_rows, gate_rows))
    mean_gate = learned.mean(); modes = {"learned": learned, "force_y": torch.zeros_like(learned),
                                         "force_h": torch.ones_like(learned),
                                         "mean_gate": torch.full_like(learned, mean_gate)}
    result = {"split": args.split, "samples": len(target), "gate": gate_statistics(learned, free[..., 3]),
              "interventions": {}}
    for name, gate in modes.items():
        prediction = fuse_velocity(free, hand, gate)
        result["interventions"][name] = {"error": error_metrics(prediction[..., :7], target),
            "behavior": behavior_metrics(prediction, target, current)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
