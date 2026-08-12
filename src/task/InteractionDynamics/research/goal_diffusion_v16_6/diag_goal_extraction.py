"""检查 V16.6 active interval、meaningful-motion 阈值与 Goal 对齐。"""
from __future__ import annotations

import argparse
import json

import torch

from src.task.InteractionDynamics.goal_interaction_diffusion import object_increments
from src.task.InteractionDynamics.train_goal_interaction_diffusion import (
    controlled_dataset, materialize, persistence_future)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset, indices, metadata = controlled_dataset(args.config, args.sequence_index, 32, 4, .2, 1., 3)
    data = materialize(dataset, indices, metadata, device, 4, 4, .015, 3, .2, 1.)
    increments = object_increments(metadata["object_poses"])
    change = (data["future"] - persistence_future(data["state"], 4)).square().mean((1, 2)).sqrt()
    print(json.dumps({"sequence": metadata["path"],
                      "active_interval": [metadata["active_start"], metadata["active_end"]],
                      "selected_active": int((data["goal"][:, 0] > .5).sum()),
                      "selected_inactive": int((data["goal"][:, 0] < .5).sum()),
                      "future_change_quantile_cm": torch.quantile(change, torch.tensor([0., .25, .5, .75, 1.])).tolist()},
                     ensure_ascii=False))
    frames = metadata["frames"]
    categories = {
        "inactive_before": [i for i, f in enumerate(frames) if f < metadata["active_start"]],
        "active_early": [i for i, f in enumerate(frames) if metadata["active_start"] <= f < (metadata["active_start"] + metadata["active_end"]) // 2],
        "active_late": [i for i, f in enumerate(frames) if (metadata["active_start"] + metadata["active_end"]) // 2 <= f <= metadata["active_end"]],
        "inactive_after": [i for i, f in enumerate(frames) if f > metadata["active_end"]],
    }
    for category, candidates in categories.items():
        for row in candidates[:5]:
            frame, onset = frames[row], int(data["onset"][row])
            current_motion = increments[frame].tolist() if frame < len(increments) else [0.] * 6
            goal_motion = data["goal"][row, 1:].reshape(4, 6)
            print(json.dumps({"category": category, "frame": frame,
                              "active": int(data["goal"][row, 0]), "current_motion": current_motion,
                              "k_star": onset, "goal_start_frame": onset,
                              "goal_dp_cm": goal_motion[:, :3].tolist(),
                              "goal_dw_rad": goal_motion[:, 3:].tolist(),
                              "hand_object_distance_cm": float(metadata["distance_cm"][frame]),
                              "future_y_change_cm": float(change[row])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
