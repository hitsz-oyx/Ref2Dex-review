"""诊断当前手物距离、anchor r 空间方差与 GT residual 的关系。"""
import json

import torch

from src.task.InteractionDynamics.residual_interaction_regression import residual_target
from src.task.InteractionDynamics.train_goal_interaction_diffusion import controlled_dataset, materialize


def main() -> None:
    dataset, indices, metadata = controlled_dataset(
        "src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml", 4, 32, 4, .2, 1., 3)
    data = materialize(dataset, indices, metadata, torch.device("cpu"), 4, 4, .015, 3, .2, 1.)
    distance = torch.as_tensor(metadata["distance_cm"])[metadata["frames"]]
    r_variance = data["state"][..., :3].var(1).mean(1)
    residual = residual_target(data["state"], data["future"], 4).square().mean((1, 2)).sqrt()
    bins = [(0, 10, "0-10 cm"), (10, 30, "10-30 cm"), (30, 60, "30-60 cm"),
            (60, 100, "60-100 cm"), (100, float("inf"), ">100 cm")]
    output = {}
    for low, high, name in bins:
        mask = (distance >= low) & (distance < high)
        output[name] = {"sample_count": int(mask.sum()),
                        "anchor_r_variance_cm2": float(r_variance[mask].mean()) if mask.any() else None,
                        "gt_residual_rms_cm": float(residual[mask].mean()) if mask.any() else None}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
