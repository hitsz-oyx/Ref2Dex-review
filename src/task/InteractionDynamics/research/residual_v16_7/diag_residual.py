"""V16.7 controlled32 residual target 数值诊断。"""
import json

import torch

from src.task.InteractionDynamics.residual_interaction_regression import residual_target
from src.task.InteractionDynamics.train_goal_interaction_diffusion import controlled_dataset, materialize
from src.task.InteractionDynamics.train_residual_interaction_regression import residual_statistics


def main() -> None:
    dataset, indices, metadata = controlled_dataset(
        "src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml", 4, 32, 4, .2, 1., 3)
    data = materialize(dataset, indices, metadata, torch.device("cpu"), 4, 4, .015, 3, .2, 1.)
    residual = residual_target(data["state"], data["future"], 4)
    magnitude = residual.square().mean((1, 2)).sqrt()
    dynamic = magnitude >= .2
    reconstruction_error = (data["future"] - (
        data["future"] - residual + residual)).abs().max()
    print(json.dumps({"sequence": metadata["path"], "frames": metadata["frames"],
                      "dynamic_count": int(dynamic.sum()), "static_count": int((~dynamic).sum()),
                      "max_reconstruction_error": float(reconstruction_error),
                      **residual_statistics(residual, dynamic)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
