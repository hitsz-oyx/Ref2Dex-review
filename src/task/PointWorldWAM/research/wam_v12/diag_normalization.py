from __future__ import annotations

import json
from pathlib import Path

import torch

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset_wam_chunk import build_wam_chunk_dataset


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    cfg = load_config(
        str(ROOT / "src/task/PointWorldWAM/configs/grab_wam_v12_debug.yaml")
    )
    dataset = build_wam_chunk_dataset(cfg.data)
    statistics = dataset.statistics()
    worlds = torch.stack([dataset[index]["world_chunk"] for index in range(len(dataset))])
    report = {"windows": len(dataset), "world": {}, "left": {}, "right": {}}
    normalized_world = (
        worlds - statistics["world_mean"]
    ) / statistics["world_std"]
    for timestep in range(dataset.chunk_size):
        report["world"][str(timestep + 1)] = {
            "displacement_mean_norm_m": float(
                torch.linalg.vector_norm(worlds[:, timestep], dim=-1).mean()
            ),
            "normalized_rms": float(normalized_world[:, timestep].square().mean().sqrt()),
        }
    for side in ("left", "right"):
        actions = torch.stack(
            [dataset[index][f"{side}_action_chunk"] for index in range(len(dataset))]
        )
        normalized = (actions - statistics[f"{side}_mean"]) / statistics[
            f"{side}_std"
        ]
        for timestep in range(dataset.chunk_size):
            report[side][str(timestep + 1)] = {
                "action_mean_norm": float(
                    torch.linalg.vector_norm(actions[:, timestep], dim=-1).mean()
                ),
                "normalized_rms": float(normalized[:, timestep].square().mean().sqrt()),
            }
    output = ROOT / "output/research/pointworld_wam/wam_v12"
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalization.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
