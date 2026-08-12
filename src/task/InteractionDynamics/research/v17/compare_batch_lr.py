"""汇总 V17 短优化 checkpoint 的配置与 validation dynamic 指标。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu")
        training = checkpoint["config"]["training"]
        diffusion = checkpoint["val"]["diffusion"]
        rows.append({"checkpoint": str(path), "epoch": checkpoint["epoch"],
                     "per_gpu_batch": training["per_gpu_batch"], "lr": training["lr"],
                     "diffusion_dynamic_r_cm": diffusion["dynamic_r_rmse_cm"],
                     "diffusion_dynamic_d_cm": diffusion["dynamic_d_rmse_cm"],
                     "selection_score": diffusion["dynamic_r_rmse_cm"]
                                        + diffusion["dynamic_d_rmse_cm"]})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
