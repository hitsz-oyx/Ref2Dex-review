"""汇总 V16.1 MANO self-inverse 产物，输出按初始化和目标分组的指标。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("output/research/InteractionDynamics/y_inverse"))
    return parser.parse_args()


def main() -> None:
    groups: dict[tuple[str, str, float], list[dict[str, float]]] = defaultdict(list)
    for path in sorted(parse_args().input_dir.glob("sample*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            history = json.loads(str(payload["history_json"].item()))
            row = history[-1]
            row["file"] = path.name
            beta_offset = (float(payload["candidate_beta_offset"].item())
                           if "candidate_beta_offset" in payload.files else 0.0)
            groups[(str(payload["initialization"].item()),
                    str(payload["target"].item()), beta_offset)].append(row)

    fields = ("surface_ade_mm", "surface_fde_mm", "joint_mpjpe_mm",
              "wrist_error_mm", "tip_mpjpe_mm", "r_rmse_cm", "u_rmse_cm",
              "g_rmse", "contact_f1")
    for (initialization, target, beta_offset), rows in sorted(groups.items()):
        print(f"\n{initialization}/{target}/beta_offset={beta_offset:g}（n={len(rows)}）")
        for row in rows:
            print(f"  {row['file']}: ADE={row['surface_ade_mm']:.4f} mm, "
                  f"FDE={row['surface_fde_mm']:.4f} mm, "
                  f"Y=({row['r_rmse_cm']:.5f}, {row['u_rmse_cm']:.5f}) cm, "
                  f"F1={row['contact_f1']:.4f}")
        mean = {field: float(np.mean([row[field] for row in rows])) for field in fields}
        print("  均值: " + ", ".join(f"{key}={value:.5f}" for key, value in mean.items()))


if __name__ == "__main__":
    main()
