from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    experiment = ROOT / "output/exp/pointworld_wam_v12_fm"
    result_dir = ROOT / "output/research/pointworld_wam/wam_v12"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, mode in zip(axes, ("inverse", "forward")):
        history = json.loads(
            (experiment / mode / "history.json").read_text(encoding="utf-8")
        )
        rows = [row for row in history if "evaluation" in row]
        steps = [row["step"] for row in rows]
        if mode == "inverse":
            axis.plot(
                steps,
                [row["evaluation"]["inverse_hand_mm"] for row in rows],
                marker="o",
                label="inverse FM (8 windows)",
            )
            axis.plot(
                steps,
                [row["evaluation"]["identity_hand_mm"] for row in rows],
                linestyle="--",
                label="identity (8 windows)",
            )
            axis.set_ylabel("hand surface error (mm)")
        else:
            axis.plot(
                steps,
                [row["evaluation"]["forward_world_mm"] for row in rows],
                marker="o",
                label="GT action (8 windows)",
            )
            axis.plot(
                steps,
                [row["evaluation"]["zero_action_world_mm"] for row in rows],
                linestyle="--",
                label="zero action (8 windows)",
            )
            axis.plot(
                steps,
                [row["evaluation"]["zero_flow_world_mm"] for row in rows],
                linestyle=":",
                label="zero flow (8 windows)",
            )
            axis.set_ylabel("object flow error (mm)")
        axis.set_title(f"{mode}-only FM")
        axis.set_xlabel("training step")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    result_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(result_dir / "fm_gates.png", dpi=180)


if __name__ == "__main__":
    main()
