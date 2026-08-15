from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    experiment = ROOT / "output/exp/pointworld_wam_v2_chunk"
    research = ROOT / "output/research/pointworld_wam/wam_v2"
    research.mkdir(parents=True, exist_ok=True)
    history = json.loads((experiment / "history.json").read_text(encoding="utf-8"))
    rows = [row for row in history if "evaluation" in row]
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        steps,
        [row["evaluation"]["inverse_hand_mm"] for row in rows],
        marker="o",
        label="generated",
    )
    axes[0].plot(
        steps,
        [row["evaluation"]["identity_hand_mm"] for row in rows],
        linestyle="--",
        label="identity",
    )
    axes[0].set_title("Inverse chunk")
    axes[0].set_ylabel("hand surface error (mm)")
    axes[1].plot(
        steps,
        [row["evaluation"]["forward_world_mm"] for row in rows],
        marker="o",
        label="clean action",
    )
    axes[1].plot(
        steps,
        [row["evaluation"]["zero_action_world_mm"] for row in rows],
        linestyle="--",
        label="zero action",
    )
    axes[1].plot(
        steps,
        [row["evaluation"]["zero_flow_world_mm"] for row in rows],
        linestyle=":",
        label="zero flow",
    )
    axes[1].set_title("Forward chunk")
    axes[1].set_ylabel("object flow error (mm)")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(research / "training_gates.png", dpi=180)


if __name__ == "__main__":
    main()
