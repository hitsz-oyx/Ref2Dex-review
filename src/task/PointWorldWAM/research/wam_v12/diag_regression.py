from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    experiment = ROOT / "output/exp/pointworld_wam_v12_regression"
    output = ROOT / "output/research/pointworld_wam/wam_v12"
    output.mkdir(parents=True, exist_ok=True)
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
                label="regression",
            )
            axis.plot(
                steps,
                [row["evaluation"]["identity_hand_mm"] for row in rows],
                linestyle="--",
                label="identity",
            )
            axis.set_ylabel("hand surface error (mm)")
        else:
            axis.plot(
                steps,
                [row["evaluation"]["forward_world_mm"] for row in rows],
                marker="o",
                label="GT action",
            )
            axis.plot(
                steps,
                [row["evaluation"]["zero_action_world_mm"] for row in rows],
                linestyle="--",
                label="zero action",
            )
            axis.plot(
                steps,
                [row["evaluation"]["zero_flow_world_mm"] for row in rows],
                linestyle=":",
                label="zero flow",
            )
            axis.set_ylabel("object flow error (mm)")
        axis.axhline(30.0, color="black", alpha=0.5, label="30 mm gate")
        axis.set_title(f"{mode} deterministic")
        axis.set_xlabel("training step")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "regression_gates.png", dpi=180)


if __name__ == "__main__":
    main()
