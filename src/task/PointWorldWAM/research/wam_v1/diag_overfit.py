from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    experiment = ROOT / "output/exp/pointworld_wam_v1"
    research = ROOT / "output/research/pointworld_wam/wam_v1"
    history = json.loads((experiment / "history.json").read_text(encoding="utf-8"))
    summary = json.loads((research / "summary.json").read_text(encoding="utf-8"))
    evaluations = [
        (int(row["step"]), float(row["eval_fm_loss"]))
        for row in history
        if "eval_fm_loss" in row
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(*zip(*evaluations), marker="o")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("fixed-noise FM loss")
    axes[0].grid(alpha=0.3)
    generation = summary["generation"]
    labels = ("generated", "identity", "action mean", "random")
    for side, offset in (("left", -0.18), ("right", 0.18)):
        values = [
            generation[side]["hand_mm"],
            generation[side]["identity_hand_mm"],
            generation[side]["mean_action_hand_mm"],
            generation[side]["random_hand_mm"],
        ]
        axes[1].bar(
            [index + offset for index in range(len(labels))],
            values,
            width=0.36,
            label=side,
        )
    axes[1].set_xticks(range(len(labels)), labels, rotation=20)
    axes[1].set_ylabel("next-hand surface error (mm)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(research / "overfit.png", dpi=180)


if __name__ == "__main__":
    main()
