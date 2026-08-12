"""四卡扫描全 GRAB stable-grasp event，并输出 Gate 0 审计与可视化。"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from src.task.InteractionDynamics.dataset_grasp_v18 import (
    all_hand_paths, detect_stable_grasp, sequence_interaction)


def event_row(event):
    return {"path": str(event.path), "grasp_frame": event.grasp_frame,
            "contact_count": event.contact_count, "stable_u_cm": event.stable_u_cm}


def visualize(row: dict, output: Path, device: torch.device) -> None:
    interaction = sequence_interaction(row["path"], device=device)
    frame = row["grasp_frame"]
    # 上图直接显示 detector 的 object-frame r anchors；下图显示关键统计时序。
    geometry = interaction["relative_geometry"][frame].cpu().numpy()
    anchors = interaction["anchors"].cpu().numpy()
    contact = interaction["contact_count"].cpu().numpy()
    velocity = interaction["u_rms_cm"].cpu().numpy()
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], s=5, c="gray")
    near = interaction["relative_distance"][frame].cpu().numpy() < .02
    ax.quiver(anchors[near, 0], anchors[near, 1], anchors[near, 2],
              geometry[near, 0], geometry[near, 1], geometry[near, 2],
              length=1., normalize=False, color="red")
    ax.set_title(f"{Path(row['path']).parent.name}/{Path(row['path']).stem}  t_g={frame}")
    bx = fig.add_subplot(122)
    bx.plot(contact, label="contact anchors (<2cm)")
    bx.plot(np.arange(len(velocity)), velocity, label="u RMS (cm/frame)")
    bx.axvline(frame, color="red", linestyle="--"); bx.axhline(4, color="gray", linestyle=":")
    bx.axhline(.3, color="gray", linestyle=":"); bx.legend(); bx.set_xlabel("frame")
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/processed_data/stage4/data/grab")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualizations", type=int, default=20)
    args = parser.parse_args()
    rank, world, local = (int(os.getenv(name, default)) for name, default in
                          (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if world > 1:
        torch.cuda.set_device(local); dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    paths = all_hand_paths(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(rank, len(paths), world):
        event = detect_stable_grasp(paths[index], device=device)
        if event is not None: rows.append(event_row(event))
        if (index - rank) // world % 50 == 0:
            print(json.dumps({"rank": rank, "scanned": index + 1,
                              "local_events": len(rows)}), flush=True)
    (args.output / f"events_rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
    if world > 1: dist.barrier()
    if rank == 0:
        events = []
        for worker in range(world):
            events += json.loads((args.output / f"events_rank{worker}.json").read_text())
        events.sort(key=lambda row: (row["path"], row["grasp_frame"]))
        (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        demonstrations = {str(path.parent) for path in paths}
        detected = {str(Path(row["path"]).parent) for row in events}
        summary = {"total_hand_files": len(paths), "total_demonstrations": len(demonstrations),
                   "events": len(events), "detected_demonstrations": len(detected),
                   "left_events": sum(Path(row["path"]).stem == "left" for row in events),
                   "right_events": sum(Path(row["path"]).stem == "right" for row in events),
                   "grasp_frame": {}, "contact_count": {}, "stable_u_cm": {}}
        for key in ("grasp_frame", "contact_count", "stable_u_cm"):
            values = np.asarray([row[key] for row in events], np.float32)
            summary[key] = ({name: float(value) for name, value in zip(
                ("min", "median", "mean", "p90", "max"),
                (values.min(), np.median(values), values.mean(), np.quantile(values, .9), values.max()))}
                if len(values) else {})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        selected = random.Random(42).sample(events, min(args.visualizations, len(events)))
        for index, row in enumerate(selected):
            visualize(row, args.output / f"event_{index:02d}.png", device)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
