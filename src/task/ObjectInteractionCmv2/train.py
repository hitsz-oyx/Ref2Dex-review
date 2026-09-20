"""Short V1.0 pilot training entry for synthetic or read-only OakInk2 data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .model import ObjectInteractionCmv2Model, object_interaction_loss
from .oakink2 import OakInk2RigidDataset
from .synthetic import make_synthetic_batch


def main():
    if "--config" in sys.argv:
        from .train_grab import main as grab_main
        return grab_main()
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--max-files", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.data_root:
        dataset = OakInk2RigidDataset(args.data_root, max_files=args.max_files, max_frames=args.max_frames)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        source = {"kind": "oakink2_stage3_readonly", "root": str(args.data_root.resolve()), "files": args.max_files, "frames_per_file": args.max_frames}
    else:
        loader = [make_synthetic_batch(batch_size=2)] * args.max_steps
        source = {"kind": "synthetic", "seed": args.seed}
    model = ObjectInteractionCmv2Model(SimpleNamespace(interaction_mode="static"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = []
    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            if step >= args.max_steps:
                break
            batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad(set_to_none=True)
            losses = object_interaction_loss(model(batch), batch)
            losses["total"].backward()
            optimizer.step()
            metrics.append({"step": step, **{k: float(v.detach()) for k, v in losses.items()}})
            step += 1
        if step >= args.max_steps:
            break
    torch.save({"model": model.state_dict(), "seed": args.seed}, args.output / "latest.pt")
    (args.output / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in metrics))
    manifest = {"task": "ObjectInteractionCmv2", "work_version": "V1.0.2", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed,
                "source": source, "epochs": args.epochs, "last_step": step, "checkpoint": "latest.pt",
                "metrics": "metrics.jsonl", "conclusion": "INCONCLUSIVE"}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
