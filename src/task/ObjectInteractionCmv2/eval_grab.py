"""V1.2 GRAB/MANO fixed-stride evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from .config import load_grab_config
from .grab import GrabManoTransitions, sha256_file
from .model import ObjectInteractionCmv2Model
from .train_grab import utc_now, write_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    cfg = load_grab_config(args.config)
    if cfg["training"].get("mode") == "smoke" and (args.max_sequences is None or not 1 <= args.max_sequences <= 3 or
       args.max_pairs is None or not 1 <= args.max_pairs <= 8):
        raise ValueError("Smoke evaluation must be bounded to three sequences and eight pairs")
    output = Path(cfg["output_root"]) / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    manifest = {"task": "ObjectInteractionCmv2", "modification_version": "V1.2.1",
                "run_id": args.run_id, "run_status": "STARTED", "created_at": utc_now(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "config": "config.json", "split": args.split, "stride": 1,
                "input": {"index": cfg["source"]["index"], "manifest": cfg["source"]["manifest"]},
                "initial_checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.checkpoint), "outputs": {"metrics": "metrics.jsonl"},
                "conclusion": "INCONCLUSIVE"}
    write_json(output / "run_manifest.json", manifest)
    dataset = GrabManoTransitions(cfg["source"]["index"], cfg["source"]["manifest"], args.split,
                                  max_sequences=args.max_sequences)
    device = torch.device(cfg["training"].get("device", "cpu"))
    model = ObjectInteractionCmv2Model(SimpleNamespace(**cfg["model"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    with torch.no_grad():
        for idx in range(min(len(dataset), args.max_pairs or len(dataset))):
            sample = dataset[idx]
            batch = {key: value[None].to(device) for key, value in sample.items() if torch.is_tensor(value)}
            pred = model(batch)["obj_flow_pred"][0]
            target = batch["obj_flow_gt"][0]
            error = torch.linalg.vector_norm(pred - target, dim=-1)
            zero = torch.linalg.vector_norm(target, dim=-1)
            row = stats[sample["sequence_id"]]
            row[0] += error.numel()
            row[1] += float(error.sum())
            row[2] += float(error.square().sum())
            row[3] += float(zero.sum())
    if not stats:
        raise ValueError("No evaluation pairs")
    total = [sum(row[i] for row in stats.values()) for i in range(4)]
    result = {"dataset": "grab", "stride": 1, "split": args.split, "sequences": len(stats),
              "points": total[0], "flow_epe_micro_mm": 1000 * total[1] / total[0],
              "flow_rmse_micro_mm": 1000 * (total[2] / total[0]) ** 0.5,
              "zero_flow_epe_micro_mm": 1000 * total[3] / total[0],
              "flow_epe_macro_mm": 1000 * sum(row[1] / row[0] for row in stats.values()) / len(stats),
              "conclusion": "INCONCLUSIVE"}
    result["zero_flow_improvement_micro_mm"] = result["zero_flow_epe_micro_mm"] - result["flow_epe_micro_mm"]
    (output / "metrics.jsonl").write_text(json.dumps(result) + "\n")
    manifest.update(run_status="COMPLETED", finished_at=utc_now(), evaluated_pairs=min(len(dataset), args.max_pairs or len(dataset)),
                    last_step=None, last_epoch=None, best_metric=None)
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
