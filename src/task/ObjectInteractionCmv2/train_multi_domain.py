"""V1.4 single-process smoke runner for Cmv2 three-domain training."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .config import load_three_domain_config
from .model import ObjectInteractionCmv2V13Model, object_interaction_v13_loss
from .multi_domain import ThreeDomainTransitions, build_source_sampler, collate_three_domain, sha256_file


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _save_checkpoint(path: Path, model, optimizer, step: int, epoch: int, cfg: dict, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "architecture_version": model.architecture_version,
        "modification_version": cfg["modification_version"],
        "step": step,
        "epoch": epoch,
        "seed": cfg["training"]["seed"],
        "metadata": metadata,
    }, temporary)
    temporary.replace(path)


def _build_dataset(cfg: dict, split: str, *, fixed_stride: int | None = None) -> ThreeDomainTransitions:
    data = cfg["data"]
    return ThreeDomainTransitions(
        cfg["sources"], split,
        num_obj_points=int(data["num_obj_points"]),
        train_stride_values=data["source_stride_values"],
        fixed_stride=fixed_stride,
        active_only=bool(data["active_only"]),
        base_seed=int(cfg["training"]["seed"]),
        max_sequences_per_domain=(
            int(cfg["training"]["max_sequences_per_domain"])
            if cfg["training"].get("max_sequences_per_domain") is not None and split == "train"
            else None
        ),
    )


def run(cfg: dict, run_id: str) -> dict:
    torch.manual_seed(int(cfg["training"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["training"]["seed"]))
    device = torch.device(str(cfg["training"]["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V1.4 requested CUDA but CUDA is unavailable")
        torch.cuda.set_device(device)
    output = Path(cfg["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    input_records = [
        {"name": item["name"], "hand_variant": item["hand_variant"], "index": item["index"],
         "index_sha256": sha256_file(item["index"]), "manifest": item["manifest"],
         "manifest_sha256": sha256_file(item["manifest"])}
        for item in cfg["sources"]
    ]
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCmv2",
        "operation": "v1_4_three_domain_smoke",
        "run_id": run_id,
        "run_status": "STARTED",
        "created_at": utc_now(),
        "modification_version": cfg["modification_version"],
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": "config.json",
        "architecture_version": ObjectInteractionCmv2V13Model.architecture_version,
        "input": {"sources": input_records},
        "coordinate_frame": "object_pose_t",
        "source_probabilities": cfg["data"]["source_probabilities"],
        "source_hand_variants": {item["name"]: item["hand_variant"] for item in cfg["sources"]},
        "source_stride_values": cfg["data"]["source_stride_values"],
        "outputs": {"metrics": "metrics.jsonl", "train_log": "train.log", "latest_checkpoint": "latest.pt"},
        "conclusion": "INCONCLUSIVE",
    }
    write_json(output / "run_manifest.json", manifest)
    metrics_path = output / "metrics.jsonl"
    log_path = output / "train.log"
    try:
        dataset = _build_dataset(cfg, "train")
        source_rows = dataset.rows_by_source()
        if any(source_rows.get(domain, 0) <= 0 for domain in cfg["data"]["source_domains"]):
            raise ValueError(f"V1.4 train source has no rows: {source_rows}")
        sampler = build_source_sampler(dataset, cfg["data"]["source_probabilities"], int(cfg["training"]["seed"]))
        loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), sampler=sampler,
                            shuffle=False, num_workers=0, drop_last=False,
                            collate_fn=collate_three_domain)
        model = ObjectInteractionCmv2V13Model(SimpleNamespace(**cfg["model"])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
        manifest.update({
            "run_status": "RUNNING", "train_rows": len(dataset), "train_rows_by_source": source_rows,
            "dropped_transitions": dataset.dropped_transitions,
            "train_rows_by_variant": dataset.rows_by_variant(),
            "batch_size": int(cfg["training"]["batch_size"]), "max_steps": int(cfg["training"]["max_steps"]),
        })
        write_json(output / "run_manifest.json", manifest)
        step, epoch, source_counts = 0, 0, Counter()
        started = time.perf_counter()
        with metrics_path.open("w", encoding="utf-8") as metrics, log_path.open("w", encoding="utf-8") as log:
            while step < int(cfg["training"]["max_steps"]):
                sampler.set_epoch(epoch)
                for raw_batch in loader:
                    if step >= int(cfg["training"]["max_steps"]):
                        break
                    batch = _move_batch(raw_batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch)
                    losses = object_interaction_v13_loss(prediction, batch)
                    if not torch.isfinite(losses["total"]):
                        raise ValueError(f"non-finite loss at step {step + 1}")
                    losses["total"].backward()
                    optimizer.step()
                    step += 1
                    values = raw_batch["source"]
                    source_counts.update(str(value) for value in values)
                    record = {
                        "step": step, "epoch": epoch,
                        "loss": float(losses["total"].detach().cpu()),
                        "translation_loss": float(losses["translation"].detach().cpu()),
                        "rotation_loss": float(losses["rotation"].detach().cpu()),
                        "flow_loss": float(losses["flow"].detach().cpu()),
                        "flow_epe_mm": float(torch.linalg.vector_norm(
                            prediction["obj_flow_pred"].detach() - batch["obj_flow_gt"], dim=-1).mean().cpu() * 1000),
                        "source_counts": dict(source_counts),
                        "step_seconds": time.perf_counter() - started,
                    }
                    line = json.dumps(record, ensure_ascii=False)
                    metrics.write(line + "\n"); metrics.flush()
                    log.write(line + "\n"); log.flush()
                    if step % int(cfg["training"]["checkpoint_interval"]) == 0 or step == int(cfg["training"]["max_steps"]):
                        _save_checkpoint(output / "latest.pt", model, optimizer, step, epoch, cfg,
                                         {"train_rows_by_source": source_rows, "source_counts": dict(source_counts)})
                epoch += 1
        manifest.update({
            "run_status": "COMPLETED", "finished_at": utc_now(), "last_step": step,
            "last_epoch": epoch, "source_counts": dict(source_counts), "latest_checkpoint": "latest.pt",
            "best_metric": None, "conclusion": "SUPPORTED",
        })
        write_json(output / "run_manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update({"run_status": "FAILED", "finished_at": utc_now(), "error": repr(error),
                         "conclusion": "INVALID_IMPLEMENTATION"})
        write_json(output / "run_manifest.json", manifest)
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    cfg = load_three_domain_config(args.config)
    run(cfg, args.run_id)


if __name__ == "__main__":
    main()
