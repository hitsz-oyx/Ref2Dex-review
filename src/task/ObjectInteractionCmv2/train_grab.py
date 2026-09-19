"""V1.2 GRAB/MANO bounded training entry."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from .config import load_grab_config
from .grab import GrabManoTransitions, sha256_file
from .model import (ObjectInteractionCmv2Model, ObjectInteractionCmv2V13Model,
                    object_interaction_loss, object_interaction_v13_loss)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    cfg = load_grab_config(args.config)
    v13 = cfg["schema_name"].endswith("v1_3")
    architecture = ObjectInteractionCmv2V13Model.architecture_version if v13 else "v1_2"
    train_cfg = cfg["training"]
    if train_cfg.get("mode") == "smoke" and (not 1 <= train_cfg["max_sequences"] <= 3 or
       not 1 <= train_cfg["max_steps"] <= 8 or not 1 <= train_cfg["batch_size"] <= 2):
        raise ValueError("Smoke budget exceeds approved bound")
    output = Path(cfg["output_root"]) / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"task": "ObjectInteractionCmv2", "work_version": cfg.get("work_version", "V1.3.2") if v13 else "V1.2.1",
                "run_id": args.run_id, "run_status": "STARTED", "created_at": utc_now(),
                "architecture_version": architecture,
                "base_commit": commit, "config": "config.json", "seed": train_cfg["seed"],
                "input": {"index": cfg["source"]["index"], "index_sha256": sha256_file(Path(cfg["source"]["index"])),
                          "manifest": cfg["source"]["manifest"], "manifest_sha256": sha256_file(Path(cfg["source"]["manifest"]))},
                "input_schema": "ref2dex_object_interaction_cm_bilateral_mano_v1_4",
                "coordinate_frame": "object_pose_t", "hand_points_per_side": 2048,
                "initial_checkpoint": str(args.resume.resolve()) if args.resume else None,
                "outputs": {"metrics": "metrics.jsonl", "train_log": "train.log", "latest_checkpoint": "latest.pt"},
                "conclusion": "INCONCLUSIVE"}
    write_json(output / "run_manifest.json", manifest)
    log_path = output / "train.log"
    try:
        run_started = time.perf_counter()
        torch.manual_seed(train_cfg["seed"])
        device = torch.device(train_cfg.get("device", "cpu"))
        dataset = GrabManoTransitions(cfg["source"]["index"], cfg["source"]["manifest"],
                                      "train", max_sequences=train_cfg["max_sequences"],
                                      direct_pose_gt=v13)
        if train_cfg.get("mode") == "formal" and (
            len(dataset.sequences) != train_cfg["expected_train_sequences"]
            or len(dataset) != train_cfg["expected_train_pairs"]
            or dataset.dropped_pairs != 0):
            raise ValueError("Full GRAB train split differs from approved contract")
        model_class = ObjectInteractionCmv2V13Model if v13 else ObjectInteractionCmv2Model
        model = model_class(SimpleNamespace(**cfg["model"])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
        step, epoch, position = 0, 0, 0
        if args.resume:
            saved = torch.load(args.resume, map_location=device)
            if saved.get("architecture_version", "v1_2") != architecture:
                raise ValueError("Resume architecture differs")
            if saved["index_sha256"] != manifest["input"]["index_sha256"] or saved["seed"] != train_cfg["seed"]:
                raise ValueError("Resume input or seed differs")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            step, epoch, position = saved["step"], saved["epoch"], saved["position"]
            torch.set_rng_state(saved["rng_cpu"].cpu())
            if device.type == "cuda" and saved.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(saved["rng_cuda"].cpu(), device)
        manifest.update(run_status="RUNNING", valid_pairs=len(dataset), dropped_pairs=dataset.dropped_pairs)
        write_json(output / "run_manifest.json", manifest)
        with log_path.open("w") as log, (output / "metrics.jsonl").open("w") as metrics:
            stopped_for_time = False
            while step < train_cfg["max_steps"]:
                order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(train_cfg["seed"] + epoch))
                if position >= len(order):
                    epoch += 1
                    position = 0
                    continue
                selected = order[position:position + train_cfg["batch_size"]].tolist()
                samples = [dataset[i] for i in selected]
                batch = {key: torch.stack([sample[key] for sample in samples]).to(device)
                         for key in samples[0] if torch.is_tensor(samples[0][key])}
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                output_value = model(batch)
                losses = (object_interaction_v13_loss(output_value, batch) if v13
                          else object_interaction_loss(output_value, batch))
                if not torch.isfinite(losses["total"]):
                    raise ValueError("Nonfinite loss")
                losses["total"].backward()
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step += 1
                position += len(selected)
                epe = torch.linalg.vector_norm(output_value["obj_flow_pred"].detach() - batch["obj_flow_gt"], dim=-1).mean()
                record = {"step": step, "epoch": epoch, "loss": float(losses["total"].detach()),
                          "flow_epe_mm": float(epe * 1000), "step_seconds": time.perf_counter() - start,
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                if v13:
                    record.update({key + "_loss": float(losses[key].detach())
                                   for key in ("translation", "rotation", "flow")})
                    mass = output_value["token_mass"].detach()
                    anchors = output_value["token_anchors"].detach()
                    mask = output_value["token_mask"].detach()
                    pairs = mask[:, :, None] & mask[:, None, :]
                    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
                    pairs = pairs & ~eye[None]
                    distances = torch.cdist(anchors, anchors)
                    record.update(
                        active_object_points=float(output_value["contact_active"].sum(1).float().mean()),
                        token_mass_min=float(mass.min()),
                        token_mass_max=float(mass.max()),
                        token_anchor_spread_m=float(distances[pairs].mean()) if pairs.any() else 0.0)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                log.write(json.dumps(record) + "\n")
                log.flush()
                saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "architecture_version": architecture,
                         "step": step, "epoch": epoch, "position": position, "seed": train_cfg["seed"],
                         "index_sha256": manifest["input"]["index_sha256"],
                         "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
                time_exceeded = (train_cfg.get("max_duration_s") is not None
                                 and time.perf_counter() - run_started >= train_cfg["max_duration_s"])
                if (step % train_cfg.get("checkpoint_interval", 1) == 0 or
                    step == train_cfg["max_steps"] or time_exceeded):
                    torch.save(saved, output / "latest.pt.tmp")
                    (output / "latest.pt.tmp").replace(output / "latest.pt")
                if step == train_cfg.get("checkpoint_step"):
                    torch.save(saved, output / f"step_{step}.pt")
                if time_exceeded and step < train_cfg["max_steps"]:
                    stopped_for_time = True
                    break
        manifest.update(run_status="STOPPED" if stopped_for_time else "COMPLETED",
                        stop_reason="max_duration_s" if stopped_for_time else None,
                        finished_at=utc_now(), last_step=step, last_epoch=epoch,
                        latest_checkpoint="latest.pt", best_metric=None, conclusion="INCONCLUSIVE")
        write_json(output / "run_manifest.json", manifest)
    except Exception as error:
        manifest.update(run_status="FAILED", finished_at=utc_now(), error=repr(error), last_step=locals().get("step", 0),
                        conclusion="INVALID_IMPLEMENTATION")
        write_json(output / "run_manifest.json", manifest)
        with log_path.open("a") as log:
            log.write(f"FAILED: {error!r}\n")
        raise
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
