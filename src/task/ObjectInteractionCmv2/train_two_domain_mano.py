"""GPU0-bounded GRAB+ARCTIC MANO continuation with domain-balanced validation."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from .config import load_two_domain_mano_config
from .model import ObjectInteractionCmv2V13Model, object_interaction_v13_loss
from .multi_domain import ThreeDomainTransitions, build_source_sampler, collate_three_domain, sha256_file
from .train_multi_domain import utc_now


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _dataset(cfg: Mapping[str, Any], split: str) -> ThreeDomainTransitions:
    data = cfg["data"]
    return ThreeDomainTransitions(
        cfg["sources"], split, num_obj_points=int(data["num_obj_points"]),
        train_stride_values=data["train_stride_values"], fixed_stride=(int(data["eval_stride"]) if split == "val" else None),
        active_only=bool(data["active_only"]), base_seed=int(cfg["training"]["seed"]),
    )


def _atomic_checkpoint(path: Path, *, model, optimizer, step: int, epoch: int,
                       cfg: Mapping[str, Any], initial_checkpoint: Path, best_metric: float | None,
                       metadata: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "architecture_version": model.architecture_version,
        "modification_version": cfg["modification_version"], "step": int(step), "epoch": int(epoch),
        "seed": int(cfg["training"]["seed"]), "initial_checkpoint": str(initial_checkpoint),
        "best_metric": best_metric, "metadata": dict(metadata),
    }, temporary)
    temporary.replace(path)


@torch.inference_mode()
def _evaluate(model, dataset: ThreeDomainTransitions, cfg: Mapping[str, Any], device: torch.device) -> dict[str, float]:
    result: dict[str, float] = {}
    model.eval()
    for domain in ("grab", "arctic"):
        indices = [index for index, (sequence_index, _) in enumerate(dataset.rows)
                   if dataset.sequences[sequence_index].domain == domain]
        if not indices:
            raise ValueError(f"validation has no {domain} transitions")
        loader = DataLoader(Subset(dataset, indices), batch_size=int(cfg["training"]["validation_batch_size"]),
                            shuffle=False, num_workers=0, collate_fn=collate_three_domain)
        total, count = 0.0, 0
        for raw in loader:
            batch = _move(raw, device)
            loss = object_interaction_v13_loss(model(batch), batch)["total"]
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite {domain} validation loss")
            size = int(batch["obj_points"].shape[0])
            total += float(loss.detach().cpu()) * size
            count += size
        result[f"val_{domain}_loss"] = total / count
    result["selection_metric"] = 0.5 * (result["val_grab_loss"] + result["val_arctic_loss"])
    return result


def run(cfg: dict[str, Any], run_id: str, initial_checkpoint: Path) -> dict[str, Any]:
    initial_checkpoint = initial_checkpoint.resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)
    device = torch.device(cfg["training"]["device"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved two-domain run")
    torch.cuda.set_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    minimum = int(cfg["training"]["minimum_free_memory_gib"]) * 2**30
    if free_bytes < minimum:
        raise RuntimeError(f"GPU0 free memory {free_bytes / 2**30:.1f} GiB is below approved minimum")
    torch.manual_seed(int(cfg["training"]["seed"])); torch.cuda.manual_seed_all(int(cfg["training"]["seed"]))
    output = Path(cfg["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", cfg)
    source_records = [{"name": item["name"], "index": item["index"], "manifest": item["manifest"],
                       "index_sha256": sha256_file(item["index"]), "manifest_sha256": sha256_file(item["manifest"])}
                      for item in cfg["sources"]]
    manifest: dict[str, Any] = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCmv2",
        "operation": "v1_4_4_two_domain_mano_weight_initialization_train", "run_id": run_id,
        "run_status": "STARTED", "created_at": utc_now(), "modification_version": cfg["modification_version"],
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": "config.json", "architecture_version": ObjectInteractionCmv2V13Model.architecture_version,
        "initial_checkpoint": str(initial_checkpoint), "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "input": {"sources": source_records}, "coordinate_frame": "object_pose_t",
        "selection_metric": "mean(val_grab_loss,val_arctic_loss)",
        "outputs": {"metrics": "metrics.jsonl", "train_log": "train.log", "latest_checkpoint": "latest.pt",
                    "best_checkpoint": "best.pt"}, "conclusion": "INCONCLUSIVE",
    }
    _write_json(output / "run_manifest.json", manifest)
    metrics_path, log_path = output / "metrics.jsonl", output / "train.log"
    try:
        train = _dataset(cfg, "train")
        validation = _dataset(cfg, "val")
        train_rows, val_rows = train.rows_by_source(), validation.rows_by_source()
        if any(train_rows.get(domain, 0) <= 0 or val_rows.get(domain, 0) <= 0 for domain in ("grab", "arctic")):
            raise ValueError(f"two-domain split lacks transitions: train={train_rows}, val={val_rows}")
        sampler = build_source_sampler(train, cfg["data"]["train_source_probabilities"], int(cfg["training"]["seed"]))
        loader = DataLoader(train, batch_size=int(cfg["training"]["batch_size"]), sampler=sampler,
                            num_workers=0, collate_fn=collate_three_domain)
        model = ObjectInteractionCmv2V13Model(SimpleNamespace(**cfg["model"])).to(device)
        initial = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        if initial.get("architecture_version") != model.architecture_version:
            raise ValueError("initial checkpoint architecture mismatch")
        model.load_state_dict(initial["model"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
        manifest.update({"run_status": "RUNNING", "train_rows_by_source": train_rows,
                         "val_rows_by_source": val_rows, "train_rows": len(train), "val_rows": len(validation),
                         "batch_size": int(cfg["training"]["batch_size"]),
                         "max_epochs": int(cfg["training"]["max_epochs"]),
                         "max_duration_s": int(cfg["training"]["max_duration_s"])})
        _write_json(output / "run_manifest.json", manifest)
        started, step, best_metric, stopped = time.perf_counter(), 0, None, False
        source_counts: Counter[str] = Counter()
        with metrics_path.open("w", encoding="utf-8") as metrics, log_path.open("w", encoding="utf-8") as log:
            for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
                model.train(); sampler.set_epoch(epoch)
                for raw in loader:
                    batch = _move(raw, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch)
                    losses = object_interaction_v13_loss(prediction, batch)
                    if not torch.isfinite(losses["total"]):
                        raise ValueError(f"non-finite loss at step {step + 1}")
                    losses["total"].backward(); optimizer.step(); step += 1
                    source_counts.update(str(value) for value in raw["source"])
                    if step % int(cfg["training"]["checkpoint_interval"]) == 0:
                        _atomic_checkpoint(output / "latest.pt", model=model, optimizer=optimizer, step=step,
                                           epoch=epoch - 1, cfg=cfg, initial_checkpoint=initial_checkpoint,
                                           best_metric=best_metric, metadata={"source_counts": dict(source_counts)})
                    if time.perf_counter() - started >= int(cfg["training"]["max_duration_s"]):
                        stopped = True
                        break
                _atomic_checkpoint(output / "latest.pt", model=model, optimizer=optimizer, step=step, epoch=epoch,
                                   cfg=cfg, initial_checkpoint=initial_checkpoint, best_metric=best_metric,
                                   metadata={"source_counts": dict(source_counts), "epoch_complete": not stopped})
                if not stopped:
                    values = _evaluate(model, validation, cfg, device)
                    improved = best_metric is None or values["selection_metric"] < best_metric
                    if improved:
                        best_metric = values["selection_metric"]
                        _atomic_checkpoint(output / "best.pt", model=model, optimizer=optimizer, step=step, epoch=epoch,
                                           cfg=cfg, initial_checkpoint=initial_checkpoint, best_metric=best_metric,
                                           metadata={"validation": values, "source_counts": dict(source_counts)})
                    record = {"epoch": epoch, "step": step, **values, "best_metric": best_metric,
                              "improved": improved, "source_counts": dict(source_counts),
                              "elapsed_seconds": time.perf_counter() - started}
                    line = json.dumps(record, ensure_ascii=False)
                    metrics.write(line + "\n"); metrics.flush(); log.write(line + "\n"); log.flush()
                if stopped:
                    break
        manifest.update({"run_status": "STOPPED" if stopped else "COMPLETED", "finished_at": utc_now(),
                         "stop_reason": "max_duration_s" if stopped else None, "last_step": step,
                         "last_epoch": epoch, "latest_checkpoint": "latest.pt", "best_checkpoint": "best.pt" if best_metric is not None else None,
                         "best_metric": best_metric, "source_counts": dict(source_counts), "conclusion": "INCONCLUSIVE"})
        _write_json(output / "run_manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update({"run_status": "FAILED", "finished_at": utc_now(), "error": repr(error),
                         "conclusion": "INVALID_IMPLEMENTATION"})
        _write_json(output / "run_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    run(load_two_domain_mano_config(args.config), args.run_id, args.initial_checkpoint)


if __name__ == "__main__":
    main()
