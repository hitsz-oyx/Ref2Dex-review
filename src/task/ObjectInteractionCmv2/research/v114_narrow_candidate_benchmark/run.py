"""Reproducible random-weight latency benchmark for the V1.14 candidate interface."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from src.task.ObjectInteractionCmv2.part_se3 import endpoint_union_topk
from src.task.ObjectInteractionCmv2.part_se3_v114 import PartSE3ObjectInteractionCmv2V114Model


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
WIDTHS = (128, 64, 32)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cuda_times(function, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    return {
        "median_ms": statistics.median(values),
        "p90_ms": _percentile(values, 0.9),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "iterations": iterations,
    }


def _inputs(device: torch.device, seed: int, hand_points: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    batch, candidates, objects, edges, parts = 1, 8, 1024, 32, 3
    points = torch.randn((batch, objects, 3), generator=generator, device=device) * 0.04
    normals = torch.nn.functional.normalize(
        torch.randn(points.shape, generator=generator, device=device), dim=-1)
    part_ids = (torch.arange(objects, device=device) % parts)[None]
    object_batch = {
        "obj_points": points,
        "obj_normals": normals,
        "obj_part_id": part_ids,
        "part_valid_mask": torch.ones((batch, parts), dtype=torch.bool, device=device),
    }
    edge_points = points[:, None, :, None] + torch.randn(
        (batch, candidates, objects, edges, 3), generator=generator, device=device) * 0.005
    candidate = {
        "edge_hand_points": edge_points,
        "edge_hand_normals": torch.nn.functional.normalize(torch.randn(
            edge_points.shape, generator=generator, device=device), dim=-1),
        "edge_hand_flow": torch.randn(
            edge_points.shape, generator=generator, device=device) * 0.001,
        "edge_source_id": torch.arange(edges, device=device, dtype=torch.int16)[
            None, None, None].expand(batch, candidates, objects, edges),
        "edge_valid_mask": torch.ones(
            (batch, candidates, objects, edges), dtype=torch.bool, device=device),
        "delta_time_s": torch.full((batch, candidates), 1 / 30, device=device),
    }
    reference_hand = torch.randn(
        (batch, candidates, hand_points, 3), generator=generator, device=device) * 0.08
    reference_flow = torch.randn(
        reference_hand.shape, generator=generator, device=device) * 0.002
    return object_batch, candidate, reference_hand, reference_flow


def _model(width: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = SimpleNamespace(
        hidden_width=128,
        interaction_dim=width,
        num_tokens=16,
        knn_k=32,
        interaction_radius_m=0.02,
        feature_scale_m=0.02,
        frame_dt_s=1 / 30,
    )
    return PartSE3ObjectInteractionCmv2V114Model(config).requires_grad_(False).eval().to(device)


def run(args) -> Path:
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be one unique directory name")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise ValueError("V1.14 performance benchmark requires CUDA")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise ValueError("tracked worktree must be clean before benchmark")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        raise ValueError("index must be clean before benchmark")

    output = HERE / "output" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    config = {
        "task": "ObjectInteractionCmv2",
        "work_version": "V1.14",
        "git_commit": commit,
        "branch": branch,
        "seed": args.seed,
        "device": args.device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "float32",
        "tf32": False,
        "batch": 1,
        "candidates": 8,
        "object_points": 1024,
        "edges": 32,
        "parts": 3,
        "endpoint_hand_points": args.hand_points,
        "interaction_dims": list(WIDTHS),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "endpoint_warmup": args.endpoint_warmup,
        "endpoint_iterations": args.endpoint_iterations,
        "initial_checkpoint": None,
        "initialization": "random",
        "coordinates": "synthetic object-frame metres; timing only",
        "split": None,
    }
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCmv2",
        "work_version": "V1.14",
        "git_commit": commit,
        "run_id": args.run_id,
        "run_status": "RUNNING",
        "operation": "random_weight_candidate_latency_benchmark",
        "created_at": started,
        "config": "config.json",
        "metrics": "metrics.jsonl",
        "log": "run.log",
        "initial_checkpoint": None,
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(output / "config.json", config)
    _write_json(output / "run_manifest.json", manifest)

    def log(message: str) -> None:
        line = datetime.now().astimezone().isoformat(timespec="seconds") + " " + message
        print(line, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        log(f"START run_id={args.run_id} commit={commit} widths={WIDTHS}")
        object_batch, candidate, reference_hand, reference_flow = _inputs(
            device, args.seed, args.hand_points)
        results = {}
        with torch.inference_mode(), (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for width in WIDTHS:
                model = _model(width, device, args.seed)
                context = model.encode_object(object_batch)
                metrics = {
                    "interaction_dim": width,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "static_encode": _cuda_times(
                        lambda: model.encode_object(object_batch), args.warmup, args.iterations),
                    "local_interaction": _cuda_times(
                        lambda: model.local_interaction.forward_candidates(context, candidate),
                        args.warmup, args.iterations),
                    "candidate_forward": _cuda_times(
                        lambda: model.forward_candidates(context, candidate),
                        args.warmup, args.iterations),
                    "encode_plus_candidate": _cuda_times(
                        lambda: model.forward_candidates(model.encode_object(object_batch), candidate),
                        args.warmup, args.iterations),
                }
                torch.cuda.reset_peak_memory_stats(device)
                model.forward_candidates(context, candidate)
                torch.cuda.synchronize()
                metrics["candidate_peak_allocated_mib"] = (
                    torch.cuda.max_memory_allocated(device) / 2**20)
                results[str(width)] = metrics
                stream.write(json.dumps({"type": "width", **metrics}, allow_nan=False) + "\n")
                stream.flush()
                log(f"WIDTH {width} candidate_median_ms={metrics['candidate_forward']['median_ms']:.6f}")
                del model, context
                torch.cuda.empty_cache()

            flat_points = object_batch["obj_points"][:, None].expand(
                -1, 8, -1, -1).reshape(8, 1024, 3)
            flat_hand = reference_hand.reshape(8, args.hand_points, 3)
            flat_flow = reference_flow.reshape_as(flat_hand)
            valid = torch.ones((8, args.hand_points), dtype=torch.bool, device=device)
            endpoint = _cuda_times(
                lambda: endpoint_union_topk(
                    flat_points, flat_hand, flat_flow, valid, k=32,
                    object_chunk=128, hand_chunk=256),
                args.endpoint_warmup, args.endpoint_iterations)
            stream.write(json.dumps({"type": "endpoint", **endpoint}, allow_nan=False) + "\n")

        baseline = results["128"]
        narrow = results["32"]
        speedups = {
            "local_interaction": (baseline["local_interaction"]["median_ms"]
                                  / narrow["local_interaction"]["median_ms"]),
            "candidate_forward": (baseline["candidate_forward"]["median_ms"]
                                  / narrow["candidate_forward"]["median_ms"]),
            "encode_plus_candidate": (baseline["encode_plus_candidate"]["median_ms"]
                                      / narrow["encode_plus_candidate"]["median_ms"]),
        }
        estimated_e2e = {
            width: endpoint["median_ms"] + metrics["encode_plus_candidate"]["median_ms"]
            for width, metrics in results.items()
        }
        speedups["endpoint_plus_model"] = estimated_e2e["128"] / estimated_e2e["32"]
        gates = {
            "local_interaction_at_least_3x": speedups["local_interaction"] >= 3.0,
            "candidate_forward_at_least_2x": speedups["candidate_forward"] >= 2.0,
            "endpoint_plus_model_at_least_1_5x": speedups["endpoint_plus_model"] >= 1.5,
            "peak_memory_not_higher": (narrow["candidate_peak_allocated_mib"]
                                       <= baseline["candidate_peak_allocated_mib"]),
        }
        conclusion = "SUPPORTED" if all(gates.values()) else "REFUTED"
        summary = {
            "hypothesis": "32D satisfies every V1.14 random-weight performance gate versus 128D",
            "results": results,
            "endpoint": endpoint,
            "estimated_endpoint_plus_model_median_ms": estimated_e2e,
            "speedups_128_over_32": speedups,
            "gates": gates,
            "conclusion": conclusion,
            "limitations": [
                "synthetic random inputs and random model weights",
                "endpoint-plus-model is the sum of separately timed sequential stages",
                "no prediction-quality or training evidence",
                "single GPU model and driver state",
            ],
        }
        _write_json(output / "summary.json", summary)
        manifest.update(
            run_status="COMPLETED",
            conclusion=conclusion,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            summary="summary.json",
        )
        _write_json(output / "run_manifest.json", manifest)
        log(f"COMPLETED conclusion={conclusion} speedups={speedups}")
        return output
    except BaseException as error:
        manifest.update(
            run_status="FAILED",
            conclusion="INCONCLUSIVE",
            error=repr(error),
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        _write_json(output / "run_manifest.json", manifest)
        log(f"FAILED error={error!r}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--hand-points", type=int, default=4096)
    parser.add_argument("--endpoint-warmup", type=int, default=3)
    parser.add_argument("--endpoint-iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.hand_points,
           args.endpoint_warmup, args.endpoint_iterations) <= 0:
        raise ValueError("benchmark budgets must be positive")
    output = run(args)
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
