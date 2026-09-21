"""CUDA-event latency profiling for one frozen V1.18 planner state."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import torch


STAGES = (
    "candidate_generation", "state_geometry", "fk", "hand_surface",
    "geometry_encoder", "swept_topk", "edge_contact",
    "token_attention_effect", "scoring", "end_to_end",
)


class CudaEventProfiler:
    """Collect non-blocking CUDA events and resolve them after one iteration."""

    def __init__(self) -> None:
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def stage(self, name: str):
        if name not in STAGES:
            raise ValueError(f"unknown latency stage: {name}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    def resolve_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        values: dict[str, float] = {}
        for name, start, end in self.events:
            if name in values:
                raise RuntimeError(f"latency stage recorded more than once: {name}")
            values[name] = float(start.elapsed_time(end))
        missing = set(STAGES) - set(values)
        if missing:
            raise RuntimeError(f"missing latency stages: {sorted(missing)}")
        return values


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not samples:
        raise ValueError("latency samples must not be empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for name in STAGES:
            grouped[name].append(sample[name])
    result = {
        name: {
            "mean_ms": mean(values),
            "median_ms": median(values),
            "p90_ms": _percentile(values, 0.90),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for name, values in grouped.items()
    }
    total = result["end_to_end"]["mean_ms"]
    accounted = sum(result[name]["mean_ms"] for name in STAGES if name != "end_to_end")
    result["accounting"] = {
        "stages_mean_ms": accounted,
        "unattributed_mean_ms": total - accounted,
        "stages_percent_of_total": 100.0 * accounted / total,
    }
    for name in STAGES:
        result[name]["percent_of_total"] = 100.0 * result[name]["mean_ms"] / total
    return result


def _tensor_sha256(values: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _slice_state(kwargs: dict[str, object], index: int) -> dict[str, object]:
    per_env = {
        "mu", "current_native", "base_target", "current_links", "object_pose",
        "reference_transport", "desired_delta_xi",
    }
    return {
        name: value[index:index + 1] if name in per_env else value
        for name, value in kwargs.items()
    }


def _max_output_error(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, float]:
    if set(left) != set(right):
        raise RuntimeError("planner parity output keys differ")
    return {
        name: float((left[name] - right[name]).abs().max().item())
        for name in sorted(left)
    }


@torch.inference_mode()
def profile_first_active_state(planner, kwargs: dict[str, object], *, warmup: int, repeats: int) -> dict[str, object]:
    """Select the first active env and profile ordered K prefixes on that state."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    mu = kwargs["mu"]
    if not isinstance(mu, torch.Tensor) or mu.device.type != "cuda":
        raise ValueError("V1.18 latency profile requires CUDA planner inputs")
    object_points, _ = planner.geometry.object(kwargs["object_pose"])
    active = planner._activation(
        kwargs["current_links"], object_points, kwargs["reference_transport"], kwargs["desired_delta_xi"])
    active_ids = active.nonzero(as_tuple=False).flatten()
    if not len(active_ids):
        raise LookupError("planner call contains no active state")
    selected_index = int(active_ids[0].item())
    state = _slice_state(kwargs, selected_index)
    fixed_rng = torch.cuda.get_rng_state(mu.device)

    torch.cuda.set_rng_state(fixed_rng, mu.device)
    production = planner.teacher(**state)
    torch.cuda.set_rng_state(fixed_rng, mu.device)
    diagnostic = planner.teacher(**state, _diagnostic_candidate_count=8)
    parity = _max_output_error(production, diagnostic)
    if any(value != 0.0 for value in parity.values()):
        raise RuntimeError(f"K=8 diagnostic path changed planner output: {parity}")

    torch.cuda.set_rng_state(fixed_rng, mu.device)
    candidates = planner._candidates(state["mu"])
    candidate_sha256 = _tensor_sha256({"candidate_actions": candidates})
    results: dict[str, object] = {}
    for candidate_count in (1, 2, 4, 8):
        for _ in range(warmup):
            torch.cuda.set_rng_state(fixed_rng, mu.device)
            planner.teacher(**state, _diagnostic_candidate_count=candidate_count)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(mu.device)
        samples = []
        for _ in range(repeats):
            torch.cuda.set_rng_state(fixed_rng, mu.device)
            profiler = CudaEventProfiler()
            with profiler.stage("end_to_end"):
                planner.teacher(
                    **state, _latency_profiler=profiler,
                    _diagnostic_candidate_count=candidate_count,
                )
            samples.append(profiler.resolve_ms())
        results[str(candidate_count)] = {
            "latency": summarize(samples),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(mu.device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(mu.device),
        }

    return {
        "schema": "ref2dex.cmresidual.v118a_latency_profile.v1",
        "selected_env_index": selected_index,
        "source_batch_size": int(mu.shape[0]),
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "candidate_counts": [1, 2, 4, 8],
        "candidate_order": ["mu", "zero", "+eps0", "+eps1", "+eps2", "-eps0", "-eps1", "-eps2"],
        "state_sha256": _tensor_sha256(state),
        "candidate_actions_sha256": candidate_sha256,
        "k8_output_parity_max_abs": parity,
        "results": results,
    }


def write_profile(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
