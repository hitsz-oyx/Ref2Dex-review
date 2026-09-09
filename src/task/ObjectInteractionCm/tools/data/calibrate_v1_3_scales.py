#!/usr/bin/env python3
"""Calibrate V1.3 scales from train-only unique-KNN-hand samples.

The cache contains full-pool uint16 KNN indices. Calibration samples the same
1024 object points as training, recomputes the 32 neighbor distances, uses
valid edges for interaction statistics, and deduplicates valid hand IDs for
the hand decoder/loss statistic. It never rewrites geometry.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.task.ObjectInteractionCm.dataset import (
    _SequenceView,
    _resolve_index_entries,
    _stable_seed,
    _world_to_frame,
)


KNN_K = 32
RADIUS_M = 0.02
OBJECT_POOL_POINTS = 4096
MODEL_OBJECT_POINTS = 1024


def _frames(view: _SequenceView, stride: int, count: int, seed: int) -> np.ndarray:
    available = max(0, int(view.frame_count) - int(stride))
    if available == 0:
        return np.empty((0,), dtype=np.int64)
    if count <= 0 or available <= count:
        return np.arange(available, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(available, size=int(count), replace=False)).astype(np.int64)


def _mean_square_norm(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    if array.size == 0:
        return None
    return float(np.square(np.linalg.norm(array, axis=-1)).mean())


def _knn_stats(
    view: _SequenceView,
    side: str,
    frame: int,
    future: int,
    object_points: np.ndarray,
    *,
    object_indices: np.ndarray | None = None,
    radius_m: float,
) -> tuple[float | None, float | None, float | None]:
    current_world, _ = view.knn_hand(side, frame)
    future_world, _ = view.knn_hand(side, future)
    pose = np.asarray(view.array("pose")[frame], dtype=np.float32)
    current = _world_to_frame(current_world, pose)
    future_hand = _world_to_frame(future_world, pose)
    indices = np.asarray(view.knn_indices(side, frame), dtype=np.int64)
    if object_indices is not None:
        indices = indices[np.asarray(object_indices, dtype=np.int64)]
    if indices.shape != (len(object_points), KNN_K):
        raise ValueError(
            f"{view.path}: expected cached KNN [{len(object_points)},{KNN_K}], got {indices.shape}"
        )
    local_current = current[indices]
    local_future = future_hand[indices]
    distances = np.linalg.norm(local_current - object_points[:, None, :], axis=-1)
    valid = distances <= float(radius_m)
    geo_value = _mean_square_norm(distances[..., None], valid)
    flow_value = _mean_square_norm(local_future - local_current, valid)
    unique_ids = np.unique(indices[valid]) if np.any(valid) else np.empty((0,), dtype=np.int64)
    unique_flow = (
        future_hand[unique_ids] - current[unique_ids]
        if unique_ids.size
        else np.empty((0, 3), dtype=np.float32)
    )
    return geo_value, flow_value, _mean_square_norm(unique_flow)


def _group_rms(groups: dict[str, list[float]], *, name: str) -> float:
    group_means = [float(np.mean(values)) for values in groups.values() if values]
    if not group_means:
        raise RuntimeError(f"No valid {name} values were collected")
    return float(np.sqrt(np.mean(group_means)))


def calibrate(
    index: Path,
    output: Path,
    *,
    max_sequences_per_source: int,
    frames_per_sequence_stride: int,
    radius_m: float,
    knn_k: int,
    seed: int,
    grab_strides: Iterable[int],
    inspire_strides: Iterable[int],
) -> dict[str, Any]:
    if int(knn_k) != KNN_K:
        raise ValueError(f"V1.3 requires knn_k={KNN_K}, got {knn_k}")
    if float(radius_m) != RADIUS_M:
        raise ValueError(f"V1.3 requires radius_m={RADIUS_M}, got {radius_m}")
    entries = _resolve_index_entries(index, "train")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    if index_payload.get("knn_k") != KNN_K:
        raise ValueError(f"{index}: index knn_k must be {KNN_K}")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_source[str(entry["source"])].append(entry)
    stride_map = {
        "grab": tuple(int(value) for value in grab_strides),
        "inspire_f1": tuple(int(value) for value in inspire_strides),
    }
    group_geo: dict[str, list[float]] = defaultdict(list)
    group_unique_hand_flow: dict[str, list[float]] = defaultdict(list)
    group_knn_flow: dict[str, list[float]] = defaultdict(list)
    group_obj_flow: dict[str, list[float]] = defaultdict(list)
    group_counts: dict[str, int] = defaultdict(int)
    selected_sequence_counts: dict[str, int] = {}

    for source, source_entries in sorted(by_source.items()):
        selected_entries = (
            source_entries[: int(max_sequences_per_source)]
            if int(max_sequences_per_source) > 0
            else source_entries
        )
        selected_sequence_counts[source] = len(selected_entries)
        for sequence_index, entry in enumerate(selected_entries):
            view = _SequenceView(Path(entry["path"]), expected_hand_points=1538)
            view.source_name = source
            if not view.has_offline_knn():
                raise ValueError(f"{view.path}: V1.3 scale calibration requires offline KNN arrays")
            side = next(iter(view.streams))
            for stride in stride_map.get(source, (2,)):
                if stride <= 0:
                    raise ValueError(f"{source}: stride must be positive, got {stride}")
                key = f"{source}/stride_{stride}"
                frames = _frames(
                    view,
                    stride,
                    int(frames_per_sequence_stride),
                    int(seed) + sequence_index * 1009 + int(stride),
                )
                for frame in frames:
                    frame = int(frame)
                    future = frame + int(stride)
                    pose = np.asarray(view.array("pose")[frame], dtype=np.float32)
                    object_world = np.asarray(view.array("obj")[frame], dtype=np.float32)
                    object_future_world = np.asarray(view.array("obj")[future], dtype=np.float32)
                    object_points_full = _world_to_frame(object_world, pose)
                    object_future_full = _world_to_frame(object_future_world, pose)
                    object_seed = _stable_seed(seed, view.path, frame, stride)
                    selected = np.random.default_rng(object_seed).choice(
                        OBJECT_POOL_POINTS,
                        size=MODEL_OBJECT_POINTS,
                        replace=False,
                    )
                    object_points = object_points_full[selected]
                    object_future = object_future_full[selected]
                    geo_value, knn_flow, unique_hand_flow = _knn_stats(
                        view,
                        side,
                        frame,
                        future,
                        object_points,
                        object_indices=selected,
                        radius_m=radius_m,
                    )
                    obj_flow = _mean_square_norm(object_future - object_points)
                    if unique_hand_flow is not None:
                        group_unique_hand_flow[key].append(unique_hand_flow)
                    if geo_value is not None:
                        group_geo[key].append(geo_value)
                    if knn_flow is not None:
                        group_knn_flow[key].append(knn_flow)
                    if obj_flow is not None:
                        group_obj_flow[key].append(obj_flow)
                    group_counts[key] += 1

    payload = {
        "schema_name": "ref2dex_object_interaction_cm_scales_v1_3",
        "schema_version": "1.0.0",
        "split": "train",
        "coordinate_frame": "object_pose_t",
        "radius_m": float(radius_m),
        "hand_supervision_radius_m": float(radius_m),
        "knn_k": int(knn_k),
        "hand_selection": "unique_hand_ids_from_valid_knn_edges",
        "object_sampling": {
            "object_pool_points": OBJECT_POOL_POINTS,
            "model_object_points": MODEL_OBJECT_POINTS,
        },
        "sampling": {
            "max_sequences_per_source": int(max_sequences_per_source),
            "frames_per_sequence_stride": int(frames_per_sequence_stride),
            "seed": int(seed),
            "stride_policy": {key: list(values) for key, values in stride_map.items()},
            "selected_sequence_counts": selected_sequence_counts,
        },
        "scales": {
            "s_geo": _group_rms(group_geo, name="geometry"),
            "s_hand_flow": _group_rms(group_unique_hand_flow, name="unique KNN hand flow"),
            "s_knn_hand_flow": _group_rms(group_knn_flow, name="KNN hand flow"),
            "s_obj_flow": _group_rms(group_obj_flow, name="object flow"),
        },
        "group_frame_counts": dict(sorted(group_counts.items())),
        "group_rms": {
            "s_geo": {
                key: float(np.sqrt(np.mean(values)))
                for key, values in sorted(group_geo.items())
                if values
            },
            "s_hand_flow": {
                key: float(np.sqrt(np.mean(values)))
                for key, values in sorted(group_unique_hand_flow.items())
                if values
            },
            "s_knn_hand_flow": {
                key: float(np.sqrt(np.mean(values)))
                for key, values in sorted(group_knn_flow.items())
                if values
            },
            "s_obj_flow": {
                key: float(np.sqrt(np.mean(values)))
                for key, values in sorted(group_obj_flow.items())
                if values
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences-per-source", type=int, default=32)
    parser.add_argument("--frames-per-sequence-stride", type=int, default=4)
    parser.add_argument("--radius-m", type=float, default=RADIUS_M)
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grab-strides", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--inspire-strides", type=int, nargs="+", default=list(range(1, 11)))
    args = parser.parse_args()
    payload = calibrate(
        args.index.resolve(),
        args.output.resolve(),
        max_sequences_per_source=args.max_sequences_per_source,
        frames_per_sequence_stride=args.frames_per_sequence_stride,
        radius_m=args.radius_m,
        knn_k=args.knn_k,
        seed=args.seed,
        grab_strides=args.grab_strides,
        inspire_strides=args.inspire_strides,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scales": payload["scales"],
                "groups": len(payload["group_frame_counts"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
