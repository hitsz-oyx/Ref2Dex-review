#!/usr/bin/env python3
"""Calibrate the V1.3 ObjectInteractionCm scales from train-only cache data.

The fixed 1538-point decoder stream and the source-specific high-resolution KNN
stream have separate flow statistics.  The cache already contains the
full-pool KNN indices, so this command only gathers the cached 32 neighbors
and recomputes their current-frame distances.  It never rewrites geometry.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.task.ObjectInteractionCm.dataset import _SequenceView, _resolve_index_entries, _world_to_frame


KNN_K = 32
RADIUS_M = 0.02
DECODER_HAND_POINTS = 1538


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


def _decoder_hand_stats(
    view: _SequenceView,
    frame: int,
    future: int,
    object_points: np.ndarray,
    *,
    radius_m: float,
) -> float | None:
    values: list[np.ndarray] = []
    for side, _, _ in view.hands(frame):
        current, _ = view.hand(side, frame)
        future_hand, _ = view.hand(side, future)
        current_local = _world_to_frame(current, view.array("pose")[frame])
        future_local = _world_to_frame(future_hand, view.array("pose")[frame])
        mask = view.hand_supervision(side, frame)
        if mask is None:
            delta = current_local[:, None, :] - object_points[None, :, :]
            nearest = np.sqrt(np.square(delta).sum(axis=-1).min(axis=1))
            mask = nearest <= float(radius_m)
        flow = future_local - current_local
        selected = flow[np.asarray(mask, dtype=bool)]
        if selected.size:
            values.append(selected)
    if not values:
        return None
    return _mean_square_norm(np.concatenate(values, axis=0))


def _knn_stats(
    view: _SequenceView,
    side: str,
    frame: int,
    future: int,
    object_points: np.ndarray,
    *,
    radius_m: float,
) -> tuple[float | None, float | None]:
    current_world, _ = view.knn_hand(side, frame)
    future_world, _ = view.knn_hand(side, future)
    pose = np.asarray(view.array("pose")[frame], dtype=np.float32)
    current = _world_to_frame(current_world, pose)
    future_hand = _world_to_frame(future_world, pose)
    indices = np.asarray(view.knn_indices(side, frame), dtype=np.int64)
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
    return geo_value, flow_value


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
    group_decoder_flow: dict[str, list[float]] = defaultdict(list)
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
            view = _SequenceView(Path(entry["path"]), expected_hand_points=DECODER_HAND_POINTS)
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
                    object_points = _world_to_frame(object_world, pose)
                    object_future = _world_to_frame(object_future_world, pose)
                    decoder_flow = _decoder_hand_stats(
                        view,
                        frame,
                        future,
                        object_points,
                        radius_m=radius_m,
                    )
                    geo_value, knn_flow = _knn_stats(
                        view,
                        side,
                        frame,
                        future,
                        object_points,
                        radius_m=radius_m,
                    )
                    obj_flow = _mean_square_norm(object_future - object_points)
                    if decoder_flow is not None:
                        group_decoder_flow[key].append(decoder_flow)
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
        "sampling": {
            "max_sequences_per_source": int(max_sequences_per_source),
            "frames_per_sequence_stride": int(frames_per_sequence_stride),
            "seed": int(seed),
            "stride_policy": {key: list(values) for key, values in stride_map.items()},
            "selected_sequence_counts": selected_sequence_counts,
        },
        "scales": {
            "s_geo": _group_rms(group_geo, name="geometry"),
            "s_hand_flow": _group_rms(group_decoder_flow, name="decoder hand flow"),
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
                for key, values in sorted(group_decoder_flow.items())
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
