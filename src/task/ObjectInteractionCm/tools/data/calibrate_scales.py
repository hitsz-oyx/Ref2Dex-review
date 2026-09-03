"""Calibrate V1.2.1 feature/target scales from train-only cache samples.

This command reads existing sequence caches and writes only a small statistics
JSON. It never exports or rewrites geometry/hand cache files.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.task.ObjectInteractionCm.dataset import _SequenceView, _resolve_index_entries, _world_to_frame


def _nearest(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    return np.asarray(cKDTree(np.asarray(query, dtype=np.float32)).query(np.asarray(points, dtype=np.float32), k=1, workers=1)[0], dtype=np.float64)


def _knn_distances(object_points: np.ndarray, hand_points: np.ndarray, k: int, radius: float) -> np.ndarray:
    from scipy.spatial import cKDTree
    distances = np.asarray(cKDTree(np.asarray(hand_points, dtype=np.float32)).query(np.asarray(object_points, dtype=np.float32), k=min(k, len(hand_points)), workers=1)[0], dtype=np.float64)
    if distances.ndim == 1:
        distances = distances[:, None]
    return distances[distances <= float(radius)]


def _frames(view: _SequenceView, stride: int, count: int, seed: int) -> np.ndarray:
    last = max(0, view.frame_count - int(stride) - 1)
    if last <= count:
        return np.arange(last, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(last, size=count, replace=False))


def calibrate(index: Path, output: Path, *, max_sequences: int, frames_per_sequence_stride: int, radius: float, knn_k: int, seed: int) -> dict:
    entries = _resolve_index_entries(index, "train")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_source[str(entry["source"])].append(entry)
    stride_map = {"grab": tuple(range(1, 11)), "inspire_f1": tuple(range(2, 21, 2))}
    group_flow: dict[str, list[float]] = defaultdict(list)
    group_geo: dict[str, list[float]] = defaultdict(list)
    group_obj: dict[str, list[float]] = defaultdict(list)
    group_counts: dict[str, int] = defaultdict(int)
    for source, source_entries in sorted(by_source.items()):
        selected_entries = source_entries[:max_sequences] if max_sequences > 0 else source_entries
        for seq_index, entry in enumerate(selected_entries):
            view = _SequenceView(Path(entry["path"]), expected_hand_points=1538)
            view.source_name = source
            for stride in stride_map.get(source, (2,)):
                key = f"{source}/stride_{stride}"
                for frame in _frames(view, stride, frames_per_sequence_stride, seed + seq_index * 1009 + stride):
                    pose = np.asarray(view.array("pose")[frame], dtype=np.float32)
                    obj_world = np.asarray(view.array("obj")[frame], dtype=np.float32)
                    obj_future_world = np.asarray(view.array("obj")[frame + stride], dtype=np.float32)
                    obj = _world_to_frame(obj_world, pose)
                    obj_future = _world_to_frame(obj_future_world, pose)
                    hands, future_hands = [], []
                    for side, _, _ in view.hands(int(frame)):
                        current, _ = view.hand(side, int(frame))
                        future, _ = view.hand(side, int(frame + stride))
                        hands.append(_world_to_frame(current, pose))
                        future_hands.append(_world_to_frame(future, pose))
                    hand = np.concatenate(hands, axis=0)
                    hand_future = np.concatenate(future_hands, axis=0)
                    flow = hand_future - hand
                    nearest = _nearest(hand, obj)
                    flow_values = np.square(np.linalg.norm(flow[nearest <= radius], axis=-1))
                    geo_values = _knn_distances(obj, hand, knn_k, radius)
                    obj_values = np.square(np.linalg.norm(obj_future - obj, axis=-1))
                    if flow_values.size:
                        group_flow[key].append(float(flow_values.mean()))
                    if geo_values.size:
                        group_geo[key].append(float(np.square(geo_values).mean()))
                    if obj_values.size:
                        group_obj[key].append(float(obj_values.mean()))
                    group_counts[key] += 1
    def group_rms(groups: dict[str, list[float]]) -> float:
        values = [float(np.mean(items)) for items in groups.values() if items]
        if not values:
            raise RuntimeError("No valid calibration values were collected")
        return float(np.sqrt(np.mean(values)))
    payload = {
        "schema_name": "ref2dex_object_interaction_cm_scales_v1_2_1",
        "split": "train",
        "radius_m": float(radius), "knn_k": int(knn_k),
        "sampling": {"max_sequences_per_source": int(max_sequences), "frames_per_sequence_stride": int(frames_per_sequence_stride), "seed": int(seed)},
        "scales": {"s_geo": group_rms(group_geo), "s_hand_flow": group_rms(group_flow), "s_obj_flow": group_rms(group_obj)},
        "group_frame_counts": dict(sorted(group_counts.items())),
        "group_rms": {
            "s_geo": {key: float(np.sqrt(np.mean(value))) for key, value in sorted(group_geo.items()) if value},
            "s_hand_flow": {key: float(np.sqrt(np.mean(value))) for key, value in sorted(group_flow.items()) if value},
            "s_obj_flow": {key: float(np.sqrt(np.mean(value))) for key, value in sorted(group_obj.items()) if value},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences-per-source", type=int, default=32)
    parser.add_argument("--frames-per-sequence-stride", type=int, default=4)
    parser.add_argument("--radius-m", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = calibrate(args.index.resolve(), args.output.resolve(), max_sequences=args.max_sequences_per_source, frames_per_sequence_stride=args.frames_per_sequence_stride, radius=args.radius_m, knn_k=args.knn_k, seed=args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "scales": payload["scales"], "groups": len(payload["group_frame_counts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
