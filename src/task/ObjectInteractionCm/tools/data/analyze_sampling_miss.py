"""Estimate 4096->1024 interaction sampling miss without writing cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.task.ObjectInteractionCm.dataset import _SequenceView, _resolve_index_entries, _stable_seed, _world_to_frame


def _active_mask(view: _SequenceView, frame: int, object_points: np.ndarray, hand_points: np.ndarray, radius: float) -> np.ndarray:
    # Existing GRAB cache stores frame-level candidate activity; recompute the
    # point-level 5 cm mask from the full pool so N_active_4096 is explicit.
    from scipy.spatial import cKDTree
    distances = cKDTree(np.asarray(hand_points, dtype=np.float32)).query(np.asarray(object_points, dtype=np.float32), k=1, workers=1)[0]
    return np.asarray(distances <= radius, dtype=bool)


def analyze(index: Path, output: Path, *, max_sequences: int, max_frames: int, num_obj_points: int, radius: float, seed: int) -> dict:
    entries = _resolve_index_entries(index, "train")
    by_source: dict[str, list[dict]] = {}
    for item in entries:
        by_source.setdefault(str(item["source"]), []).append(item)
    source_stats = {}
    all_counts = []
    for source, source_entries in sorted(by_source.items()):
        samples = []
        for seq_idx, entry in enumerate(source_entries[:max_sequences] if max_sequences > 0 else source_entries):
            view = _SequenceView(Path(entry["path"]), expected_hand_points=1538)
            last = max(0, view.frame_count - 21)
            frames = np.arange(last, dtype=np.int64)
            if max_frames > 0 and len(frames) > max_frames:
                frames = np.sort(np.random.default_rng(seed + seq_idx).choice(frames, max_frames, replace=False))
            for frame in frames:
                pose = np.asarray(view.array("pose")[frame], dtype=np.float32)
                obj_world = np.asarray(view.array("obj")[frame], dtype=np.float32)
                obj = _world_to_frame(obj_world, pose)
                hands = [_world_to_frame(view.hand(side, int(frame))[0], pose) for side, _, _ in view.hands(int(frame))]
                hand = np.concatenate(hands, axis=0)
                active = _active_mask(view, int(frame), obj, hand, radius)
                rng = np.random.default_rng(_stable_seed(seed, entry["path"], int(frame)))
                selected = rng.choice(len(obj), size=num_obj_points, replace=False)
                n_full, n_sample = int(active.sum()), int(active[selected].sum())
                samples.append({"n_active_4096": n_full, "n_active_1024": n_sample})
                all_counts.append((n_full, n_sample))
    full = np.asarray([item[0] for item in all_counts], dtype=np.int64)
    sampled = np.asarray([item[1] for item in all_counts], dtype=np.int64)
    active_full = full > 0
    miss = active_full & (sampled == 0)
    payload = {
        "schema_name": "ref2dex_object_interaction_cm_sampling_miss_v1_2_1",
        "split": "train", "radius_m": float(radius), "num_obj_pool": 4096, "num_obj_points": int(num_obj_points),
        "sampling": {"max_sequences_per_source": int(max_sequences), "max_frames_per_sequence": int(max_frames), "seed": int(seed)},
        "total_samples": int(len(full)), "active_full_samples": int(active_full.sum()),
        "n_active_4096_mean": float(full.mean()) if len(full) else 0.0,
        "n_active_1024_mean": float(sampled.mean()) if len(sampled) else 0.0,
        "sampling_miss_ratio_conditional": float(miss.sum() / max(active_full.sum(), 1)),
        "true_no_interaction_ratio": float((~active_full).mean()) if len(full) else 0.0,
        "source_samples": {source: len(items) for source, items in sorted(by_source.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences-per-source", type=int, default=32)
    parser.add_argument("--max-frames-per-sequence", type=int, default=16)
    parser.add_argument("--num-obj-points", type=int, default=1024)
    parser.add_argument("--radius-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = analyze(args.index.resolve(), args.output.resolve(), max_sequences=args.max_sequences_per_source, max_frames=args.max_frames_per_sequence, num_obj_points=args.num_obj_points, radius=args.radius_m, seed=args.seed)
    print(json.dumps({key: payload[key] for key in ("total_samples", "active_full_samples", "n_active_4096_mean", "n_active_1024_mean", "sampling_miss_ratio_conditional", "true_no_interaction_ratio")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
