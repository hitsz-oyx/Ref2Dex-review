"""Calibrate hand-flow normalization for a source-balanced Cm training mix.

The statistic matches training: train split only, sources sampled equally,
rows sampled uniformly within each source, and strides sampled uniformly from
``min_stride`` through ``max_stride``.  Every hand point is supervised, so no
point subsampling is applied.  Flow norms are computed in world coordinates;
they are unchanged by the rigid current-hand-root transform used by the
runtime dataset.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from src.base.run_manifest import build_run_manifest, write_run_manifest
from src.task.Cm.dataset.hrdexdb import _manifest_specs
from src.task.Cm.tools.data.compute_flow_scale import object_v2_train_sequence_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grab-root", required=True, type=Path)
    parser.add_argument("--grab-split", required=True, type=Path)
    parser.add_argument("--hrdexdb-root", required=True, type=Path)
    parser.add_argument("--hrdexdb-manifest", required=True, type=Path)
    parser.add_argument("--hrdexdb-source-prefix", default="inspire_f1")
    parser.add_argument("--min-stride", type=int, default=1)
    parser.add_argument("--max-stride", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="scale JSON path; defaults to this experiment's output/<run_id>/hand_flow_scale.json",
    )
    return parser.parse_args()


def _new_state(min_stride: int, max_stride: int) -> dict[int, list[float]]:
    return {stride: [0.0, 0.0] for stride in range(min_stride, max_stride + 1)}


def _accumulate_hand_stream(
    hand_path: Path,
    active_rows: np.ndarray,
    state: dict[int, list[float]],
    *,
    chunk_rows: int,
) -> None:
    hand = np.load(hand_path, mmap_mode="r")
    if hand.ndim != 3 or hand.shape[-1] != 3:
        raise ValueError(f"Expected [T,N,3] hand points at {hand_path}, got {hand.shape}")
    for stride, totals in state.items():
        for start in range(0, active_rows.size, chunk_rows):
            current = active_rows[start : start + chunk_rows]
            now = np.asarray(hand[current], dtype=np.float64)
            future = np.asarray(hand[current + stride], dtype=np.float64)
            squared_norm = np.einsum("...d,...d->...", future - now, future - now)
            totals[0] += float(squared_norm.sum(dtype=np.float64))
            totals[1] += float(squared_norm.size)


def _grab_state(
    root: Path,
    split: Path,
    *,
    min_stride: int,
    max_stride: int,
    chunk_rows: int,
) -> tuple[dict[int, list[float]], int]:
    state = _new_state(min_stride, max_stride)
    stream_count = 0
    for sequence in object_v2_train_sequence_dirs(root, split):
        for side in ("left", "right"):
            side_dir = sequence / side
            hand_path = side_dir / "hand_points_world.npy"
            offsets_path = side_dir / "candidate_offsets.npy"
            if not hand_path.is_file() or not offsets_path.is_file():
                continue
            offsets = np.load(offsets_path, mmap_mode="r")
            upper = max(0, len(offsets) - 1 - max_stride)
            active = np.flatnonzero(np.diff(offsets[: upper + 1]) > 0).astype(np.int64)
            if active.size == 0:
                continue
            _accumulate_hand_stream(hand_path, active, state, chunk_rows=chunk_rows)
            stream_count += 1
    return state, stream_count


def _inspire_state(
    root: Path,
    manifest: Path,
    source_prefix: str,
    *,
    min_stride: int,
    max_stride: int,
    chunk_rows: int,
) -> tuple[dict[int, list[float]], int]:
    state = _new_state(min_stride, max_stride)
    episodes = _manifest_specs(root, manifest, "train", source_prefix)
    for _, episode_dir in episodes:
        geometry = episode_dir / "geometry"
        candidate = np.load(geometry / "obj_candidate_mask_5cm.npy", mmap_mode="r")
        upper = max(0, candidate.shape[0] - max_stride)
        active = np.flatnonzero(np.asarray(candidate[:upper]).any(axis=1)).astype(np.int64)
        if active.size == 0:
            continue
        _accumulate_hand_stream(
            geometry / "hand_points_world.npy", active, state, chunk_rows=chunk_rows
        )
    return state, len(episodes)


def _source_payload(state: dict[int, list[float]]) -> dict[str, object]:
    per_stride = {
        str(stride): {
            "num_points": int(count),
            "hand_flow_rms_m": float(np.sqrt(squared_sum / count)),
        }
        for stride, (squared_sum, count) in state.items()
    }
    mean_square = float(
        np.mean([squared_sum / count for squared_sum, count in state.values()])
    )
    return {"hand_flow_rms_m": float(np.sqrt(mean_square)), "per_stride": per_stride}


def main() -> None:
    args = _parse_args()
    if args.min_stride <= 0 or args.max_stride < args.min_stride:
        raise ValueError("Require 0 < min_stride <= max_stride")
    grab, grab_streams = _grab_state(
        args.grab_root.resolve(),
        args.grab_split.resolve(),
        min_stride=args.min_stride,
        max_stride=args.max_stride,
        chunk_rows=args.chunk_rows,
    )
    inspire, inspire_episodes = _inspire_state(
        args.hrdexdb_root.resolve(),
        args.hrdexdb_manifest.resolve(),
        args.hrdexdb_source_prefix,
        min_stride=args.min_stride,
        max_stride=args.max_stride,
        chunk_rows=args.chunk_rows,
    )
    source_states = {"grab": grab, args.hrdexdb_source_prefix: inspire}
    source_payloads = {
        name: _source_payload(state) for name, state in source_states.items()
    }
    source_stride_mean_squares = [
        squared_sum / count
        for state in source_states.values()
        for squared_sum, count in state.values()
    ]
    rms_m = float(np.sqrt(np.mean(source_stride_mean_squares)))
    payload = {
        "hand_flow_target_rms_m": rms_m,
        "hand_flow_target_scale": float(1.0 / rms_m),
        "statistics_split": "train",
        "statistics_sources": ["grab", args.hrdexdb_source_prefix],
        "statistics_source_weighting": "equal_per_source",
        "statistics_stride_distribution": (
            f"uniform_{args.min_stride}_to_{args.max_stride}"
        ),
        "statistics_stride_weighting": "equal_per_stride",
        "statistics_point_weighting": "all_1538_hand_points",
        "grab_train_hand_streams": grab_streams,
        "inspire_train_episodes": inspire_episodes,
        "sources": source_payloads,
    }
    output = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parent
        / "output"
        / datetime.now().strftime("hand_flow_%Y%m%d_%H%M%S")
        / "hand_flow_scale.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "grab_root": str(args.grab_root.resolve()),
        "grab_split": str(args.grab_split.resolve()),
        "hrdexdb_root": str(args.hrdexdb_root.resolve()),
        "hrdexdb_manifest": str(args.hrdexdb_manifest.resolve()),
        "hrdexdb_source_prefix": str(args.hrdexdb_source_prefix),
        "min_stride": int(args.min_stride),
        "max_stride": int(args.max_stride),
        "chunk_rows": int(args.chunk_rows),
    }
    (output.parent / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output.parent / "metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "cm_hand_flow_scale",
                "dataset_split": "train",
                "statistics_sources": payload["statistics_sources"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    payload["output"] = str(output)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = build_run_manifest(
        task="Cm",
        run_name=output.parent.name,
        output_dir=output.parent,
        mode="calibration",
        config=config_snapshot,
        metadata={
            "schema_name": "cm_hand_flow_scale",
            "dataset_split": "train",
            "root": str(args.grab_root.resolve()),
            "hrdexdb_root": str(args.hrdexdb_root.resolve()),
            "hrdexdb_manifest": str(args.hrdexdb_manifest.resolve()),
        },
        config_source=Path(__file__).with_name("experiment.yaml"),
        config_snapshot=output.parent / "config.json",
        metadata_snapshot=output.parent / "metadata.json",
        repo_root=Path(__file__).resolve().parents[5],
    )
    write_run_manifest(output.parent / "run_manifest.json", manifest)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
