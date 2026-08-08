"""Build a non-destructive dominant-hand manifest for the Cm Stage 4 cache.

The output references existing ``left.npz``/``right.npz`` files.  It never
copies, modifies, or deletes the Stage 4 arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


SCHEMA_NAME = "ref2dex_cm_dominant_hand_manifest"
SCHEMA_VERSION = 1


def coupling_error(
    object_current: np.ndarray,
    object_future: np.ndarray,
    hand_current: np.ndarray,
    hand_future: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    """Median local residual between object flow and nearest hand-point flow."""
    object_indices = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    if object_indices.size == 0:
        return float("inf")
    nearest_hand = cKDTree(np.asarray(hand_current, dtype=np.float64)).query(
        np.asarray(object_current[object_indices], dtype=np.float64), k=1, workers=1,
    )[1]
    object_flow = object_future[object_indices] - object_current[object_indices]
    hand_flow = hand_future[nearest_hand] - hand_current[nearest_hand]
    residual = np.linalg.norm(object_flow - hand_flow, axis=-1)
    return float(np.median(residual))


def _prepare_local_correspondence(
    object_current: np.ndarray,
    hand_current: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache current-frame nearest hand points across all requested strides."""
    object_indices = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    nearest_hand = cKDTree(np.asarray(hand_current, dtype=np.float64)).query(
        np.asarray(object_current[object_indices], dtype=np.float64), k=1, workers=1,
    )[1]
    return object_indices, np.asarray(nearest_hand, dtype=np.int64)


def _coupling_error_from_correspondence(
    object_current: np.ndarray,
    object_future: np.ndarray,
    hand_current: np.ndarray,
    hand_future: np.ndarray,
    object_indices: np.ndarray,
    nearest_hand: np.ndarray,
) -> float:
    object_flow = object_future[object_indices] - object_current[object_indices]
    hand_flow = hand_future[nearest_hand] - hand_current[nearest_hand]
    return float(np.median(np.linalg.norm(object_flow - hand_flow, axis=-1)))


def choose_dominant_hand(
    left_error_m: float,
    right_error_m: float,
    *,
    stride: int,
    dominance_ratio: float,
    min_error_gap_m_per_step: float,
    max_main_error_m_per_step: float,
) -> tuple[str | None, str, float]:
    """Return a side only when one hand is a conservative, credible winner."""
    errors = {"left": float(left_error_m), "right": float(right_error_m)}
    main_side = min(errors, key=errors.__getitem__)
    other_side = "right" if main_side == "left" else "left"
    main_error, other_error = errors[main_side], errors[other_side]
    confidence = other_error / max(main_error, 1.0e-12)
    credible_error = max_main_error_m_per_step * stride
    if not np.isfinite(main_error) or main_error > credible_error:
        return None, "neither_explains_motion", confidence
    # Both hands matching local object flow is evidence of genuine bilateral
    # participation, not a license to keep whichever numerical error is lower.
    if other_error <= credible_error:
        return None, "bilateral_cooperation", confidence
    if main_error * dominance_ratio > other_error:
        return None, "ambiguous_ratio", confidence
    if other_error - main_error < min_error_gap_m_per_step * stride:
        return None, "ambiguous_gap", confidence
    return main_side, f"bilateral_{main_side}_dominant", confidence


def _resolve_sequences(data_root: Path, split_file: Path | None, max_sequences: int | None) -> list[Path]:
    if split_file is None:
        sequences = sorted({path.parent for path in data_root.glob("**/*.npz") if path.name in {"left.npz", "right.npz"}})
    else:
        entries = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        hand_paths = [(data_root / entry).resolve() for entry in entries]
        missing = [path for path in hand_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Split references missing Stage 4 files: {missing[:3]}")
        sequences = sorted({path.parent for path in hand_paths})
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    if not sequences:
        raise ValueError("No Stage 4 Cm sequences selected")
    return sequences


def _process_sequence(
    payload: tuple[Path, Path, tuple[int, ...], float, float, float],
) -> tuple[list[str], Counter[str], Counter[int], int]:
    sequence_dir, data_root, strides, dominance_ratio, min_gap, max_main_error = payload
    shared_path = sequence_dir / "shared.npz"
    side_paths = {side: sequence_dir / f"{side}.npz" for side in ("left", "right")}
    if not shared_path.is_file() or not all(path.is_file() for path in side_paths.values()):
        raise FileNotFoundError(f"{sequence_dir}: dominant-hand filtering requires shared/left/right NPZ files")
    with np.load(shared_path, allow_pickle=False) as shared_npz:
        shared = {key: np.asarray(shared_npz[key]) for key in shared_npz.files}
    hands: dict[str, dict[str, np.ndarray]] = {}
    for side, path in side_paths.items():
        with np.load(path, allow_pickle=False) as hand_npz:
            hands[side] = {key: np.asarray(hand_npz[key]) for key in hand_npz.files}

    object_points = np.asarray(shared["obj_points_world"], dtype=np.float32)
    frame_count = int(object_points.shape[0])
    max_stride = max(strides)
    seq_id = str(np.asarray(shared["seq_id"]).item())
    rows: list[str] = []
    reason_counts: Counter[str] = Counter()
    stride_kept: Counter[int] = Counter()
    frames_considered = 0
    for current in range(frame_count - max_stride):
        frames_considered += 1
        active = {
            side: bool(np.asarray(hands[side]["obj_candidate_mask_5cm"][current], dtype=bool).any())
            for side in ("left", "right")
        }
        if not any(active.values()):
            reason_counts["both_inactive"] += len(strides)
            continue
        local_correspondence = {
            side: _prepare_local_correspondence(
                object_points[current], hands[side]["hand_points_world"][current],
                hands[side]["obj_candidate_mask_5cm"][current],
            )
            for side in ("left", "right") if active[side]
        }
        for stride in strides:
            if active["left"] != active["right"]:
                side = "left" if active["left"] else "right"
                reason = f"unilateral_{side}"
                errors = {"left": None, "right": None}
                confidence = None
            else:
                errors = {
                    side: _coupling_error_from_correspondence(
                        object_points[current], object_points[current + stride],
                        hands[side]["hand_points_world"][current],
                        hands[side]["hand_points_world"][current + stride],
                        *local_correspondence[side],
                    )
                    for side in ("left", "right")
                }
                side, reason, confidence = choose_dominant_hand(
                    errors["left"], errors["right"], stride=stride,
                    dominance_ratio=dominance_ratio,
                    min_error_gap_m_per_step=min_gap,
                    max_main_error_m_per_step=max_main_error,
                )
            reason_counts[reason] += 1
            if side is None:
                continue
            row = {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "decision": "keep",
                "hand_path": side_paths[side].resolve().relative_to(data_root).as_posix(),
                "seq_id": seq_id,
                "current_frame": current,
                "raw_frame_id": int(shared["raw_frame_id"][current]),
                "stride": int(stride),
                "dominant_side": side,
                "reason": reason,
                "left_error_m": errors["left"],
                "right_error_m": errors["right"],
                "confidence_ratio": confidence,
            }
            rows.append(json.dumps(row, separators=(",", ":"), allow_nan=False))
            stride_kept[int(stride)] += 1
    return rows, reason_counts, stride_kept, frames_considered


def build_manifest(
    *,
    data_root: Path,
    output_dir: Path,
    sequences: Sequence[Path],
    strides: Sequence[int],
    dominance_ratio: float,
    min_error_gap_m_per_step: float,
    max_main_error_m_per_step: float,
    num_workers: int = 1,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dominant_hand.jsonl"
    reason_counts: Counter[str] = Counter()
    stride_kept: Counter[int] = Counter()
    rows_written = 0
    frames_considered = 0

    payloads = [
        (sequence_dir, data_root, tuple(strides), dominance_ratio,
         min_error_gap_m_per_step, max_main_error_m_per_step)
        for sequence_dir in sequences
    ]
    with manifest_path.open("w", encoding="utf-8") as output, ProcessPoolExecutor(max_workers=num_workers) as executor:
        for sequence_index, (rows, sequence_reasons, sequence_strides, sequence_frames) in enumerate(
            executor.map(_process_sequence, payloads, chunksize=1), start=1,
        ):
            output.writelines(f"{row}\n" for row in rows)
            rows_written += len(rows)
            reason_counts.update(sequence_reasons)
            stride_kept.update(sequence_strides)
            frames_considered += sequence_frames
            if sequence_index % 50 == 0 or sequence_index == len(sequences):
                print(
                    f"[dominant-hand] sequences={sequence_index}/{len(sequences)} "
                    f"kept={rows_written}", flush=True,
                )

    total_decisions = int(sum(reason_counts.values()))
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path.resolve()),
        "num_sequences": len(sequences),
        "num_current_frames_considered": frames_considered,
        "num_stride_decisions": total_decisions,
        "num_kept_samples": rows_written,
        "keep_rate": rows_written / total_decisions if total_decisions else 0.0,
        "strides": list(strides),
        "thresholds": {
            "dominance_ratio": dominance_ratio,
            "min_error_gap_m_per_step": min_error_gap_m_per_step,
            "max_main_error_m_per_step": max_main_error_m_per_step,
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "kept_by_stride": {str(key): value for key, value in sorted(stride_kept.items())},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_data/stage4/data"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--strides", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--dominance-ratio", type=float, default=2.0)
    parser.add_argument("--min-error-gap-m-per-step", type=float, default=0.0005)
    parser.add_argument("--max-main-error-m-per-step", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    strides = tuple(sorted(set(args.strides)))
    if not strides or strides[0] <= 0:
        raise ValueError("strides must contain positive integers")
    if args.dominance_ratio <= 1.0:
        raise ValueError("dominance-ratio must be greater than 1")
    if args.min_error_gap_m_per_step < 0.0 or args.max_main_error_m_per_step <= 0.0:
        raise ValueError("error thresholds must be non-negative, with a positive maximum")
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive")
    data_root = args.data_root.resolve()
    sequences = _resolve_sequences(data_root, args.split_file, args.max_sequences)
    summary = build_manifest(
        data_root=data_root,
        output_dir=args.output_dir,
        sequences=sequences,
        strides=strides,
        dominance_ratio=args.dominance_ratio,
        min_error_gap_m_per_step=args.min_error_gap_m_per_step,
        max_main_error_m_per_step=args.max_main_error_m_per_step,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
