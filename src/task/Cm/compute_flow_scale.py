"""Calibrate one rotation-invariant point-flow scale from Cm training data.

The calibration deliberately reads *only* the train split.  It enumerates
every legal current-frame/stride pair and always uses current-frame 5cm
candidates, matching ``Stage4CmDataset`` sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(data[key]).item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, type=Path, help="Cm Stage 4 train split root.")
    parser.add_argument("--min-stride", type=int, default=1)
    parser.add_argument("--max-stride", type=int, default=10)
    parser.add_argument(
        "--num-obj-points", type=int, default=512,
        help="Maximum valid object points sampled per training pair.",
    )
    parser.add_argument(
        "--include-inactive", action="store_false", dest="active_only", default=True,
        help="Include non-contact current frames instead of the training default active-only filter.",
    )
    parser.add_argument(
        "--statistics-split", choices=("train",), default="train",
        help="Fixed to train to make accidental val/test calibration explicit.",
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=None,
        help="Output JSON path; defaults to <train-path>/metadata.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print statistics without writing metadata.")
    return parser.parse_args()


def _load_existing_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON; refusing to overwrite it.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def calibrate_flow_scale(
    train_path: Path,
    *,
    min_stride: int,
    max_stride: int,
    active_only: bool,
    num_obj_points: int = 512,
) -> dict[str, Any]:
    """Return RMS flow scale statistics over the exact stride population.

    Each stride is fully enumerated for every legal current frame, so strides
    receive equal weight when their valid-pair counts are equal (as they are
    for a complete sequence with a fixed maximum stride).  Norms are computed
    in world coordinates because a hand-root rigid transform preserves them.
    """
    if min_stride <= 0 or max_stride < min_stride:
        raise ValueError("Require 0 < min_stride <= max_stride.")
    if num_obj_points <= 0:
        raise ValueError("num_obj_points must be positive.")
    train_path = train_path.resolve()
    hand_paths = sorted(
        path for path in train_path.glob("**/*.npz") if path.name in {"left.npz", "right.npz"}
    )
    if not hand_paths:
        raise FileNotFoundError(f"No Cm left/right NPZ files found under {train_path}")

    squared_sum = 0.0
    point_count = 0
    pair_count = 0
    sequence_ids: set[str] = set()
    stride_counts = {stride: {"pairs": 0, "points": 0, "squared_sum": 0.0} for stride in range(min_stride, max_stride + 1)}

    for hand_path in hand_paths:
        shared_path = hand_path.parent / "shared.npz"
        if not shared_path.exists():
            raise FileNotFoundError(f"{hand_path}: missing sibling shared.npz")
        with np.load(shared_path, allow_pickle=False) as shared, np.load(hand_path, allow_pickle=False) as hand:
            sequence_ids.add(_scalar(shared, "seq_id"))
            obj_points = np.asarray(shared["obj_points_world"], dtype=np.float64)
            candidate_mask = np.asarray(hand["obj_candidate_mask_5cm"], dtype=bool)
            if obj_points.ndim != 3 or obj_points.shape[-1] != 3:
                raise ValueError(f"{shared_path}: expected obj_points_world [T,N,3]")
            if candidate_mask.shape != obj_points.shape[:2]:
                raise ValueError(f"{hand_path}: candidate mask shape does not match object points")
            frame_count = obj_points.shape[0]
            # Match Stage4CmDataset: every sampled current frame must support
            # every configured stride, including max_stride.
            for current in range(frame_count - max_stride):
                candidate = candidate_mask[current]
                if active_only and not candidate.any():
                    continue
                for stride in range(min_stride, max_stride + 1):
                    # A current hand-root transform is rigid, hence it leaves
                    # ||O_{t+s} - O_t|| unchanged.  Avoiding the transform
                    # makes this full calibration pass inexpensive.
                    flow = obj_points[current + stride] - obj_points[current]
                    # ``active_only`` determines whether an empty candidate
                    # frame is retained; it never changes the runtime point
                    # sampler, which always draws from 5cm candidates.
                    flow = flow[candidate]
                    squared_norm = np.einsum("ij,ij->i", flow, flow)
                    available_count = int(squared_norm.size)
                    if available_count == 0:
                        continue
                    # Runtime sampling is uniform without replacement among
                    # candidates and caps each pair at num_obj_points.  Its
                    # expected mean-square flow is the candidate mean; use
                    # that expectation with the same effective pair weight.
                    count = min(available_count, int(num_obj_points))
                    sum_value = float(squared_norm.mean(dtype=np.float64)) * count
                    squared_sum += sum_value
                    point_count += count
                    pair_count += 1
                    stride_counts[stride]["pairs"] += 1
                    stride_counts[stride]["points"] += count
                    stride_counts[stride]["squared_sum"] += sum_value

    if point_count == 0:
        raise ValueError("No valid object-flow points were found for calibration.")
    # Training samples the stride uniformly, not proportional to the number
    # of valid pairs near a sequence boundary.  First average points within
    # each stride, then give every stride equal weight.
    per_stride_mean_square = [
        values["squared_sum"] / values["points"]
        for values in stride_counts.values()
        if values["points"] > 0
    ]
    if len(per_stride_mean_square) != len(stride_counts):
        raise ValueError("At least one requested stride has no valid object-flow points.")
    rms_m = float(np.sqrt(np.mean(per_stride_mean_square)))
    if not np.isfinite(rms_m) or rms_m <= 0.0:
        raise ValueError(f"Invalid flow RMS {rms_m!r}")
    per_stride = {
        str(stride): {
            "num_pairs": int(values["pairs"]),
            "num_points": int(values["points"]),
            "flow_rms_m": float(np.sqrt(values["squared_sum"] / values["points"])) if values["points"] else None,
        }
        for stride, values in stride_counts.items()
    }
    return {
        "flow_target_rms_m": rms_m,
        "flow_target_scale": float(1.0 / rms_m),
        "statistics_split": "train",
        "statistics_stride_distribution": f"uniform_{min_stride}_to_{max_stride}",
        "statistics_stride_weighting": "equal_per_stride",
        "statistics_active_only": bool(active_only),
        "statistics_num_obj_points": int(num_obj_points),
        "statistics_point_weighting": "per_pair_capped_at_num_obj_points",
        "statistics_num_sequences": len(sequence_ids),
        "statistics_num_hand_streams": len(hand_paths),
        "statistics_num_pairs": pair_count,
        "statistics_num_points": point_count,
        "statistics_per_stride": per_stride,
    }


def main() -> None:
    args = _parse_args()
    result = calibrate_flow_scale(
        args.train_path,
        min_stride=args.min_stride,
        max_stride=args.max_stride,
        active_only=args.active_only,
        num_obj_points=args.num_obj_points,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    metadata_path = args.metadata_path or (args.train_path / "metadata.json")
    metadata_path = metadata_path.resolve()
    metadata = _load_existing_metadata(metadata_path)
    metadata.update(result)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[cm-flow-scale] wrote {metadata_path}")
    print(
        "[cm-flow-scale] config values: "
        f"meta.flow_target_rms_m={result['flow_target_rms_m']:.9g} "
        f"meta.object_flow_target_scale={result['flow_target_scale']:.9g}"
    )


if __name__ == "__main__":
    main()
