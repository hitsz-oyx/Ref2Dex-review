from __future__ import annotations

from typing import Any

from .mesh_io import import_numpy


DEFAULT_THRESHOLDS = {
    "eps_mm": 1.0,
    "minor_depth_mm": 2.0,
    "minor_inside_ratio": 0.002,
    "significant_depth_mm": 5.0,
    "significant_inside_ratio": 0.01,
    "significant_inside_count": 20,
}


def summarize_frame_metrics(frame_rows: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    np = import_numpy()
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if not frame_rows:
        return {
            "num_frames_checked": 0,
            "num_query_points_total": 0,
            "num_inside_points_total": 0,
            "total_inside_ratio": 0.0,
            "mean_frame_max_negative_depth_mm": 0.0,
            "var_frame_max_negative_depth_mm": 0.0,
            "max_frame_max_negative_depth_mm": 0.0,
            "mean_frame_mean_negative_depth_mm": 0.0,
            "var_frame_mean_negative_depth_mm": 0.0,
            "frames_with_any_negative": 0,
            "raw_negative_frame_ratio": 0.0,
            "frames_with_minor_penetration": 0,
            "minor_penetration_frame_ratio": 0.0,
            "frames_with_significant_penetration": 0,
            "significant_penetration_frame_ratio": 0.0,
        }
    depths = np.asarray([float(row["frame_max_negative_depth_mm"]) for row in frame_rows], dtype="float64")
    mean_depths = np.asarray([float(row["frame_mean_negative_depth_mm"]) for row in frame_rows], dtype="float64")
    inside_counts = np.asarray([int(row["inside_count"]) for row in frame_rows], dtype="int64")
    query_counts = np.asarray([int(row["num_query_points"]) for row in frame_rows], dtype="int64")
    inside_ratios = np.divide(
        inside_counts,
        np.maximum(query_counts, 1),
        dtype="float64",
    )
    any_negative = inside_counts > 0
    minor = (depths >= float(thresholds["minor_depth_mm"])) & (
        inside_ratios >= float(thresholds["minor_inside_ratio"])
    )
    significant = (
        (depths >= float(thresholds["significant_depth_mm"]))
        & (inside_ratios >= float(thresholds["significant_inside_ratio"]))
        & (inside_counts >= int(thresholds["significant_inside_count"]))
    )
    total_query = int(query_counts.sum())
    total_inside = int(inside_counts.sum())
    frame_count = int(len(frame_rows))
    return {
        "num_frames_checked": frame_count,
        "num_query_points_total": total_query,
        "num_inside_points_total": total_inside,
        "total_inside_ratio": float(total_inside / total_query) if total_query else 0.0,
        "mean_frame_max_negative_depth_mm": float(depths.mean()),
        "var_frame_max_negative_depth_mm": float(depths.var()),
        "max_frame_max_negative_depth_mm": float(depths.max()),
        "mean_frame_mean_negative_depth_mm": float(mean_depths.mean()),
        "var_frame_mean_negative_depth_mm": float(mean_depths.var()),
        "frames_with_any_negative": int(any_negative.sum()),
        "raw_negative_frame_ratio": float(any_negative.mean()),
        "frames_with_minor_penetration": int(minor.sum()),
        "minor_penetration_frame_ratio": float(minor.mean()),
        "frames_with_significant_penetration": int(significant.sum()),
        "significant_penetration_frame_ratio": float(significant.mean()),
    }


def aggregate_sequence_summaries(sequence_summaries: list[dict[str, Any]], failures: int = 0) -> dict[str, Any]:
    np = import_numpy()
    succeeded = [row for row in sequence_summaries if row.get("status", "ok") == "ok"]
    aggregate: dict[str, Any] = {
        "num_sequences_sampled": int(len(sequence_summaries) + failures),
        "num_sequences_succeeded": int(len(succeeded)),
        "num_sequences_failed": int(failures + len(sequence_summaries) - len(succeeded)),
    }
    if not succeeded:
        aggregate.update(
            {
                "mean_of_sequence_mean_frame_max_negative_depth_mm": 0.0,
                "var_of_sequence_mean_frame_max_negative_depth_mm": 0.0,
                "mean_of_sequence_var_frame_max_negative_depth_mm": 0.0,
                "mean_of_sequence_max_frame_max_negative_depth_mm": 0.0,
                "max_of_sequence_max_frame_max_negative_depth_mm": 0.0,
                "mean_total_inside_ratio": 0.0,
                "mean_raw_negative_frame_ratio": 0.0,
                "mean_minor_penetration_frame_ratio": 0.0,
                "mean_significant_penetration_frame_ratio": 0.0,
            }
        )
        return aggregate
    mean_depth = np.asarray([float(row["mean_frame_max_negative_depth_mm"]) for row in succeeded], dtype="float64")
    var_depth = np.asarray([float(row["var_frame_max_negative_depth_mm"]) for row in succeeded], dtype="float64")
    max_depth = np.asarray([float(row["max_frame_max_negative_depth_mm"]) for row in succeeded], dtype="float64")
    aggregate.update(
        {
            "mean_of_sequence_mean_frame_max_negative_depth_mm": float(mean_depth.mean()),
            "var_of_sequence_mean_frame_max_negative_depth_mm": float(mean_depth.var()),
            "mean_of_sequence_var_frame_max_negative_depth_mm": float(var_depth.mean()),
            "mean_of_sequence_max_frame_max_negative_depth_mm": float(max_depth.mean()),
            "max_of_sequence_max_frame_max_negative_depth_mm": float(max_depth.max()),
            "mean_total_inside_ratio": float(np.asarray([float(row["total_inside_ratio"]) for row in succeeded]).mean()),
            "mean_raw_negative_frame_ratio": float(
                np.asarray([float(row["raw_negative_frame_ratio"]) for row in succeeded]).mean()
            ),
            "mean_minor_penetration_frame_ratio": float(
                np.asarray([float(row["minor_penetration_frame_ratio"]) for row in succeeded]).mean()
            ),
            "mean_significant_penetration_frame_ratio": float(
                np.asarray([float(row["significant_penetration_frame_ratio"]) for row in succeeded]).mean()
            ),
        }
    )
    return aggregate

