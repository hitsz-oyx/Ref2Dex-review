"""Measure whether whole-hand pose alone predicts contact topology.

This is deliberately a *data probe*, not a correspondence model.  It holds
out complete sequences, encodes only the hand's canonical-to-runtime surface
deformation, and predicts hand contact labels without reading object points.
The random-match and shuffled-label controls distinguish a genuine pose prior
from a gain due merely to the target distribution.

Example
-------
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.analyze_hand_pose_prior \\
  --data-root processed_data/generated/stage3/grab_subset100_initonly_4096_ds4_smoke12 \\
  --output outputs/analysis/hand_pose_prior_smoke12.json
"""

# This is an archival analysis utility; it is intentionally not imported by
# the production correspondence task.
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CONTACT_RADIUS_M = 0.02


def _sequence_group_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    stem = relative.stem
    if stem.endswith("_left") or stem.endswith("_right"):
        stem = stem.rsplit("_", 1)[0]
    parent = relative.parent.as_posix()
    return stem if parent in {"", "."} else f"{parent}/{stem}"


def _hand_side(path: Path) -> str:
    if path.stem.endswith("_left"):
        return "left"
    if path.stem.endswith("_right"):
        return "right"
    raise ValueError(f"Cannot infer hand side from {path.name}; expected *_left.npz or *_right.npz.")


def _split_files_by_sequence(
    paths: list[Path],
    *,
    root: Path,
    val_split: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[str]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_sequence_group_key(path, root), []).append(path)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(f"Need at least two sequence groups, got {len(keys)}.")
    num_val = min(max(1, int(round(len(keys) * val_split))), len(keys) - 1)
    rng = np.random.default_rng(seed)
    val_keys = {keys[int(index)] for index in rng.permutation(len(keys))[:num_val]}
    train = [path for path in paths if _sequence_group_key(path, root) not in val_keys]
    val = [path for path in paths if _sequence_group_key(path, root) in val_keys]
    return train, val, sorted(val_keys)


def _contact_target(distance: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - distance / CONTACT_RADIUS_M, 0.0, 1.0).astype(np.float32)


def _anchor_indices(num_hand_points: int, anchors: int) -> np.ndarray:
    if anchors <= 0:
        raise ValueError("anchors must be positive.")
    if anchors > num_hand_points:
        raise ValueError(f"anchors={anchors} exceeds num_hand_points={num_hand_points}.")
    return np.linspace(0, num_hand_points - 1, anchors, dtype=np.int64)


def _finger_topology_target(contact: np.ndarray, finger_id: np.ndarray) -> np.ndarray:
    """Return per-finger mean, active fraction, and upper-tail contact score."""
    outputs: list[np.ndarray] = []
    for finger in sorted(np.unique(finger_id).tolist()):
        values = contact[:, finger_id == finger]
        outputs.extend(
            [
                values.mean(axis=1),
                (values > 0.0).mean(axis=1),
                np.quantile(values, 0.9, axis=1),
            ]
        )
    return np.stack(outputs, axis=1).astype(np.float32)


def _load_samples(paths: list[Path], *, anchors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pose_parts: list[np.ndarray] = []
    topology_parts: list[np.ndarray] = []
    heatmap_parts: list[np.ndarray] = []
    expected_finger_id: np.ndarray | None = None
    expected_anchor_idx: np.ndarray | None = None
    num_frames = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            hand = np.asarray(payload["hand_points"], dtype=np.float32)
            canonical = np.asarray(payload["hand_cano_points"], dtype=np.float32)
            finger_id = np.asarray(payload["hand_finger_id"], dtype=np.int64)
            contact = _contact_target(np.asarray(payload["hand_to_obj_min_dist"], dtype=np.float32))
        if hand.ndim != 3 or hand.shape[-1] != 3:
            raise ValueError(f"{path}: invalid hand_points shape {hand.shape}.")
        if canonical.shape != hand.shape[1:]:
            raise ValueError(f"{path}: canonical shape {canonical.shape} != {hand.shape[1:]}.")
        if contact.shape != hand.shape[:2]:
            raise ValueError(f"{path}: contact shape {contact.shape} != {hand.shape[:2]}.")
        if expected_finger_id is None:
            expected_finger_id = finger_id
            expected_anchor_idx = _anchor_indices(hand.shape[1], anchors)
        elif not np.array_equal(expected_finger_id, finger_id):
            raise ValueError(
                f"{path}: hand_finger_id differs across files; a cross-sequence pose probe "
                "requires a stable surface-point semantic layout."
            )
        assert expected_anchor_idx is not None and expected_finger_id is not None
        # This is strictly hand-only. In hand-root coordinates, the residual
        # relative to the canonical surface represents the articulated pose.
        pose_delta = hand[:, expected_anchor_idx] - canonical[expected_anchor_idx]
        pose_parts.append(pose_delta.reshape(hand.shape[0], -1))
        topology_parts.append(_finger_topology_target(contact, expected_finger_id))
        heatmap_parts.append(contact[:, expected_anchor_idx])
        num_frames += hand.shape[0]

    if not pose_parts:
        raise ValueError("No frames loaded.")
    metadata = {
        "num_frames": int(num_frames),
        "pose_feature_dim": int(pose_parts[0].shape[1]),
        "topology_target_dim": int(topology_parts[0].shape[1]),
        "heatmap_anchor_dim": int(heatmap_parts[0].shape[1]),
        "num_fingers": int(np.unique(expected_finger_id).size),
    }
    return (
        np.concatenate(pose_parts, axis=0),
        np.concatenate(topology_parts, axis=0),
        np.concatenate(heatmap_parts, axis=0),
        metadata,
    )


def _standardize(train: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-5)
    return (train - mean) / std, (value - mean) / std, mean.squeeze(0)


def _pca_reduce(train: np.ndarray, value: np.ndarray, *, components: int) -> tuple[np.ndarray, np.ndarray]:
    """Project poses to a train-fitted low-dimensional kinematic subspace."""
    if components <= 0:
        return train, value
    rank = min(int(components), train.shape[0] - 1, train.shape[1])
    if rank <= 0:
        return train, value
    # ``train`` has already been standardized with train-only statistics.
    # The validation pose never influences this PCA basis.
    _, _, right_vectors = np.linalg.svd(train, full_matrices=False)
    projection = right_vectors[:rank].T
    return train @ projection, value @ projection


def _knn_predict(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, *, k: int) -> np.ndarray:
    if not 1 <= k <= train_x.shape[0]:
        raise ValueError(f"k={k} must lie in [1, {train_x.shape[0]}].")
    predictions: list[np.ndarray] = []
    train_norm = np.sum(train_x * train_x, axis=1)
    chunk_size = 256
    for start in range(0, val_x.shape[0], chunk_size):
        query = val_x[start : start + chunk_size]
        distance_sq = np.maximum(
            np.sum(query * query, axis=1, keepdims=True) + train_norm[None, :] - 2.0 * query @ train_x.T,
            0.0,
        )
        nearest = np.argpartition(distance_sq, kth=k - 1, axis=1)[:, :k]
        nearest_distance = np.take_along_axis(distance_sq, nearest, axis=1)
        weight = 1.0 / np.maximum(nearest_distance, 1e-6)
        weight = weight / weight.sum(axis=1, keepdims=True)
        predictions.append(np.einsum("bk,bkd->bd", weight, train_y[nearest]))
    return np.concatenate(predictions, axis=0)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative.")
    x_train = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1)
    x_val = np.concatenate([val_x, np.ones((val_x.shape[0], 1), dtype=val_x.dtype)], axis=1)
    regularizer = np.eye(x_train.shape[1], dtype=np.float64) * float(ridge_lambda)
    regularizer[-1, -1] = 0.0
    weight = np.linalg.solve(
        x_train.T.astype(np.float64) @ x_train.astype(np.float64) + regularizer,
        x_train.T.astype(np.float64) @ train_y.astype(np.float64),
    )
    # Contact targets are probabilities. Clipping makes this a fair linear
    # probe rather than letting extrapolation dominate its MSE.
    return np.clip(x_val.astype(np.float64) @ weight, 0.0, 1.0).astype(np.float32)


def _metrics(prediction: np.ndarray, target: np.ndarray, *, positive_mask: np.ndarray | None = None) -> dict[str, float]:
    error = prediction - target
    flattened_prediction = prediction.reshape(-1)
    flattened_target = target.reshape(-1)
    if float(flattened_prediction.std()) < 1e-12 or float(flattened_target.std()) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flattened_prediction, flattened_target)[0, 1])
    result = {
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
        "pearson": correlation if np.isfinite(correlation) else 0.0,
    }
    if positive_mask is not None:
        if not bool(positive_mask.any()):
            raise ValueError("Positive-mask metric requested with no positive targets.")
        result["positive_mae"] = float(np.abs(error)[positive_mask].mean())
    return result


def _improvement(reference_mse: float, value_mse: float) -> float:
    return float(100.0 * (reference_mse - value_mse) / max(reference_mse, 1e-12))


def _run_probe(
    train_pose: np.ndarray,
    val_pose: np.ndarray,
    train_target: np.ndarray,
    val_target: np.ndarray,
    *,
    k: int,
    ridge_lambda: float,
    random_trials: int,
    seed: int,
    pose_pca_dim: int,
) -> dict[str, Any]:
    standard_train_pose, standard_val_pose, _ = _standardize(train_pose, val_pose)
    standard_train_pose, standard_val_pose = _pca_reduce(
        standard_train_pose,
        standard_val_pose,
        components=pose_pca_dim,
    )
    mean_prediction = np.broadcast_to(train_target.mean(axis=0, keepdims=True), val_target.shape).copy()
    positive_mask = val_target > 0.0
    result: dict[str, Any] = {
        "mean": _metrics(mean_prediction, val_target, positive_mask=positive_mask),
        "pose_knn": _metrics(
            _knn_predict(standard_train_pose, train_target, standard_val_pose, k=k),
            val_target,
            positive_mask=positive_mask,
        ),
        "pose_ridge": _metrics(
            _ridge_predict(
                standard_train_pose,
                train_target,
                standard_val_pose,
                ridge_lambda=ridge_lambda,
            ),
            val_target,
            positive_mask=positive_mask,
        ),
    }
    rng = np.random.default_rng(seed)
    random_metrics = []
    for _ in range(random_trials):
        random_prediction = train_target[rng.integers(0, train_target.shape[0], size=val_target.shape[0])]
        random_metrics.append(_metrics(random_prediction, val_target, positive_mask=positive_mask))
    result["random_match"] = {
        key: float(np.mean([item[key] for item in random_metrics]))
        for key in random_metrics[0]
    }
    result["random_match_std"] = {
        key: float(np.std([item[key] for item in random_metrics]))
        for key in random_metrics[0]
    }
    mean_mse = result["mean"]["mse"]
    for method in ("pose_knn", "pose_ridge"):
        result[method]["mse_improvement_vs_mean_percent"] = _improvement(mean_mse, result[method]["mse"])
    return result


def _file_identity(path: Path) -> tuple[str, str, str]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            str(np.asarray(payload["object_name"]).item()),
            str(np.asarray(payload["seq_id"]).item()),
            str(np.asarray(payload["side"]).item()),
        )


def _stable_choice_index(*, seed: int, key: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % size


def _weighted_method_summary(
    group_reports: list[dict[str, Any]],
    *,
    target_name: str,
) -> dict[str, Any]:
    methods = ("mean", "random_match", "pose_knn", "pose_ridge")
    total_frames = sum(int(item["num_val_frames"]) for item in group_reports)
    summary: dict[str, Any] = {
        "num_object_side_groups": len(group_reports),
        "num_val_frames": total_frames,
        "methods": {},
    }
    for method in methods:
        first = group_reports[0][target_name][method]
        summary["methods"][method] = {
            metric: float(
                sum(
                    item["num_val_frames"] * item[target_name][method][metric]
                    for item in group_reports
                )
                / total_frames
            )
            for metric in first
        }
    mean_mse = summary["methods"]["mean"]["mse"]
    for method in ("pose_knn", "pose_ridge"):
        summary["methods"][method]["mse_improvement_vs_object_mean_percent"] = _improvement(
            mean_mse,
            summary["methods"][method]["mse"],
        )
    summary["knn_beats_object_mean_mse_group_fraction"] = float(
        np.mean(
            [
                item[target_name]["pose_knn"]["mse"] < item[target_name]["mean"]["mse"]
                for item in group_reports
            ]
        )
    )
    return summary


def _run_object_conditional_probe(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    """Probe pose after conditioning the label distribution on object identity.

    For every (object, side) with at least two distinct sequences, one entire
    sequence is held out and all other sequences of that same object/side form
    its training pool. The baseline is therefore the object-conditional mean,
    not a global contact-rate prior.
    """
    object_side_sequences: dict[tuple[str, str], dict[str, list[Path]]] = {}
    for path in paths:
        object_name, seq_id, side = _file_identity(path)
        object_side_sequences.setdefault((object_name, side), {}).setdefault(seq_id, []).append(path)

    group_reports: list[dict[str, Any]] = []
    for (object_name, side), sequences in sorted(object_side_sequences.items()):
        sequence_ids = sorted(sequences)
        if len(sequence_ids) < 2:
            continue
        val_seq = sequence_ids[
            _stable_choice_index(
                seed=int(args.seed),
                key=f"{object_name}:{side}",
                size=len(sequence_ids),
            )
        ]
        val_paths = sequences[val_seq]
        train_paths = [
            path
            for seq_id, sequence_paths in sequences.items()
            if seq_id != val_seq
            for path in sequence_paths
        ]
        train_pose, train_topology, train_heatmap, train_meta = _load_samples(
            train_paths,
            anchors=int(args.anchors),
        )
        val_pose, val_topology, val_heatmap, val_meta = _load_samples(
            val_paths,
            anchors=int(args.anchors),
        )
        train_layout = {key: value for key, value in train_meta.items() if key != "num_frames"}
        val_layout = {key: value for key, value in val_meta.items() if key != "num_frames"}
        if train_layout != val_layout:
            raise ValueError(
                f"{object_name}/{side}: train/validation layouts differ: "
                f"{train_layout} vs {val_layout}."
            )
        group_reports.append(
            {
                "object_name": object_name,
                "side": side,
                "val_seq": val_seq,
                "train_sequences": [seq_id for seq_id in sequence_ids if seq_id != val_seq],
                "num_train_frames": int(train_pose.shape[0]),
                "num_val_frames": int(val_pose.shape[0]),
                "topology_probe": _run_probe(
                    train_pose,
                    val_pose,
                    train_topology,
                    val_topology,
                    k=min(int(args.knn_k), int(train_pose.shape[0])),
                    ridge_lambda=float(args.ridge_lambda),
                    random_trials=int(args.random_trials),
                    seed=int(args.seed) + len(group_reports) * 10 + 1,
                    pose_pca_dim=int(args.pose_pca_dim),
                ),
                "heatmap_anchor_probe": _run_probe(
                    train_pose,
                    val_pose,
                    train_heatmap,
                    val_heatmap,
                    k=min(int(args.knn_k), int(train_pose.shape[0])),
                    ridge_lambda=float(args.ridge_lambda),
                    random_trials=int(args.random_trials),
                    seed=int(args.seed) + len(group_reports) * 10 + 2,
                    pose_pca_dim=int(args.pose_pca_dim),
                ),
            }
        )
    if not group_reports:
        raise ValueError("No object-side pair has at least two distinct sequences.")
    return {
        "protocol": "object_conditional_sequence_holdout",
        "num_object_side_groups": len(group_reports),
        "topology_summary": _weighted_method_summary(group_reports, target_name="topology_probe"),
        "heatmap_anchor_summary": _weighted_method_summary(group_reports, target_name="heatmap_anchor_probe"),
        "groups": group_reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data_root).expanduser().resolve()
    paths = sorted(root.glob("**/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files under {root}.")
    if args.protocol == "object_conditional":
        return {
            "data_root": str(root),
            "seed": int(args.seed),
            "knn_k": int(args.knn_k),
            "ridge_lambda": float(args.ridge_lambda),
            "random_trials": int(args.random_trials),
            "pose_pca_dim": int(args.pose_pca_dim),
            "object_conditional": _run_object_conditional_probe(paths, args),
        }

    train_paths, val_paths, val_groups = _split_files_by_sequence(
        paths,
        root=root,
        val_split=float(args.val_split),
        seed=int(args.seed),
    )
    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        side_train_paths = [path for path in train_paths if _hand_side(path) == side]
        side_val_paths = [path for path in val_paths if _hand_side(path) == side]
        if not side_train_paths or not side_val_paths:
            continue
        train_pose, train_topology, train_heatmap, train_meta = _load_samples(
            side_train_paths,
            anchors=int(args.anchors),
        )
        val_pose, val_topology, val_heatmap, val_meta = _load_samples(
            side_val_paths,
            anchors=int(args.anchors),
        )
        train_layout = {key: value for key, value in train_meta.items() if key != "num_frames"}
        val_layout = {key: value for key, value in val_meta.items() if key != "num_frames"}
        if train_layout != val_layout:
            raise ValueError(f"{side}: train/validation layouts differ: {train_layout} vs {val_layout}.")
        by_side[side] = {
            "num_train_files": len(side_train_paths),
            "num_val_files": len(side_val_paths),
            "num_train_frames": int(train_pose.shape[0]),
            "num_val_frames": int(val_pose.shape[0]),
            "layout": train_layout,
            "topology_probe": _run_probe(
                train_pose,
                val_pose,
                train_topology,
                val_topology,
                k=int(args.knn_k),
                ridge_lambda=float(args.ridge_lambda),
                random_trials=int(args.random_trials),
                seed=int(args.seed) + (1 if side == "left" else 11),
                pose_pca_dim=int(args.pose_pca_dim),
            ),
            "heatmap_anchor_probe": _run_probe(
                train_pose,
                val_pose,
                train_heatmap,
                val_heatmap,
                k=int(args.knn_k),
                ridge_lambda=float(args.ridge_lambda),
                random_trials=int(args.random_trials),
                seed=int(args.seed) + (2 if side == "left" else 12),
                pose_pca_dim=int(args.pose_pca_dim),
            ),
        }
    if not by_side:
        raise ValueError("No valid left/right train-validation partitions were found.")
    report = {
        "data_root": str(root),
        "seed": int(args.seed),
        "val_split": float(args.val_split),
        "knn_k": int(args.knn_k),
        "ridge_lambda": float(args.ridge_lambda),
        "random_trials": int(args.random_trials),
        "pose_pca_dim": int(args.pose_pca_dim),
        "val_groups": val_groups,
        "num_train_files": len(train_paths),
        "num_val_files": len(val_paths),
        "by_side": by_side,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence-held-out H-only contact-prior probe.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol",
        choices=("global_h_only", "object_conditional"),
        default="global_h_only",
        help="global_h_only holds out arbitrary sequences; object_conditional holds out one sequence per object+side.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1000.0)
    parser.add_argument("--random-trials", type=int, default=32)
    parser.add_argument(
        "--pose-pca-dim",
        type=int,
        default=16,
        help="Train-fitted PCA pose components; use 0 to retain raw surface residuals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "object_conditional" in report:
        conditional = report["object_conditional"]
        for name in ("topology_summary", "heatmap_anchor_summary"):
            summary = conditional[name]
            methods = summary["methods"]
            print(
                f"object_conditional/{name} groups={summary['num_object_side_groups']} "
                f"mean_mse={methods['mean']['mse']:.6g} "
                f"knn_mse={methods['pose_knn']['mse']:.6g} "
                f"knn_gain={methods['pose_knn']['mse_improvement_vs_object_mean_percent']:.2f}% "
                f"ridge_mse={methods['pose_ridge']['mse']:.6g} "
                f"ridge_gain={methods['pose_ridge']['mse_improvement_vs_object_mean_percent']:.2f}%"
            )
    else:
        for side, side_report in report["by_side"].items():
            for name in ("topology_probe", "heatmap_anchor_probe"):
                probe = side_report[name]
                print(
                    f"{side}/{name} mean_mse={probe['mean']['mse']:.6g} "
                    f"knn_mse={probe['pose_knn']['mse']:.6g} "
                    f"knn_gain={probe['pose_knn']['mse_improvement_vs_mean_percent']:.2f}% "
                    f"ridge_mse={probe['pose_ridge']['mse']:.6g} "
                    f"ridge_gain={probe['pose_ridge']['mse_improvement_vs_mean_percent']:.2f}%"
                )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
