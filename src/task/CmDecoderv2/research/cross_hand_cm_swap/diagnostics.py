"""源窗口、离线配对门槛与 parent 聚类统计；不消费未来目标状态。"""
from __future__ import annotations

import numpy as np
import torch

from src.task.CmDecoderv2.dataset import (
    _stable_seed, _points_world_to_frame, _normals_world_to_frame,
)
from src.task.CmDecoderv2.research.inspire_rollout_effect.run import _compact_knn_edge_stream


LIMITS = dict(effect_absolute_mm=1.0, effect_relative=0.25,
              contact_iou=0.25, contact_centroid_mm=20.0, moving_rms_mm=1.0,
              wrong_time_gap=20, seed=42, bootstrap_repeats=2000)
CONDITIONS = ("correct", "identity", "mano_sync", "mano_shift", "mano_transported")


def source_window(sequence, start):
    """与正式 loader 同 seed/采样；每个 flow 的两端都在该 transition 起点坐标。"""
    seed = _stable_seed(42, sequence.id, int(sequence.source_frame[start]), 0)
    frames = []
    for offset in range(4):
        t = start + offset
        pose = sequence.object_pose[t]
        chosen = np.random.default_rng(seed ^ _stable_seed("object", offset)).choice(
            sequence.object_points.shape[1], 1024, replace=False)
        obj = _points_world_to_frame(sequence.object_points[t, chosen], pose)
        hand = _points_world_to_frame(sequence.knn_hand_points[t], pose)
        future = _points_world_to_frame(sequence.knn_hand_points[t + 1], pose)
        fields = _compact_knn_edge_stream(
            object_points=obj, full_hand_points=hand,
            full_hand_normals=_normals_world_to_frame(sequence.knn_hand_normals[t], pose),
            full_hand_flow=future - hand, edge_global_ids=sequence.knn_edge_indices[t, chosen], radius_m=.02)
        fields.pop("global_hand_ids")
        fields.update(obj_points=obj,
                      obj_normals=_normals_world_to_frame(sequence.object_normals[t, chosen], pose),
                      obj_valid_mask=np.ones(1024, dtype=bool))
        frames.append(fields)
    out = {}
    variable = {"hand_points", "hand_normals", "hand_flow", "hand_valid_mask"}
    count = max(len(f["hand_points"]) for f in frames)
    for key in frames[0]:
        values = [f[key] for f in frames]
        if key in variable:
            array = np.zeros((4, count, *values[0].shape[1:]), dtype=values[0].dtype)
            for t, value in enumerate(values):
                array[t, :len(value)] = value
        else:
            array = np.stack(values)
        out[key] = torch.from_numpy(array)
    return out


def local_object_flow(points, poses):
    """Rigid object endpoint displacement, in each current canonical object frame."""
    relative = np.linalg.inv(poses[:-1].astype(np.float64)) @ poses[1:].astype(np.float64)
    return (points[None] @ relative[:, :3, :3].transpose(0, 2, 1)
            + relative[:, None, :3, 3] - points[None]) * 1000


def contact_comparison(points, first, second):
    union = (first | second).sum(-1)
    iou = (first & second).sum(-1) / np.maximum(union, 1)
    a = first @ points / np.maximum(first.sum(-1, keepdims=True), 1)
    b = second @ points / np.maximum(second.sum(-1, keepdims=True), 1)
    gap = np.linalg.norm(a - b, axis=-1) * 1000
    gap[(first.sum(-1) == 0) | (second.sum(-1) == 0)] = 1e6
    return iou, gap


def gates(bank):
    valid = bank["target_valid"].all(-1) & bank["mano_valid"].all(-1)
    limit = np.maximum(LIMITS["effect_absolute_mm"], LIMITS["effect_relative"] *
                       np.maximum(bank["actual_rms_mm"], bank["reference_rms_mm"]))
    effect = valid & (bank["effect_epe_mm"] <= limit).all(-1)
    strict = effect & (bank["contact_iou"] >= LIMITS["contact_iou"]).all(-1)
    strict &= (bank["contact_centroid_mm"] <= LIMITS["contact_centroid_mm"]).all(-1)
    return dict(valid=valid, effect=effect, strict=strict,
                strict_moving=strict & (bank["actual_rms_mm"].mean(-1) >= LIMITS["moving_rms_mm"]))


def shifted_donors(sequence, frame, valid):
    donors = np.full(len(frame), -1, dtype=np.int64)
    for i in range(len(frame)):
        candidates = np.flatnonzero((sequence == sequence[i]) & valid &
                                    (np.abs(frame - frame[i]) >= LIMITS["wrong_time_gap"]))
        if len(candidates):
            donors[i] = np.random.default_rng(_stable_seed(42, int(sequence[i]), int(frame[i]))).choice(candidates)
    return donors


def paired_statistics(values, baseline, sequence):
    """对同一 recipient 配对；bootstrap 保留整个 parent，含其共享错时供体。"""
    if not len(values):
        return None
    values, baseline = np.asarray(values), np.asarray(baseline)
    unique = np.unique(sequence)
    n = np.array([(sequence == s).sum() for s in unique])
    sums = np.array([(values - baseline)[sequence == s].sum() for s in unique])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(unique), (LIMITS["bootstrap_repeats"], len(unique)))
    boot = sums[draws].sum(-1) / n[draws].sum(-1)
    return dict(windows=len(values), parents=len(unique), micro=float(values.mean()),
                paired_baseline_micro=float(baseline.mean()), delta=float((values-baseline).mean()),
                delta_ci95=np.quantile(boot, [.025, .975]).tolist(),
                macro=float(np.mean([values[sequence == s].mean() for s in unique])),
                macro_delta=float((sums/n).mean()))


def summarize(pair_bank, metric_arrays):
    selection = gates(pair_bank)
    summaries = {}
    for group, mask in selection.items():
        summaries[group] = {}
        for c in CONDITIONS:
            usable = mask & metric_arrays[c]["available"]
            result = {}
            for key in ("hand_epe_mm", "q_mae_rad", "wrist_translation_mm", "wrist_rotation_deg", "output_change_mm"):
                result[key] = paired_statistics(metric_arrays[c][key][usable],
                    metric_arrays["correct"][key][usable], pair_bank["sequence_number"][usable])
            result["gain_vs_identity_mm"] = paired_statistics(
                metric_arrays["identity"]["hand_epe_mm"][usable]-metric_arrays[c]["hand_epe_mm"][usable],
                np.zeros(usable.sum()), pair_bank["sequence_number"][usable])
            summaries[group][c] = result
    return summaries
