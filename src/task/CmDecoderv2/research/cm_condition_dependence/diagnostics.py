"""Predeclared matching and paired statistics; no learned models or target GT."""
from __future__ import annotations

import numpy as np


CONDITIONS = ("correct", "identity", "zero_all", "swap", "matched_swap")
MATCH_LIMITS = dict(min_frame_gap=20, translation_m=0.03, rotation_deg=30.0,
                    q_rms_rad=0.25, active_fraction=0.10,
                    flow_rms_ratio_min=0.5, flow_rms_ratio_max=2.0,
                    mean_flow_difference_min_mm=1.0, mean_flow_difference_relative=0.5)


def rotation_difference_deg(first, second):
    relative = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def pair_differences(bank, recipient, candidates):
    candidates = np.asarray(candidates, dtype=np.int64)
    recipient_pose = np.linalg.inv(bank["object_pose"][recipient]) @ bank["current_wrist"][recipient]
    donor_pose = np.linalg.inv(bank["object_pose"][candidates]) @ bank["current_wrist"][candidates]
    translation = np.linalg.norm(donor_pose[:, :3, 3] - recipient_pose[:3, 3], axis=-1)
    rotation = rotation_difference_deg(recipient_pose[:3, :3], donor_pose[:, :3, :3])
    q = np.sqrt(np.mean((bank["current_q"][candidates] - bank["current_q"][recipient]) ** 2, axis=-1))
    contact = np.abs(bank["active_fraction"][candidates] - bank["active_fraction"][recipient])
    rms = bank["flow_rms_mm"][candidates] / max(float(bank["flow_rms_mm"][recipient]), 1e-8)
    motion = np.linalg.norm(bank["flow_mean_mm"][candidates] - bank["flow_mean_mm"][recipient], axis=-1)
    return dict(translation_m=translation, rotation_deg=rotation, q_rms_rad=q,
                active_fraction=contact, flow_rms_ratio=rms, mean_flow_difference_mm=motion)


def select_donors(bank, *, eligible=None, seed=42):
    """Return real same-sequence windows; -1 means there is no qualifying donor.

    Only present state, input hand motion, validity and sample IDs are read.
    Targets, model predictions and errors cannot affect donor selection.
    """
    count = len(bank["start_frame"])
    valid = np.asarray(bank["cm_valid"], dtype=bool).all(axis=1)
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)
    broad = np.full(count, -1, dtype=np.int64)
    matched = np.full(count, -1, dtype=np.int64)
    available = np.zeros((count, 2), dtype=np.int64)
    for i in np.flatnonzero(valid):
        candidates = np.flatnonzero(valid & (bank["sequence_number"] == bank["sequence_number"][i]) &
                                   (np.abs(bank["start_frame"] - bank["start_frame"][i]) >= MATCH_LIMITS["min_frame_gap"]))
        available[i, 0] = len(candidates)
        if not len(candidates):
            continue
        # Per-row RNG makes selection stable when other sequences are omitted in smoke.
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(bank["sequence_number"][i]), int(bank["start_frame"][i])]))
        broad[i] = int(rng.choice(candidates))
        diff = pair_differences(bank, i, candidates)
        threshold = max(MATCH_LIMITS["mean_flow_difference_min_mm"],
                        MATCH_LIMITS["mean_flow_difference_relative"] * float(np.linalg.norm(bank["flow_mean_mm"][i])))
        mask = ((diff["translation_m"] <= MATCH_LIMITS["translation_m"]) &
                (diff["rotation_deg"] <= MATCH_LIMITS["rotation_deg"]) &
                (diff["q_rms_rad"] <= MATCH_LIMITS["q_rms_rad"]) &
                (diff["active_fraction"] <= MATCH_LIMITS["active_fraction"]) &
                (diff["flow_rms_ratio"] >= MATCH_LIMITS["flow_rms_ratio_min"]) &
                (diff["flow_rms_ratio"] <= MATCH_LIMITS["flow_rms_ratio_max"]) &
                (diff["mean_flow_difference_mm"] >= threshold))
        available[i, 1] = int(mask.sum())
        if mask.any():
            score = sum((diff[key] / MATCH_LIMITS[key]) ** 2
                        for key in ("translation_m", "rotation_deg", "q_rms_rad", "active_fraction"))
            matched[i] = int(candidates[np.flatnonzero(mask)[np.argmin(score[mask])]])
    return {"swap": broad, "matched_swap": matched, "candidate_counts": available}


def continuous_starts(bank, steps=16):
    lookup = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(bank["sequence_number"], bank["start_frame"]))}
    valid = np.asarray(bank["cm_valid"], dtype=bool).all(axis=1)
    eligible = np.zeros(len(valid), dtype=bool)
    for i in np.flatnonzero(valid):
        sequence, start = int(bank["sequence_number"][i]), int(bank["start_frame"][i])
        eligible[i] = all((sequence, start + h) in lookup and valid[lookup[sequence, start + h]] for h in range(steps))
    return eligible, lookup


def spread_starts(bank, eligible, maximum=3, gap=16):
    selected = []
    for sequence in np.unique(bank["sequence_number"]):
        candidates = np.flatnonzero(eligible & (bank["sequence_number"] == sequence))
        # Time-spaced quantiles fixed before observing any model errors.
        chosen = []
        for position in np.linspace(0, max(len(candidates) - 1, 0), maximum).round().astype(int):
            if not len(candidates):
                break
            index = int(candidates[position])
            if all(abs(int(bank["start_frame"][index]) - int(bank["start_frame"][j])) >= gap for j in chosen):
                chosen.append(index)
        selected.extend(chosen)
    return selected


def paired_summary(rows, value_key="hand_epe_mm", repeats=2000, seed=42):
    """Micro/macro estimates on each control's SAME recipients, clustered by sequence."""
    result = {}
    lookup = {(r["sample_id"], r["condition"]): r for r in rows}
    for condition in CONDITIONS:
        members = [r for r in rows if r["condition"] == condition]
        if not members:
            result[condition] = None
            continue
        groups = {}
        for r in members:
            correct = lookup[r["sample_id"], "correct"]
            identity = lookup[r["sample_id"], "identity"]
            values = [r[value_key], correct[value_key], identity[value_key],
                      r[value_key] - correct[value_key], identity[value_key] - r[value_key]]
            groups.setdefault(r["sequence_id"], []).append(values)
        keys = ("epe_mm", "paired_correct_epe_mm", "paired_identity_epe_mm",
                "penalty_vs_correct_mm", "gain_over_identity_mm")
        sums = np.asarray([np.sum(g, axis=0) for g in groups.values()], dtype=np.float64)
        counts = np.asarray([len(g) for g in groups.values()])
        macro = (sums / counts[:, None]).mean(axis=0)
        micro = sums.sum(axis=0) / counts.sum()
        rng = np.random.default_rng(seed)
        draws = rng.integers(len(counts), size=(repeats, len(counts)))
        estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
        result[condition] = dict(samples=len(members), sequences=len(groups),
                                 frame_micro=dict(zip(keys, map(float, micro))),
                                 sequence_macro=dict(zip(keys, map(float, macro))),
                                 micro_ci95={k: np.quantile(estimates[:, j], [.025, .975]).tolist() for j, k in enumerate(keys)})
    return result
