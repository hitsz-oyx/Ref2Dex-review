"""Temporal motion-first segmentation helpers for the OakInk2 pilot.

The module deliberately does not replace the v1/v1.1 selector.  It first
derives robust motion components from object SE(3) trajectories and only then
asks a caller-provided distance source for cached hand/object distances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalSegmentationConfig:
    window_radius: int = 3
    window_translation_m: float = 0.002
    window_rotation_deg: float = 2.0
    part_consensus_ratio: float = 0.60
    motion_max_gap: int = 1
    motion_min_evidence_frames: int = 3
    contact_threshold_m: float = 0.02
    contact_max_gap: int = 2

    def validate(self) -> None:
        if self.window_radius < 1:
            raise ValueError("window_radius must be positive")
        if self.window_translation_m <= 0.0 or self.window_rotation_deg <= 0.0:
            raise ValueError("motion thresholds must be positive")
        if not 0.0 < self.part_consensus_ratio <= 1.0:
            raise ValueError("part_consensus_ratio must be in (0, 1]")
        if self.motion_max_gap < 0 or self.contact_max_gap < 0:
            raise ValueError("gap limits must be non-negative")
        if self.motion_min_evidence_frames < 1:
            raise ValueError("motion_min_evidence_frames must be positive")
        if self.contact_threshold_m <= 0.0:
            raise ValueError("contact_threshold_m must be positive")


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _components(
    frame_ids: Sequence[int],
    mask: Sequence[bool],
    *,
    max_gap: int,
    min_evidence_frames: int,
    timeline_positions: Sequence[int] | None = None,
) -> list[dict]:
    frames = [int(value) for value in frame_ids]
    positions = frames if timeline_positions is None else [int(value) for value in timeline_positions]
    values = np.asarray(mask, dtype=bool)
    if values.shape != (len(frames),):
        raise ValueError(f"mask shape {values.shape} does not match {len(frames)} frames")
    if len(positions) != len(frames):
        raise ValueError("timeline_positions must match frame_ids")
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("timeline_positions must be strictly increasing")
    true_indices = np.flatnonzero(values).tolist()
    if not true_indices:
        return []
    groups: list[list[int]] = [[true_indices[0]]]
    for index in true_indices[1:]:
        previous = groups[-1][-1]
        index_gap = index - previous - 1
        timeline_gap = positions[index] - positions[previous] - 1
        if index_gap <= max_gap and timeline_gap == index_gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [
        {
            "start_frame": frames[group[0]],
            "end_frame": frames[group[-1]],
            "evidence_frame_ids": [frames[index] for index in group],
            "evidence_frame_count": len(group),
        }
        for group in groups
        if len(group) >= min_evidence_frames
    ]


def motion_evidence(
    pose_by_part: Mapping[str, Mapping[int, np.ndarray]],
    frame_ids: Sequence[int],
    config: TemporalSegmentationConfig,
    *,
    timeline_positions: Sequence[int] | None = None,
) -> dict:
    """Return centered-window per-part, root-consensus, and part-only motion."""

    config.validate()
    frames = [int(value) for value in frame_ids]
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise ValueError("frame_ids must be strictly increasing")
    positions = frames if timeline_positions is None else [int(value) for value in timeline_positions]
    if len(positions) != len(frames):
        raise ValueError("timeline_positions must match frame_ids")
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("timeline_positions must be strictly increasing")
    parts = sorted(str(value) for value in pose_by_part)
    if not parts:
        return {
            "part_ids": [],
            "required_part_count": 0,
            "part_evidence": {},
            "root_mask": [False] * len(frames),
            "part_only_mask": [False] * len(frames),
            "motion_components": [],
            "part_only_components": [],
        }

    radius = config.window_radius
    part_evidence: dict[str, dict] = {}
    active_matrix = np.zeros((len(parts), len(frames)), dtype=bool)
    for part_index, part_id in enumerate(parts):
        poses = pose_by_part[part_id]
        translations = np.full(len(frames), np.nan, dtype=np.float64)
        rotations = np.full(len(frames), np.nan, dtype=np.float64)
        for center in range(radius, len(frames) - radius):
            left_frame = frames[center - radius]
            right_frame = frames[center + radius]
            if left_frame not in poses or right_frame not in poses:
                continue
            left = np.asarray(poses[left_frame], dtype=np.float64)
            right = np.asarray(poses[right_frame], dtype=np.float64)
            if left.shape != (4, 4) or right.shape != (4, 4):
                raise ValueError(f"invalid pose shape for {part_id}")
            translations[center] = float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
            rotations[center] = _rotation_angle_deg(left[:3, :3], right[:3, :3])
            active_matrix[part_index, center] = bool(
                translations[center] >= config.window_translation_m
                or rotations[center] >= config.window_rotation_deg
            )
        part_evidence[part_id] = {
            "translation_m": [None if not np.isfinite(value) else float(value) for value in translations],
            "rotation_deg": [None if not np.isfinite(value) else float(value) for value in rotations],
            "active_frame_ids": [frames[index] for index in np.flatnonzero(active_matrix[part_index])],
        }

    required = max(1, int(math.ceil(len(parts) * config.part_consensus_ratio)))
    root_mask = active_matrix.sum(axis=0) >= required
    any_part_mask = active_matrix.any(axis=0)
    part_only_mask = np.logical_and(any_part_mask, ~root_mask)
    motion_components = _components(
        frames,
        root_mask,
        max_gap=config.motion_max_gap,
        min_evidence_frames=config.motion_min_evidence_frames,
        timeline_positions=positions,
    )
    part_only_components = _components(
        frames,
        part_only_mask,
        max_gap=config.motion_max_gap,
        min_evidence_frames=config.motion_min_evidence_frames,
        timeline_positions=positions,
    )
    return {
        "part_ids": parts,
        "required_part_count": required,
        "part_evidence": part_evidence,
        "root_mask": root_mask.tolist(),
        "part_only_mask": part_only_mask.tolist(),
        "motion_components": motion_components,
        "part_only_components": part_only_components,
    }


def _contact_components(
    frame_ids: Sequence[int],
    distances: Mapping[int, float],
    config: TemporalSegmentationConfig,
    *,
    timeline_positions: Sequence[int] | None = None,
) -> list[dict]:
    contact_mask = [
        float(distances.get(int(frame), float("inf"))) < config.contact_threshold_m
        for frame in frame_ids
    ]
    components = _components(
        frame_ids,
        contact_mask,
        max_gap=config.contact_max_gap,
        min_evidence_frames=1,
        timeline_positions=timeline_positions,
    )
    for component in components:
        selected = component.pop("evidence_frame_ids")
        component["selected_frame_ids"] = selected
        component["selected_frame_count"] = len(selected)
        component["min_distance_m"] = min(float(distances[frame]) for frame in selected)
    return components


def segment_motion_then_contact(
    pose_by_part: Mapping[str, Mapping[int, np.ndarray]],
    frame_ids: Sequence[int],
    distance_provider: Callable[[Sequence[int]], Mapping[int, float]],
    config: TemporalSegmentationConfig | None = None,
    *,
    timeline_positions: Sequence[int] | None = None,
) -> dict:
    """Segment reliable root motion, then expand cached 2 cm contact around it.

    ``distance_provider`` is never called when no reliable root-motion
    component exists.  This makes the motion-first computation boundary
    explicit and testable.
    """

    config = config or TemporalSegmentationConfig()
    config.validate()
    frames = [int(value) for value in frame_ids]
    evidence = motion_evidence(
        pose_by_part,
        frames,
        config,
        timeline_positions=timeline_positions,
    )
    motion_components = evidence["motion_components"]
    if not motion_components:
        return {
            "config": asdict(config),
            "distance_provider_called": False,
            **evidence,
            "contact_components": [],
            "selected_components": [],
            "selected_frame_ids": [],
            "selected_frame_count": 0,
            "status": "part_only_motion" if evidence["part_only_components"] else "no_reliable_motion",
        }

    distances = {int(key): float(value) for key, value in distance_provider(frames).items()}
    contacts = _contact_components(
        frames,
        distances,
        config,
        timeline_positions=timeline_positions,
    )
    selected_components = []
    for contact in contacts:
        selected = contact["selected_frame_ids"]
        overlapping_motion = [
            motion
            for motion in motion_components
            if any(motion["start_frame"] <= frame <= motion["end_frame"] for frame in selected)
        ]
        if not overlapping_motion:
            continue
        selected_components.append({
            **contact,
            "motion_component_ranges": [
                [motion["start_frame"], motion["end_frame"]] for motion in overlapping_motion
            ],
        })
    selected_frames = sorted({
        frame for component in selected_components for frame in component["selected_frame_ids"]
    })
    return {
        "config": asdict(config),
        "distance_provider_called": True,
        **evidence,
        "contact_components": contacts,
        "selected_components": selected_components,
        "selected_frame_ids": selected_frames,
        "selected_frame_count": len(selected_frames),
        "status": "selected" if selected_frames else "motion_without_2cm_seed",
    }
