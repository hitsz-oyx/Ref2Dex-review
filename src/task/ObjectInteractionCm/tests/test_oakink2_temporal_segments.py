from __future__ import annotations

import numpy as np

from src.task.ObjectInteractionCm.tools.data.oakink2_temporal_segments import (
    TemporalSegmentationConfig,
    motion_evidence,
    segment_motion_then_contact,
)
from src.task.ObjectInteractionCm.research.oakink2_temporal_segmentation.run import (
    _official_30hz_timeline,
)


def _pose(x: float = 0.0, angle_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    value = np.eye(4, dtype=np.float32)
    value[:3, :3] = np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    value[0, 3] = x
    return value


def _trajectory(values: list[float]) -> dict[int, np.ndarray]:
    return {index: _pose(x=value) for index, value in enumerate(values)}


def test_no_reliable_motion_never_reads_distance() -> None:
    frames = list(range(15))
    stable = _trajectory([0.0] * len(frames))
    jitter = {index: _pose(angle_deg=(1.7 if index == 7 else 0.0)) for index in frames}
    calls = []

    result = segment_motion_then_contact(
        {"base": stable, "part": jitter},
        frames,
        lambda requested: calls.append(list(requested)) or {},
    )

    assert result["status"] in {"no_reliable_motion", "part_only_motion"}
    assert not result["distance_provider_called"]
    assert calls == []
    assert result["selected_frame_ids"] == []


def test_multi_part_consensus_rejects_part_only_motion() -> None:
    frames = list(range(17))
    stable = _trajectory([0.0] * len(frames))
    moving = _trajectory([index * 0.001 for index in frames])

    evidence = motion_evidence(
        {"base": stable, "articulated_part": moving},
        frames,
        TemporalSegmentationConfig(),
    )

    assert evidence["required_part_count"] == 2
    assert evidence["motion_components"] == []
    assert evidence["part_only_components"]


def test_motion_first_contact_expands_both_sides_and_keeps_strict_frames() -> None:
    frames = list(range(21))
    moving = _trajectory([index * 0.001 for index in frames])
    distances = {frame: 0.05 for frame in frames}
    for frame in range(2, 18):
        distances[frame] = 0.01
    distances[8] = 0.03
    distances[9] = 0.03
    calls = []

    result = segment_motion_then_contact(
        {"object": moving},
        frames,
        lambda requested: calls.append(list(requested)) or distances,
    )

    assert result["status"] == "selected"
    assert result["distance_provider_called"]
    assert len(calls) == 1
    assert result["selected_frame_ids"] == [
        *range(2, 8), *range(10, 18),
    ]
    assert result["selected_components"][0]["start_frame"] == 2
    assert result["selected_components"][0]["end_frame"] == 17


def test_contact_gap_larger_than_limit_splits_components() -> None:
    frames = list(range(25))
    moving = _trajectory([index * 0.001 for index in frames])
    distances = {frame: 0.05 for frame in frames}
    for frame in list(range(1, 8)) + list(range(11, 23)):
        distances[frame] = 0.01

    result = segment_motion_then_contact(
        {"object": moving}, frames, lambda _: distances,
    )

    assert len(result["selected_components"]) == 2
    assert result["selected_components"][0]["selected_frame_ids"] == list(range(1, 8))
    assert result["selected_components"][1]["selected_frame_ids"] == list(range(11, 23))


def test_motion_without_contact_seed_is_rejected() -> None:
    frames = list(range(17))
    moving = _trajectory([index * 0.001 for index in frames])
    result = segment_motion_then_contact(
        {"object": moving}, frames, lambda _: {frame: 0.03 for frame in frames},
    )
    assert result["status"] == "motion_without_2cm_seed"
    assert result["selected_frame_count"] == 0


def test_official_30hz_timeline_uses_annotation_alignment_not_fixed_modulo() -> None:
    mocap = np.arange(0, 52, dtype=np.int64)
    video_aligned = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50]

    frame_ids, positions = _official_30hz_timeline(mocap[3:49], video_aligned)

    np.testing.assert_array_equal(frame_ids, np.asarray([5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46]))
    np.testing.assert_array_equal(positions, np.arange(1, 12))
    assert 46 % 4 == 2


def test_irregular_raw_ids_use_contiguous_official_timeline_for_components() -> None:
    frames = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50, 54, 58, 62, 66]
    positions = list(range(len(frames)))
    poses = {frame: _pose(x=index * 0.001) for index, frame in enumerate(frames)}
    distances = {frame: 0.01 for frame in frames}

    result = segment_motion_then_contact(
        {"object": poses},
        frames,
        lambda _: distances,
        timeline_positions=positions,
    )

    assert result["status"] == "selected"
    assert result["motion_components"]
    assert result["selected_frame_ids"] == frames
