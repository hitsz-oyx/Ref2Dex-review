from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from process.GRAB.filter_cm_dominant_hand import choose_dominant_hand, coupling_error
from src.task.Cm.dataset import Stage4CmDataset


def _write_sequence(root: Path, name: str, *, hand_count: int = 4) -> Path:
    sequence_dir = root / "grab" / "s1" / name
    sequence_dir.mkdir(parents=True)
    frames, object_count = 12, 4096
    obj = np.zeros((frames, object_count, 3), dtype=np.float32)
    hand = np.zeros((frames, hand_count, 3), dtype=np.float32)
    normals_obj = np.zeros_like(obj)
    normals_obj[..., 2] = 1.0
    normals_hand = np.zeros_like(hand)
    normals_hand[..., 2] = 1.0
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (frames, 4, 4)).copy()
    np.savez(
        sequence_dir / "shared.npz",
        schema_name=np.asarray("ref2dex_cm_sequence_shared"),
        coordinate_frame=np.asarray("world"),
        ds_rate=np.asarray(4, dtype=np.int32),
        source_fps=np.asarray(120.0, dtype=np.float32),
        seq_id=np.asarray(f"s1/{name}"),
        raw_frame_id=np.arange(frames, dtype=np.int32) * 4,
        obj_points_world=obj,
        obj_normals_world=normals_obj,
        obj_point_id=np.arange(object_count, dtype=np.int32),
    )
    for side in ("left", "right"):
        np.savez(
            sequence_dir / f"{side}.npz",
            schema_name=np.asarray("ref2dex_cm_sequence_hand"),
            side=np.asarray(side),
            hand_points_world=hand,
            hand_normals_world=normals_hand,
            hand_root_pose_world=pose,
            obj_candidate_mask_5cm=np.ones((frames, object_count), dtype=bool),
        )
    return sequence_dir


def test_coupling_score_and_conservative_bilateral_decisions() -> None:
    object_current = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    object_future = object_current + np.asarray([0.01, 0, 0], dtype=np.float32)
    hand_current = object_current.copy()
    matching_hand = hand_current + np.asarray([0.01, 0, 0], dtype=np.float32)
    static_hand = hand_current.copy()
    candidate = np.ones(2, dtype=bool)
    assert coupling_error(object_current, object_future, hand_current, matching_hand, candidate) < 1e-7
    assert np.isclose(coupling_error(object_current, object_future, hand_current, static_hand, candidate), 0.01)

    common = dict(
        stride=1, dominance_ratio=2.0,
        min_error_gap_m_per_step=0.0005,
        max_main_error_m_per_step=0.005,
    )
    assert choose_dominant_hand(0.001, 0.010, **common)[0] == "left"
    assert choose_dominant_hand(0.001, 0.004, **common)[1] == "bilateral_cooperation"
    assert choose_dominant_hand(0.003, 0.004, **common)[0] is None
    assert choose_dominant_hand(0.007, 0.020, **common)[1] == "neither_explains_motion"


def test_dataset_manifest_fixes_side_frame_stride_and_respects_file_split(tmp_path: Path) -> None:
    data_root = tmp_path / "stage4"
    first = _write_sequence(data_root, "first")
    second = _write_sequence(data_root, "second")
    rows = [
        {"hand_path": "grab/s1/first/left.npz", "current_frame": 0, "stride": 1, "dominant_side": "left"},
        {"hand_path": "grab/s1/first/right.npz", "current_frame": 1, "stride": 5, "dominant_side": "right"},
        {"hand_path": "grab/s1/second/left.npz", "current_frame": 2, "stride": 1, "dominant_side": "left"},
    ]
    manifest = tmp_path / "dominant_hand.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first_files = [first / "left.npz", first / "right.npz"]

    dataset = Stage4CmDataset(
        data_root, file_list=first_files, num_hand_points=4, num_obj_points=8,
        min_stride=1, max_stride=5, dominant_hand_manifest=manifest,
    )
    assert len(dataset) == 2
    assert [(path.stem, frame) for path, frame in map(dataset.sample_location, range(2))] == [
        ("left", 0), ("right", 1),
    ]
    assert [int(dataset[index]["stride"]) for index in range(2)] == [1, 5]
    assert all(path.parent == first for path in dataset.file_paths)
    assert all(path.parent != second for path in dataset.file_paths)

    stride_five = Stage4CmDataset(
        data_root, file_list=first_files, num_hand_points=4, num_obj_points=8,
        min_stride=1, max_stride=5, fixed_stride=5, dominant_hand_manifest=manifest,
    )
    assert len(stride_five) == 1
    assert stride_five.sample_location(0) == (first / "right.npz", 1)
