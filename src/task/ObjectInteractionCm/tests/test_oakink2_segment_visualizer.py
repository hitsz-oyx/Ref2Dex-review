import numpy as np

from src.task.ObjectInteractionCm.research.oakink2_segment_visualizer.run import (
    SegmentRecord,
    _bounds_from_frame_def,
    _frame_ids_for_mode,
    _records_for_category,
    _transform_points,
)


def _record(*, reason="bilateral_multi_active_split", status="selected", new=True):
    return SegmentRecord(
        ordinal=0,
        status=status,
        new_in_v1_1=new,
        row={
            "sequence": "scene_01__A001++seq__abcdef__date",
            "frame_range_def": "((2, 5), (3, 6))",
            "selected_root_id": "root",
            "selected_object_ids": ["part"],
            "selected_object_name": "tool",
            "selected_frame_ids": [3, 5],
            "selected_frame_count": 2,
            "selection_reason": reason,
        },
    )


def test_frame_modes_keep_exact_selected_ids_and_primitive_union():
    record = _record()
    assert _bounds_from_frame_def(record.row["frame_range_def"]) == (2, 6)
    np.testing.assert_array_equal(
        _frame_ids_for_mode(record, range(1, 8), "selected"), np.asarray([3, 5])
    )
    np.testing.assert_array_equal(
        _frame_ids_for_mode(record, range(1, 8), "primitive"), np.arange(2, 7)
    )


def test_static_control_always_uses_primitive_frames():
    record = _record(status="rejected_static")
    record.row["selected_frame_ids"] = []
    np.testing.assert_array_equal(
        _frame_ids_for_mode(record, range(1, 8), "selected"), np.arange(2, 7)
    )


def test_world_transform_uses_row_vector_se3_convention():
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose[:3, 3] = [0.5, 1.0, 2.0]
    np.testing.assert_allclose(
        _transform_points(points, pose),
        np.asarray([[0.5, 2.0, 2.0], [-1.5, 1.0, 2.0]], dtype=np.float32),
    )


def test_category_filters_do_not_mix_static_controls_with_selected_set():
    bilateral = _record()
    override = _record(reason="semantic_object_override")
    override = SegmentRecord(1, override.row, override.status, override.new_in_v1_1)
    existing = _record(new=False)
    existing = SegmentRecord(2, existing.row, existing.status, existing.new_in_v1_1)
    static = _record(status="rejected_static")
    static = SegmentRecord(3, static.row, static.status, static.new_in_v1_1)
    records = [bilateral, override, existing, static]
    assert len(_records_for_category(records, "new")) == 2
    assert len(_records_for_category(records, "bilateral")) == 1
    assert len(_records_for_category(records, "override")) == 1
    assert len(_records_for_category(records, "static")) == 1
    assert len(_records_for_category(records, "all")) == 3
