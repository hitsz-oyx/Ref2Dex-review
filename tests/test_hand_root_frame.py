"""Tests for the hand-root coordinate frame support.

Covers three layers:
  1. Stage 2 ``pack_stage2_hand`` passes through ``hand_root_pose`` if the
     preprocessor supplied it; legacy Stage 2 payloads without the field
     still build (no spurious requirement).
  2. Stage 3 ``build_stage3_sequence`` with ``coordinate_frame="hand_root"``
     correctly transforms world -> hand_root, and ``coordinate_frame="object"``
     still produces the legacy schema (with ``T_world_from_obj``).
  3. Distance / KNN fields are frame-invariant (equal under both frames).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process.stage3 import prepare_corr_static as stage3_mod  # noqa: E402
from process.stage3.prepare_corr_static import (  # noqa: E402
    _normals_world_to_hand_root,
    _points_world_to_hand_root,
)
from process.common.stage2 import pack_stage2_hand  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic Stage 2 payload
# ---------------------------------------------------------------------------


def _axis_angle(rot_axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a (3, 3) rotation matrix from a unit axis and an angle (rad)."""
    axis = rot_axis / np.linalg.norm(rot_axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float32,
    )


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def _make_minimal_source(
    T: int,
    *,
    num_obj: int,
    num_hand: int,
    with_hand_root_pose: bool,
) -> dict:
    """Build a source dict shaped like process/ARCTIC/raw.py output."""
    rng = np.random.default_rng(0)
    obj_pts = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals /= np.clip(np.linalg.norm(obj_normals, axis=-1, keepdims=True), 1e-8, None)
    hand_pts = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals /= np.clip(np.linalg.norm(hand_normals, axis=-1, keepdims=True), 1e-8, None)
    obj_root_pose = np.stack(
        [_make_se3(np.eye(3, dtype=np.float32), rng.standard_normal(3).astype(np.float32)) for _ in range(T)]
    )
    out = {
        "dataset_name": "synthetic",
        "seq_id": "s1/synthetic_seq",
        "subject_id": "s1",
        "seq_name": "synthetic_seq",
        "object_name": "synth",
        "raw_frame_id": np.arange(T, dtype=np.int32),
        "obj_points_world": obj_pts,
        "obj_normals_world": obj_normals,
        "obj_point_id": np.arange(num_obj, dtype=np.int32),
        "obj_root_pose": obj_root_pose,
        "right_hand_points_world": hand_pts,
        "right_hand_normals_world": hand_normals,
        "right_hand_point_id": np.arange(num_hand, dtype=np.int32),
        "right_hand_cano_points": rng.standard_normal((num_hand, 3)).astype(np.float32),
        "right_hand_finger_id": np.zeros((num_hand,), dtype=np.int32),
        "right_hand_region_id": np.zeros((num_hand,), dtype=np.int32),
        "right_hand_to_obj_nn_id": np.zeros((T, num_hand), dtype=np.int32),
        "right_hand_min_dist_to_obj": np.zeros((T,), dtype=np.float32),
        "left_hand_points_world": hand_pts.copy(),
        "left_hand_normals_world": hand_normals.copy(),
        "left_hand_point_id": np.arange(num_hand, dtype=np.int32),
        "left_hand_cano_points": rng.standard_normal((num_hand, 3)).astype(np.float32),
        "left_hand_finger_id": np.zeros((num_hand,), dtype=np.int32),
        "left_hand_region_id": np.zeros((num_hand,), dtype=np.int32),
        "left_hand_to_obj_nn_id": np.zeros((T, num_hand), dtype=np.int32),
        "left_hand_min_dist_to_obj": np.zeros((T,), dtype=np.float32),
    }
    if with_hand_root_pose:
        # Use a non-trivial rotation + translation per frame
        R = _axis_angle(np.array([0.0, 0.0, 1.0]), 0.3)  # 17° about z
        out["right_hand_root_pose"] = np.stack(
            [_make_se3(R, np.array([0.1, 0.2, 0.3], dtype=np.float32)) for _ in range(T)]
        )
        out["left_hand_root_pose"] = out["right_hand_root_pose"].copy()
    return out


# ---------------------------------------------------------------------------
# Stage 2: pack_stage2_hand
# ---------------------------------------------------------------------------


def test_stage2_passes_through_hand_root_pose_when_present() -> None:
    source = _make_minimal_source(T=3, num_obj=8, num_hand=5, with_hand_root_pose=True)
    payload = pack_stage2_hand(
        source,
        side="right",
        source_raw_file="s1/synthetic_seq.mano.npy",
        processing_mode="init_only",
        frame_keep_threshold=1.0,
        config={},
    )
    assert payload is not None
    assert "hand_root_pose" in payload, "hand_root_pose must be propagated when source provides it"
    assert payload["hand_root_pose"].shape == (3, 4, 4)
    # After [keep] filtering the pose is preserved.
    assert np.allclose(payload["hand_root_pose"], source["right_hand_root_pose"])


def test_stage2_skips_hand_root_pose_when_absent() -> None:
    source = _make_minimal_source(T=3, num_obj=8, num_hand=5, with_hand_root_pose=False)
    payload = pack_stage2_hand(
        source,
        side="right",
        source_raw_file="s1/synthetic_seq.mano.npy",
        processing_mode="init_only",
        frame_keep_threshold=1.0,
        config={},
    )
    assert payload is not None
    assert "hand_root_pose" not in payload, "legacy payloads must remain valid"


# ---------------------------------------------------------------------------
# Stage 3: hand_root transformation math
# ---------------------------------------------------------------------------


def test_hand_root_round_trip_transform() -> None:
    """For a single hand_root pose, world -> hand_root -> world must be identity."""
    T = 4
    rng = np.random.default_rng(1)
    R = _axis_angle(np.array([0.0, 1.0, 0.0]), 0.4)
    t = rng.standard_normal(3).astype(np.float32)
    hand_root_poses = np.stack([_make_se3(R, t) for _ in range(T)])

    pts_world = rng.standard_normal((T, 7, 3)).astype(np.float32)
    pts_hand = _points_world_to_hand_root(pts_world, hand_root_poses)

    # Back: world = R @ pts_hand + t
    pts_back = np.einsum("tij,tnj->tni", hand_root_poses[:, :3, :3], pts_hand) + hand_root_poses[:, None, :3, 3]
    assert np.allclose(pts_back, pts_world, atol=1e-5), "world -> hand_root -> world is not identity"

    # Wrist itself: in hand_root frame it should be at origin.
    wrist_in_hand = _points_world_to_hand_root(t[None, None, :], hand_root_poses[:1])
    assert np.allclose(wrist_in_hand, np.zeros_like(wrist_in_hand), atol=1e-5)


def test_normals_hand_root_pure_rotation() -> None:
    """Normal transformation must be pure rotation, no translation; and remain unit length."""
    T = 3
    R = _axis_angle(np.array([1.0, 0.0, 0.0]), 0.7)
    t = np.array([5.0, -2.0, 0.5], dtype=np.float32)
    hand_root_poses = np.stack([_make_se3(R, t) for _ in range(T)])
    rng = np.random.default_rng(2)
    n_world = rng.standard_normal((T, 6, 3)).astype(np.float32)
    n_world /= np.linalg.norm(n_world, axis=-1, keepdims=True)
    n_hand = _normals_world_to_hand_root(n_world, hand_root_poses)
    norms = np.linalg.norm(n_hand, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), "hand_root normals must remain unit length"


# ---------------------------------------------------------------------------
# Stage 3: build_stage3_sequence with both coordinate frames
# ---------------------------------------------------------------------------


def _build_payload_for_stage3(num_obj: int = 16, num_hand: int = 6, T: int = 2) -> dict:
    """Build a Stage 2-shaped payload that satisfies `_validate_stage2_geometry`."""
    rng = np.random.default_rng(3)
    obj_pts = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals /= np.clip(np.linalg.norm(obj_normals, axis=-1, keepdims=True), 1e-8, None)
    hand_pts = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals /= np.clip(np.linalg.norm(hand_normals, axis=-1, keepdims=True), 1e-8, None)
    obj_root_pose = np.stack(
        [_make_se3(np.eye(3, dtype=np.float32), rng.standard_normal(3).astype(np.float32)) for _ in range(T)]
    )
    R = _axis_angle(np.array([0.0, 0.0, 1.0]), 0.0)  # identity rotation
    t = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    hand_root_poses = np.stack([_make_se3(R, t) for _ in range(T)])
    return {
        "schema_name": "ref2dex_opti",
        "schema_version": "1.0.0",
        "processing_mode": "init_only",
        "dataset_name": "synthetic",
        "seq_id": "s1/synthetic_seq",
        "subject_id": "s1",
        "seq_name": "synthetic_seq",
        "object_name": "synth",
        "side": "right",
        "raw_frame_id": np.arange(T, dtype=np.int32),
        "stage2_frame_idx": np.arange(T, dtype=np.int32),
        "obj_points_world": obj_pts,
        "obj_normals_world": obj_normals,
        "obj_point_id": np.arange(num_obj, dtype=np.int32),
        "obj_root_pose": obj_root_pose,
        "hand_points_world": hand_pts,
        "hand_normals_world": hand_normals,
        "hand_point_id": np.arange(num_hand, dtype=np.int32),
        "hand_cano_points": rng.standard_normal((num_hand, 3)).astype(np.float32),
        "hand_finger_id": np.zeros((num_hand,), dtype=np.int32),
        "hand_region_id": np.zeros((num_hand,), dtype=np.int32),
        "hand_root_pose": hand_root_poses,
    }


def test_stage3_object_frame_keeps_legacy_schema() -> None:
    payload = _build_payload_for_stage3()
    out = stage3_mod.build_stage3_sequence(
        payload,
        source_path=Path("/tmp/synthetic.pkl"),
        stage2_root=Path("/tmp"),
        num_obj_pool=16,
        num_hand_points=6,
        k_cross=4,
        candidate_threshold=0.05,
        frame_batch_size=1,
        device=torch.device("cpu"),
        mirror_left_to_right=False,
        coordinate_frame="object",
    )
    assert "T_world_from_obj" in out
    assert "T_world_from_hand_root" not in out
    assert str(out["coordinate_frame"].item()) == "object"


def test_stage3_hand_root_frame_emits_pose_and_label() -> None:
    payload = _build_payload_for_stage3()
    out = stage3_mod.build_stage3_sequence(
        payload,
        source_path=Path("/tmp/synthetic.pkl"),
        stage2_root=Path("/tmp"),
        num_obj_pool=16,
        num_hand_points=6,
        k_cross=4,
        candidate_threshold=0.05,
        frame_batch_size=1,
        device=torch.device("cpu"),
        mirror_left_to_right=False,
        coordinate_frame="hand_root",
    )
    assert "T_world_from_hand_root" in out
    assert "T_world_from_obj" not in out
    assert str(out["coordinate_frame"].item()) == "hand_root"
    # Pose shape sanity
    assert out["T_world_from_hand_root"].shape == (2, 4, 4)


def test_stage3_hand_root_requires_hand_root_pose_in_payload() -> None:
    payload = _build_payload_for_stage3()
    del payload["hand_root_pose"]
    with pytest.raises(KeyError, match="hand_root_pose"):
        stage3_mod.build_stage3_sequence(
            payload,
            source_path=Path("/tmp/synthetic.pkl"),
            stage2_root=Path("/tmp"),
            num_obj_pool=16,
            num_hand_points=6,
            k_cross=4,
            candidate_threshold=0.05,
            frame_batch_size=1,
            device=torch.device("cpu"),
            mirror_left_to_right=False,
            coordinate_frame="hand_root",
        )


def test_stage3_distances_are_frame_invariant() -> None:
    """obj_to_hand_min_dist, candidate mask and KNN index must be identical
    under object frame and hand_root frame (rigid SE(3) preserves distances)."""
    payload = _build_payload_for_stage3(num_obj=32, num_hand=10, T=2)
    common_kwargs = dict(
        source_path=Path("/tmp/synthetic.pkl"),
        stage2_root=Path("/tmp"),
        num_obj_pool=32,
        num_hand_points=10,
        k_cross=4,
        candidate_threshold=1.0,  # accept all
        frame_batch_size=1,
        device=torch.device("cpu"),
        mirror_left_to_right=False,
    )
    obj_out = stage3_mod.build_stage3_sequence(payload, coordinate_frame="object", **common_kwargs)
    hand_out = stage3_mod.build_stage3_sequence(payload, coordinate_frame="hand_root", **common_kwargs)
    np.testing.assert_allclose(obj_out["obj_to_hand_min_dist"], hand_out["obj_to_hand_min_dist"], atol=1e-5)
    np.testing.assert_array_equal(obj_out["obj_candidate_mask_5cm"], hand_out["obj_candidate_mask_5cm"])
    np.testing.assert_array_equal(obj_out["gt_obj_to_hand_knn_idx"], hand_out["gt_obj_to_hand_knn_idx"])


def test_stage3_rejects_invalid_coordinate_frame() -> None:
    payload = _build_payload_for_stage3()
    with pytest.raises(ValueError, match="coordinate_frame"):
        stage3_mod.build_stage3_sequence(
            payload,
            source_path=Path("/tmp/synthetic.pkl"),
            stage2_root=Path("/tmp"),
            num_obj_pool=16,
            num_hand_points=6,
            k_cross=4,
            candidate_threshold=0.05,
            frame_batch_size=1,
            device=torch.device("cpu"),
            mirror_left_to_right=False,
            coordinate_frame="banana",
        )
