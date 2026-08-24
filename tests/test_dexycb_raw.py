from __future__ import annotations

import numpy as np
import pytest
import torch

from process.DexYCB.raw import (
    build_SE3,
    expand_mano_pca_pose,
    find_reference_camera,
    quat_to_rotmat,
    trim_invalid_mano_prefix,
    transform_normals,
    transform_points,
)


def test_dexycb_quaternion_is_xyzw() -> None:
    half = np.sqrt(0.5)
    rotation = quat_to_rotmat(np.array([[0.0, 0.0, half, half]], dtype=np.float32))
    expected = np.array(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(rotation, expected, atol=1e-6)


def test_dexycb_transform_uses_source_to_target_rotation() -> None:
    half = np.sqrt(0.5)
    rotation = quat_to_rotmat(np.array([[0.0, 0.0, half, half]], dtype=np.float32))
    pose = build_SE3(rotation, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    points = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    normals = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)

    np.testing.assert_allclose(
        transform_points(points, pose), np.array([[[1.0, 3.0, 3.0]]]), atol=1e-6
    )
    np.testing.assert_allclose(
        transform_normals(normals, rotation), np.array([[[0.0, 1.0, 0.0]]]), atol=1e-6
    )


def test_find_reference_camera_requires_unique_identity() -> None:
    identity = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    translated = identity.copy()
    translated[0, 3] = 0.1
    extrinsics = {"reference": identity, "other": translated, "apriltag": identity}

    assert find_reference_camera(extrinsics, ["other", "reference"]) == "reference"
    with pytest.raises(ValueError, match="exactly one"):
        find_reference_camera(extrinsics, ["other"])
    with pytest.raises(ValueError, match="exactly one"):
        find_reference_camera(extrinsics, ["reference", "apriltag"])


def test_trim_invalid_mano_prefix_preserves_contiguous_raw_ids() -> None:
    raw_ids = np.arange(5, dtype=np.int32)
    pose_y = np.zeros((5, 2, 7), dtype=np.float32)
    pose_m = np.zeros((5, 1, 51), dtype=np.float32)
    pose_m[2:, 0, 0] = 1.0

    kept_ids, kept_y, kept_m = trim_invalid_mano_prefix(raw_ids, pose_y, pose_m)

    np.testing.assert_array_equal(kept_ids, [2, 3, 4])
    assert kept_y.shape[0] == kept_m.shape[0] == 3


def test_trim_invalid_mano_prefix_rejects_internal_gap() -> None:
    raw_ids = np.arange(4, dtype=np.int32)
    pose_y = np.zeros((4, 1, 7), dtype=np.float32)
    pose_m = np.zeros((4, 1, 51), dtype=np.float32)
    pose_m[[0, 2, 3], 0, 0] = 1.0

    with pytest.raises(ValueError, match="internal gaps"):
        trim_invalid_mano_prefix(raw_ids, pose_y, pose_m)


def test_expand_mano_pca_pose_applies_component_basis() -> None:
    coefficients = torch.arange(45, dtype=torch.float32).reshape(1, 45)
    components = 2.0 * torch.eye(45, dtype=torch.float32)

    expanded = expand_mano_pca_pose(coefficients, components)

    torch.testing.assert_close(expanded, 2.0 * coefficients)
