from __future__ import annotations

import numpy as np

from src.task.ObjectInteractionCm.tools.data.build_bilateral_mano_v1_4_cache import (
    KNN_POINTS_PER_SIDE,
    SURFACE_SEED,
    _sample_surface_chunk,
    _surface_correspondence,
)


def test_surface_correspondence_is_fixed_and_area_weighted() -> None:
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [0, 3, 0]], dtype=np.float32
    )
    faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    first = _surface_correspondence(vertices, faces, count=KNN_POINTS_PER_SIDE, seed=SURFACE_SEED)
    second = _surface_correspondence(vertices, faces, count=KNN_POINTS_PER_SIDE, seed=SURFACE_SEED)
    np.testing.assert_array_equal(first["face_ids"], second["face_ids"])
    np.testing.assert_array_equal(first["barycentric"], second["barycentric"])
    np.testing.assert_allclose(first["barycentric"].sum(axis=1), 1.0, atol=1e-6)
    assert int((first["face_ids"] == 1).sum()) > int((first["face_ids"] == 0).sum()) * 5


def test_surface_sampling_preserves_cross_frame_correspondence_and_normals() -> None:
    vertices = np.asarray(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[2, 3, 4], [3, 3, 4], [2, 4, 4]],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    correspondence = {
        "face_ids": np.zeros((4,), dtype=np.int64),
        "barycentric": np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.3, 0.5]], dtype=np.float32
        ),
    }
    points, normals = _sample_surface_chunk(vertices, faces, correspondence)
    np.testing.assert_allclose(
        points[1] - points[0],
        np.broadcast_to(np.asarray([2, 3, 4], dtype=np.float32), (4, 3)),
        atol=5e-7,
    )
    np.testing.assert_allclose(
        normals,
        np.broadcast_to(np.asarray([0, 0, 1], dtype=np.float32), normals.shape),
        atol=1e-6,
    )
