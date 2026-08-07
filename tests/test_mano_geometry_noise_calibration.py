from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.calibrate_mano_geometry_noise import root_aligned_vertices, vertex_rms_mm


def test_root_alignment_removes_independent_wrist_translation() -> None:
    vertices = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]], dtype=torch.float32
    )
    output = SimpleNamespace(
        vertices=vertices,
        joints=torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32),
    )
    shifted = SimpleNamespace(
        vertices=vertices + 7.0,
        joints=output.joints + 7.0,
    )

    torch.testing.assert_close(
        root_aligned_vertices(output), root_aligned_vertices(shifted)
    )


def test_vertex_rms_uses_euclidean_vertex_distance_in_mm() -> None:
    reference = torch.zeros((1, 2, 3), dtype=torch.float32)
    value = torch.tensor(
        [[[0.003, 0.004, 0.0], [0.0, 0.0, 0.005]]], dtype=torch.float32
    )

    # Both vertices move by 5 mm, so their RMS Euclidean displacement is 5 mm.
    torch.testing.assert_close(vertex_rms_mm(reference, value), torch.tensor([5.0]))
