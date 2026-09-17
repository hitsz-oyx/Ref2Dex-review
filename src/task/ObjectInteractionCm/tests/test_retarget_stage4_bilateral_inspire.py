from pathlib import Path

import numpy as np
import pytest

from src.task.ObjectInteractionCm.research.hand_region_sampling.run import (
    _build_inspire_pool,
    _sample_uniform_surface,
)
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (
    InspireUrdfModel,
    _fk_surface,
)
from src.task.ObjectInteractionCm.tools.data.retarget_stage4_bilateral_inspire import (
    canonical_cloud_to_visual_local,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
INSPIRE_URDF = (
    REPO_ROOT.parent
    / "dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
)


@pytest.mark.skipif(not INSPIRE_URDF.is_file(), reason="Dexplore Inspire asset is unavailable")
def test_canonical_cloud_is_localized_before_fk() -> None:
    model = InspireUrdfModel(INSPIRE_URDF)
    pool, _ = _build_inspire_pool(INSPIRE_URDF, 0.20)
    cloud = _sample_uniform_surface(pool, 2024, 1538)

    local_points, local_normals = canonical_cloud_to_visual_local(
        model, cloud.points, cloud.normals, cloud.source_visual_ids
    )
    repaired_points, repaired_normals = _fk_surface(
        model,
        np.zeros((1, 18), dtype=np.float32),
        local_points,
        local_normals,
        cloud.source_visual_ids,
    )
    broken_points, _ = _fk_surface(
        model,
        np.zeros((1, 18), dtype=np.float32),
        cloud.points,
        cloud.normals,
        cloud.source_visual_ids,
    )

    np.testing.assert_allclose(repaired_points[0], cloud.points, atol=2e-7)
    np.testing.assert_allclose(repaired_normals[0], cloud.normals, atol=2e-6)
    assert np.max(np.abs(broken_points[0] - cloud.points)) > 0.1


@pytest.mark.skipif(not INSPIRE_URDF.is_file(), reason="Dexplore Inspire asset is unavailable")
def test_localization_preserves_sample_identity_and_normal_units() -> None:
    model = InspireUrdfModel(INSPIRE_URDF)
    pool, _ = _build_inspire_pool(INSPIRE_URDF, 0.20)
    cloud = _sample_uniform_surface(pool, 2024, 1538)
    local_points, local_normals = canonical_cloud_to_visual_local(
        model, cloud.points, cloud.normals, cloud.source_visual_ids
    )

    assert local_points.shape == (1538, 3)
    assert local_normals.shape == (1538, 3)
    assert np.isfinite(local_points).all()
    np.testing.assert_allclose(np.linalg.norm(local_normals, axis=1), 1.0, atol=1e-6)
    assert len(np.unique(cloud.source_visual_ids)) == len(model.visuals)
