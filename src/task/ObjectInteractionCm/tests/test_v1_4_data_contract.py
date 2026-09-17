import numpy as np
import pytest

from src.task.ObjectInteractionCm.dataset import SourceBalancedSampler
from src.task.ObjectInteractionCm.tools.data.compute_v1_4_max_steps import compute_max_steps
from src.task.ObjectInteractionCm.tools.data.export_bilateral_geometry import export_bilateral_geometry


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def test_max_steps_uses_ceiling():
    assert compute_max_steps(120, 509 * 100, 32) == 190875
    assert compute_max_steps(1, 33, 32) == 2
    with pytest.raises(ValueError):
        compute_max_steps(0, 10, 1)


def test_source_balanced_sampler_equalizes_domain_probability():
    sampler = SourceBalancedSampler(
        ["grab"] * 2 + ["arctic"] * 3 + ["oakink2"] * 5,
        {"grab": 1.0 / 3.0, "arctic": 1.0 / 3.0, "oakink2": 1.0 / 3.0},
        seed=42,
        strict=True,
    )
    totals = {
        source: sum(float(weight) for item_source, weight in zip(sampler.sources, sampler._weights) if item_source == source)
        for source in ("grab", "arctic", "oakink2")
    }
    assert all(abs(value - 1.0 / 3.0) < 1e-12 for value in totals.values())
    with pytest.raises(ValueError, match="exactly the train sources"):
        SourceBalancedSampler(["grab"], {"grab": 1.0 / 3.0, "arctic": 1.0 / 3.0}, seed=42, strict=True)


def test_export_merges_bilateral_and_drops_stale_knn(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    for name, value in {
        "obj_points_pool_world.npy": np.zeros((2, 4, 3), np.float32),
        "obj_normals_pool_world.npy": np.ones((2, 4, 3), np.float32),
        "obj_pose_world.npy": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "source_frame_id.npy": np.arange(2, dtype=np.int32),
        "frame_time.npy": np.arange(2, dtype=np.float32),
    }.items():
        _write(source / "shared" / name, value)
    for side, n, active in (("left", 2, [True, False]), ("right", 3, [False, True])):
        _write(source / side / "hand_points_world.npy", np.full((2, n, 3), n, np.float32))
        _write(source / side / "hand_normals_world.npy", np.ones((2, n, 3), np.float32))
        _write(source / side / "candidate_active_5cm.npy", np.asarray(active))
    _write(source / "geometry" / "obj_knn_indices.npy", np.zeros((2, 4, 32), np.uint16))

    meta = export_bilateral_geometry(source, output)
    merged = np.load(output / "geometry" / "hand_points_world.npy")
    assert merged.shape == (2, 5, 3)
    assert np.all(merged[:, :2] == 2) and np.all(merged[:, 2:] == 3)
    assert np.array_equal(np.load(output / "geometry" / "obj_candidate_mask_5cm.npy"), [True, True])
    assert meta["hand_side"] == "bilateral_merged_left_then_right"
    assert not (output / "geometry" / "obj_knn_indices.npy").exists()
