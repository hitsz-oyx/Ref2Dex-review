from __future__ import annotations

import numpy as np

from src.task.Cm.dataset_hrdexdb import HrdexdbGeometryDataset
from src.task.Cm.runner import CmActionRunner


def _write_geometry(root, *, frames: int = 8, pool: int = 4096) -> tuple[str, object]:
    episode = root / "episodes" / "human" / "unit" / "0"
    geometry = episode / "geometry"
    geometry.mkdir(parents=True)
    hand = np.zeros((frames, 1538, 3), dtype=np.float32)
    hand[..., 0] = 0.01
    obj = np.full((frames, pool, 3), 0.2, dtype=np.float32)
    obj[:, :6, 0] = 0.03
    obj[1:, :, 0] += 0.001
    normals_hand = np.zeros_like(hand)
    normals_hand[..., 2] = 1.0
    normals_obj = np.zeros_like(obj)
    normals_obj[..., 2] = 1.0
    candidate = np.zeros((frames, pool), dtype=bool)
    candidate[:, :6] = True
    np.save(geometry / "hand_points_world.npy", hand)
    np.save(geometry / "hand_normals_world.npy", normals_hand)
    np.save(geometry / "obj_points_pool_world.npy", obj)
    np.save(geometry / "obj_normals_pool_world.npy", normals_obj)
    np.save(geometry / "obj_candidate_mask_5cm.npy", candidate)
    np.save(geometry / "wrist_pose_world.npy", np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)))
    np.save(geometry / "frame_time.npy", np.arange(frames, dtype=np.float64) / 30.0)
    np.save(geometry / "source_frame_id.npy", np.arange(frames, dtype=np.int64))
    return "human/unit/0", episode


def test_hrdexdb_uses_candidate_pool_and_padding_mask(tmp_path) -> None:
    episode_name, episode = _write_geometry(tmp_path)
    dataset = HrdexdbGeometryDataset(
        [(episode_name, episode)], min_stride=1, max_stride=2, base_seed=7,
    )
    sample = dataset[0]
    assert sample["obj_points"].shape == (512, 3)
    assert int(sample["obj_valid_mask"].sum()) == 6
    assert np.all(sample["obj_flow_gt"].numpy()[~sample["obj_valid_mask"].numpy()] == 0.0)
    assert sample["selected_obj_idx"].shape == (512,)


def test_cm_stride_summary_supports_source_qualified_panels() -> None:
    metrics = {
        "val/stride_1/grab/flow/epe_mm": 1.0,
        "val/stride_5/grab/flow/epe_mm": 5.0,
        "val/stride_10/grab/flow/epe_mm": 10.0,
        "val/stride_1/hrdexdb/flow/epe_mm": 2.0,
        "val/stride_5/hrdexdb/flow/epe_mm": 6.0,
        "val/stride_10/hrdexdb/flow/epe_mm": 11.0,
    }
    summary = CmActionRunner._summarize_stride_metrics(metrics, split="val")
    assert summary["val/mean_stride_epe_mm"] == (1 + 5 + 10 + 2 + 6 + 11) / 6
    assert summary["val/grab/mean_stride_epe_mm"] == (1 + 5 + 10) / 3
    assert summary["val/hrdexdb/mean_stride_epe_mm"] == (2 + 6 + 11) / 3
