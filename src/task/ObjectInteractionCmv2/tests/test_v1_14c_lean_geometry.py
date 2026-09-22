from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.task.ObjectInteractionCmv2.multi_domain import (
    LEAN_CACHE_MANIFEST_SCHEMA,
    LEAN_GEOMETRY_SCHEMA,
    ThreeDomainTransitions,
    fixed_bilateral_hand_indices,
    hand_index_sha256,
)
from src.task.ObjectInteractionCmv2.part_se3_v114_training import load_v114_config
from src.task.ObjectInteractionCmv2.part_se3_data import RigidArticulatedPartTransitions


def _write_lean(root: Path, variant: str) -> tuple[Path, Path]:
    sequence = root / "sequence"
    geometry = sequence / "geometry"
    geometry.mkdir(parents=True)
    frames = 4
    rng = np.random.default_rng(4)
    obj = np.broadcast_to(
        rng.normal(size=(1, 4096, 3)).astype(np.float32) * 0.01,
        (frames, 4096, 3)).copy()
    hand = rng.normal(size=(frames, 4096, 3)).astype(np.float32) * 0.01
    np.save(geometry / "obj_points_pool_world.npy", obj)
    np.save(geometry / "obj_normals_pool_world.npy", np.zeros_like(obj))
    np.save(geometry / "obj_pose_world.npy", np.broadcast_to(np.eye(4, dtype=np.float32), (frames, 4, 4)))
    np.save(geometry / "source_frame_id.npy", np.arange(frames, dtype=np.int32) * 4)
    np.save(geometry / "frame_time.npy", np.arange(frames, dtype=np.float32) / 30)
    np.save(geometry / "knn_hand_points_world.npy", hand)
    np.save(geometry / "knn_hand_normals_world.npy", np.zeros_like(hand))
    np.save(geometry / "obj_candidate_mask_2cm.npy", np.ones((frames, 4096), dtype=bool))
    (geometry / "manifest.json").write_text(json.dumps({
        "schema_name": LEAN_GEOMETRY_SCHEMA, "dataset": "grab", "source_dataset": "grab",
        "source": variant, "hand_variant": variant, "split": "train",
        "coordinate_frame": "object_pose_t", "hand_side": "bilateral_merged_left_then_right",
        "frame_count": frames, "object_pool_points": 4096, "knn_hand_points": 4096,
        "knn_points_per_side": 2048, "hand_sampling_contract": "bilateral_fixed_random_2048_per_side_v1",
        "hand_sampling_seed": 42, "effective_fps": 30.0,
    }))
    index = root / "index.json"
    index.write_text(json.dumps({"sequences": {"train": [{"id": "grab/x", "path": str(sequence),
        "dataset": "grab", "source": variant, "frame_count": frames}], "val": [], "test": []}}))
    manifest = root / "cache_manifest.json"
    manifest.write_text(json.dumps({"schema_name": LEAN_CACHE_MANIFEST_SCHEMA,
        "knn_hand_points_per_stream": {variant: 4096}, "validation": {"bad_count": 0}}))
    return index, manifest


def test_lean_inspire_is_not_resampled_twice(tmp_path: Path) -> None:
    index, manifest = _write_lean(tmp_path, "inspire_f1")
    dataset = ThreeDomainTransitions([
        {"name": "grab", "hand_variant": "inspire_f1", "index": str(index), "manifest": str(manifest)}
    ], "train", num_obj_points=16, train_stride_values={"grab": [1]},
       hand_points_per_side=2048, hand_sampling_seed=42)
    assert dataset.sequences[0].hand_preselected
    assert dataset.hand_indices == [None]
    assert dataset[0]["hand_points"].shape == (4096, 3)
    rigid = RigidArticulatedPartTransitions([{
        "name": "grab", "hand_variant": "inspire_f1", "path": str(tmp_path / "sequence"),
        "id": "grab/x", "articulation": {"num_links": 1, "joints": []},
    }], "train", num_obj_points=1024, stride_values=(1,), hand_points_per_side=2048)
    assert rigid[0]["hand_points"].shape == (4096, 3)


def test_frozen_hand_index_hashes_are_variant_specific() -> None:
    assert hand_index_sha256(fixed_bilateral_hand_indices("mano")) != hand_index_sha256(
        fixed_bilateral_hand_indices("inspire_f1"))


def test_v114c_config_is_approved_lean_run() -> None:
    path = Path(__file__).parents[1] / "configs/active/mixed_part_se3_v1_14c_local_lean.yaml"
    config = load_v114_config(path, allow_approved_run=True)
    assert config["data_backend"] == "lean_geometry"
    assert config["active_groups"] == ["grab/mano", "grab/inspire_f1"]
