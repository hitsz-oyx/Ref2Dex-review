from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from process.common.object_cache_v2 import SCHEMA_NAME, convert_root
from src.task.Cm.tools.data.build_object_sampling_bank import build_bank
from src.task.Cm.dataset.object_v2 import CmObjectV2Dataset, dataset_statistics, make_dataloaders


def _write_stage4(root: Path, dataset_name: str, subject: str, sequence: str) -> None:
    out = root / dataset_name / subject / sequence
    out.mkdir(parents=True)
    frames = 14
    hand_points = 1538
    drift = np.arange(frames, dtype=np.float32)[:, None, None] * np.array([0.001, 0, 0], np.float32)
    obj = np.zeros((frames, 4096, 3), np.float32) + drift
    hand = np.zeros((frames, hand_points, 3), np.float32)
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    np.savez(out / "shared.npz", raw_frame_id=np.arange(frames), obj_points_world=obj,
             obj_normals_world=np.zeros_like(obj), obj_point_id=np.arange(4096), ds_rate=np.array(1),
             source_fps=np.array(30.0), dataset_name=np.asarray(dataset_name),
             seq_id=np.asarray(f"{subject}/{sequence}"), subject_id=np.asarray(subject),
             seq_name=np.asarray(sequence), object_name=np.asarray("unit"))
    for side in ("left", "right"):
        candidate = np.zeros((frames, 4096), bool)
        candidate[:, 100:800] = True
        np.savez(out / f"{side}.npz", side=np.asarray(side), hand_points_world=hand,
                 hand_normals_world=np.zeros_like(hand), hand_root_pose_world=pose,
                 obj_candidate_mask_5cm=candidate)


def _cache(tmp_path: Path) -> Path:
    source = tmp_path / "stage4"
    for dataset_name in ("grab", "arctic"):
        for index in range(5):
            _write_stage4(source, dataset_name, f"s{index}", "seq")
    root = tmp_path / "object_v2"
    convert_root(source, root)
    for meta in root.glob("**/shared/meta.json"):
        build_bank(meta.parent.parent)
    return root


def test_object_v2_conversion_sampling_and_statistics(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    assert json.loads((root / "meta.json").read_text())["schema_name"] == SCHEMA_NAME
    dataset = CmObjectV2Dataset(root, min_stride=1, max_stride=3)
    sample = dataset[0]
    assert sample["obj_points"].shape == (512, 3)
    assert sample["hand_points"].shape == (1538, 3)
    assert sample["obj_valid_mask"].all()
    assert sample["dataset_id"] in {"grab", "arctic"}
    assert np.isfinite(sample["obj_flow_gt"].numpy()).all()
    stats = dataset_statistics(dataset)
    assert set(stats) == {"grab", "arctic"}
    assert stats["grab"]["flow_rms_m"] > 0


def test_object_v2_sequence_disjoint_dataloaders(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    data = SimpleNamespace(root=str(root), train_path=str(root), val_split=0.2, test_fraction=0.2,
                           batch_size=2, val_batch_size=2, test_batch_size=2, num_workers=0,
                           drop_last=False, pin_memory=False, persistent_workers=False, prefetch_factor=2,
                           min_stride=1, max_stride=3, active_only=True, val_stride=1, test_stride=1,
                           val_strides=(1, 3), sampling_bank_size=4, fixed_eval_bank=0)
    meta = SimpleNamespace(num_obj_points=512, num_hand_points=1538)
    train, val, test, metadata, val_loaders, _ = make_dataloaders(data, 42, meta_cfg=meta)
    assert val is not None and test is not None
    assert metadata["dataset_split"] == {"train_sequences": 6, "val_sequences": 2, "test_sequences": 2}
    train_dirs = {row[0] for row in train.dataset.rows}
    val_dirs = {row[0] for row in val.dataset.rows}
    test_dirs = {row[0] for row in test.dataset.rows}
    assert not (train_dirs & val_dirs or train_dirs & test_dirs or val_dirs & test_dirs)
    assert set(val_loaders) == {"val/stride_1/", "val/stride_3/"}
