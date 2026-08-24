import argparse
import json
from pathlib import Path

import numpy as np

from src.task.CmDecoder.build_cache import _load_pose_sequence, _split_object_disjoint
from src.task.CmDecoder.build_horizon_cache import build as build_horizon_cache


def test_load_compact_pose_sequence_uses_numeric_frame_order(tmp_path: Path):
    path = tmp_path / "poses.npz"
    np.savez(
        path,
        frame_2=np.full((4, 4), 2, dtype=np.float32),
        frame_0=np.full((4, 4), 0, dtype=np.float32),
        frame_1=np.full((4, 4), 1, dtype=np.float32),
    )

    poses = _load_pose_sequence([path])

    assert poses.shape == (3, 4, 4)
    assert poses.dtype == np.float32
    assert poses[:, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_object_disjoint_split_never_crosses_objects():
    episodes = [
        f"inspire_f1/object_{object_index}/{scene_index}"
        for object_index in range(20)
        for scene_index in range(6)
    ]

    splits, split_objects = _split_object_disjoint(
        episodes, seed=42, val_fraction=0.1, test_fraction=0.1
    )

    assert sum(map(len, splits.values())) == len(episodes)
    assert set(split_objects["train"]).isdisjoint(split_objects["val"])
    assert set(split_objects["train"]).isdisjoint(split_objects["test"])
    assert set(split_objects["val"]).isdisjoint(split_objects["test"])
    for split, values in splits.items():
        assert {Path(value).parts[-2] for value in values} == set(split_objects[split])


def test_horizon_cache_excludes_episode_without_contiguous_pairs(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    relative = "v4/episodes/example"
    geometry = source_root / relative / "geometry"
    geometry.mkdir(parents=True)
    frame_count = 3
    arrays = {
        "q_full": np.zeros((frame_count, 12), dtype=np.float32),
        "frame_time": np.arange(frame_count, dtype=np.float64) / 30.0,
        "source_frame_id": np.array([1, 3, 5], dtype=np.int64),
        "hand_points_world": np.zeros((frame_count, 2, 3), dtype=np.float32),
        "hand_normals_world": np.zeros((frame_count, 2, 3), dtype=np.float32),
        "obj_points_world": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "obj_normals_world": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "wrist_pose_world": np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0),
        "hand_point_link_index": np.zeros((2,), dtype=np.int16),
        "hand_points_local": np.zeros((2, 3), dtype=np.float32),
    }
    for name, value in arrays.items():
        np.save(geometry / f"{name}.npy", value)
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({
        "schema": "cmdecoder_layered_v4",
        "episode_count": 1,
        "splits": {"train": ["inspire_f1/example/0"], "val": [], "test": []},
        "cache_dirs": {"inspire_f1/example/0": relative},
        "split_objects": {"train": ["example"], "val": [], "test": []},
    }))
    output_manifest = tmp_path / "output_manifest.json"

    build_horizon_cache(argparse.Namespace(
        source_root=source_root,
        source_manifest=source_manifest,
        output_root=output_root,
        output_manifest=output_manifest,
        output_version="v2",
        stride=10,
        source_hz=30.0,
    ))

    manifest = json.loads(output_manifest.read_text())
    assert manifest["episode_count"] == 0
    assert manifest["cache_dirs"] == {}
    assert manifest["splits"] == {"train": [], "val": [], "test": []}


def test_horizon_hand_flow_keeps_target_in_current_wrist_frame(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    relative = "v4/episodes/example"
    geometry = source_root / relative / "geometry"
    geometry.mkdir(parents=True)
    wrist = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    wrist[1, 0, 3] = 1.0
    arrays = {
        "q_full": np.zeros((2, 12), dtype=np.float32),
        "frame_time": np.array([0.0, 1.0 / 30.0]),
        "source_frame_id": np.array([1, 2], dtype=np.int64),
        "hand_points_world": np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=np.float32),
        "hand_normals_world": np.zeros((2, 1, 3), dtype=np.float32),
        "obj_points_world": np.zeros((2, 1, 3), dtype=np.float32),
        "obj_normals_world": np.zeros((2, 1, 3), dtype=np.float32),
        "wrist_pose_world": wrist,
        "hand_point_link_index": np.zeros((1,), dtype=np.int16),
        "hand_points_local": np.zeros((1, 3), dtype=np.float32),
    }
    for name, value in arrays.items():
        np.save(geometry / f"{name}.npy", value)
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({
        "schema": "cmdecoder_layered_v4",
        "episode_count": 1,
        "splits": {"train": ["inspire_f1/example/0"], "val": [], "test": []},
        "cache_dirs": {"inspire_f1/example/0": relative},
    }))
    output_manifest = tmp_path / "output_manifest.json"

    build_horizon_cache(argparse.Namespace(
        source_root=source_root,
        source_manifest=source_manifest,
        output_root=output_root,
        output_manifest=output_manifest,
        output_version="v2",
        stride=1,
        source_hz=30.0,
    ))

    manifest = json.loads(output_manifest.read_text())
    task = output_root / manifest["cache_dirs"]["inspire_f1/example/0"] / "task"
    flow = np.load(task / "hand_flow.npy")
    np.testing.assert_allclose(flow[0, 0], np.array([1.0, 0.0, 0.0]), atol=1e-7)
    assert manifest["hand_flow_frame"] == "current_wrist"
    assert json.loads((task / "manifest.json").read_text())["hand_flow_frame"] == "current_wrist"
