from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import SequentialSampler

from src.base.distributed import DistributedState, make_default_eval_sampler
from src.task.Cm.dataset import Stage4CmDataset
from process.stage4.prepare_cm import build_hand_sequence, build_shared_sequence


def test_runtime_stride_uses_current_hand_frame_and_epoch_seed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sequence_dir = Path(tmp) / "s1" / "unit"
        sequence_dir.mkdir(parents=True)
        shared_path = sequence_dir / "shared.npz"
        hand_path = sequence_dir / "right.npz"
        frames, obj_count, hand_count = 16, 4096, 4
        obj = np.zeros((frames, obj_count, 3), dtype=np.float32)
        hand = np.zeros((frames, hand_count, 3), dtype=np.float32)
        # Object moves in world X; both endpoints are expressed in the same
        # current hand frame, so root motion itself is not duplicated in flow.
        obj[..., 0] = np.arange(frames, dtype=np.float32)[:, None]
        pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
        pose[:, 1, 3] = np.arange(frames, dtype=np.float32)
        raw_frame_id = np.arange(frames, dtype=np.int32) * 4
        np.savez(
            shared_path, schema_name=np.asarray("ref2dex_cm_sequence_shared"), coordinate_frame=np.asarray("world"),
            ds_rate=np.asarray(4, dtype=np.int32), source_fps=np.asarray(120.0, dtype=np.float32),
            seq_id=np.asarray("unit"), raw_frame_id=raw_frame_id,
            obj_points_world=obj, obj_normals_world=np.broadcast_to(np.array([0, 0, 1], np.float32), obj.shape).copy(),
            obj_point_id=np.arange(obj_count, dtype=np.int32),
        )
        np.savez(
            hand_path, schema_name=np.asarray("ref2dex_cm_sequence_hand"), side=np.asarray("right"), hand_points_world=hand,
            hand_normals_world=np.broadcast_to(np.array([0, 0, 1], np.float32), hand.shape).copy(),
            hand_root_pose_world=pose,
            obj_candidate_mask_5cm=np.ones((frames, obj_count), bool),
        )
        dataset = Stage4CmDataset(sequence_dir, num_obj_points=8, num_hand_points=hand_count, min_stride=1, max_stride=10)
        sample = dataset[0]
        stride = int(sample["stride"])
        np.testing.assert_allclose(
            sample["obj_flow_gt"].numpy()[sample["obj_valid_mask"].numpy()],
            np.tile([stride, 0, 0], (8, 1)),
        )
        np.testing.assert_allclose(sample["hand_flow"].numpy(), np.zeros((hand_count, 3)))
        assert int(sample["raw_frame_id"]) == 0
        assert int(sample["next_raw_frame_id"]) == 4 * stride
        assert dataset.ds_rate == 4
        assert dataset.source_fps == 120.0
        assert dataset.effective_fps == 30.0
        dataset.set_epoch(3)
        assert 1 <= int(dataset[0]["stride"]) <= 10


def test_distributed_eval_sampler_shards_without_padding() -> None:
    dataset = list(range(7))
    rank0 = make_default_eval_sampler(
        dataset,
        distributed=DistributedState(enabled=True, world_size=2, rank=0),
    )
    rank1 = make_default_eval_sampler(
        dataset,
        distributed=DistributedState(enabled=True, world_size=2, rank=1),
    )
    assert rank0 is not None and rank1 is not None
    assert set(rank0).isdisjoint(set(rank1))
    assert sorted([*rank0, *rank1]) == list(range(7))
    # The base sampler is sequential, preserving deterministic validation.
    assert isinstance(rank0.sampler, SequentialSampler)


def test_shared_hand_cache_schema_omits_distance_arrays() -> None:
    frames, obj_count, hand_count = 2, 4096, 4
    obj = np.zeros((frames, obj_count, 3), dtype=np.float32)
    hand = np.zeros((frames, hand_count, 3), dtype=np.float32)
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    source = {
        "dataset_name": "grab", "seq_id": "unit", "subject_id": "s1", "seq_name": "unit",
        "object_name": "cube", "raw_frame_id": np.asarray([0, 4], dtype=np.int32),
        "obj_points_world": obj, "obj_normals_world": np.broadcast_to(np.asarray([0, 0, 1], np.float32), obj.shape).copy(),
        "obj_point_id": np.arange(obj_count, dtype=np.int32), "right_hand_root_pose": pose,
        "right_hand_points_world": hand,
        "right_hand_normals_world": np.broadcast_to(np.asarray([0, 0, 1], np.float32), hand.shape).copy(),
        "right_hand_cano_points": hand[0], "right_hand_point_id": np.arange(hand_count, dtype=np.int32),
        "right_hand_finger_id": np.zeros(hand_count, dtype=np.int32), "right_hand_region_id": np.zeros(hand_count, dtype=np.int32),
    }
    shared = build_shared_sequence(source, source_path=Path("/tmp/grab/s1/unit.npz"), grab_root=Path("/tmp/grab"), ds_rate=4)
    hand_cache = build_hand_sequence(source, side="right", candidate_threshold=0.05, frame_batch_size=2, device="cpu")
    assert shared["schema_name"].item() == "ref2dex_cm_sequence_shared"
    assert hand_cache["schema_name"].item() == "ref2dex_cm_sequence_hand"
    assert "obj_to_hand_min_dist" not in hand_cache
    assert "hand_to_obj_min_dist" not in hand_cache
    assert hand_cache["obj_candidate_mask_5cm"].shape == (frames, obj_count)
