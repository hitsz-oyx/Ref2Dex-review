from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import SequentialSampler

from src.base.distributed import DistributedState, make_default_eval_sampler
from src.task.Cm.dataset import Stage4CmDataset


def test_runtime_stride_uses_current_hand_frame_and_epoch_seed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sequence_right.npz"
        frames, obj_count, hand_count = 16, 4096, 4
        obj = np.zeros((frames, obj_count, 3), dtype=np.float32)
        hand = np.zeros((frames, hand_count, 3), dtype=np.float32)
        # Object moves in world X; both endpoints are expressed in the same
        # current hand frame, so root motion itself is not duplicated in flow.
        obj[..., 0] = np.arange(frames, dtype=np.float32)[:, None]
        pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
        pose[:, 1, 3] = np.arange(frames, dtype=np.float32)
        np.savez(
            path, schema_name=np.asarray("ref2dex_cm_sequence"), coordinate_frame=np.asarray("world"),
            ds_rate=np.asarray(1, dtype=np.int32),
            seq_id=np.asarray("unit"), side=np.asarray("right"), raw_frame_id=np.arange(frames, dtype=np.int32),
            obj_points_world=obj, obj_normals_world=np.broadcast_to(np.array([0, 0, 1], np.float32), obj.shape).copy(),
            obj_point_id=np.arange(obj_count, dtype=np.int32), hand_points_world=hand,
            hand_normals_world=np.broadcast_to(np.array([0, 0, 1], np.float32), hand.shape).copy(),
            hand_root_pose_world=pose, obj_to_hand_min_dist=np.zeros((frames, obj_count), np.float32),
            obj_candidate_mask_5cm=np.ones((frames, obj_count), bool),
        )
        dataset = Stage4CmDataset(path, num_obj_points=8, num_hand_points=hand_count, min_stride=1, max_stride=12)
        sample = dataset[0]
        stride = int(sample["stride"])
        np.testing.assert_allclose(
            sample["obj_flow_gt"].numpy()[sample["obj_valid_mask"].numpy()],
            np.tile([stride, 0, 0], (8, 1)),
        )
        np.testing.assert_allclose(sample["hand_flow"].numpy(), np.zeros((hand_count, 3)))
        dataset.set_epoch(3)
        assert 1 <= int(dataset[0]["stride"]) <= 12


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
