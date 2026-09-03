from pathlib import Path

import numpy as np
import pytest
import torch

from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2
from src.task.correspondence_ptv3_v2.research.io_cache.build_uncompressed_npz_cache import (
    build_cache,
)


def _write_stage3(path: Path) -> None:
    rng = np.random.default_rng(7)
    obj = rng.normal(size=(2, 8, 3)).astype(np.float32)
    hand = rng.normal(size=(2, 6, 3)).astype(np.float32)
    distance = np.linalg.norm(hand[:, :, None] - obj[:, None, :], axis=-1).min(axis=-1)
    np.savez_compressed(
        path,
        seq_id=np.asarray("seq"),
        side=np.asarray("right"),
        raw_frame_id=np.arange(2, dtype=np.int64),
        obj_points=obj,
        obj_normals=np.ones_like(obj),
        hand_points=hand,
        hand_normals=np.ones_like(hand),
        hand_to_obj_min_dist=distance.astype(np.float32),
        coordinate_frame=np.asarray("hand_root"),
    )


def test_uncompressed_array_cache_preserves_dataset_sample(tmp_path: Path) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    _write_stage3(source / "sample.npz")
    summary = build_cache(
        source_root=source,
        cache_root=cache,
        workers=1,
        limit_files=0,
        verify_existing=True,
    )
    assert summary["written"] == 1

    kwargs = dict(
        num_obj_points=4,
        num_hand_points=6,
        num_supervision_edges=3,
        apply_obj_perturb=False,
        apply_hand_perturb=False,
        runtime_resample_object=True,
    )
    source_dataset = CorrStaticDatasetV2(source, **kwargs)
    cache_dataset = CorrStaticDatasetV2(
        source,
        array_cache_path=cache,
        array_cache_required=True,
        **kwargs,
    )
    source_sample = source_dataset[0]
    cache_sample = cache_dataset[0]
    assert source_sample.keys() == cache_sample.keys()
    for key in source_sample:
        if torch.is_tensor(source_sample[key]):
            assert torch.equal(source_sample[key], cache_sample[key]), key
        else:
            assert source_sample[key] == cache_sample[key]


def test_required_array_cache_fails_on_missing_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_stage3(source / "sample.npz")
    with pytest.raises(FileNotFoundError, match="Required uncompressed array cache"):
        CorrStaticDatasetV2(
            source,
            num_obj_points=4,
            num_hand_points=6,
            num_supervision_edges=3,
            apply_obj_perturb=False,
            apply_hand_perturb=False,
            array_cache_path=tmp_path / "missing",
            array_cache_required=True,
        )
