from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.task.ObjectInteractionCmv2.multi_domain import (
    KNN_HAND_POINTS,
    SourceBalancedSampler,
    ThreeDomainTransitions,
    collate_three_domain,
)


def _make_domain(root: Path, domain: str, variant: str, frames: int = 12) -> tuple[Path, Path, Path]:
    hand_points = KNN_HAND_POINTS[variant]
    source_name = "mano" if variant == "mano" else "inspire_f1"
    sequence = root / domain / f"sequences/train/{source_name}" / domain / "demo"
    geometry = sequence / "geometry"
    geometry.mkdir(parents=True)
    manifest = {
        "schema_name": (
            "ref2dex_object_interaction_cm_bilateral_mano_v1_4"
            if variant == "mano" else "ref2dex_object_interaction_cm_oakink2_inspire_v1_4"
        ),
        "schema_version": "1.0.0", "source_dataset": domain, "split": "train",
        "source": source_name, "coordinate_frame": "object_pose_t",
        "hand_side": "bilateral_merged_left_then_right", "object_pool_points": 4096,
        "decoder_hand_points": 3076, "knn_hand_points": hand_points,
        "knn_points_per_side": hand_points // 2, "knn_k": 32,
        "effective_fps": 30.0,
    }
    (geometry / "manifest.json").write_text(json.dumps(manifest))
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    pose[:, 0, 3] = np.arange(frames, dtype=np.float32) * 0.01
    local = np.zeros((4096, 3), dtype=np.float32)
    local[:, 0] = np.linspace(-0.02, 0.02, 4096)
    points = local[None] + pose[:, None, :3, 3]
    normals = np.broadcast_to([0, 0, 1], points.shape).copy()
    hands = np.zeros((frames, hand_points, 3), dtype=np.float32)
    hands[:, :, 2] = 0.01
    hand_normals = np.broadcast_to([0, 0, -1], hands.shape).copy()
    np.save(geometry / "frame_time.npy", np.arange(frames, dtype=np.float32) / 30)
    np.save(geometry / "source_frame_id.npy", np.arange(frames, dtype=np.int32))
    np.save(geometry / "obj_points_pool_world.npy", points)
    np.save(geometry / "obj_normals_pool_world.npy", normals)
    np.save(geometry / "obj_pose_world.npy", pose)
    np.save(geometry / "knn_hand_points_world.npy", hands)
    np.save(geometry / "knn_hand_normals_world.npy", hand_normals)
    np.save(geometry / "obj_candidate_mask_2cm.npy", np.ones((frames, 4096), dtype=bool))
    entry = {"dataset": domain, "source": source_name, "variant": variant,
             "id": f"{domain}/demo", "path": str(sequence), "split": "train", "frame_count": frames}
    index = {"schema_name": "ref2dex_object_interaction_cm_index_v1_2",
             "sequences": {"train": [entry], "val": [], "test": []}}
    index_path = root / f"{domain}_index.json"
    index_path.write_text(json.dumps(index))
    manifest_path = root / f"{domain}_cache_manifest.json"
    manifest_path.write_text(json.dumps({
        "validation": {"bad_count": 0},
        "knn_hand_points_per_stream": {variant: hand_points},
        "total_sequences": 1,
    }))
    return sequence, index_path, manifest_path


def _specs(tmp_path: Path):
    specs = []
    for domain, variant in (("grab", "mano"), ("arctic", "mano"), ("oakink2", "inspire_f1")):
        _, index, manifest = _make_domain(tmp_path, domain, variant)
        specs.append({"name": domain, "hand_variant": variant,
                      "index": str(index), "manifest": str(manifest)})
    return specs


def test_three_domain_stride_and_direct_pose_gt(tmp_path):
    dataset = ThreeDomainTransitions(
        _specs(tmp_path), "train", fixed_stride=2, active_only=True, base_seed=7,
        train_stride_values={domain: tuple(range(1, 11)) for domain in ("grab", "arctic", "oakink2")},
    )
    assert dataset.rows_by_source() == {"grab": 10, "arctic": 10, "oakink2": 10}
    assert dataset.rows_by_variant() == {"mano": 20, "inspire_f1": 10}
    sample = dataset[0]
    assert sample["stride"].item() == 2
    assert np.isclose(sample["delta_time_s"].item(), 2 / 30, atol=1e-6)
    assert np.allclose(sample["delta_translation_gt"].numpy(), [0.02, 0, 0], atol=1e-6)
    assert np.allclose(sample["delta_rotation_gt"].numpy(), np.eye(3), atol=1e-6)
    assert sample["obj_points"].shape == (1024, 3)
    assert sample["hand_points"].shape == (4096, 3)
    assert sample["hand_valid_mask"].all()
    assert sample["pose_flow_residual_max_m"].item() < 1e-5


def test_collate_pads_mano_and_inspire_streams(tmp_path):
    dataset = ThreeDomainTransitions(
        _specs(tmp_path), "train", fixed_stride=1, active_only=True,
        train_stride_values={domain: (1,) for domain in ("grab", "arctic", "oakink2")},
    )
    batch = collate_three_domain([dataset[0], dataset[22]])
    assert batch["hand_points"].shape == (2, 20270, 3)
    assert int(batch["hand_valid_mask"][0].sum()) == 4096
    assert int(batch["hand_valid_mask"][1].sum()) == 20270
    assert torch_is_zero(batch["hand_points"][0, 4096:])


@pytest.mark.parametrize("stride", (1, 2, 3))
def test_oakink2_timeline_gap_excludes_only_crossing_actual_stride(tmp_path, stride):
    sequence, index, manifest = _make_domain(tmp_path, "oakink2", "mano")
    geometry = sequence / "geometry"
    frame_time = np.load(geometry / "frame_time.npy")
    frame_time[6:] += 0.5
    np.save(geometry / "frame_time.npy", frame_time)
    dataset = ThreeDomainTransitions(
        [{"name": "oakink2", "hand_variant": "mano", "index": str(index), "manifest": str(manifest)}],
        "train", fixed_stride=stride, active_only=False,
    )
    assert len(dataset) == 12 - 2 * stride
    assert dataset.dropped_timeline_transitions == stride
    assert all(frame not in range(6 - stride, 6) for _, frame in dataset.rows)


def torch_is_zero(value):
    return bool(np.allclose(value.numpy(), 0.0))


def test_source_balanced_sampler_assigns_one_third_mass():
    sources = ["grab"] * 2 + ["arctic"] * 3 + ["oakink2"] * 5
    sampler = SourceBalancedSampler(
        sources, {"grab": 1 / 3, "arctic": 1 / 3, "oakink2": 1 / 3}, seed=42)
    masses = {
        domain: sum(float(sampler._weights[i]) for i, value in enumerate(sources) if value == domain)
        for domain in ("grab", "arctic", "oakink2")
    }
    assert masses == pytest.approx({"grab": 1 / 3, "arctic": 1 / 3, "oakink2": 1 / 3})
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second


def test_source_balanced_sampler_rejects_missing_domain():
    with pytest.raises(ValueError, match="strict source probabilities"):
        SourceBalancedSampler(["grab", "arctic"], {"grab": 0.5}, seed=1)
