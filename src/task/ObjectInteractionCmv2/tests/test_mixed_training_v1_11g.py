from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.task.ObjectInteractionCmv2.articulated import ArticulatedObjectInteractionCmv2V15Model, analytic_fk_flow, collate_articulated
from src.task.ObjectInteractionCmv2.mixed_training import GROUPS, batch_counts, combine_entries, oakink_holdout, recording_id, sample_batch
from src.task.ObjectInteractionCmv2.multi_domain import ThreeDomainTransitions
from src.task.ObjectInteractionCmv2.oakink2_parts import OakInkPartTransitions, PART_ADAPTER_SCHEMA
from src.task.ObjectInteractionCmv2.tests.test_v1_4_three_domain import _make_domain
from src.task.ObjectInteractionCmv2.tests.test_v1_5_articulated import _sample
from src.task.ObjectInteractionCmv2.train_mixed_articulated_ddp import _evaluate_part_group, initialize_model, validation_indices


def grouped_entries():
    grouped = {}
    for group in GROUPS:
        domain = group.split("/")[0]
        if domain == "oakink2":
            grouped[group] = {"train": [{"id": f"oakink2/record{record:02d}/{segment:04d}"}
                                         for record in range(20) for segment in range(3)], "val": [], "test": []}
        else:
            grouped[group] = {split: [{"id": f"{domain}/{split}"}] for split in ("train", "val", "test")}
    return grouped


def test_recording_holdout_has_no_segment_or_variant_leakage():
    grouped = grouped_entries()
    original = copy.deepcopy(grouped)
    sequences, holdout = combine_entries(grouped)
    assert grouped == original
    assert len(holdout) == 2
    assert set(holdout) == oakink_holdout(reversed([row["id"] for row in grouped["oakink2/mano"]["train"]]))
    for split, rows in sequences.items():
        for group in GROUPS:
            selected = {row["id"] for row in rows if row["group"] == group}
            if group.startswith("oakink2"):
                paired = {row["id"] for row in rows if row["group"] == "oakink2/mano"}
                assert selected == paired
                assert all((recording_id(value) in holdout) == (split == "val") for value in selected)
            else:
                assert selected == {f"{group.split('/')[0]}/{split}"}


def test_split_refuses_variant_mismatch_or_duplicate():
    grouped = grouped_entries()
    grouped["grab/inspire_f1"]["train"][0]["id"] = "grab/wrong"
    with pytest.raises(ValueError, match="variants disagree"):
        combine_entries(grouped)
    grouped = grouped_entries()
    grouped["arctic/mano"]["val"] = grouped["arctic/mano"]["train"]
    with pytest.raises(ValueError, match="duplicate or leaking"):
        combine_entries(grouped)


def test_balanced_batch_exact_hand_ratio_and_three_step_domain_ratio():
    totals = Counter()
    for step in range(3):
        counts = batch_counts(step)
        assert sum(counts.values()) == 64
        assert sum(count for group, count in counts.items() if group.endswith("/mano")) == 32
        assert counts["grab/inspire_f1"] == counts["oakink2/inspire_f1"] == 16
        totals.update(counts)
    assert [totals[group] for group in GROUPS[:3]] == [32, 32, 32]


def test_rank_generators_are_reproducible_and_independent():
    datasets = {group: [{"identity": index} for index in range(100)] for group in GROUPS}
    first = sample_batch(datasets, torch.Generator().manual_seed(42), 0)
    assert first == sample_batch(datasets, torch.Generator().manual_seed(42), 0)
    assert first != sample_batch(datasets, torch.Generator().manual_seed(100045), 0)
    assert Counter(sample["hand_variant"] for sample in first) == {"mano": 32, "inspire_f1": 32}


def make_part_adapter(root: Path, sequence: Path, sequence_id: str, parts: int = 2) -> Path:
    geometry = sequence / "geometry"
    frame_ids = np.load(geometry / "source_frame_id.npy")
    base_pose = np.load(geometry / "obj_pose_world.npy")
    points = np.load(geometry / "obj_points_pool_world.npy")[0] - base_pose[0, :3, 3]
    normals = np.load(geometry / "obj_normals_pool_world.npy")[0]
    adapter = root / "adapter"
    records = []
    for part_index in range(parts):
        part = adapter / "sequences" / "demo" / f"part_{part_index:02d}"
        part.mkdir(parents=True)
        pose = np.tile(np.eye(4, dtype=np.float32), (len(frame_ids), 1, 1))
        pose[:, part_index % 3, 3] = np.arange(len(frame_ids), dtype=np.float32) * (0.01 + 0.01 * part_index)
        np.save(part / "object_points_local.npy", points)
        np.save(part / "object_normals_local.npy", normals)
        np.save(part / "obj_pose_world.npy", pose)
        np.save(part / "source_frame_id.npy", frame_ids)
        records.append({"part_index": part_index, "object_id": f"part-{part_index}",
                        "path": str(part.relative_to(adapter))})
    (adapter / "index.json").write_text(json.dumps({"schema_name": PART_ADAPTER_SCHEMA,
        "sequences": [{"id": sequence_id, "parts": records}]}))
    (adapter / "cache_manifest.json").write_text(json.dumps({"schema_name": PART_ADAPTER_SCHEMA,
        "validation": {"bad_count": 0}}))
    return adapter


@pytest.mark.parametrize("count", [1, 2, 3, 127, 128, 129])
def test_validation_shards_do_not_pad_or_drop(count):
    first, second = (validation_indices(count, rank, 2) for rank in range(2))
    assert not set(first) & set(second)
    assert sorted(first + second) == list(range(count))


def test_model_only_initialization_resets_optimizer(tmp_path):
    config = {"model": {"hidden_width": 16, "knn_k": 4, "interaction_radius_m": 0.02,
                         "feature_scale_m": 0.02, "frame_dt_s": 1 / 30},
              "training": {"learning_rate": 0.001}, "checkpoint": str(tmp_path / "best.pt")}
    original = ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**config["model"]))
    torch.save({"architecture_version": original.architecture_version, "model": original.state_dict(),
                "optimizer": {"invalid_old_state": True}, "epoch": 9, "step": 72, "best_metric": 0.1}, config["checkpoint"])
    loaded, optimizer, metadata = initialize_model(config, torch.device("cpu"))
    assert not optimizer.state
    assert metadata == {"epoch": 9, "step": 72, "best_metric": 0.1}
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in original.state_dict().items())


@pytest.mark.parametrize("variant", ["mano", "inspire_f1"])
def test_oakink_adapter_preserves_rigid_gt_and_timeline_guard(tmp_path, variant):
    sequence, index, manifest = _make_domain(tmp_path, "oakink2", variant, frames=6)
    np.save(sequence / "geometry/frame_time.npy", np.array([0, 1, 2, 20, 21, 22], dtype=np.float32) / 30)
    dataset = ThreeDomainTransitions([{"name": "oakink2", "hand_variant": variant, "index": str(index),
                                       "manifest": str(manifest)}], "train", num_obj_points=4, fixed_stride=1,
                                      train_stride_values={"oakink2": [1]}, active_only=False)
    adapter = make_part_adapter(tmp_path, sequence, "oakink2/demo")
    wrapped = OakInkPartTransitions(dataset, adapter)
    assert len(wrapped) == 4
    assert wrapped.dropped_timeline_transitions == 1
    assert wrapped.component_count(0) == 2
    sample = wrapped.sample_component(0, 1)
    mixed = collate_articulated([_sample("arctic", 8, 1), sample])
    assert mixed["joint_valid_mask"].tolist() == [[True], [False]]
    batch = collate_articulated([sample])
    flow = analytic_fk_flow(batch["obj_points"], batch["obj_link_id"], batch["link_valid_mask"],
                            batch["joint_parent"], batch["joint_child"], batch["joint_axis_root"],
                            batch["joint_origin_root"], batch["joint_valid_mask"], batch["delta_xi_root_gt"], batch["delta_q_gt"])
    assert torch.allclose(flow, batch["obj_flow_gt"], atol=1e-6)
    assert torch.allclose(sample["delta_xi_root_gt"][:3], torch.tensor([0.0, 0.02, 0.0]), atol=1e-6)


def test_oakink_part_sampler_draws_component_after_transition(tmp_path):
    sequence, index, manifest = _make_domain(tmp_path, "oakink2", "mano", frames=8)
    base = ThreeDomainTransitions([{"name": "oakink2", "hand_variant": "mano", "index": str(index),
                                    "manifest": str(manifest)}], "train", num_obj_points=4, fixed_stride=1,
                                   train_stride_values={"oakink2": [1]}, active_only=False)
    oakink = OakInkPartTransitions(base, make_part_adapter(tmp_path, sequence, "oakink2/demo", parts=3))
    datasets = {group: [{"identity": index} for index in range(10)] for group in GROUPS}
    datasets["oakink2/mano"] = oakink
    samples = sample_batch(datasets, torch.Generator().manual_seed(7), 0)
    selected = [sample for sample in samples if sample.get("source") == "oakink2"]
    assert len(selected) == batch_counts(0)["oakink2/mano"]
    assert {sample["sequence_id"].split("#part=")[1] for sample in selected} <= {"part-0", "part-1", "part-2"}


def test_part_validation_means_components_within_each_transition():
    class Dataset:
        magnitudes = ((0.01, 0.03), (0.0,))

        def component_count(self, transition_index):
            return len(self.magnitudes[transition_index])

        def sample_component(self, transition_index, component_index):
            sample = _sample("oakink2", 8, 0)
            sample["obj_flow_gt"][:, 0] = self.magnitudes[transition_index][component_index]
            return sample

    class ZeroModel:
        def __call__(self, batch):
            return {"obj_flow_pred": torch.zeros_like(batch["obj_flow_gt"]),
                    "delta_xi_root": torch.zeros_like(batch["delta_xi_root_gt"])}

    metrics, _, components = _evaluate_part_group(ZeroModel(), Dataset(), [0, 1], torch.device("cpu"))
    assert metrics.samples == 2
    assert components == 3
    assert metrics.summary()["flow_epe_micro_mm"] == pytest.approx(10.0)
