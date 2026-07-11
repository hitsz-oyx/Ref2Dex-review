from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.base import load_config
from src.task.correspondence_ptv3.config import (
    Config as CorrConfig,
    resolve_contact_supervision_config,
    resolve_edge_sampler_config,
)
from src.task.correspondence_ptv3.data.stage3 import Stage3Store
from src.task.correspondence_ptv3.supervision import build_contact_supervision


def _write_stage3_npz(root: Path) -> Path:
    obj_points = np.asarray(
        [[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]],
        dtype=np.float32,
    )
    hand_points = np.asarray(
        [[[0.0, 0.0, 0.005], [0.02, 0.0, 0.04]]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        obj_points[:, :, None, :] - hand_points[:, None, :, :],
        axis=-1,
    )
    path = root / "sample_right.npz"
    np.savez(
        path,
        schema_name=np.asarray("train_corr_static"),
        raw_frame_id=np.asarray([3], dtype=np.int64),
        seq_id=np.asarray("seq/test"),
        side=np.asarray("right"),
        obj_points=obj_points,
        obj_normals=np.zeros_like(obj_points),
        obj_point_id=np.asarray([10, 11], dtype=np.int64),
        hand_points=hand_points,
        hand_normals=np.zeros_like(hand_points),
        hand_point_id=np.asarray([0, 1], dtype=np.int64),
        hand_cano_points=np.zeros((2, 3), dtype=np.float32),
        hand_finger_id=np.asarray([-1, -1], dtype=np.int64),
        hand_region_id=np.asarray([-1, -1], dtype=np.int64),
        obj_to_hand_min_dist=distance.min(axis=-1).astype(np.float32),
        obj_candidate_mask_5cm=np.asarray([[True, False]], dtype=bool),
        gt_obj_to_hand_knn_idx=np.asarray([[[0, 1], [1, 0]]], dtype=np.int64),
    )
    return path


class Stage3StoreTests(unittest.TestCase):
    def test_load_frame_returns_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage3_store_") as tmpdir:
            path = _write_stage3_npz(Path(tmpdir))
            store = Stage3Store(path)
            self.assertEqual(len(store), 1)
            frame = store.load_frame(0)
            self.assertEqual(frame.seq_id, "seq/test")
            self.assertEqual(frame.side, "right")
            self.assertEqual(frame.raw_frame_id, 3)
            self.assertEqual(frame.obj_points.shape, (2, 3))
            self.assertEqual(frame.hand_points.shape, (2, 3))
            self.assertEqual(frame.obj_point_id.tolist(), [10, 11])


class ConfigResolverTests(unittest.TestCase):
    def test_new_edge_sampler_conflicts_with_explicit_legacy_override(self) -> None:
        cfg = CorrConfig()
        cfg.meta.logit_sampling_mode = "balanced"
        cfg.meta.edge_sampler = SimpleNamespace(name="dense")
        with self.assertRaisesRegex(ValueError, "Conflicting edge sampler configuration"):
            resolve_edge_sampler_config(
                cfg.meta,
                explicit_override_keys={"meta.logit_sampling_mode"},
            )

    def test_new_contact_supervision_conflicts_with_explicit_legacy_override(self) -> None:
        cfg = CorrConfig()
        cfg.meta.contact_supervision_mode = "bin"
        cfg.meta.contact_supervision = SimpleNamespace(name="soft")
        with self.assertRaisesRegex(ValueError, "Conflicting contact supervision configuration"):
            resolve_contact_supervision_config(
                cfg.meta,
                explicit_override_keys={"meta.contact_supervision_mode"},
            )

    def test_canonical_soft_strat128_yaml_resolves_components(self) -> None:
        cfg = load_config("src/task/correspondence_ptv3/configs/soft_strat128.yaml")
        edge_cfg = resolve_edge_sampler_config(cfg.meta)
        supervision_cfg = resolve_contact_supervision_config(cfg.meta)
        self.assertEqual(edge_cfg.name, "stratified")
        self.assertEqual(edge_cfg.params["quotas"], (16, 32, 32, 32, 16))
        self.assertEqual(supervision_cfg.name, "soft")


class SupervisionFactoryTests(unittest.TestCase):
    def test_soft_and_bin_supervision_build_and_compute(self) -> None:
        cfg = CorrConfig()
        cfg.meta.contact_supervision = SimpleNamespace(name="soft")
        soft = build_contact_supervision(cfg.meta)
        self.assertEqual(soft.output_dim, 1)
        soft_result = soft.compute_edge_loss(
            logits=torch.zeros(1, 1, 2, 1),
            target_probability=torch.tensor([[[0.2, 0.8]]]),
            edge_weight=torch.ones(1, 1, 2),
            obj_valid_mask=torch.tensor([[True]]),
        )
        self.assertIn("excess_bce", soft_result.metrics)

        cfg.meta.contact_supervision = SimpleNamespace(
            name="bin",
            num_contact_bins=10,
            contact_bin_decode_mode="expectation",
            contact_bin_weights=None,
            edge_contact_bin_weights=None,
            contact_bin_weight_path=None,
        )
        cfg.meta.contact_supervision_mode = "bin"
        binary = build_contact_supervision(cfg.meta)
        self.assertEqual(binary.output_dim, 10)
        bin_result = binary.compute_object_loss(
            logits=torch.zeros(1, 2, 10),
            target_probability=torch.tensor([[0.1, 0.9]]),
            valid_mask=torch.ones(1, 2),
        )
        self.assertIn("ce", bin_result.metrics)


if __name__ == "__main__":
    unittest.main()
