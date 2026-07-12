from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from src.base.cli import build_train_parser, load_train_config_from_args
from src.base import load_config
from src.task.correspondence_ptv3.composition import (
    EdgeSamplerConfig,
    resolve_contact_supervision_config,
    resolve_correspondence_components,
    resolve_edge_sampler_config,
)
from src.task.correspondence_ptv3.config import Config as CorrConfig
from src.task.correspondence_ptv3.data.dataset import make_dataloaders
from src.task.correspondence_ptv3.data.stage3 import Stage3Store
from src.task.correspondence_ptv3.objectives import CrossEdgeRankKObjective
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner
from src.task.correspondence_ptv3.supervision import build_contact_supervision
from src.task.correspondence_ptv3.targets import (
    build_clean_correspondence_targets,
    build_dynamic_edge_contact_targets,
)


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
        hand_cano_points=np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        hand_finger_id=np.asarray([4, 5], dtype=np.int64),
        hand_region_id=np.asarray([1, 2], dtype=np.int64),
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


class CompositionTests(unittest.TestCase):
    def test_canonical_config_wins_over_non_explicit_legacy_fallback(self) -> None:
        cfg = CorrConfig()
        cfg.meta.edge_sampler = SimpleNamespace(
            name="STRATIFIED",
            distance_edges=[0.005, 0.015, 0.03, 0.06],
            quotas=[16, 32, 32, 32, 16],
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
        )
        cfg.meta.logit_sampling_mode = "balanced"
        cfg.meta.logit_near_radius = 0.123
        cfg.meta.logit_far_min_radius = 0.456

        components = resolve_correspondence_components(cfg)

        self.assertEqual(components.edge_sampler.name, "stratified")
        self.assertEqual(
            components.edge_sampler.params["distance_edges"],
            (0.005, 0.015, 0.03, 0.06),
        )
        self.assertEqual(
            components.edge_sampler.params["quotas"],
            (16, 32, 32, 32, 16),
        )
        self.assertEqual(components.edge_sampler.params["logit_near_radius"], 0.005)
        self.assertEqual(components.edge_sampler.params["logit_far_min_radius"], 0.03)

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
        components = resolve_correspondence_components(cfg)
        self.assertEqual(components.model.name, "ptv3_concat")
        self.assertEqual(components.edge_sampler.name, "stratified")
        self.assertEqual(components.edge_sampler.params["quotas"], (16, 32, 32, 32, 16))
        self.assertEqual(components.edge_sampler.params["logit_near_radius"], 0.005)
        self.assertEqual(components.edge_sampler.params["logit_far_min_radius"], 0.03)
        self.assertEqual(components.contact_supervision.name, "soft")

    def test_canonical_yaml_drops_legacy_duplicates(self) -> None:
        text = Path(
            "src/task/correspondence_ptv3/configs/soft_strat128.yaml"
        ).read_text(encoding="utf-8")
        for removed in (
            "contact_supervision_mode:",
            "logit_sampling_mode:",
            "logit_stratified_distance_edges:",
            "logit_stratified_quotas:",
        ):
            self.assertNotIn(removed, text)


class DataloaderCompositionTests(unittest.TestCase):
    def test_dataloader_uses_canonical_params_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="composition_dataloader_") as tmpdir:
            _write_stage3_npz(Path(tmpdir))
            cfg = CorrConfig()
            cfg.data.train_path = tmpdir
            cfg.data.val_split = 0.0
            cfg.data.batch_size = 1
            cfg.data.num_workers = 0
            cfg.data.shuffle = False
            cfg.data.pin_memory = False
            cfg.meta.edge_sampler = SimpleNamespace(
                name="stratified",
                distance_edges=[0.005, 0.015, 0.03, 0.06],
                quotas=[16, 32, 32, 32, 16],
                logit_near_radius=0.005,
                logit_far_min_radius=0.03,
            )
            cfg.meta.logit_sampling_mode = "balanced"
            cfg.meta.logit_near_radius = 0.123
            cfg.meta.logit_far_min_radius = 0.456

            components = resolve_correspondence_components(cfg)
            train_loader, val_loader, metadata, val_loaders = make_dataloaders(
                cfg.data,
                seed=7,
                meta_cfg=cfg.meta,
                components=components,
            )

            self.assertIsNone(val_loader)
            self.assertEqual(val_loaders, {})
            self.assertEqual(train_loader.dataset.edge_sampler.logit_near_radius, 0.005)
            self.assertEqual(train_loader.dataset.edge_sampler.logit_far_min_radius, 0.03)
            self.assertEqual(metadata["resolved_components"]["edge_sampler"]["name"], "stratified")
            self.assertEqual(
                metadata["resolved_components"]["edge_sampler"]["params"]["quotas"],
                (16, 32, 32, 32, 16),
            )
            self.assertEqual(metadata["logit_sampling_mode"], "stratified")
            self.assertEqual(metadata["logit_near_radius"], 0.005)
            self.assertEqual(metadata["logit_far_min_radius"], 0.03)


class SupervisionFactoryTests(unittest.TestCase):
    def test_soft_and_bin_supervision_build_and_compute(self) -> None:
        cfg = CorrConfig()
        cfg.meta.contact_supervision = SimpleNamespace(name="soft")
        soft = build_contact_supervision(resolve_contact_supervision_config(cfg.meta))
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
        binary = build_contact_supervision(resolve_contact_supervision_config(cfg.meta))
        self.assertEqual(binary.output_dim, 10)
        bin_result = binary.compute_object_loss(
            logits=torch.zeros(1, 2, 10),
            target_probability=torch.tensor([[0.1, 0.9]]),
            valid_mask=torch.ones(1, 2),
        )
        self.assertIn("ce", bin_result.metrics)

    def test_runner_builds_contact_supervision_once(self) -> None:
        cfg = CorrConfig()
        cfg.meta.contact_supervision = SimpleNamespace(name="soft")
        runner = object.__new__(CorrespondencePTV3Runner)
        runner.cfg = cfg
        runner.explicit_override_keys = set()

        original = build_contact_supervision
        with mock.patch(
            "src.task.correspondence_ptv3.runner.build_contact_supervision",
            side_effect=original,
        ) as patched:
            first = runner.contact_supervision
            second = runner.contact_supervision

        self.assertIs(first, second)
        self.assertEqual(patched.call_count, 1)


class ObjectiveAndTargetTests(unittest.TestCase):
    def test_rankk_objective_characterization(self) -> None:
        objective = CrossEdgeRankKObjective(k=1, label_gap=0.15, margin=0.2)
        pred = torch.tensor([[[0.4, 0.35, 0.2]]], dtype=torch.float32)
        target = torch.tensor([[[1.0, 0.7, 0.1]]], dtype=torch.float32)
        valid_mask = torch.ones_like(target, dtype=torch.bool)

        loss, pair_count = objective.compute(pred, target, valid_mask)

        self.assertAlmostEqual(float(loss), 0.0225, places=6)
        self.assertEqual(float(pair_count), 2.0)

    def test_target_builders_characterization(self) -> None:
        batch = {
            "points": torch.tensor(
                [[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.04]]],
                dtype=torch.float32,
            ),
            "runtime_obj_valid_mask": torch.tensor([[True, True]]),
            "input_obj_to_hand_logit_idx": torch.tensor([[[0, 1], [0, 1]]]),
            "input_obj_to_hand_logit_valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "gt_obj_to_hand_knn_idx": torch.tensor([[[1, 0], [0, 1]]]),
            "obj_contact_label": torch.tensor([[0.2, 0.05]], dtype=torch.float32),
            "hand_cano_points": torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32),
            "hand_finger_id": torch.tensor([[3, 5]], dtype=torch.int64),
            "hand_region_id": torch.tensor([[1, 2]], dtype=torch.int64),
        }

        edge_target = build_dynamic_edge_contact_targets(
            batch,
            num_obj_points=2,
            num_hand_points=2,
            d_pos=0.005,
            d_neg=0.03,
            gamma=1.0,
        )
        target_cano, corr_valid, target_finger, target_region = build_clean_correspondence_targets(
            batch,
            corr_contact_label_min=0.1,
        )

        self.assertEqual(tuple(edge_target.shape), (1, 2, 2))
        self.assertEqual(float(edge_target[0, 0, 0]), 1.0)
        self.assertEqual(float(edge_target[0, 0, 1]), 0.0)
        self.assertTrue(torch.equal(corr_valid, torch.tensor([[1.0, 0.0]])))
        self.assertTrue(torch.equal(target_cano[0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertEqual(int(target_finger[0, 0]), 5)
        self.assertEqual(int(target_region[0, 0]), 2)


class BaseCliTests(unittest.TestCase):
    def test_base_cli_preserves_override_tracking(self) -> None:
        parser = build_train_parser(
            description="test",
            default_config="src.task.correspondence_ptv3.config:Config",
        )
        args = parser.parse_args(
            [
                "--set",
                "name=exp_name",
                "--set",
                "wandb.name=wandb_run",
                "--set",
                "meta.gamma=1.5",
                "--data",
                "/tmp/stage3",
                "--output-dir",
                "/tmp/out",
                "--device",
                "cpu",
                "--distributed",
            ]
        )
        cfg = load_train_config_from_args(args)

        self.assertEqual(cfg.name, "exp_name")
        self.assertEqual(cfg.wandb.name, "wandb_run")
        self.assertEqual(cfg.data.train_path, "/tmp/stage3")
        self.assertEqual(cfg.train.output_dir, "/tmp/out")
        self.assertEqual(cfg.train.device, "cpu")
        self.assertTrue(cfg.train.distributed.enable)
        self.assertIn("name", cfg._explicit_override_keys)
        self.assertIn("wandb.name", cfg._explicit_override_keys)
        self.assertIn("meta.gamma", cfg._explicit_override_keys)
        self.assertIn("data.train_path", cfg._explicit_override_keys)
        self.assertIn("train.output_dir", cfg._explicit_override_keys)
        self.assertIn("train.device", cfg._explicit_override_keys)
        self.assertIn("train.distributed.enable", cfg._explicit_override_keys)
        self.assertTrue(cfg._explicit_name)
        self.assertTrue(cfg.wandb._explicit_name)
        self.assertTrue(cfg.train._explicit_output_dir)


if __name__ == "__main__":
    unittest.main()
