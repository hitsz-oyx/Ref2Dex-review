from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate

from src.base import load_checkpoint, save_config
from src.base.cli import build_train_parser
from src.task.correspondence_ptv3.config_loader import (
    DEFAULT_CONFIG_NAME,
    load_correspondence_config,
    load_correspondence_config_from_args,
)
from src.task.correspondence_ptv3.models.common import resolve_ptv3_repo_path
from src.task.correspondence_ptv3.models.ptv3_concat import PTv3ConcatModel
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner


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


class FakeBackbone(nn.Module):
    def __init__(self, ptv3_cfg, in_channels: int) -> None:
        del ptv3_cfg, in_channels
        super().__init__()
        self.output_dim = 32

    def forward(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del coord, valid_mask
        return feat.new_zeros(feat.shape[0], feat.shape[1], self.output_dim)


class HydraConfigTests(unittest.TestCase):
    def test_model_config_owns_ptv3_structure_and_repo_path_defaults(self) -> None:
        cfg = load_correspondence_config("base")
        meta_dict = cfg.meta.to_dict()

        self.assertNotIn("point_feat_dim", meta_dict)
        self.assertNotIn("use_cross_attn", meta_dict)
        self.assertNotIn("use_cano_head", meta_dict)
        self.assertNotIn("use_finger_region_head", meta_dict)
        self.assertNotIn("ptv3_grid_size", meta_dict)
        self.assertNotIn("ptv3_enc_depths", meta_dict)
        self.assertIn("point_feat_dim", cfg.model)
        self.assertIn("ptv3", cfg.model)
        self.assertIsNone(cfg.model["ptv3"]["repo_path"])
        self.assertEqual(
            resolve_ptv3_repo_path(cfg.model["ptv3"]["repo_path"]),
            Path(__file__).resolve().parents[1] / "third_party" / "PointTransformerV3",
        )

    def test_hydra_composition_exposes_only_component_specific_fields(self) -> None:
        balanced = load_correspondence_config("base")
        self.assertEqual(
            balanced.edge_sampler["_target_"],
            "src.task.correspondence_ptv3.sampling.balanced.BalancedEdgeSampler",
        )
        self.assertIn("k_near_logit", balanced.edge_sampler)
        self.assertIn("k_far_logit", balanced.edge_sampler)
        self.assertIn("logit_near_radius", balanced.edge_sampler)
        self.assertIn("logit_far_min_radius", balanced.edge_sampler)

        stratified = load_correspondence_config("soft_strat128")
        self.assertEqual(
            stratified.edge_sampler["_target_"],
            "src.task.correspondence_ptv3.sampling.stratified.StratifiedEdgeSampler",
        )
        self.assertIn("distance_edges", stratified.edge_sampler)
        self.assertIn("quotas", stratified.edge_sampler)
        self.assertNotIn("logit_near_radius", stratified.edge_sampler)
        self.assertNotIn("logit_far_min_radius", stratified.edge_sampler)

        dense = load_correspondence_config("base", overrides=["edge_sampler=dense"])
        self.assertEqual(
            dense.edge_sampler["_target_"],
            "src.task.correspondence_ptv3.sampling.dense.DenseEdgeSampler",
        )
        self.assertEqual(set(dense.edge_sampler.keys()), {"_target_"})

    def test_hydra_instantiate_components(self) -> None:
        balanced = instantiate(load_correspondence_config("base").edge_sampler)
        stratified = instantiate(load_correspondence_config("soft_strat128").edge_sampler)
        dense = instantiate(load_correspondence_config("base", overrides=["edge_sampler=dense"]).edge_sampler)
        soft = instantiate(load_correspondence_config("soft_contact_baseline").contact_supervision)
        binary = instantiate(load_correspondence_config("bin_contact_legacy").contact_supervision)

        self.assertEqual(type(balanced).__name__, "BalancedEdgeSampler")
        self.assertEqual(type(stratified).__name__, "StratifiedEdgeSampler")
        self.assertEqual(type(dense).__name__, "DenseEdgeSampler")
        self.assertEqual(type(soft).__name__, "SoftContactSupervision")
        self.assertEqual(type(binary).__name__, "BinContactSupervision")

    def test_saved_config_preserves_targets_and_can_reload(self) -> None:
        cfg = load_correspondence_config("soft_strat128")
        with tempfile.TemporaryDirectory(prefix="corr_cfg_") as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_config(cfg, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("_target_", payload["edge_sampler"])
            self.assertIn("_target_", payload["contact_supervision"])
            self.assertIn("_target_", payload["model"])

            reloaded = load_correspondence_config(path)
            self.assertEqual(type(instantiate(reloaded.edge_sampler)).__name__, "StratifiedEdgeSampler")

    def test_resolved_component_override_requires_recipe_source(self) -> None:
        cfg = load_correspondence_config("base")
        with self.assertRaisesRegex(ValueError, "Hydra recipe source"):
            load_correspondence_config(cfg.to_dict(), overrides=["edge_sampler=stratified"])

    def test_external_yaml_with_defaults_must_live_under_config_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="corr_ext_yaml_") as tmpdir:
            path = Path(tmpdir) / "external.yaml"
            path.write_text("defaults:\n  - base\nname: external\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "External YAML with Hydra defaults"):
                load_correspondence_config(path)

    def test_cli_override_loader_tracks_explicit_keys(self) -> None:
        parser = build_train_parser(
            description="test parser",
            default_config=DEFAULT_CONFIG_NAME,
        )
        with tempfile.TemporaryDirectory(prefix="corr_cli_data_") as data_dir, tempfile.TemporaryDirectory(
            prefix="corr_cli_out_"
        ) as out_dir:
            args = parser.parse_args(
                [
                    "--set",
                    "edge_sampler=stratified",
                    "--set",
                    "contact_supervision=soft",
                    "--set",
                    "meta.gamma=1.5",
                    "--data",
                    data_dir,
                    "--output-dir",
                    out_dir,
                    "--device",
                    "cpu",
                    "--distributed",
                ]
            )
            cfg = load_correspondence_config_from_args(args)

        self.assertEqual(type(instantiate(cfg.edge_sampler)).__name__, "StratifiedEdgeSampler")
        self.assertEqual(type(instantiate(cfg.contact_supervision)).__name__, "SoftContactSupervision")
        self.assertEqual(cfg.meta.gamma, 1.5)
        self.assertEqual(cfg.data.train_path, data_dir)
        self.assertEqual(cfg.train.output_dir, out_dir)
        self.assertEqual(cfg.train.device, "cpu")
        self.assertTrue(cfg.train.distributed.enable)
        self.assertEqual(
            getattr(cfg, "_explicit_override_keys"),
            {
                "edge_sampler",
                "contact_supervision",
                "meta.gamma",
                "data.train_path",
                "train.output_dir",
                "train.device",
                "train.distributed.enable",
            },
        )
        self.assertTrue(getattr(cfg.train, "_explicit_output_dir"))


class CheckpointReloadTests(unittest.TestCase):
    def _train_cfg(self, data_root: str, output_dir: str):
        cfg = load_correspondence_config("soft_contact_baseline")
        cfg.wandb.enable = False
        cfg.data.train_path = data_root
        cfg.data.val_split = 0.0
        cfg.data.batch_size = 1
        cfg.data.num_workers = 0
        cfg.data.shuffle = False
        cfg.data.drop_last = False
        cfg.data.pin_memory = False
        cfg.train.output_dir = output_dir
        cfg.train.compile = False
        cfg.train.amp = False
        cfg.meta.num_obj_points = 2
        cfg.meta.num_hand_points = 2
        cfg.meta.num_fingers = 6
        cfg.meta.num_regions = 6
        setattr(cfg.train, "_explicit_output_dir", True)
        return cfg

    def test_checkpoint_reload_uses_saved_component_dicts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="corr_ckpt_data_") as data_dir, tempfile.TemporaryDirectory(
            prefix="corr_ckpt_out_"
        ) as out_dir:
            _write_stage3_npz(Path(data_dir))
            cfg = self._train_cfg(data_dir, out_dir)
            with mock.patch.object(PTv3ConcatModel, "backbone_cls", FakeBackbone):
                runner = CorrespondencePTV3Runner(cfg, mode="train")
                checkpoint_path = runner.save(epoch=0, is_best=False)
                checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
                reloaded_cfg = load_correspondence_config(checkpoint["config"])
                eval_runner = CorrespondencePTV3Runner(
                    reloaded_cfg,
                    mode="eval",
                    checkpoint=checkpoint_path,
                    build_data=False,
                )
                eval_runner.setup_inference(checkpoint_path)

            self.assertEqual(type(eval_runner.edge_sampler).__name__, "BalancedEdgeSampler")
            self.assertEqual(type(eval_runner.contact_supervision).__name__, "SoftContactSupervision")
            self.assertEqual(type(eval_runner.model).__name__, "PTv3ConcatModel")
            self.assertIn("_target_", checkpoint["config"]["edge_sampler"])
            self.assertIn("_target_", checkpoint["config"]["contact_supervision"])
            self.assertIn("_target_", checkpoint["config"]["model"])


if __name__ == "__main__":
    unittest.main()
