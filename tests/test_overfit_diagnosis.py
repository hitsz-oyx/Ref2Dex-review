from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.correspondence_ptv3.config_loader import load_correspondence_config
from src.task.correspondence_ptv3.data import CorrStaticDataset
from src.task.correspondence_ptv3.models.ptv3_concat import PTv3ConcatModel
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner
from src.task.correspondence_ptv3.sampling import (
    BalancedEdgeSampler,
    DenseEdgeSampler,
    StratifiedEdgeSampler,
)


class DummyRunner(BaseRunner):
    @classmethod
    def configure_overfit_mode(
        cls,
        cfg: TaskConfig,
        explicit_override_keys: set[str],
    ) -> None:
        super().configure_overfit_mode(cfg, explicit_override_keys)
        cfg.meta.child_hook_saw_base = bool(cfg.data.drop_last is False)

    def make_dataloaders(self, data_cfg, seed: int):
        del data_cfg, seed
        train_loader = DataLoader([{"x": torch.tensor([0.0])}], batch_size=1, shuffle=False)
        return train_loader, None, {"dummy": True}

    def build_model(self, model_cfg):
        del model_cfg
        return nn.Linear(1, 1)

    def step(self, model: torch.nn.Module, batch, mode: str = "train") -> RunnerOutput:
        del mode
        x = batch["x"].float()
        pred = model(x)
        loss = pred.sum() * 0.0
        return RunnerOutput(loss=loss, metrics={"loss": float(loss.detach().cpu())}, batch_size=int(x.shape[0]))


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


def _make_stage3_arrays() -> dict[str, np.ndarray]:
    obj_points = np.asarray(
        [
            [
                [0.000, 0.000, 0.000],
                [0.020, 0.000, 0.000],
                [0.040, 0.000, 0.000],
                [0.060, 0.000, 0.000],
                [0.080, 0.000, 0.000],
                [0.100, 0.000, 0.000],
            ]
        ],
        dtype=np.float32,
    )
    hand_points = np.asarray(
        [
            [
                [0.000, 0.000, 0.005],
                [0.020, 0.000, 0.010],
                [0.040, 0.000, 0.040],
                [0.070, 0.000, 0.006],
                [0.120, 0.000, 0.030],
            ]
        ],
        dtype=np.float32,
    )
    obj_normals = np.tile(np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32), (1, 6, 1))
    hand_normals = np.tile(np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32), (1, 5, 1))
    obj_point_id = np.arange(6, dtype=np.int64)
    hand_point_id = np.arange(5, dtype=np.int64)
    distance = np.linalg.norm(
        obj_points[:, :, None, :] - hand_points[:, None, :, :],
        axis=-1,
    )
    obj_to_hand_min_dist = distance.min(axis=-1).astype(np.float32)
    obj_candidate_mask = obj_to_hand_min_dist <= 0.05
    knn_order = np.argsort(distance, axis=-1)[..., :2].astype(np.int64)
    gt_knn = np.full((1, 6, 2), -1, dtype=np.int64)
    gt_knn[obj_candidate_mask] = knn_order[obj_candidate_mask]
    return {
        "schema_name": np.asarray("train_corr_static"),
        "raw_frame_id": np.asarray([5], dtype=np.int64),
        "seq_id": np.asarray("seq/test"),
        "side": np.asarray("right"),
        "obj_points": obj_points,
        "obj_normals": obj_normals,
        "obj_point_id": obj_point_id,
        "hand_points": hand_points,
        "hand_normals": hand_normals,
        "hand_point_id": hand_point_id,
        "hand_cano_points": np.zeros((5, 3), dtype=np.float32),
        "hand_finger_id": np.full((5,), -1, dtype=np.int64),
        "hand_region_id": np.full((5,), -1, dtype=np.int64),
        "obj_to_hand_min_dist": obj_to_hand_min_dist,
        "obj_candidate_mask_5cm": obj_candidate_mask,
        "gt_obj_to_hand_knn_idx": gt_knn,
    }


def _write_stage3_npz(root: Path) -> Path:
    path = root / "sample_right.npz"
    np.savez(path, **_make_stage3_arrays())
    return path


class OverfitModeTests(unittest.TestCase):
    def test_base_and_child_overfit_profile_and_saved_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runner_overfit_") as tmpdir:
            cfg = TaskConfig()
            cfg.wandb.enable = False
            cfg.train.output_dir = tmpdir
            setattr(cfg.train, "_explicit_output_dir", True)
            cfg.train.overfit_mode = True
            cfg.data.shuffle = True
            cfg.data.drop_last = True
            cfg.data.num_workers = 4
            cfg.data.persistent_workers = True
            cfg.train.weight_decay = 0.25
            cfg.train.scheduler = "cosine"
            cfg.train.warmup_ratio = 0.2
            cfg.train.grad_clip_norm = 3.0
            cfg.train.amp = True
            cfg.train.compile = True
            cfg.train.early_stopping_patience = 7
            cfg.train.max_steps = 1
            cfg._explicit_override_keys = {"train.weight_decay", "data.shuffle"}
            cfg.train.weight_decay = 1e-5
            cfg.data.shuffle = True

            DummyRunner(cfg, mode="train", build_data=True)

            self.assertTrue(cfg.data.shuffle)
            self.assertFalse(cfg.data.drop_last)
            self.assertEqual(cfg.data.num_workers, 0)
            self.assertFalse(cfg.data.persistent_workers)
            self.assertAlmostEqual(cfg.train.weight_decay, 1e-5)
            self.assertIsNone(cfg.train.scheduler)
            self.assertEqual(cfg.train.warmup_ratio, 0.0)
            self.assertEqual(cfg.train.warmup_steps, 0)
            self.assertIsNone(cfg.train.grad_clip_norm)
            self.assertFalse(cfg.train.amp)
            self.assertFalse(cfg.train.compile)
            self.assertIsNone(cfg.train.early_stopping_patience)
            self.assertTrue(cfg.meta.child_hook_saw_base)

            saved = json.loads((Path(tmpdir) / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["data"]["shuffle"])
            self.assertFalse(saved["data"]["drop_last"])
            self.assertAlmostEqual(saved["train"]["weight_decay"], 1e-5)
            self.assertIsNone(saved["train"]["scheduler"])
            self.assertTrue(saved["meta"]["child_hook_saw_base"])

    def test_correspondence_overfit_hook_respects_explicit_overrides(self) -> None:
        cfg = load_correspondence_config("bin_contact_legacy")
        cfg.train.overfit_mode = True
        cfg.meta.augment = True
        cfg.model["ptv3"]["drop_path"] = 0.2
        explicit = {"meta.augment", "model.ptv3.drop_path"}

        CorrespondencePTV3Runner.configure_overfit_mode(cfg, explicit)

        self.assertTrue(cfg.meta.fix_overfit_seed)
        self.assertTrue(cfg.meta.augment)
        self.assertFalse(cfg.meta.apply_hand_perturb)
        self.assertFalse(cfg.meta.augment_rotation)
        self.assertFalse(cfg.meta.augment_translation)
        self.assertFalse(cfg.meta.augment_scale)
        self.assertEqual(cfg.meta.hand_perturb_prob, 0.0)
        self.assertFalse(cfg.meta.val_augment)
        self.assertAlmostEqual(cfg.model["ptv3"]["drop_path"], 0.2)
        self.assertFalse(cfg.model["ptv3"]["shuffle_orders"])


class DatasetAndSamplerTests(unittest.TestCase):
    def _dataset(
        self,
        root: str,
        *,
        edge_sampler: object,
        fix_overfit_seed: bool = False,
    ) -> CorrStaticDataset:
        return CorrStaticDataset(
            root,
            num_obj_points=4,
            num_hand_points=5,
            k_cross=2,
            k_ctx=2,
            ctx_radius=0.04,
            base_seed=13,
            augment=True,
            apply_hand_perturb=True,
            hand_perturb_prob=1.0,
            hand_rot_std_deg=10.0,
            hand_trans_std=0.01,
            edge_sampler=edge_sampler,
            fix_overfit_seed=fix_overfit_seed,
        )

    def test_dataset_accepts_injected_samplers_without_batch_stats(self) -> None:
        samplers = [
            BalancedEdgeSampler(
                k_near_logit=2,
                k_far_logit=2,
                logit_near_radius=0.025,
                logit_far_min_radius=0.025,
            ),
            StratifiedEdgeSampler(
                distance_edges=(0.005, 0.015, 0.03, 0.06),
                quotas=(16, 32, 32, 32, 16),
            ),
            DenseEdgeSampler(),
        ]
        with tempfile.TemporaryDirectory(prefix="corr_dataset_") as tmpdir:
            _write_stage3_npz(Path(tmpdir))
            for sampler in samplers:
                dataset = self._dataset(tmpdir, edge_sampler=sampler)
                sample = dataset[0]
                self.assertIn("input_obj_to_hand_logit_idx", sample)
                self.assertIn("input_obj_to_hand_logit_valid_mask", sample)
                self.assertIn("input_obj_to_hand_logit_loss_weight", sample)
                self.assertNotIn("input_obj_to_hand_logit_near_count", sample)
                self.assertNotIn("input_obj_to_hand_logit_far_count", sample)

    def test_dense_sampler_contract_masks_invalid_rows(self) -> None:
        sample = DenseEdgeSampler().sample(
            gt_obj_points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            gt_hand_points=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            obj_valid=np.asarray([True, False]),
            seed=0,
        )
        self.assertEqual(sample.idx.shape, (2, 3))
        self.assertTrue(sample.valid_mask[0].all())
        self.assertFalse(sample.valid_mask[1].any())
        self.assertTrue(np.all(sample.idx[1] == -1))
        self.assertTrue(np.allclose(sample.loss_weight[1], 0.0))

    def test_stratified_refill_priority_characterization(self) -> None:
        hand_z = (
            [0.003] * 4
            + [0.010] * 8
            + [0.020] * 100
            + [0.040] * 100
            + [0.080] * 100
        )
        hand_points = np.asarray([[0.0, 0.0, z] for z in hand_z], dtype=np.float32)
        sample = StratifiedEdgeSampler(
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
        ).sample(
            gt_obj_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            gt_hand_points=hand_points,
            obj_valid=np.asarray([True]),
            seed=7,
        )
        chosen = hand_points[sample.idx[0, sample.valid_mask[0]], 2]
        counts = (
            int(np.count_nonzero(chosen <= 0.005)),
            int(np.count_nonzero((chosen > 0.005) & (chosen <= 0.015))),
            int(np.count_nonzero((chosen > 0.015) & (chosen < 0.03))),
            int(np.count_nonzero((chosen >= 0.03) & (chosen <= 0.06))),
            int(np.count_nonzero(chosen > 0.06)),
        )
        self.assertEqual(counts, (4, 8, 50, 50, 16))

    def test_stratified_sampler_accepts_non_128_quota_sum(self) -> None:
        sample = StratifiedEdgeSampler(
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(1, 2, 1, 2, 2),
        ).sample(
            gt_obj_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            gt_hand_points=np.asarray(
                [[0.0, 0.0, z] for z in (0.003, 0.010, 0.020, 0.040, 0.080, 0.090, 0.100, 0.110)],
                dtype=np.float32,
            ),
            obj_valid=np.asarray([True]),
            seed=11,
        )
        self.assertEqual(sample.idx.shape, (1, 8))
        self.assertEqual(int(sample.valid_mask.sum()), 8)

    def test_fixed_overfit_seed_keeps_sampling_augmentation_and_edges_identical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="corr_overfit_seed_") as tmpdir:
            _write_stage3_npz(Path(tmpdir))
            dataset = self._dataset(
                tmpdir,
                edge_sampler=BalancedEdgeSampler(
                    k_near_logit=2,
                    k_far_logit=2,
                    logit_near_radius=0.025,
                    logit_far_min_radius=0.025,
                ),
                fix_overfit_seed=True,
            )
            dataset.set_epoch(0)
            first = dataset[0]
            dataset.set_epoch(5)
            second = dataset[0]

        torch.testing.assert_close(first["selected_obj_idx"], second["selected_obj_idx"])
        torch.testing.assert_close(first["points"], second["points"])
        torch.testing.assert_close(
            first["input_obj_to_hand_logit_idx"],
            second["input_obj_to_hand_logit_idx"],
        )
        torch.testing.assert_close(
            first["input_obj_to_hand_logit_valid_mask"],
            second["input_obj_to_hand_logit_valid_mask"],
        )

    def test_model_build_regression_soft_and_bin(self) -> None:
        soft_cfg = load_correspondence_config("soft_contact_baseline")
        soft_cfg.meta.num_obj_points = 2
        soft_cfg.meta.num_hand_points = 2
        soft_cfg.meta.num_fingers = 6
        soft_cfg.meta.num_regions = 6

        bin_cfg = load_correspondence_config("bin_contact_legacy")
        bin_cfg.meta.num_obj_points = 2
        bin_cfg.meta.num_hand_points = 2
        bin_cfg.meta.num_fingers = 6
        bin_cfg.meta.num_regions = 6

        with unittest.mock.patch.object(PTv3ConcatModel, "backbone_cls", FakeBackbone):
            soft_model = instantiate(
                soft_cfg.model,
                _recursive_=False,
                _convert_="object",
                task_meta=soft_cfg.meta,
                contact_supervision=instantiate(soft_cfg.contact_supervision),
            )
            bin_model = instantiate(
                bin_cfg.model,
                _recursive_=False,
                _convert_="object",
                task_meta=bin_cfg.meta,
                contact_supervision=instantiate(bin_cfg.contact_supervision),
            )

        self.assertEqual(soft_model.contact_output_dim, 1)
        self.assertEqual(bin_model.contact_output_dim, 10)
        self.assertEqual(soft_model.contact_supervision_mode, "soft")
        self.assertEqual(bin_model.contact_supervision_mode, "bin")
        self.assertEqual(bin_model.contact_bin_decode_mode, "expectation")


if __name__ == "__main__":
    unittest.main()
