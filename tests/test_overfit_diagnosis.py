from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from render.eval_visualize import _build_runtime_frame
from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.correspondence_ptv3.config import Config as CorrConfig
from src.task.correspondence_ptv3.dataset import (
    CorrStaticDataset,
    _compute_dense_logit_neighbors,
    _compute_runtime_logit_neighbors,
    _compute_stratified_logit_neighbors,
)
from src.task.correspondence_ptv3.model import StaticHOCPTv3
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner
import src.task.correspondence_ptv3.model as correspondence_model


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
    def __init__(self, meta, in_channels: int) -> None:
        del meta, in_channels
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


def _corr_cfg(mode: str = "soft") -> CorrConfig:
    cfg = CorrConfig()
    cfg.wandb.enable = False
    cfg.meta.contact_supervision_mode = mode
    cfg.meta.use_cross_attn = False
    cfg.meta.use_cano_head = False
    cfg.meta.use_finger_region_head = False
    cfg.meta.loss_cross_edge_rankk_weight = 0.0
    cfg.meta.loss_cano_weight = 0.0
    cfg.meta.loss_finger_weight = 0.0
    cfg.meta.loss_region_weight = 0.0
    return cfg


class OverfitModeTests(unittest.TestCase):
    def test_base_overfit_mode_disabled_keeps_values(self) -> None:
        cfg = TaskConfig()
        cfg.wandb.enable = False
        cfg.train.output_dir = tempfile.mkdtemp(prefix="runner_no_overfit_")
        setattr(cfg.train, "_explicit_output_dir", True)
        cfg.data.shuffle = True
        cfg.data.drop_last = True
        cfg.data.num_workers = 2
        cfg.train.weight_decay = 0.123
        cfg.train.scheduler = "cosine"
        cfg.train.grad_clip_norm = 1.5
        cfg.train.overfit_mode = False

        DummyRunner(cfg, mode="train", build_data=False)

        self.assertTrue(cfg.data.shuffle)
        self.assertTrue(cfg.data.drop_last)
        self.assertEqual(cfg.data.num_workers, 2)
        self.assertEqual(cfg.train.weight_decay, 0.123)
        self.assertEqual(cfg.train.scheduler, "cosine")
        self.assertEqual(cfg.train.grad_clip_norm, 1.5)

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
        cfg = _corr_cfg(mode="bin")
        cfg.train.overfit_mode = True
        cfg.meta.augment = True
        cfg.meta.ptv3_drop_path = 0.2
        explicit = {"meta.augment", "meta.ptv3_drop_path"}

        CorrespondencePTV3Runner.configure_overfit_mode(cfg, explicit)

        self.assertTrue(cfg.meta.fix_overfit_seed)
        self.assertTrue(cfg.meta.augment)
        self.assertFalse(cfg.meta.apply_hand_perturb)
        self.assertFalse(cfg.meta.augment_rotation)
        self.assertFalse(cfg.meta.augment_translation)
        self.assertFalse(cfg.meta.augment_scale)
        self.assertEqual(cfg.meta.hand_perturb_prob, 0.0)
        self.assertFalse(cfg.meta.val_augment)
        self.assertAlmostEqual(cfg.meta.ptv3_drop_path, 0.2)
        self.assertFalse(cfg.meta.ptv3_shuffle_orders)


class DatasetModeTests(unittest.TestCase):
    @staticmethod
    def _bucket_counts(
        hand_points: np.ndarray,
        selected_idx: np.ndarray,
    ) -> tuple[int, int, int, int, int]:
        valid_idx = selected_idx[selected_idx >= 0]
        selected_dist = hand_points[valid_idx, 2]
        return (
            int(np.count_nonzero(selected_dist <= 0.005)),
            int(np.count_nonzero((selected_dist > 0.005) & (selected_dist <= 0.015))),
            int(np.count_nonzero((selected_dist > 0.015) & (selected_dist < 0.03))),
            int(np.count_nonzero((selected_dist >= 0.03) & (selected_dist <= 0.06))),
            int(np.count_nonzero(selected_dist > 0.06)),
        )

    @staticmethod
    def _make_hand_points_by_distance(counts: tuple[int, int, int, int, int]) -> np.ndarray:
        ranges = (
            (0.0010, 0.0048),
            (0.0060, 0.0148),
            (0.0160, 0.0296),
            (0.0310, 0.0596),
            (0.0610, 0.1800),
        )
        distances: list[float] = []
        for count, (low, high) in zip(counts, ranges):
            if int(count) <= 0:
                continue
            values = np.linspace(low, high, int(count), endpoint=True, dtype=np.float32)
            distances.extend(float(value) for value in values.tolist())
        return np.asarray(
            [[0.0, 0.0, float(distance)] for distance in distances],
            dtype=np.float32,
        )

    def test_balanced_logit_neighbors_shape_and_padding(self) -> None:
        obj_points = np.asarray(
            [
                [0.000, 0.000, 0.000],
                [0.050, 0.000, 0.000],
                [0.200, 0.000, 0.000],
            ],
            dtype=np.float32,
        )
        hand_points = np.asarray(
            [
                [0.000, 0.000, 0.010],
                [0.000, 0.000, 0.020],
                [0.050, 0.000, 0.010],
                [0.050, 0.000, 0.040],
                [0.250, 0.000, 0.000],
            ],
            dtype=np.float32,
        )
        obj_valid = np.asarray([True, True, False], dtype=bool)
        logit_idx, logit_valid, logit_weight, near_count, far_count = _compute_runtime_logit_neighbors(
            obj_points,
            hand_points,
            obj_valid,
            k_near_logit=2,
            k_far_logit=2,
            logit_near_radius=0.025,
            logit_far_min_radius=0.025,
            seed=123,
        )
        self.assertEqual(logit_idx.shape, (3, 4))
        self.assertEqual(logit_valid.shape, (3, 4))
        self.assertEqual(logit_weight.shape, (3, 4))
        self.assertEqual(tuple(near_count.tolist()), (2, 1, 0))
        self.assertEqual(tuple(far_count.tolist()), (2, 2, 0))
        self.assertTrue(np.all(logit_valid[0, :2]))
        self.assertTrue(np.all(logit_valid[1, :1]))
        self.assertTrue(np.all(logit_valid[1, 2:4]))
        self.assertTrue(np.all(logit_idx[2] == -1))
        self.assertTrue(np.all(~logit_valid[2]))
        self.assertTrue(np.all(logit_weight[2] == 0))

    def test_stratified_logit_neighbors_shape_and_quota(self) -> None:
        obj_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        hand_points = self._make_hand_points_by_distance((20, 40, 40, 40, 20))
        obj_valid = np.asarray([True], dtype=bool)
        logit_idx, logit_valid, logit_weight, near_count, far_count = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            obj_valid,
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=123,
        )
        self.assertEqual(logit_idx.shape, (1, 128))
        self.assertTrue(np.all(logit_valid))
        self.assertTrue(np.all(logit_weight == 1.0))
        self.assertEqual(
            self._bucket_counts(hand_points, logit_idx[0]),
            (16, 32, 32, 32, 16),
        )
        self.assertEqual(int(near_count[0]), 16)
        self.assertEqual(int(far_count[0]), 48)

    def test_stratified_logit_neighbors_never_duplicate_hand_points(self) -> None:
        obj_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        hand_points = self._make_hand_points_by_distance((12, 24, 48, 48, 32))
        obj_valid = np.asarray([True, True], dtype=bool)
        logit_idx, logit_valid, _logit_weight, _near_count, _far_count = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            obj_valid,
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=456,
        )
        for row in range(logit_idx.shape[0]):
            valid_idx = logit_idx[row, logit_valid[row]]
            self.assertEqual(len(valid_idx.tolist()), len(set(valid_idx.tolist())))

    def test_stratified_logit_neighbors_round_robin_refill_priority(self) -> None:
        obj_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        hand_points = self._make_hand_points_by_distance((4, 8, 100, 100, 100))
        obj_valid = np.asarray([True], dtype=bool)
        logit_idx, logit_valid, _logit_weight, _near_count, _far_count = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            obj_valid,
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=789,
        )
        self.assertEqual(int(logit_valid[0].sum()), 128)
        self.assertEqual(len(set(logit_idx[0, logit_valid[0]].tolist())), 128)
        self.assertEqual(
            self._bucket_counts(hand_points, logit_idx[0]),
            (4, 8, 50, 50, 16),
        )

    def test_stratified_refill_uses_s0_before_s4(self) -> None:
        obj_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        hand_points = self._make_hand_points_by_distance((100, 8, 8, 8, 100))
        logit_idx, logit_valid, *_ = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            np.asarray([True]),
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=7,
        )
        self.assertEqual(int(logit_valid[0].sum()), 128)
        self.assertEqual(self._bucket_counts(hand_points, logit_idx[0]), (88, 8, 8, 8, 16))

    def test_stratified_refill_uses_s4_only_after_other_layers(self) -> None:
        obj_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        hand_points = self._make_hand_points_by_distance((20, 8, 8, 8, 100))
        logit_idx, logit_valid, *_ = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            np.asarray([True]),
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=8,
        )
        self.assertEqual(self._bucket_counts(hand_points, logit_idx[0]), (20, 8, 8, 8, 84))

    def test_stratified_small_hand_pool_and_boundaries(self) -> None:
        obj_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        hand_points = np.asarray(
            [[0.0, 0.0, value] for value in (0.005, 0.015, 0.03, 0.06)],
            dtype=np.float32,
        )
        logit_idx, logit_valid, logit_weight, *_ = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            np.asarray([True]),
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=9,
        )
        self.assertEqual(self._bucket_counts(hand_points, logit_idx[0]), (1, 1, 0, 2, 0))
        self.assertEqual(int(logit_valid[0].sum()), 4)
        self.assertEqual(len(set(logit_idx[0, logit_valid[0]].tolist())), 4)
        self.assertTrue(np.all(logit_idx[0, ~logit_valid[0]] == -1))
        self.assertTrue(np.all(logit_weight[0, ~logit_valid[0]] == 0.0))

    def test_stratified_logit_neighbors_invalid_object_rows_are_padding(self) -> None:
        obj_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        hand_points = self._make_hand_points_by_distance((20, 40, 40, 40, 20))
        obj_valid = np.asarray([True, False], dtype=bool)
        logit_idx, logit_valid, logit_weight, _near_count, _far_count = _compute_stratified_logit_neighbors(
            obj_points,
            hand_points,
            obj_valid,
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
            seed=123,
        )
        self.assertTrue(np.all(logit_idx[1] == -1))
        self.assertTrue(np.all(~logit_valid[1]))
        self.assertTrue(np.all(logit_weight[1] == 0.0))

    def test_stratified_seed_behavior(self) -> None:
        obj_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        hand_points = self._make_hand_points_by_distance((24, 48, 64, 96, 64))
        obj_valid = np.asarray([True], dtype=bool)
        args = dict(
            gt_obj_points=obj_points,
            gt_hand_points=hand_points,
            obj_valid=obj_valid,
            distance_edges=(0.005, 0.015, 0.03, 0.06),
            quotas=(16, 32, 32, 32, 16),
            logit_near_radius=0.005,
            logit_far_min_radius=0.03,
        )
        same0 = _compute_stratified_logit_neighbors(seed=42, **args)[0]
        same1 = _compute_stratified_logit_neighbors(seed=42, **args)[0]
        diff = _compute_stratified_logit_neighbors(seed=43, **args)[0]
        np.testing.assert_array_equal(same0, same1)
        self.assertFalse(np.array_equal(same0, diff))

    def test_dataset_accepts_stratified_mode_and_epoch_changes_edges_without_fixed_seed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="corr_dataset_stratified_") as tmpdir:
            hand_points = self._make_hand_points_by_distance((24, 48, 64, 96, 64))
            num_hand = int(hand_points.shape[0])
            obj_points = np.asarray(
                [
                    [0.000, 0.000, 0.000],
                    [0.004, 0.000, 0.000],
                    [0.008, 0.000, 0.000],
                    [0.012, 0.000, 0.000],
                    [0.016, 0.000, 0.000],
                    [0.020, 0.000, 0.000],
                    [0.024, 0.000, 0.000],
                    [0.028, 0.000, 0.000],
                ],
                dtype=np.float32,
            )
            obj_points = obj_points[None]
            obj_normals = np.tile(
                np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32),
                (1, obj_points.shape[1], 1),
            )
            hand_points = hand_points[None]
            hand_normals = np.tile(
                np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32),
                (1, num_hand, 1),
            )
            distance = np.linalg.norm(
                obj_points[:, :, None, :] - hand_points[:, None, :, :],
                axis=-1,
            )
            obj_to_hand_min_dist = distance.min(axis=-1).astype(np.float32)
            obj_candidate_mask = np.ones_like(obj_to_hand_min_dist, dtype=bool)
            knn_order = np.argsort(distance, axis=-1)[..., :2].astype(np.int64)
            npz_path = Path(tmpdir) / "sample_right.npz"
            np.savez(
                npz_path,
                schema_name=np.asarray("train_corr_static"),
                raw_frame_id=np.asarray([5], dtype=np.int64),
                seq_id=np.asarray("seq/test"),
                side=np.asarray("right"),
                obj_points=obj_points,
                obj_normals=obj_normals,
                obj_point_id=np.arange(obj_points.shape[1], dtype=np.int64),
                hand_points=hand_points,
                hand_normals=hand_normals,
                hand_point_id=np.arange(num_hand, dtype=np.int64),
                hand_cano_points=np.zeros((num_hand, 3), dtype=np.float32),
                hand_finger_id=np.full((num_hand,), -1, dtype=np.int64),
                hand_region_id=np.full((num_hand,), -1, dtype=np.int64),
                obj_to_hand_min_dist=obj_to_hand_min_dist,
                obj_candidate_mask_5cm=obj_candidate_mask,
                gt_obj_to_hand_knn_idx=knn_order,
            )
            dataset = CorrStaticDataset(
                npz_path,
                num_obj_points=4,
                num_hand_points=num_hand,
                logit_sampling_mode="stratified",
                logit_stratified_distance_edges=(0.005, 0.015, 0.03, 0.06),
                logit_stratified_quotas=(16, 32, 32, 32, 16),
                augment=False,
                apply_hand_perturb=False,
                fix_overfit_seed=False,
            )
            dataset.set_epoch(0)
            sample0 = dataset[0]
            dataset.set_epoch(1)
            sample1 = dataset[0]
            self.assertFalse(
                torch.equal(
                    sample0["input_obj_to_hand_logit_idx"],
                    sample1["input_obj_to_hand_logit_idx"],
                )
            )

    def test_dense_logit_neighbors_expected_rows(self) -> None:
        obj_points = np.asarray(
            [
                [0.000, 0.000, 0.000],
                [0.050, 0.000, 0.000],
                [0.100, 0.000, 0.000],
            ],
            dtype=np.float32,
        )
        hand_points = np.asarray(
            [
                [0.000, 0.000, 0.005],
                [0.010, 0.000, 0.005],
                [0.020, 0.000, 0.005],
                [0.030, 0.000, 0.005],
                [0.040, 0.000, 0.005],
            ],
            dtype=np.float32,
        )
        obj_valid = np.asarray([True, False, True], dtype=bool)
        logit_idx, logit_valid, logit_weight, near_count, far_count = _compute_dense_logit_neighbors(
            obj_points,
            hand_points,
            obj_valid,
            logit_near_radius=0.025,
            logit_far_min_radius=0.025,
        )
        self.assertEqual(logit_idx.shape, (3, 5))
        np.testing.assert_array_equal(logit_idx[0], np.asarray([0, 1, 2, 3, 4], dtype=np.int64))
        np.testing.assert_array_equal(logit_idx[2], np.asarray([0, 1, 2, 3, 4], dtype=np.int64))
        self.assertTrue(np.all(logit_valid[0]))
        self.assertTrue(np.all(logit_valid[2]))
        self.assertTrue(np.all(logit_weight[0] == 1))
        self.assertTrue(np.all(logit_weight[2] == 1))
        self.assertTrue(np.all(logit_idx[1] == -1))
        self.assertTrue(np.all(~logit_valid[1]))
        self.assertTrue(np.all(logit_weight[1] == 0))
        self.assertGreaterEqual(int(near_count[0]), 1)
        self.assertGreaterEqual(int(far_count[0]), 1)

    def test_fix_overfit_seed_makes_dataset_epoch_invariant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="corr_dataset_") as tmpdir:
            npz_path = _write_stage3_npz(Path(tmpdir))
            dataset = CorrStaticDataset(
                npz_path,
                num_obj_points=4,
                num_hand_points=5,
                k_near_logit=2,
                k_far_logit=2,
                logit_sampling_mode="balanced",
                augment=True,
                apply_hand_perturb=True,
                hand_rot_std_deg=10.0,
                hand_trans_std=0.01,
                hand_perturb_prob=1.0,
                fix_overfit_seed=True,
            )
            dataset.set_epoch(0)
            sample0 = dataset[0]
            dataset.set_epoch(10)
            sample1 = dataset[0]
            torch.testing.assert_close(sample0["selected_obj_idx"], sample1["selected_obj_idx"])
            torch.testing.assert_close(
                sample0["input_obj_to_hand_logit_idx"],
                sample1["input_obj_to_hand_logit_idx"],
            )
            torch.testing.assert_close(sample0["points"], sample1["points"])

    def test_invalid_stratified_quota_configuration_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 128"):
            CorrStaticDataset(
                data_path=Path("."),
                file_list=[Path(__file__)],
                logit_sampling_mode="stratified",
                logit_stratified_distance_edges=(0.005, 0.015, 0.03, 0.06),
                logit_stratified_quotas=(16, 32, 32, 32, 15),
            )


class DynamicKTests(unittest.TestCase):
    def test_soft_cross_edge_loss_backward_for_dynamic_k(self) -> None:
        for k in (64, 1538):
            logits = torch.zeros(1, 2, k, 1, dtype=torch.float32, requires_grad=True)
            target = torch.linspace(0.0, 1.0, k, dtype=torch.float32).view(1, 1, k).expand(1, 2, k)
            edge_weight = torch.ones(1, 2, k, dtype=torch.float32)
            edge_weight[:, 0, k // 2 :] = 0.0
            obj_valid_mask = torch.tensor([[True, True]])
            loss = CorrespondencePTV3Runner._masked_bce_with_logits_per_object(
                logits,
                target,
                edge_weight,
                obj_valid_mask,
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_model_forward_supports_dense_k_without_hardcoded_64(self) -> None:
        for k in (128, 1538):
            cfg = _corr_cfg(mode="soft")
            cfg.meta.num_obj_points = 16
            cfg.meta.num_hand_points = 1538
            cfg.meta.logit_sampling_mode = "stratified" if k == 128 else "dense"
            with mock.patch.object(correspondence_model, "PTv3DenseBackbone", FakeBackbone):
                model = StaticHOCPTv3(cfg)
                points = torch.randn(1, 16 + 1538, 3, dtype=torch.float32)
                normals = torch.randn(1, 16 + 1538, 3, dtype=torch.float32)
                batch = {
                    "points": points,
                    "normals": normals,
                    "point_valid_mask": torch.ones(1, 16 + 1538, dtype=torch.bool),
                    "runtime_obj_valid_mask": torch.ones(1, 16, dtype=torch.bool),
                    "input_obj_to_hand_ctx_idx": torch.zeros(1, 16, 1, dtype=torch.long),
                    "input_obj_to_hand_ctx_valid_mask": torch.ones(1, 16, 1, dtype=torch.bool),
                    "input_obj_to_hand_logit_idx": torch.arange(k, dtype=torch.long).view(1, 1, k).expand(1, 16, k),
                    "input_obj_to_hand_logit_valid_mask": torch.ones(1, 16, k, dtype=torch.bool),
                }
                outputs = model(batch)
                self.assertEqual(outputs["pred_cross_contact_logits"].shape, (1, 16, k, 1))
                loss = outputs["pred_cross_contact_logits"].sum()
                loss.backward()
                self.assertIsNotNone(model.cross_edge_head.weight.grad)


class VisualizerSeedTests(unittest.TestCase):
    def test_runtime_frame_uses_fixed_seed_across_epochs(self) -> None:
        data = _make_stage3_arrays()
        args = SimpleNamespace(
            base_seed=42,
            num_obj_points=4,
            fix_overfit_seed=True,
            logit_sampling_mode="balanced",
            k_ctx=2,
            ctx_radius=0.04,
            k_near_logit=2,
            k_far_logit=2,
            logit_near_radius=0.025,
            logit_far_min_radius=0.025,
            hand_perturb=True,
            hand_rot_std_deg=10.0,
            hand_trans_std=0.01,
            hand_perturb_prob=1.0,
            augment=True,
            augment_rotation=True,
            rotation_range=180.0,
            augment_translation=False,
            translation_range=0.1,
            augment_scale=False,
            scale_range=(0.9, 1.1),
            d_pos=0.005,
            d_neg=0.03,
            gamma=1.0,
        )
        runtime0 = _build_runtime_frame(data, frame=0, epoch=0, args=args)
        runtime10 = _build_runtime_frame(data, frame=0, epoch=10, args=args)
        np.testing.assert_array_equal(runtime0.selected_obj_idx, runtime10.selected_obj_idx)
        np.testing.assert_array_equal(runtime0.input_logit_idx, runtime10.input_logit_idx)
        np.testing.assert_allclose(runtime0.noisy_hand_points, runtime10.noisy_hand_points)


if __name__ == "__main__":
    unittest.main()
