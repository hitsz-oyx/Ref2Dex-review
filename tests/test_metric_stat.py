from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import torch

from src.base import BaseRunner, MetricAverager, MetricStat, TaskConfig, load_config
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner


class MetricRunner(BaseRunner):
    def make_dataloaders(self, data_cfg, seed: int):
        raise NotImplementedError

    def build_model(self, model_cfg):
        raise NotImplementedError

    def step(self, model, batch, mode="train"):
        raise NotImplementedError


class MetricStatTests(unittest.TestCase):
    def test_float_metrics_keep_batch_size_weighting(self) -> None:
        averager = MetricAverager()
        averager.update({"value": 2.0}, n=2)
        averager.update({"value": 4.0}, n=1)
        self.assertEqual(averager.compute()["value"], 8.0 / 3.0)

    def test_mixed_validity_and_real_zero(self) -> None:
        averager = MetricAverager()
        averager.update({"ap": MetricStat(0.9, 1, expose_validity=True)})
        averager.update({"ap": MetricStat.invalid(expose_validity=True)})
        averager.update({"ap": MetricStat(0.7, 1, expose_validity=True)})
        metrics = averager.compute()
        self.assertAlmostEqual(metrics["ap"], 0.8)
        self.assertEqual(metrics["ap_valid_count"], 2.0)
        self.assertEqual(metrics["ap_valid"], 1.0)

        zero = MetricAverager()
        zero.update({"score": MetricStat(0.0, 1, expose_validity=True)})
        self.assertEqual(zero.compute()["score"], 0.0)

    def test_all_invalid_omits_value_but_exposes_validity(self) -> None:
        averager = MetricAverager()
        averager.update({"ap": MetricStat.invalid(expose_validity=True)})
        metrics = averager.compute()
        self.assertNotIn("ap", metrics)
        self.assertEqual(metrics["ap_valid_count"], 0.0)
        self.assertEqual(metrics["ap_valid"], 0.0)

    def test_step_ddp_uses_total_and_count(self) -> None:
        cfg = TaskConfig()
        cfg.wandb.enable = False
        cfg.train.output_dir = tempfile.mkdtemp(prefix="metric_stat_")
        runner = MetricRunner(cfg, build_data=False)
        remote = iter(({"ap": 0.9}, {"ap": 1.0}))
        with mock.patch("src.base.base_runner.reduce_dict", side_effect=lambda *_args, **_kwargs: next(remote)):
            metrics = runner._reduce_step_metrics(
                {"ap": MetricStat(0.0, 0.0, expose_validity=True)}
            )
        self.assertEqual(metrics["ap"], 0.9)
        self.assertEqual(metrics["ap_valid_count"], 1.0)
        self.assertEqual(metrics["ap_valid"], 1.0)


class CorrespondenceMetricTests(unittest.TestCase):
    def test_auprc_and_rank_invalidity(self) -> None:
        runner = object.__new__(CorrespondencePTV3Runner)
        scores = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
        valid = torch.ones_like(scores, dtype=torch.bool)
        invalid = runner._batched_binary_auprc_stat(scores, torch.zeros_like(scores, dtype=torch.bool), valid)
        self.assertEqual(invalid.count, 0.0)
        valid_ap = runner._batched_binary_auprc_stat(scores, torch.tensor([[True, False]]), valid)
        self.assertEqual(valid_ap.count, 1.0)
        self.assertAlmostEqual(valid_ap.total, 1.0)

        target = torch.zeros(1, 1, 2)
        rank = runner._batched_cross_edge_rank_at_k_stat(
            scores.view(1, 1, 2), target, valid.view(1, 1, 2), k=1
        )
        self.assertEqual(rank.count, 0.0)

    def test_entropy_excess_and_gradient(self) -> None:
        target = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
        entropy = CorrespondencePTV3Runner._binary_entropy_floor_map(target)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertEqual(float(entropy[0]), 0.0)
        self.assertEqual(float(entropy[-1]), 0.0)
        self.assertTrue(bool((entropy[1:-1] > 0).all()))

        logits = torch.tensor([-12.0, -1.3862944, 0.0, 1.3862944, 12.0], requires_grad=True)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        excess = raw - entropy.mean().detach()
        raw_grad = torch.autograd.grad(raw, logits, retain_graph=True)[0]
        excess_grad = torch.autograd.grad(excess, logits)[0]
        torch.testing.assert_close(raw_grad, excess_grad)
        self.assertLess(float(excess.detach()), 2e-5)

    def test_per_object_excess_uses_matching_reduction(self) -> None:
        logits = torch.zeros(1, 2, 8, 1, requires_grad=True)
        target = torch.tensor([[[0.2, 0.8, 0, 0, 0, 0, 0, 0], [0.5] * 8]])
        weight = torch.ones(1, 2, 8)
        weight[:, 0, 2:] = 0
        valid = torch.tensor([[True, True]])
        raw = CorrespondencePTV3Runner._masked_bce_with_logits_per_object(logits, target, weight, valid)
        floor = CorrespondencePTV3Runner._reduce_loss_map_per_object(
            CorrespondencePTV3Runner._binary_entropy_floor_map(target), weight, valid
        )
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros(2), torch.tensor([0.2, 0.8])).mean() + torch.log(torch.tensor(2.0))) / 2
        self.assertTrue(torch.isfinite(raw - floor))
        self.assertAlmostEqual(float(raw), float(expected), places=6)

    def test_soft_loss_uses_excess_bce_and_bin_remains_finite(self) -> None:
        runner = object.__new__(CorrespondencePTV3Runner)
        cfg = load_config("src/task/correspondence_ptv3/configs/bin_contact_baseline.yaml")
        cfg.meta.num_obj_points = 2
        cfg.meta.num_hand_points = 2
        cfg.meta.loss_obj_contact_weight = 1.0
        cfg.meta.loss_cross_edge_weight = 1.0
        cfg.meta.d_pos = 0.005
        cfg.meta.d_neg = 0.03
        runner.cfg = cfg
        obj_logits = torch.zeros(1, 2, 1, requires_grad=True)
        edge_logits = torch.zeros(1, 2, 2, 1, requires_grad=True)
        points = torch.tensor([[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.0, 0.02]]])
        batch = {
            "runtime_obj_valid_mask": torch.tensor([[True, True]]),
            "input_obj_to_hand_logit_valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "input_obj_to_hand_logit_loss_weight": torch.ones(1, 2, 2),
            "input_obj_to_hand_logit_idx": torch.tensor([[[0, 1], [0, 1]]]),
            "obj_contact_label": torch.tensor([[1.0, 0.4]]),
            "gt_points": points,
            "points": points,
        }
        preds = {
            "pred_obj_contact_logits": obj_logits,
            "pred_cross_contact_logits": edge_logits,
            "pred_obj_contact_prob": torch.sigmoid(obj_logits).squeeze(-1),
            "pred_cross_contact_prob": torch.sigmoid(edge_logits).squeeze(-1),
        }
        losses, metrics = runner._compute_losses(preds, batch)
        self.assertIn("cross_edge_excess_bce", metrics)
        self.assertLess(float(losses["cross_edge_contact"].detach()), float(metrics["cross_edge_raw_bce"]))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(obj_logits.grad).all())
        self.assertTrue(torch.isfinite(edge_logits.grad).all())

        runner = object.__new__(CorrespondencePTV3Runner)
        cfg.meta.contact_supervision_mode = "bin"
        runner.cfg = cfg
        bin_obj = torch.zeros(1, 2, 10, requires_grad=True)
        bin_edge = torch.zeros(1, 2, 2, 10, requires_grad=True)
        bin_preds = {
            "pred_obj_contact_logits": bin_obj,
            "pred_cross_contact_logits": bin_edge,
            "pred_obj_contact_prob": torch.full((1, 2), 0.5),
            "pred_cross_contact_prob": torch.full((1, 2, 2), 0.5),
        }
        bin_losses, bin_metrics = runner._compute_losses(bin_preds, batch)
        self.assertNotIn("cross_edge_excess_bce", bin_metrics)
        self.assertTrue(torch.isfinite(sum(bin_losses.values())))

    def test_formal_yamls_default_to_soft(self) -> None:
        for path in (
            "src/task/correspondence_ptv3/configs/bin_contact_baseline.yaml",
            "src/task/correspondence_ptv3/configs/bin_contact_multigpu_gb256.yaml",
        ):
            cfg = load_config(path)
            self.assertEqual(cfg.meta.contact_supervision_mode, "soft")
            self.assertIsNone(cfg.meta.contact_bin_weights)
            self.assertIsNone(cfg.meta.edge_contact_bin_weights)
            self.assertIn("soft_contact", cfg.name)
            self.assertIn("soft-contact", cfg.wandb.tags)


if __name__ == "__main__":
    unittest.main()
