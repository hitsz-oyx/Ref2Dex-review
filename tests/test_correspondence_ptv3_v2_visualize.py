from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.task.correspondence_ptv3_v2.visualize import (
    HtmlVisualizer,
    ViewerState,
    _prepare_single_batch,
)


class _DatasetStub:
    def __init__(self) -> None:
        obj_points = torch.tensor(
            [[0.02, 0.00, 0.00], [0.03, 0.00, 0.00], [0.04, 0.00, 0.00]]
        )
        hand_points = torch.tensor(
            [[0.00, 0.00, 0.00], [0.00, 0.01, 0.00], [0.00, 0.02, 0.00], [0.00, 0.03, 0.00]]
        )
        points = torch.cat([obj_points, hand_points], dim=0)
        normals = torch.zeros_like(points)
        normals[:, 2] = 1.0
        self.sample = {
            "points": points,
            "normals": normals,
            "gt_points": points.clone(),
            "gt_normals": normals.clone(),
            "runtime_obj_valid_mask": torch.tensor([True, False, True]),
            "num_obj_points": torch.tensor(3),
            "num_hand_points": torch.tensor(4),
            "hand_min_dist": torch.tensor([0.02, 0.022, 0.028, 0.04]),
        }
        self.epoch = 0

    def __len__(self) -> int:
        return 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        return self.sample


class _ManoDatasetStub(_DatasetStub):
    def __init__(self, *, use_pca: bool) -> None:
        super().__init__()
        pose_dim = 24 if use_pca else 45
        self.sample.update(
            {
                "has_mano": torch.tensor(True),
                "apply_hand_perturb": torch.tensor(False),
                "hand_perturb_seed": torch.tensor(1234),
                "__mano_side__": "right",
                "mano_use_pca": torch.tensor(use_pca),
                "mano_num_pca_comps": torch.tensor(pose_dim),
                "mano_pose_dim": torch.tensor(pose_dim),
                "mano_pose": torch.zeros((45,)),
            }
        )


class _ManoRunnerStub:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            meta=SimpleNamespace(
                apply_hand_perturb=False,
                hand_perturb_prob=0.8,
                hand_pca_std=0.2,
                hand_pca_noise_scale=0.1,
                hand_axis_angle_std_rad=0.05,
            )
        )
        self.calls: list[dict[str, float | int | bool]] = []

    def _reconstruct_hand_from_mano(self, batch, *, side) -> None:
        del side
        use_pca = bool(batch["mano_use_pca"].reshape(-1)[0].item())
        magnitude = (
            float(self.cfg.meta.hand_pca_std)
            if use_pca
            else float(self.cfg.meta.hand_axis_angle_std_rad)
        )
        seed = int(batch["hand_perturb_seed"].reshape(-1)[0].item())
        self.calls.append(
            {
                "use_pca": use_pca,
                "magnitude": magnitude,
                "seed": seed,
                "prob": float(self.cfg.meta.hand_perturb_prob),
            }
        )
        num_hand = int(batch["num_hand_points"].reshape(-1)[0].item())
        batch["points"][:, -num_hand:, 0] += magnitude


def _make_gt_visualizer() -> HtmlVisualizer:
    return HtmlVisualizer(
        runner=None,
        dataset=_DatasetStub(),
        state=ViewerState(),
        marker_radius=0.003,
        vis_contact_radius=0.02,
        perturb_translation_step=0.005,
        perturb_rotation_step_deg=5.0,
    )


def _make_mano_visualizer(*, use_pca: bool) -> tuple[HtmlVisualizer, _ManoRunnerStub]:
    mano_runner = _ManoRunnerStub()
    visualizer = HtmlVisualizer(
        runner=None,
        dataset=_ManoDatasetStub(use_pca=use_pca),
        state=ViewerState(),
        marker_radius=0.003,
        vis_contact_radius=0.02,
        perturb_translation_step=0.005,
        perturb_rotation_step_deg=5.0,
        mano_runner=mano_runner,
        hand_pca_std=0.2,
        hand_pca_noise_scale=0.1,
        hand_pca_noise_clip=3.0,
        hand_axis_angle_std_rad=0.05,
        hand_axis_angle_clip_rad=0.15,
        hand_perturb_prob=0.8,
    )
    return visualizer, mano_runner


def test_gt_only_visualizer_filters_invalid_points_and_has_finite_scene() -> None:
    visualizer = _make_gt_visualizer()
    state = visualizer.get_state_dict()
    scene = visualizer.get_scene_dict()

    assert state["has_model"] is False
    assert state["has_hand_heatmap_head"] is False
    assert len(scene["obj_points"]) == 2
    assert len(scene["obj_colors"]) == 2
    assert len(scene["hand_points"]) == 4
    assert len(scene["hand_colors"]) == 4
    for key in ("obj_points", "obj_colors", "hand_points", "hand_colors"):
        assert np.isfinite(np.asarray(scene[key], dtype=np.float32)).all()


def test_gt_only_visualizer_allows_gt_heatmap_but_rejects_eval() -> None:
    visualizer = _make_gt_visualizer()

    state = visualizer.apply_action("set_mode", {"mode": "hand_heatmap"})
    assert state["display_mode"] == "hand_heatmap"
    scene = visualizer.get_scene_dict()
    assert len(scene["hand_colors"]) == 4

    state = visualizer.apply_action("set_show_gt", {"show_gt": False})
    assert state["show_gt"] is True
    state = visualizer.apply_action("toggle_gt", {})
    assert state["show_gt"] is True


def test_gt_only_visualizer_object_perturbation_moves_visible_object() -> None:
    visualizer = _make_gt_visualizer()
    before = np.asarray(visualizer.get_scene_dict()["obj_points"], dtype=np.float32)

    visualizer.apply_action("toggle_perturb", {})
    visualizer.apply_action("set_translation", {"translation": [0.01, 0.0, 0.0]})
    after = np.asarray(visualizer.get_scene_dict()["obj_points"], dtype=np.float32)

    np.testing.assert_allclose(
        after - before,
        np.broadcast_to(np.asarray([0.01, 0.0, 0.0]), after.shape),
        atol=1e-7,
    )


def test_prepare_single_batch_preserves_mano_strings_and_batches_scalars() -> None:
    prepared = _prepare_single_batch(
        {
            "num_obj_points": torch.tensor(512),
            "points": torch.zeros((2050, 3)),
            "__mano_side__": "left",
            "mano_v_template_sha": "abc123",
        }
    )

    assert prepared["num_obj_points"].shape == (1,)
    assert prepared["points"].shape == (1, 2050, 3)
    assert prepared["__mano_side__"] == ["left"]
    assert prepared["mano_v_template_sha"] == ["abc123"]


@pytest.mark.parametrize(
    ("use_pca", "representation", "base_magnitude"),
    [(True, "pca", 0.2), (False, "axis_angle", 0.05)],
)
def test_gt_only_mano_perturbation_uses_representation_and_yaml_strength(
    use_pca: bool,
    representation: str,
    base_magnitude: float,
) -> None:
    visualizer, mano_runner = _make_mano_visualizer(use_pca=use_pca)
    before = visualizer.get_scene_dict()

    state = visualizer.apply_action("toggle_hand_perturb", {})
    after = visualizer.get_scene_dict()

    assert state["hand_perturbation_available"] is True
    assert state["hand_perturbation_representation"] == representation
    assert state["hand_perturbation_enabled"] is True
    assert mano_runner.calls[-1]["use_pca"] is use_pca
    assert mano_runner.calls[-1]["magnitude"] == pytest.approx(base_magnitude)
    assert mano_runner.calls[-1]["prob"] == 1.0
    np.testing.assert_allclose(after["obj_points"], before["obj_points"])
    np.testing.assert_allclose(
        np.asarray(after["hand_points"])[:, 0] - np.asarray(before["hand_points"])[:, 0],
        base_magnitude,
    )

    state = visualizer.apply_action("set_hand_perturb_strength", {"strength": 2.0})
    assert state["hand_perturbation_strength"] == 2.0
    assert mano_runner.calls[-1]["magnitude"] == pytest.approx(base_magnitude * 2.0)
    # Temporary visualization overrides must not leak into the runner config.
    assert mano_runner.cfg.meta.hand_perturb_prob == 0.8
    assert mano_runner.cfg.meta.hand_pca_std == 0.2
    assert mano_runner.cfg.meta.hand_axis_angle_std_rad == 0.05


def test_gt_only_mano_variant_changes_seed_and_reset_restores_clean_hand() -> None:
    visualizer, mano_runner = _make_mano_visualizer(use_pca=True)
    clean_hand = np.asarray(visualizer.get_scene_dict()["hand_points"])
    visualizer.apply_action("toggle_hand_perturb", {})
    seed_zero = int(mano_runner.calls[-1]["seed"])

    state = visualizer.apply_action("next_hand_variant", {})
    assert state["hand_perturbation_variant"] == 1
    assert int(mano_runner.calls[-1]["seed"]) != seed_zero

    state = visualizer.apply_action("reset_hand_perturb", {})
    reset_hand = np.asarray(visualizer.get_scene_dict()["hand_points"])
    assert state["hand_perturbation_enabled"] is False
    assert state["hand_perturbation_strength"] == 1.0
    assert state["hand_perturbation_variant"] == 0
    np.testing.assert_allclose(reset_hand, clean_hand)
