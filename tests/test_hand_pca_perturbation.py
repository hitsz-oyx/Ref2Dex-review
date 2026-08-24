from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.base import cleanup_distributed, load_config
from src.task.correspondence_ptv3_v2.mano_recon import (
    MANOConfig,
    MANOLayerCache,
    resolve_mano_model_dir,
)
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner


def _runner_without_init() -> CorrespondencePTV3V2Runner:
    return object.__new__(CorrespondencePTV3V2Runner)


def _standard_noise(seeds: list[int], width: int) -> torch.Tensor:
    rows = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        rows.append(torch.randn(width, generator=generator))
    return torch.stack(rows)


def test_relative_mano_model_dir_is_anchored_to_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected = (repo_root / "dataset/arctic/data/body_models/mano").resolve()
    assert resolve_mano_model_dir("dataset/arctic/data/body_models/mano") == expected


def test_pca_scalar_std_is_not_replaced_by_noise_scale() -> None:
    runner = _runner_without_init()
    seeds = [7, 11]
    pose = torch.zeros((2, 5), dtype=torch.float32)

    actual = runner._apply_pca_noise(
        hand_pose_g=pose,
        num_pca_comps=3,
        indices=[0, 1],
        seeds_list=seeds,
        per_sample_apply=[True, True],
        hand_pca_std=0.5,
        hand_pca_scale=0.1,
        hand_pca_clip=0.25,
        per_dim_std_t=None,
        hand_perturb_prob=1.0,
    )

    expected = _standard_noise(seeds, 3).clamp(-0.25, 0.25) * 0.5
    torch.testing.assert_close(actual[:, :3], expected)
    torch.testing.assert_close(actual[:, 3:], torch.zeros((2, 2)))


def test_pca_per_dim_std_uses_scale_and_sigma_clip() -> None:
    runner = _runner_without_init()
    seeds = [13, 17]
    pose = torch.zeros((2, 3), dtype=torch.float32)
    train_std = torch.tensor([1.0, 2.0, 4.0])

    actual = runner._apply_pca_noise(
        hand_pose_g=pose,
        num_pca_comps=3,
        indices=[0, 1],
        seeds_list=seeds,
        per_sample_apply=[True, True],
        hand_pca_std=0.5,
        hand_pca_scale=0.1,
        hand_pca_clip=0.25,
        per_dim_std_t=train_std,
        hand_perturb_prob=1.0,
    )

    expected = _standard_noise(seeds, 3).clamp(-0.25, 0.25) * train_std * 0.1
    torch.testing.assert_close(actual, expected)
    assert torch.all(actual.abs() <= train_std * 0.1 * 0.25 + 1e-7)


def test_geometry_calibrated_pca_std_overrides_legacy_coefficient_scale() -> None:
    runner = _runner_without_init()
    seeds = [13]
    geometry_std = torch.tensor([0.1, 0.2, 0.4])
    actual = runner._apply_pca_noise(
        hand_pose_g=torch.zeros((1, 3)),
        num_pca_comps=3,
        indices=[0],
        seeds_list=seeds,
        per_sample_apply=[True],
        hand_pca_std=99.0,
        hand_pca_scale=99.0,
        hand_pca_clip=99.0,
        per_dim_std_t=torch.full((3,), 99.0),
        hand_perturb_prob=1.0,
        geometry_std_t=geometry_std,
        geometry_clip_sigma=0.25,
    )
    expected = _standard_noise(seeds, 3).clamp(-0.25, 0.25) * geometry_std
    torch.testing.assert_close(actual, expected)


def test_geometry_calibrated_axis_angle_uses_per_dimension_std() -> None:
    runner = _runner_without_init()
    seed = 23
    geometry_std = torch.tensor([0.01, 0.02, 0.04])
    actual = runner._apply_axis_angle_noise(
        hand_pose_g=torch.zeros((1, 3)),
        indices=[0],
        seeds_list=[seed],
        per_sample_apply=[True],
        hand_aa_std=99.0,
        hand_aa_clip=99.0,
        hand_perturb_prob=1.0,
        geometry_std_t=geometry_std,
        geometry_clip_sigma=0.25,
    )
    expected = _standard_noise([seed ^ 0xAA], 3).clamp(-0.25, 0.25) * geometry_std
    torch.testing.assert_close(actual, expected)


def test_pca_gate_respects_per_sample_flag() -> None:
    runner = _runner_without_init()
    actual = runner._apply_pca_noise(
        hand_pose_g=torch.zeros((2, 3)),
        num_pca_comps=3,
        indices=[0, 1],
        seeds_list=[19, 23],
        per_sample_apply=[False, True],
        hand_pca_std=0.5,
        hand_pca_scale=0.1,
        hand_pca_clip=3.0,
        per_dim_std_t=None,
        hand_perturb_prob=1.0,
    )

    torch.testing.assert_close(actual[0], torch.zeros(3))
    assert not torch.equal(actual[1], torch.zeros(3))
    assert not runner._sample_passed_gate(
        i_global=0,
        seed=0,
        per_sample_apply=[True],
        hand_perturb_prob=0.0,
    )


def test_v_template_sha_accepts_dataloader_string_batches() -> None:
    normalize = CorrespondencePTV3V2Runner._normalize_v_template_sha
    assert normalize(["sha-a", "sha-b"], 2) == ["sha-a", "sha-b"]
    assert normalize(("sha-a", "sha-b"), 2) == ["sha-a", "sha-b"]
    assert normalize("sha-a", 2) == ["sha-a", "sha-a"]
    assert normalize(None, 2) == [None, None]


def test_prepare_batch_runs_geometry_pipeline_without_grad(monkeypatch) -> None:
    runner = _runner_without_init()
    runner.device = torch.device("cpu")
    runner.cfg = SimpleNamespace(
        meta=SimpleNamespace(
            use_mano_reconstruction=True,
            apply_hand_perturb=True,
        )
    )
    calls: list[str] = []

    def reconstruct(batch, *, side) -> None:
        assert side == ["right"]
        assert not torch.is_grad_enabled()
        calls.append("reconstruct")

    def resample(batch) -> None:
        assert not torch.is_grad_enabled()
        calls.append("resample")

    def build_supervision(batch, *, contact_seed_list) -> None:
        assert not torch.is_grad_enabled()
        assert contact_seed_list == [41]
        calls.append("supervision")

    monkeypatch.setattr(runner, "_reconstruct_hand_from_mano", reconstruct)
    monkeypatch.setattr(runner, "_resample_object_from_perturbed_hand", resample)
    monkeypatch.setattr(runner, "_build_supervision_gpu", build_supervision)

    batch = {
        "__mano_side__": ["right"],
        "contact_seed": torch.tensor([41]),
        "has_mano": torch.tensor([True]),
        "mano_global_orient": torch.zeros((1, 3)),
        "mano_transl": torch.zeros((1, 3)),
        "mano_pose": torch.zeros((1, 24)),
        "mano_betas": torch.zeros((1, 10)),
    }
    runner.prepare_batch(batch)
    assert calls == ["reconstruct", "resample", "supervision"]


def test_mano_cache_freezes_new_and_subject_specific_layers(tmp_path, monkeypatch) -> None:
    class FakeMANO(torch.nn.Module):
        def __init__(self, _path: str, **kwargs) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            value = kwargs.get("v_template")
            if value is None:
                value = np.zeros((4, 3), dtype=np.float32)
            self.register_buffer("v_template", torch.as_tensor(value, dtype=torch.float32))

    fake_smplx = ModuleType("smplx")
    fake_smplx.MANO = FakeMANO
    monkeypatch.setitem(sys.modules, "smplx", fake_smplx)
    (tmp_path / "MANO_RIGHT.pkl").touch()

    cache = MANOLayerCache(model_dir=tmp_path, device="cpu")
    cfg = MANOConfig(
        side="right",
        use_pca=True,
        num_pca_comps=3,
        flat_hand_mean=True,
        v_template_sha=None,
        v_template_shape=None,
    )
    base_layer = cache.get(cfg)
    assert not base_layer.training
    assert all(not parameter.requires_grad for parameter in base_layer.parameters())

    subject_template = np.ones((4, 3), dtype=np.float32)
    subject_layer, _ = cache.get_or_build_for_v_template(
        side="right",
        use_pca=True,
        num_pca_comps=3,
        flat_hand_mean=True,
        v_template=subject_template,
    )
    assert not subject_layer.training
    assert all(not parameter.requires_grad for parameter in subject_layer.parameters())
    subject_sha, subject_shape = cache._v_template_id(subject_template)
    cached = cache.get_cached_for_v_template_id(
        side="right",
        use_pca=True,
        num_pca_comps=3,
        flat_hand_mean=True,
        v_template_sha=subject_sha,
        v_template_shape=subject_shape,
    )
    assert cached is not None and cached[0] is subject_layer


def test_runner_prepare_batch_gpu_smoke() -> None:
    data_root_value = os.environ.get("REF2DEX_STAGE3_GPU_SMOKE_ROOT")
    if not data_root_value:
        pytest.skip("REF2DEX_STAGE3_GPU_SMOKE_ROOT is unset")
    data_root = Path(data_root_value)
    if not data_root.exists():
        pytest.skip(f"REF2DEX_STAGE3_GPU_SMOKE_ROOT does not exist: {data_root}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    cfg = load_config(
        "src/task/correspondence_ptv3_v2/configs/smoke_test_v21_mano.yaml"
    )
    cfg.data.train_path = str(data_root)
    cfg.data.val_path = str(data_root)
    cfg.data.batch_size = 2
    cfg.data.val_batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.meta.hand_perturb_prob = 1.0
    model_dir = os.environ.get("REF2DEX_MANO_MODEL_DIR")
    if model_dir:
        cfg.meta.mano_model_dir = model_dir
    cfg.train.device = "cuda"
    cfg.train.distributed.enable = False

    runner = CorrespondencePTV3V2Runner(
        cfg, mode="train", device="cuda", build_data=False
    )
    try:
        (
            train_loader,
            _val_loader,
            _test_loader,
            metadata,
            val_loaders,
            _test_loaders,
        ) = runner.make_dataloaders(cfg.data, seed=cfg.train.seed)
        runner.train_dataset = train_loader.dataset
        runner.configure_data(metadata, train_loader.dataset)

        train_batch = runner.prepare_batch(next(iter(train_loader)))
        val_clean_batch = runner.prepare_batch(next(iter(val_loaders["val_clean/"])))

        for batch in (train_batch, val_clean_batch):
            batch_size = int(batch["points"].shape[0])
            assert batch["points"].shape == (batch_size, 512 + 1538, 3)
            assert torch.isfinite(batch["points"]).all()
            assert torch.isfinite(batch["normals"]).all()
            assert batch["runtime_obj_valid_mask"].all()
            assert not batch["points"].requires_grad
            selected = batch["runtime_obj_selected_idx"]
            pool_size = int(batch["full_input_obj_points"].shape[1])
            assert ((selected >= 0) & (selected < pool_size)).all()
            for row in selected:
                assert torch.unique(row).numel() == row.numel()

        train_hand = train_batch["points"][:, -1538:]
        train_gt_hand = train_batch["gt_points"][:, -1538:]
        assert not torch.allclose(train_hand, train_gt_hand, atol=1e-5)

        val_hand = val_clean_batch["points"][:, -1538:]
        val_gt_hand = val_clean_batch["gt_points"][:, -1538:]
        torch.testing.assert_close(val_hand, val_gt_hand, atol=1e-5, rtol=0.0)
    finally:
        cleanup_distributed()


# ---------------------------------------------------------------------------
# Fix #3 / Fix #4 (docs/指导.md):
#   * Path resolution for hand-noise calibration must be repo-relative.
#   * Legacy mode (use_mano_reconstruction=False) must be incompatible with
#     runtime_resample_object=True at runner construction time.
# ---------------------------------------------------------------------------


def test_hand_noise_profiles_resolve_relative_to_base_dir() -> None:
    """Fix #3: relative ``hand_geometry_calibration_paths`` must resolve
    against the supplied ``base_dir`` (the repo root), NOT ``Path.cwd()``."""
    from src.task.correspondence_ptv3_v2.hand_noise_profiles import (
        HandGeometryNoiseProfiles,
    )

    repo_root = Path(__file__).resolve().parents[1]
    relative = "src/task/correspondence_ptv3_v2/calibration/grab_pca24_target_9mm.json"
    absolute = repo_root / relative

    # Sanity: the file we depend on is actually shipped with the repo now.
    assert absolute.exists(), (
        f"Calibration JSON missing from tracked location: {absolute}. "
        "Re-run tools/calibrate_mano_geometry_noise.py or check the "
        "fix-2 commit."
    )

    # Run the loader from a different cwd; the relative path must still
    # be found because base_dir is the repo root.
    cwd = os.getcwd()
    try:
        os.chdir("/tmp")
        profiles = HandGeometryNoiseProfiles.from_paths(
            {"grab": relative},
            base_dir=repo_root,
        )
    finally:
        os.chdir(cwd)

    assert len(profiles._profiles) > 0
    for profile in profiles._profiles.values():
        assert profile.source_path == str(absolute)


def test_runner_requires_stored_proxy_for_legacy_runtime_resample() -> None:
    """Fix #4: ``use_mano_reconstruction=False`` + ``runtime_resample_object=True``
    is supported only with an explicit stored-hand proxy index.  Without the
    index, construction must fail instead of silently instantiating MANO or
    falling back to linspace indices."""
    from src.base import TaskConfig

    cfg = TaskConfig()
    cfg.meta.use_mano_reconstruction = False
    cfg.meta.runtime_resample_object = True

    runner = _runner_without_init()
    runner.cfg = cfg
    with pytest.raises(ValueError, match="requires meta.stored_hand_proxy_indices_path"):
        CorrespondencePTV3V2Runner.__init__(runner, cfg, mode="train", build_data=False)
