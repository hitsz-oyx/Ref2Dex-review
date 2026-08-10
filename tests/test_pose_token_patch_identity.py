import torch

from src.task.Posetoken.dataset import build_canonical_patch_map


def test_patch_map_depends_only_on_canonical_surface():
    generator = torch.Generator().manual_seed(7)
    canonical = torch.randn(128, 3, generator=generator)
    patch = build_canonical_patch_map(canonical, num_patches=8, patch_size=12)
    posed_surfaces = [canonical, canonical + torch.randn(canonical.shape, generator=generator) * .1,
                      canonical * torch.tensor([1.0, .7, 1.2])]
    sample_patch_indices = [patch for _ in posed_surfaces]
    assert all(torch.equal(sample_patch_indices[0], value) for value in sample_patch_indices[1:])
    assert patch.shape == (8, 12)


def test_morphology_pairs_share_pose_target_but_have_different_inputs():
    from src.task.Posetoken.dataset import SyntheticManoPoseDataset
    dataset = SyntheticManoPoseDataset(
        mano_path="dataset/arctic/data/body_models/mano", side="right",
        num_samples=2, seed=9, beta_std=.75, pose_group_size=2,
        generation_batch_size=2)
    first, second = dataset[0], dataset[1]
    torch.testing.assert_close(first["mano_pose"], second["mano_pose"])
    torch.testing.assert_close(first["hand_target_points_root"],
                               second["hand_target_points_root"])
    assert not torch.allclose(first["hand_points_root"], second["hand_points_root"])
    assert torch.equal(first["patch_knn_idx"], second["patch_knn_idx"])


def test_parameter_dataset_keeps_only_mano_parameters_and_pair_identity():
    from src.task.Posetoken.dataset import ManoPoseParameterDataset
    dataset = ManoPoseParameterDataset(
        mano_path="dataset/arctic/data/body_models/mano", side="right",
        num_samples=2, seed=9, beta_std=.75, pose_group_size=2)
    assert not hasattr(dataset, "hand_points_root")
    first, second = dataset[0], dataset[1]
    torch.testing.assert_close(first["mano_pose"], second["mano_pose"])
    assert not torch.allclose(first["mano_beta"], second["mano_beta"])
    assert first["patch_knn_idx"].shape == (64, 32)
