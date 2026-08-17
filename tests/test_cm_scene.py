"""Parity tests for Cm Scene Cache V1 (docs/指导/V1.md §26).

Test A  object-only regression: legacy NPZ dataset vs scene dataset (env
        disabled) must produce identical samples for the same seed/frame/
        stride/bank.
Test B  ragged candidate parity: np.nonzero(mask) vs candidate_indices[
        offsets[t]:offsets[t+1]] must be element-wise identical.
Test C  sampling bank determinism and overlap diversity across banks.
Test D  dense cache parity: online stub encoder vs cached FP16 features,
        plus strict fingerprint validation.
Test E  loss parity: with environment disabled, the head loss on legacy and
        scene samples agrees to floating-point error.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from process.GRAB.stage4_cm_scene import compute_scene_candidate_ragged
from src.task.Cm.build_dense_cache import build_dense_cache
from src.task.Cm.build_sampling_bank import build_all_banks
from src.task.Cm.cache_schema import SCHEMA_NAME, SCHEMA_VERSION, load_mmap
from src.task.Cm.compute_flow_scale import calibrate_flow_scale, calibrate_flow_scale_scene
from src.task.Cm.dataset import Stage4CmDataset
from src.task.Cm.dataset_scene import Stage4CmSceneDataset
from src.task.Cm.model import CmFlowHead
from src.task.Cm.runner import scaled_flow_smooth_l1


FRAMES = 16
OBJ_POOL = 4096
HAND_POINTS = 32
DS_RATE = 4
SOURCE_FPS = 120.0
SEQ_ID = "unit"


def _ball(rng: np.random.Generator, count: int, center: np.ndarray, radius: float) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.clip(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8, None)
    radii = radius * rng.random(count) ** (1.0 / 3.0)
    return center[None, :] + directions * radii[:, None]


def _build_geometry(
    *,
    num_obj_near: int,
    num_obj_far: int,
    num_env_near: int,
    num_env_far: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obj_near = _ball(rng, num_obj_near, np.array([0.03, 0.0, 0.0], np.float32), 0.015)
    obj_far = _ball(rng, num_obj_far, np.array([0.5, 0.0, 0.0], np.float32), 0.05)
    obj_base = np.concatenate([obj_near, obj_far], axis=0).astype(np.float32)
    env_near = _ball(rng, num_env_near, np.array([-0.03, 0.01, 0.0], np.float32), 0.01)
    env_far = _ball(rng, num_env_far, np.array([0.0, -0.5, 0.0], np.float32), 0.05)
    env = np.concatenate([env_near, env_far], axis=0).astype(np.float32)
    # Object translates along +Y (perpendicular to the hand offset, so the
    # near-candidate count stays stable across frames); environment static.
    drift = np.arange(FRAMES, dtype=np.float32)[:, None, None] * np.array([0, 0.001, 0], np.float32)[None, None]
    obj_world = (obj_base[None] + drift).astype(np.float32)
    hand_base = _ball(rng, HAND_POINTS, np.zeros(3, np.float32), 0.01).astype(np.float32)
    hand_world = (hand_base[None] + np.arange(FRAMES, dtype=np.float32)[:, None, None] * np.array([0, 0.001, 0], np.float32)[None, None]).astype(np.float32)
    rotation = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]], np.float32)
    pose = np.tile(np.eye(4, dtype=np.float32), (FRAMES, 1, 1))
    pose[:, :3, :3] = rotation
    pose[:, 2, 3] = 0.01 * np.arange(FRAMES, dtype=np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (obj_base.shape[0], 1))
    return {
        "obj_world": obj_world,
        "obj_normals": normals,
        "env_points": env,
        "env_normals": np.tile(np.array([0.0, 1.0, 0.0], np.float32), (env.shape[0], 1)),
        "hand_world": hand_world,
        "hand_normals": np.tile(np.array([0.0, 0.0, -1.0], np.float32), (hand_world.shape[1], 1)),
        "hand_pose": pose,
    }


def _write_scene_sequence(
    root: Path,
    subject: str,
    name: str,
    geometry: dict[str, np.ndarray],
    *,
    threshold: float = 0.05,
) -> Path:
    sequence_dir = root / subject / name
    shared_dir = sequence_dir / "shared"
    shared_dir.mkdir(parents=True)
    env = geometry["env_points"]
    frames = geometry["obj_world"].shape[0]
    num_env = env.shape[0]
    scene_world = np.concatenate(
        [geometry["obj_world"], np.broadcast_to(env[None], (frames,) + env.shape)], axis=1
    ).astype(np.float32)
    offsets, indices = compute_scene_candidate_ragged(
        scene_world,
        geometry["hand_world"],
        geometry["hand_pose"],
        candidate_threshold=threshold,
        frame_batch_size=4,
        device=torch.device("cpu"),
    )
    raw_frame_id = np.arange(frames, dtype=np.int32) * DS_RATE
    obj_normals = np.broadcast_to(
        geometry["obj_normals"][None].astype(np.float32), geometry["obj_world"].shape
    )
    np.save(shared_dir / "raw_frame_id.npy", raw_frame_id)
    np.save(shared_dir / "obj_points_world.npy", geometry["obj_world"].astype(np.float32))
    np.save(shared_dir / "obj_normals_world.npy", np.ascontiguousarray(obj_normals))
    np.save(shared_dir / "env_points_world.npy", env.astype(np.float32))
    np.save(shared_dir / "env_normals_world.npy", geometry["env_normals"].astype(np.float32))
    scene_source_id = np.concatenate([
        np.zeros(OBJ_POOL, np.uint8),
        np.ones(num_env, np.uint8),
    ])
    np.save(shared_dir / "scene_source_id.npy", scene_source_id)
    (shared_dir / "meta.json").write_text(
        json.dumps({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "seq_id": SEQ_ID,
            "dataset_name": "grab",
            "subject_id": subject,
            "seq_name": name,
            "object_name": "unit",
            "source_raw_file": f"{subject}/{name}.npz",
            "ds_rate": DS_RATE,
            "source_fps": SOURCE_FPS,
            "coordinate_frame": "world",
            "environment_storage": "static_world",
            "environment_assets": ["table"] if num_env else [],
        }),
        encoding="utf-8",
    )
    side_dir = sequence_dir / "right"
    side_dir.mkdir()
    np.save(side_dir / "hand_points_world.npy", geometry["hand_world"].astype(np.float32))
    np.save(
        side_dir / "hand_normals_world.npy",
        np.broadcast_to(
            geometry["hand_normals"][None].astype(np.float32),
            (frames, geometry["hand_normals"].shape[0], 3),
        ),
    )
    np.save(side_dir / "hand_root_pose_world.npy", geometry["hand_pose"].astype(np.float32))
    np.save(side_dir / "candidate_offsets.npy", offsets)
    np.save(side_dir / "candidate_indices.npy", indices)
    return sequence_dir


def _write_root_meta(root: Path, num_env: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(
        json.dumps({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "scene_pool": {
                "object_points": OBJ_POOL,
                "environment_points": int(num_env),
                "total_points": OBJ_POOL + int(num_env),
            },
            "candidate_threshold_m": 0.05,
            "num_hand_points": HAND_POINTS,
            "ds_rate": DS_RATE,
            "source_fps": SOURCE_FPS,
            "environment_storage": "static_world",
        }),
        encoding="utf-8",
    )


def _write_legacy_sequence(sequence_dir: Path, geometry: dict[str, np.ndarray]) -> Path:
    legacy_dir = sequence_dir.parent / (sequence_dir.name + "_legacy")
    legacy_dir.mkdir(parents=True, exist_ok=True)
    frames = geometry["obj_world"].shape[0]
    scene_world = np.concatenate(
        [
            geometry["obj_world"],
            np.broadcast_to(geometry["env_points"][None], (frames,) + geometry["env_points"].shape),
        ],
        axis=1,
    ).astype(np.float32)
    from process.common.stage4_cm import compute_current_candidate_mask

    scene_mask = compute_current_candidate_mask(
        scene_world,
        geometry["hand_world"],
        geometry["hand_pose"],
        candidate_threshold=0.05,
        frame_batch_size=4,
        device=torch.device("cpu"),
    )
    raw_frame_id = np.arange(frames, dtype=np.int32) * DS_RATE
    np.savez(
        legacy_dir / "shared.npz",
        schema_name=np.asarray("ref2dex_cm_sequence_shared"),
        schema_version=np.asarray("3.0.0"),
        coordinate_frame=np.asarray("world"),
        ds_rate=np.asarray(DS_RATE, dtype=np.int32),
        source_fps=np.asarray(SOURCE_FPS, dtype=np.float32),
        seq_id=np.asarray(SEQ_ID),
        dataset_name=np.asarray("grab"),
        subject_id=np.asarray(sequence_dir.parent.name),
        seq_name=np.asarray(sequence_dir.name),
        object_name=np.asarray("unit"),
        source_raw_file=np.asarray("unit.npz"),
        raw_frame_id=raw_frame_id,
        obj_points_world=geometry["obj_world"].astype(np.float32),
        obj_normals_world=np.broadcast_to(
            geometry["obj_normals"][None].astype(np.float32), geometry["obj_world"].shape
        ),
        obj_point_id=np.arange(OBJ_POOL, dtype=np.int32),
    )
    np.savez(
        legacy_dir / "right.npz",
        schema_name=np.asarray("ref2dex_cm_sequence_hand"),
        schema_version=np.asarray("3.0.0"),
        side=np.asarray("right"),
        hand_points_world=geometry["hand_world"].astype(np.float32),
        hand_normals_world=np.broadcast_to(
            geometry["hand_normals"][None].astype(np.float32),
            (frames, geometry["hand_normals"].shape[0], 3),
        ),
        hand_root_pose_world=geometry["hand_pose"].astype(np.float32),
        obj_candidate_mask_5cm=scene_mask[:, :OBJ_POOL],
    )
    return legacy_dir


def _geometry_with_env(num_env_near: int, num_env_far: int, *, seed: int = 7):
    return _build_geometry(
        num_obj_near=600,
        num_obj_far=OBJ_POOL - 600,
        num_env_near=num_env_near,
        num_env_far=num_env_far,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Test B: ragged candidate parity (V1.md §26 B)
# ---------------------------------------------------------------------------
def test_ragged_candidate_parity() -> None:
    from process.common.stage4_cm import compute_current_candidate_mask

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        geometry = _geometry_with_env(100, 412)
        sequence_dir = _write_scene_sequence(root, "s1", "unit", geometry)
        offsets = np.load(sequence_dir / "right" / "candidate_offsets.npy")
        indices = np.load(sequence_dir / "right" / "candidate_indices.npy")
        num_env = geometry["env_points"].shape[0]
        frames = geometry["obj_world"].shape[0]
        scene_world = np.concatenate(
            [
                geometry["obj_world"],
                np.broadcast_to(geometry["env_points"][None], (frames,) + geometry["env_points"].shape),
            ],
            axis=1,
        ).astype(np.float32)
        mask = compute_current_candidate_mask(
            scene_world,
            geometry["hand_world"],
            geometry["hand_pose"],
            candidate_threshold=0.05,
            frame_batch_size=4,
            device=torch.device("cpu"),
        )
        assert offsets.shape == (frames + 1,)
        assert int(offsets[0]) == 0 and int(offsets[-1]) == indices.shape[0]
        for frame in range(frames):
            decoded = indices[int(offsets[frame]):int(offsets[frame + 1])]
            reference = np.flatnonzero(mask[frame])
            np.testing.assert_array_equal(decoded, reference.astype(decoded.dtype))
            # Environment candidates are exactly the >= OBJ_POOL tail.
            np.testing.assert_array_equal(
                decoded[decoded >= OBJ_POOL],
                np.flatnonzero(mask[frame][OBJ_POOL:]) + OBJ_POOL,
            )
        assert int(mask[:, OBJ_POOL:].sum()) > 0, "fixture must include environment candidates"
        assert OBJ_POOL + num_env < 65535


# ---------------------------------------------------------------------------
# Test C: sampling bank determinism + diversity (V1.md §26 C)
# ---------------------------------------------------------------------------
def test_sampling_bank_determinism_and_diversity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        geometry = _geometry_with_env(100, 412)
        main_dir = _write_scene_sequence(root, "s1", "unit", geometry)
        small_geometry = _build_geometry(
            num_obj_near=80,
            num_obj_far=OBJ_POOL - 80,
            num_env_near=100,
            num_env_far=412,
            seed=11,
        )
        small_dir = _write_scene_sequence(root, "s1", "small", small_geometry)
        _write_root_meta(root, num_env=512)
        stats = build_all_banks(root, bank_size=4, num_points=512, sampling_seed=42)
        assert stats["sides"] == 2

        bank_path = main_dir / "sampling_bank" / "right_indices.npy"
        first = np.load(bank_path)
        np.testing.assert_array_equal(first, np.load(bank_path))
        assert first.shape == (FRAMES, 4, 512)
        assert first.dtype == np.uint16
        # Same (sequence, side, frame, bank) always yields identical indices.
        rng = np.random.default_rng(0)
        for _ in range(5):
            frame = int(rng.integers(0, FRAMES))
            bank = int(rng.integers(0, 4))
            np.testing.assert_array_equal(load_mmap(bank_path)[frame, bank], first[frame, bank])
        # Candidate count 700 > 512: banks are complete and show diversity.
        offsets = np.load(main_dir / "right" / "candidate_offsets.npy")
        counts = np.diff(offsets)
        assert counts.min() >= 700
        for frame in (0, 3, 7):
            sets = [set(first[frame, b].tolist()) for b in range(4)]
            for bank_set in sets:
                assert len(bank_set) == 512  # all slots filled, no sentinel
            overlaps = [
                len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                for a in range(4) for b in range(a + 1, 4)
            ]
            assert 0.05 < float(np.mean(overlaps)) < 0.95, overlaps
            assert len(set().union(*sets)) > 512
        # Candidate count 180 < 512: unfilled slots use the uint16 sentinel.
        small_bank = np.load(small_dir / "sampling_bank" / "right_indices.npy")
        valid = small_bank != 0xFFFF
        small_counts = np.diff(np.load(small_dir / "right" / "candidate_offsets.npy"))
        assert small_counts.max() <= 180
        assert bool((~valid).any())
        expected_valid = np.broadcast_to(
            np.minimum(small_counts, 512)[:, None], (FRAMES, 4)
        )
        np.testing.assert_array_equal(valid.sum(axis=-1), expected_valid)


# ---------------------------------------------------------------------------
# Test A: object-only regression vs legacy dataset (V1.md §26 A)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bank", [0, 1, 2, 3])
def test_object_only_regression_matches_legacy_dataset(bank: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Environment disabled: the scene pool degenerates to the object pool.
        geometry = _build_geometry(
            num_obj_near=600,
            num_obj_far=OBJ_POOL - 600,
            num_env_near=0,
            num_env_far=0,
            seed=7,
        )
        sequence_dir = _write_scene_sequence(root, "s1", "unit", geometry)
        legacy_dir = _write_legacy_sequence(sequence_dir, geometry)
        _write_root_meta(root, num_env=0)
        build_all_banks(root, bank_size=4, num_points=512, sampling_seed=42)

        legacy = Stage4CmDataset(
            legacy_dir, num_obj_points=512, num_hand_points=HAND_POINTS,
            base_seed=42, min_stride=1, max_stride=10, fixed_stride=3,
        )
        legacy.set_epoch(bank)
        scene = Stage4CmSceneDataset(
            root, file_list=[sequence_dir], num_obj_points=512, num_hand_points=HAND_POINTS,
            base_seed=42, min_stride=1, max_stride=10, fixed_stride=3,
            sampling_bank_size=4, fixed_bank=bank,
        )
        assert len(legacy) == len(scene) > 0
        for index in range(len(legacy)):
            old_sample = legacy[index]
            new_sample = scene[index]
            assert int(old_sample["stride"]) == int(new_sample["stride"]) == 3
            for key in (
                "selected_obj_idx", "obj_valid_mask", "obj_points", "obj_normals",
                "obj_flow_gt", "hand_points", "hand_normals", "hand_flow",
            ):
                np.testing.assert_array_equal(
                    old_sample[key].numpy(), new_sample[key].numpy(), err_msg=f"{key}@{index}"
                )
            # Environment-disabled scene samples are pure object points.
            assert int(new_sample["scene_source_id"].max()) == 0


# ---------------------------------------------------------------------------
# Test D: dense cache parity online vs cached FP16 (V1.md §26 D)
# ---------------------------------------------------------------------------
class _StubDenseEncoder(torch.nn.Module):
    """Deterministic stand-in for the frozen PTv3 dense-token encoder."""

    token_dim = 16

    def forward(self, *, obj_points, obj_normals, hand_points, hand_normals, obj_valid_mask):
        def features(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
            base = (points * normals).sum(dim=-1, keepdim=True) / 3.0
            offsets = torch.linspace(0.0, 0.15, self.token_dim, device=points.device)
            return base + offsets[None, None, :]

        z_obj = features(obj_points.float(), obj_normals.float())
        z_hand = features(hand_points.float(), hand_normals.float())
        contact = torch.sigmoid(hand_points.float().norm(dim=-1) - 4.5)
        return z_obj, z_hand, contact


def test_dense_cache_parity_and_fingerprint_guard() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        from src.task.Cm.cache_schema import validate_scene_root

        root = Path(tmp)
        geometry = _geometry_with_env(100, 412)
        sequence_dir = _write_scene_sequence(root, "s1", "unit", geometry)
        _write_root_meta(root, num_env=512)
        build_all_banks(root, bank_size=4, num_points=512, sampling_seed=42)
        encoder = _StubDenseEncoder()
        stats = build_dense_cache(
            root,
            encoder=encoder,
            checkpoint_sha="stub-sha",
            bank_size=4,
            num_scene_points=512,
            batch_size=8,
            device="cpu",
            dtype="float16",
            token_dim=encoder.token_dim,
        )
        assert stats["sides"] == 1

        dataset = Stage4CmSceneDataset(
            root, file_list=[sequence_dir], num_obj_points=512, num_hand_points=HAND_POINTS,
            min_stride=1, max_stride=10, fixed_stride=2, sampling_bank_size=4,
            fixed_bank=0, use_dense_cache=True,
        )
        sample = dataset[0]
        online_inputs = {
            "obj_points": sample["obj_points"][None],
            "obj_normals": sample["obj_normals"][None],
            "hand_points": sample["hand_points"][None],
            "hand_normals": sample["hand_normals"][None],
            "obj_valid_mask": sample["obj_valid_mask"][None],
        }
        with torch.no_grad():
            z_obj, z_hand, contact = encoder(**online_inputs)
        # FP16 rounding is the only allowed difference between the online
        # stub path and the cached features (parity Test D).
        assert float((sample["cached_z_obj"] - z_obj[0]).abs().max()) < 2e-3
        assert float((sample["cached_z_hand"] - z_hand[0]).abs().max()) < 2e-3
        assert float((sample["cached_hand_contact"] - contact[0]).abs().max()) < 2e-3
        assert sample["cached_z_obj"].shape == (512, encoder.token_dim)
        assert sample["cached_z_hand"].shape == (HAND_POINTS, encoder.token_dim)

        # Strict fingerprint validation: mismatch must raise, never fall back.
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        meta["dense_cache"]["fingerprint"] = "tampered"
        (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError):
            validate_scene_root(
                root, num_scene_points=512, sampling_bank_size=4,
                use_dense_cache=True, dense_fingerprint="expected",
            )


# ---------------------------------------------------------------------------
# Test E: loss parity with environment disabled (V1.md §26 E)
# ---------------------------------------------------------------------------
def test_loss_parity_object_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        geometry = _build_geometry(
            num_obj_near=600,
            num_obj_far=OBJ_POOL - 600,
            num_env_near=0,
            num_env_far=0,
            seed=7,
        )
        sequence_dir = _write_scene_sequence(root, "s1", "unit", geometry)
        legacy_dir = _write_legacy_sequence(sequence_dir, geometry)
        _write_root_meta(root, num_env=0)
        build_all_banks(root, bank_size=4, num_points=512, sampling_seed=42)

        legacy = Stage4CmDataset(
            legacy_dir, num_obj_points=512, num_hand_points=HAND_POINTS,
            base_seed=42, min_stride=1, max_stride=10, fixed_stride=4,
        )
        legacy.set_epoch(2)
        scene = Stage4CmSceneDataset(
            root, file_list=[sequence_dir], num_obj_points=512, num_hand_points=HAND_POINTS,
            base_seed=42, min_stride=1, max_stride=10, fixed_stride=4,
            sampling_bank_size=4, fixed_bank=2,
        )
        torch.manual_seed(0)
        head = CmFlowHead(
            dense_token_dim=16, cm_dim=32, num_cm_tokens=4,
            use_slot_gate=False, use_time_condition=True,
        )
        head.eval()
        z_obj = torch.randn(1, 512, 16)
        z_hand = torch.randn(1, HAND_POINTS, 16)
        contact = torch.rand(1, HAND_POINTS)
        losses = []
        for sample in (legacy[0], scene[0]):
            with torch.no_grad():
                prediction = head(
                    z_obj=z_obj,
                    z_hand=z_hand,
                    dense_hand_contact=contact,
                    obj_points=sample["obj_points"][None],
                    obj_normals=sample["obj_normals"][None],
                    hand_points=sample["hand_points"][None],
                    hand_normals=sample["hand_normals"][None],
                    hand_flow=sample["hand_flow"][None],
                    obj_valid_mask=sample["obj_valid_mask"][None],
                    delta_time_s=sample["delta_time_s"][None],
                )
                loss = scaled_flow_smooth_l1(
                    prediction["pred_obj_flow"],
                    sample["obj_flow_gt"][None].float(),
                    sample["obj_valid_mask"][None].bool(),
                    beta_m=0.005,
                    target_scale=100.0,
                )
                losses.append(float(loss))
        assert abs(losses[0] - losses[1]) < 1e-6


# ---------------------------------------------------------------------------
# Flow calibration on the scene root (V1.md §23)
# ---------------------------------------------------------------------------
def test_scene_flow_scale_matches_legacy_without_env() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        geometry = _build_geometry(
            num_obj_near=600,
            num_obj_far=OBJ_POOL - 600,
            num_env_near=0,
            num_env_far=0,
            seed=7,
        )
        sequence_dir = _write_scene_sequence(root, "s1", "unit", geometry)
        legacy_dir = _write_legacy_sequence(sequence_dir, geometry)
        _write_root_meta(root, num_env=0)
        scene_stats = calibrate_flow_scale_scene(
            root, min_stride=1, max_stride=4, active_only=True, num_obj_points=512,
        )
        legacy_stats = calibrate_flow_scale(
            legacy_dir, min_stride=1, max_stride=4, active_only=True, num_obj_points=512,
        )
        assert abs(scene_stats["flow_target_rms_m"] - legacy_stats["flow_target_rms_m"]) < 1e-9

        # Zero-flow environment candidates must dilute the scene RMS.
        env_geometry = _geometry_with_env(100, 412)
        env_root = Path(tmp) / "env_root"
        _write_scene_sequence(env_root, "s1", "unit", env_geometry)
        _write_root_meta(env_root, num_env=512)
        env_stats = calibrate_flow_scale_scene(
            env_root, min_stride=1, max_stride=4, active_only=True, num_obj_points=512,
        )
        assert env_stats["flow_target_rms_m"] < scene_stats["flow_target_rms_m"]
        assert env_stats["statistics_scene_environment_candidate_points"] > 0
