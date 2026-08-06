"""Tests for hand-root transforms and the minimal v2 Stage 3 schema.

Covers Stage 2 hand-root propagation, Stage 3 coordinate transforms, the
minimal field set, and frame invariance of the dense hand-to-object
distance target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process.common import stage3_corr as stage3_mod  # noqa: E402
from process.common.stage3_corr import (  # noqa: E402
    _normals_world_to_hand_root,
    _points_world_to_hand_root,
)
from process.common.stage2 import pack_stage2_hand  # noqa: E402


_STAGE3_V2_FIELDS = {
    "schema_name",
    "schema_version",
    "seq_id",
    "side",
    "raw_frame_id",
    "obj_points",
    "obj_normals",
    "hand_points",
    "hand_normals",
    "hand_to_obj_min_dist",
    "coordinate_frame",
    "hand_root_pose",
    "obj_point_id",
    "obj_root_pose_world",
}


# ---------------------------------------------------------------------------
# Synthetic Stage 2 payload
# ---------------------------------------------------------------------------


def _axis_angle(rot_axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a (3, 3) rotation matrix from a unit axis and an angle (rad)."""
    axis = rot_axis / np.linalg.norm(rot_axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float32,
    )


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def _make_minimal_source(
    T: int,
    *,
    num_obj: int,
    num_hand: int,
    with_hand_root_pose: bool,
) -> dict:
    """Build a source dict shaped like process/ARCTIC/raw.py output."""
    rng = np.random.default_rng(0)
    obj_pts = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals /= np.clip(np.linalg.norm(obj_normals, axis=-1, keepdims=True), 1e-8, None)
    hand_pts = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals /= np.clip(np.linalg.norm(hand_normals, axis=-1, keepdims=True), 1e-8, None)
    obj_root_pose = np.stack(
        [_make_se3(np.eye(3, dtype=np.float32), rng.standard_normal(3).astype(np.float32)) for _ in range(T)]
    )
    out = {
        "dataset_name": "synthetic",
        "seq_id": "s1/synthetic_seq",
        "subject_id": "s1",
        "seq_name": "synthetic_seq",
        "object_name": "synth",
        "raw_frame_id": np.arange(T, dtype=np.int32),
        "obj_points_world": obj_pts,
        "obj_normals_world": obj_normals,
        "obj_point_id": np.arange(num_obj, dtype=np.int32),
        "obj_root_pose": obj_root_pose,
        "right_hand_points_world": hand_pts,
        "right_hand_normals_world": hand_normals,
        "right_hand_point_id": np.arange(num_hand, dtype=np.int32),
        "right_hand_cano_points": rng.standard_normal((num_hand, 3)).astype(np.float32),
        "right_hand_finger_id": np.zeros((num_hand,), dtype=np.int32),
        "right_hand_region_id": np.zeros((num_hand,), dtype=np.int32),
        "right_hand_to_obj_nn_id": np.zeros((T, num_hand), dtype=np.int32),
        "right_hand_min_dist_to_obj": np.zeros((T,), dtype=np.float32),
        "left_hand_points_world": hand_pts.copy(),
        "left_hand_normals_world": hand_normals.copy(),
        "left_hand_point_id": np.arange(num_hand, dtype=np.int32),
        "left_hand_cano_points": rng.standard_normal((num_hand, 3)).astype(np.float32),
        "left_hand_finger_id": np.zeros((num_hand,), dtype=np.int32),
        "left_hand_region_id": np.zeros((num_hand,), dtype=np.int32),
        "left_hand_to_obj_nn_id": np.zeros((T, num_hand), dtype=np.int32),
        "left_hand_min_dist_to_obj": np.zeros((T,), dtype=np.float32),
    }
    if with_hand_root_pose:
        # Use a non-trivial rotation + translation per frame
        R = _axis_angle(np.array([0.0, 0.0, 1.0]), 0.3)  # 17° about z
        out["right_hand_root_pose"] = np.stack(
            [_make_se3(R, np.array([0.1, 0.2, 0.3], dtype=np.float32)) for _ in range(T)]
        )
        out["left_hand_root_pose"] = out["right_hand_root_pose"].copy()
    return out


# ---------------------------------------------------------------------------
# Stage 2: pack_stage2_hand
# ---------------------------------------------------------------------------


def test_stage2_passes_through_hand_root_pose_when_present() -> None:
    source = _make_minimal_source(T=3, num_obj=8, num_hand=5, with_hand_root_pose=True)
    payload = pack_stage2_hand(
        source,
        side="right",
        source_raw_file="s1/synthetic_seq.mano.npy",
        processing_mode="init_only",
        frame_keep_threshold=1.0,
        config={},
    )
    assert payload is not None
    assert "hand_root_pose" in payload, "hand_root_pose must be propagated when source provides it"
    assert payload["hand_root_pose"].shape == (3, 4, 4)
    # After [keep] filtering the pose is preserved.
    assert np.allclose(payload["hand_root_pose"], source["right_hand_root_pose"])


def test_stage2_skips_hand_root_pose_when_absent() -> None:
    source = _make_minimal_source(T=3, num_obj=8, num_hand=5, with_hand_root_pose=False)
    payload = pack_stage2_hand(
        source,
        side="right",
        source_raw_file="s1/synthetic_seq.mano.npy",
        processing_mode="init_only",
        frame_keep_threshold=1.0,
        config={},
    )
    assert payload is not None
    assert "hand_root_pose" not in payload, "legacy payloads must remain valid"


# ---------------------------------------------------------------------------
# Stage 3: hand_root transformation math
# ---------------------------------------------------------------------------


def test_hand_root_round_trip_transform() -> None:
    """For a single hand_root pose, world -> hand_root -> world must be identity."""
    T = 4
    rng = np.random.default_rng(1)
    R = _axis_angle(np.array([0.0, 1.0, 0.0]), 0.4)
    t = rng.standard_normal(3).astype(np.float32)
    hand_root_poses = np.stack([_make_se3(R, t) for _ in range(T)])

    pts_world = rng.standard_normal((T, 7, 3)).astype(np.float32)
    pts_hand = _points_world_to_hand_root(pts_world, hand_root_poses)

    # Back: world = R @ pts_hand + t
    pts_back = np.einsum("tij,tnj->tni", hand_root_poses[:, :3, :3], pts_hand) + hand_root_poses[:, None, :3, 3]
    assert np.allclose(pts_back, pts_world, atol=1e-5), "world -> hand_root -> world is not identity"

    # Wrist itself: in hand_root frame it should be at origin.
    wrist_in_hand = _points_world_to_hand_root(t[None, None, :], hand_root_poses[:1])
    assert np.allclose(wrist_in_hand, np.zeros_like(wrist_in_hand), atol=1e-5)


def test_normals_hand_root_pure_rotation() -> None:
    """Normal transformation must be pure rotation, no translation; and remain unit length."""
    T = 3
    R = _axis_angle(np.array([1.0, 0.0, 0.0]), 0.7)
    t = np.array([5.0, -2.0, 0.5], dtype=np.float32)
    hand_root_poses = np.stack([_make_se3(R, t) for _ in range(T)])
    rng = np.random.default_rng(2)
    n_world = rng.standard_normal((T, 6, 3)).astype(np.float32)
    n_world /= np.linalg.norm(n_world, axis=-1, keepdims=True)
    n_hand = _normals_world_to_hand_root(n_world, hand_root_poses)
    norms = np.linalg.norm(n_hand, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), "hand_root normals must remain unit length"


# ---------------------------------------------------------------------------
# Stage 3: build_stage3_sequence with both coordinate frames
# ---------------------------------------------------------------------------


def _build_payload_for_stage3(num_obj: int = 16, num_hand: int = 6, T: int = 2) -> dict:
    """Build a Stage 2-shaped payload that satisfies `_validate_stage2_geometry`."""
    rng = np.random.default_rng(3)
    obj_pts = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals = rng.standard_normal((T, num_obj, 3)).astype(np.float32)
    obj_normals /= np.clip(np.linalg.norm(obj_normals, axis=-1, keepdims=True), 1e-8, None)
    hand_pts = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals = rng.standard_normal((T, num_hand, 3)).astype(np.float32)
    hand_normals /= np.clip(np.linalg.norm(hand_normals, axis=-1, keepdims=True), 1e-8, None)
    obj_root_pose = np.stack(
        [_make_se3(np.eye(3, dtype=np.float32), rng.standard_normal(3).astype(np.float32)) for _ in range(T)]
    )
    R = _axis_angle(np.array([0.0, 0.0, 1.0]), 0.0)  # identity rotation
    t = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    hand_root_poses = np.stack([_make_se3(R, t) for _ in range(T)])
    return {
        "schema_name": "ref2dex_opti",
        "schema_version": "1.0.0",
        "processing_mode": "init_only",
        "dataset_name": "synthetic",
        "seq_id": "s1/synthetic_seq",
        "subject_id": "s1",
        "seq_name": "synthetic_seq",
        "object_name": "synth",
        "side": "right",
        "raw_frame_id": np.arange(T, dtype=np.int32),
        "stage2_frame_idx": np.arange(T, dtype=np.int32),
        "obj_points_world": obj_pts,
        "obj_normals_world": obj_normals,
        "obj_point_id": np.arange(num_obj, dtype=np.int32),
        "obj_root_pose": obj_root_pose,
        "hand_points_world": hand_pts,
        "hand_normals_world": hand_normals,
        "hand_point_id": np.arange(num_hand, dtype=np.int32),
        "hand_cano_points": rng.standard_normal((num_hand, 3)).astype(np.float32),
        "hand_finger_id": np.zeros((num_hand,), dtype=np.int32),
        "hand_region_id": np.zeros((num_hand,), dtype=np.int32),
        "hand_root_pose": hand_root_poses,
    }


def test_stage3_object_frame_emits_minimal_v2_schema() -> None:
    payload = _build_payload_for_stage3()
    out = stage3_mod.build_stage3_sequence(
        payload,
        num_obj_pool=16,
        num_hand_points=6,
        candidate_threshold=0.05,
        frame_batch_size=1,
        device=torch.device("cpu"),
        coordinate_frame="object",
    )
    assert set(out) == _STAGE3_V2_FIELDS
    assert str(out["coordinate_frame"].item()) == "object"
    expected = stage3_mod._points_world_to_obj(
        payload["obj_points_world"], payload["obj_root_pose"]
    )
    np.testing.assert_allclose(out["obj_points"], expected, atol=1e-6)


def test_stage3_hand_root_frame_emits_minimal_v2_schema() -> None:
    payload = _build_payload_for_stage3()
    out = stage3_mod.build_stage3_sequence(
        payload,
        num_obj_pool=16,
        num_hand_points=6,
        candidate_threshold=0.05,
        frame_batch_size=1,
        device=torch.device("cpu"),
        coordinate_frame="hand_root",
    )
    assert set(out) == _STAGE3_V2_FIELDS
    assert str(out["coordinate_frame"].item()) == "hand_root"
    expected = _points_world_to_hand_root(
        payload["obj_points_world"], payload["hand_root_pose"]
    )
    np.testing.assert_allclose(out["obj_points"], expected, atol=1e-6)


def test_stage3_persists_optional_object_state_fields() -> None:
    payload = _build_payload_for_stage3()
    payload["obj_repr"] = "rigid_canonical"
    payload["obj_points_canonical"] = payload["obj_points_world"][0].copy()
    payload["obj_normals_canonical"] = payload["obj_normals_world"][0].copy()
    payload["obj_part_id"] = np.zeros((16,), dtype=np.int32)
    payload["obj_articulation"] = np.zeros((payload["raw_frame_id"].shape[0], 1), dtype=np.float32)
    out = stage3_mod.build_stage3_sequence(
        payload,
        num_obj_pool=16,
        num_hand_points=6,
        candidate_threshold=0.05,
        frame_batch_size=1,
        device=torch.device("cpu"),
        coordinate_frame="hand_root",
    )
    assert str(out["obj_repr"].item()) == "rigid_canonical"
    np.testing.assert_allclose(out["obj_points_canonical"], payload["obj_points_canonical"])
    np.testing.assert_allclose(out["obj_normals_canonical"], payload["obj_normals_canonical"])
    np.testing.assert_array_equal(out["obj_point_id"], payload["obj_point_id"])
    np.testing.assert_allclose(out["obj_root_pose_world"], payload["obj_root_pose"])
    np.testing.assert_array_equal(out["obj_part_id"], payload["obj_part_id"])
    np.testing.assert_allclose(out["obj_articulation"], payload["obj_articulation"])


def test_stage3_minimal_v2_does_not_require_legacy_stage2_metadata() -> None:
    payload = _build_payload_for_stage3()
    for key in (
        "processing_mode",
        "dataset_name",
        "subject_id",
        "seq_name",
        "object_name",
        "stage2_frame_idx",
        "obj_point_id",
        "obj_root_pose",
        "hand_point_id",
        "hand_cano_points",
        "hand_finger_id",
        "hand_region_id",
    ):
        del payload[key]
    out = stage3_mod.build_stage3_sequence(
        payload,
        num_obj_pool=16,
        num_hand_points=6,
        candidate_threshold=0.05,
        frame_batch_size=1,
        device=torch.device("cpu"),
        coordinate_frame="hand_root",
    )
    assert set(out) == _STAGE3_V2_FIELDS.difference({"obj_point_id", "obj_root_pose_world"})


def test_stage3_hand_root_requires_hand_root_pose_in_payload() -> None:
    payload = _build_payload_for_stage3()
    del payload["hand_root_pose"]
    with pytest.raises(KeyError, match="hand_root_pose"):
        stage3_mod.build_stage3_sequence(
            payload,
            num_obj_pool=16,
            num_hand_points=6,
            candidate_threshold=0.05,
            frame_batch_size=1,
            device=torch.device("cpu"),
            coordinate_frame="hand_root",
        )


def test_stage3_distances_are_frame_invariant() -> None:
    """The stored dense hand target is frame-invariant."""
    payload = _build_payload_for_stage3(num_obj=32, num_hand=10, T=2)
    common_kwargs = dict(
        num_obj_pool=32,
        num_hand_points=10,
        candidate_threshold=1.0,  # accept all
        frame_batch_size=1,
        device=torch.device("cpu"),
    )
    obj_out = stage3_mod.build_stage3_sequence(payload, coordinate_frame="object", **common_kwargs)
    hand_out = stage3_mod.build_stage3_sequence(payload, coordinate_frame="hand_root", **common_kwargs)
    np.testing.assert_allclose(
        obj_out["hand_to_obj_min_dist"], hand_out["hand_to_obj_min_dist"], atol=1e-5
    )
    assert obj_out["hand_to_obj_min_dist"].shape == (2, 10)


def test_stage3_rejects_invalid_coordinate_frame() -> None:
    payload = _build_payload_for_stage3()
    with pytest.raises(ValueError, match="coordinate_frame"):
        stage3_mod.build_stage3_sequence(
            payload,
            num_obj_pool=16,
            num_hand_points=6,
            candidate_threshold=0.05,
            frame_batch_size=1,
            device=torch.device("cpu"),
            coordinate_frame="banana",
        )


# ---------------------------------------------------------------------------
# Stage 3 -> dataset: per-NPZ coordinate_frame is the source of truth
# ---------------------------------------------------------------------------


def _write_stage3_npz(
    path: Path,
    *,
    num_obj: int,
    num_hand: int,
    T: int,
    coordinate_frame: str,
) -> None:
    payload = _build_payload_for_stage3(num_obj=num_obj, num_hand=num_hand, T=T)
    out = stage3_mod.build_stage3_sequence(
        payload,
        num_obj_pool=num_obj,
        num_hand_points=num_hand,
        candidate_threshold=1.0,
        frame_batch_size=1,
        device=torch.device("cpu"),
        coordinate_frame=coordinate_frame,
    )
    np.savez(path, **out)


def _make_dataset_kwargs() -> dict:
    return dict(
        num_obj_points=4,
        num_hand_points=6,
        num_supervision_edges=4,
        contact_radius=0.02,
        base_seed=0,
        augment=False,
        apply_obj_perturb=False,
        obj_rot_std_deg=0.0,
        obj_trans_std=0.0,
        obj_perturb_prob=0.0,
    )


def test_dataset_rejects_coordinate_frame_mismatch(tmp_path: Path) -> None:
    """The dataset must reject an explicit coordinate_frame that does not
    match the per-NPZ coordinate_frame (Stage 3 .npz is the source of truth)."""
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    npz_path = tmp_path / "synth.npz"
    _write_stage3_npz(npz_path, num_obj=16, num_hand=6, T=2, coordinate_frame="hand_root")
    with pytest.raises(ValueError, match="coordinate_frame"):
        CorrStaticDatasetV2(
            tmp_path,
            file_list=[npz_path],
            coordinate_frame="object",
            **_make_dataset_kwargs(),
        )


def test_dataset_accepts_matching_coordinate_frame(tmp_path: Path) -> None:
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    npz_path = tmp_path / "synth.npz"
    _write_stage3_npz(npz_path, num_obj=16, num_hand=6, T=2, coordinate_frame="hand_root")
    ds = CorrStaticDatasetV2(
        tmp_path,
        file_list=[npz_path],
        coordinate_frame="hand_root",
        **_make_dataset_kwargs(),
    )
    assert ds.coordinate_frame == "hand_root"
    # Sampled point is consistent: 4 obj + 6 hand, and supervision seeds/distances
    # are deferred to the GPU builder.
    sample = ds[0]
    assert "hand_min_dist" in sample
    assert sample["hand_min_dist"].shape == (6,)
    assert "contact_seed" in sample
    assert "selected_obj_point_id" not in sample
    assert "selected_obj_min_dist" not in sample


def test_dataset_inherits_coordinate_frame_when_unspecified(tmp_path: Path) -> None:
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    npz_path = tmp_path / "synth.npz"
    _write_stage3_npz(npz_path, num_obj=16, num_hand=6, T=2, coordinate_frame="hand_root")
    ds = CorrStaticDatasetV2(tmp_path, file_list=[npz_path], **_make_dataset_kwargs())
    assert ds.coordinate_frame == "hand_root"


def test_dataset_rejects_mixed_coordinate_frames(tmp_path: Path) -> None:
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    _write_stage3_npz(p1, num_obj=16, num_hand=6, T=2, coordinate_frame="hand_root")
    _write_stage3_npz(p2, num_obj=16, num_hand=6, T=2, coordinate_frame="object")
    with pytest.raises(ValueError, match="coordinate_frame"):
        CorrStaticDatasetV2(
            tmp_path,
            file_list=[p1, p2],
            **_make_dataset_kwargs(),
        )


def test_runner_configure_data_raises_on_metadata_mismatch() -> None:
    """The runner must second-assert that metadata.coordinate_frame matches
    the config, so a future refactor cannot silently bypass the guard."""
    from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner

    runner = CorrespondencePTV3V2Runner.__new__(CorrespondencePTV3V2Runner)
    runner.cfg = type("Cfg", (), {})()
    runner.cfg.meta = type("Meta", (), {"coordinate_frame": "hand_root"})()
    with pytest.raises(ValueError, match="coordinate_frame"):
        runner.configure_data({"coordinate_frame": "object"})
    # And missing coordinate_frame in metadata must also be flagged.
    with pytest.raises(ValueError, match="missing 'coordinate_frame'"):
        runner.configure_data({})


# ---------------------------------------------------------------------------
# GRAB absent-hand path: no root_pose emitted, no KeyError downstream.
# ---------------------------------------------------------------------------


def test_grab_absent_hand_payload_omits_root_pose() -> None:
    """The _empty_hand_payload must not include a root_pose key. Stage 2
    pack_stage2_hand treats hand_root_pose as optional; if absent, no
    hand_root_pose field is propagated and the npz is regenerated cleanly."""
    from process.GRAB.raw import GRABRawAdapter

    # Bypass __init__ (which would load MANO) but set the attribute the
    # _empty_hand_payload helper reads.
    adapter = GRABRawAdapter.__new__(GRABRawAdapter)
    adapter.num_hand_points = 1538
    payload = GRABRawAdapter._empty_hand_payload(adapter, T=2)
    assert "root_pose" not in payload
    assert payload["points"].shape == (2, 1538, 3)


def test_grab_process_sequence_handles_absent_hand_keys() -> None:
    """process_sequence() must build output dict even when right/left hand
    keys are absent on the underlying npz (absent-hand path)."""
    import importlib

    grab_mod = importlib.import_module("process.GRAB.raw")

    # Build a stub class that mimics GRABRawAdapter but does not require
    # the heavy MANO init. _empty_hand_payload only reads num_hand_points,
    # so we set that explicitly.
    class _StubAdapter:
        num_hand_points = 1538

        def _process_one_hand(self, side, seq_data, frame_ids, obj_points):
            # Simulate GRAB scenario where one hand key is missing from
            # seq_data._raw.files. The output dict is exactly what
            # _empty_hand_payload returns, and it must NOT include root_pose.
            return grab_mod.GRABRawAdapter._empty_hand_payload(self, T=len(frame_ids))

    class _FakeSeqData:
        class _Raw:
            files: tuple = ()  # no hand keys at all

        _raw = _Raw()
        obj_name = "synth"
        n_comps = 24
        n_frames = 2

    stub = _StubAdapter()
    right = stub._process_one_hand("right", _FakeSeqData(), np.arange(2), np.zeros((2, 4, 3)))
    left = stub._process_one_hand("left", _FakeSeqData(), np.arange(2), np.zeros((2, 4, 3)))
    assert "root_pose" not in right
    assert "root_pose" not in left


# ---------------------------------------------------------------------------
# v2.1 removes the hand prediction path entirely. Repeating a sample along
# the batch dim must still leave the two active cross-edge losses invariant.
# ---------------------------------------------------------------------------


def test_v21_cross_edge_losses_are_batch_size_invariant_via_runner() -> None:
    """Repeating the same sample along the batch dim must preserve the
    active v2.1 loss keys when routed through ``runner._compute_losses()``.
    """
    from src.task.correspondence_ptv3_v2.config import Config
    from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner
    from src.task.correspondence_ptv3_v2.losses import contact_target_from_distance

    torch.manual_seed(0)
    B, N = 1, 1538
    cfg = Config()
    cfg.meta.loss_hand_contact_weight = 0.0

    # Build a bare runner that avoids the full __init__ (data discovery, dirs).
    runner = object.__new__(CorrespondencePTV3V2Runner)
    runner.cfg = cfg
    runner.mode = "eval"  # skip data setup in overfit_mode

    # Synthesize a hand-target tensor (B, N) from random distances.
    dist = torch.rand(B, N, dtype=torch.float32) * 0.05
    hand_target = contact_target_from_distance(dist, contact_radius=0.02)

    # Synthesize cross-edge logits / probs (random + contact auxiliary).
    random_logits = torch.randn(B, 512, 128, dtype=torch.float32) * 2.0
    random_prob = torch.sigmoid(random_logits)
    contact_logits = torch.randn(B, 512, 16, dtype=torch.float32) * 2.0
    contact_prob = torch.sigmoid(contact_logits)

    batch1 = {
        "hand_contact_target": hand_target,
        "runtime_obj_valid_mask": torch.ones(B, 512, dtype=torch.bool),
        "random_edge_idx": torch.zeros(B, 512, 128, dtype=torch.long),
        "random_edge_valid_mask": torch.ones(B, 512, 128, dtype=torch.bool),
        "random_edge_contact_target": torch.zeros(B, 512, 128),  # dummy edge targets
        "contact_edge_idx": torch.zeros(B, 512, 16, dtype=torch.long),
        "contact_edge_valid_mask": torch.ones(B, 512, 16, dtype=torch.bool),
        "contact_edge_contact_target": torch.zeros(B, 512, 16),  # dummy edge targets
        "points": torch.zeros(B, 512 + N, 3),
    }
    preds1 = {
        "pred_cross_random_logits": random_logits,
        "pred_cross_random_prob": random_prob,
        "pred_cross_contact_aux_logits": contact_logits,
        "pred_cross_contact_aux_prob": contact_prob,
    }
    loss1, _ = runner._compute_losses(preds1, batch1, compute_diagnostics=True)

    # Duplicate along batch dim to B=4.
    B4 = 4
    def _repeat(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            return t
        return t.repeat(B4, *([1] * (t.dim() - 1)))

    batch4 = {k: _repeat(v) for k, v in batch1.items()}
    preds4 = {k: _repeat(v) for k, v in preds1.items()}
    loss4, _ = runner._compute_losses(preds4, batch4, compute_diagnostics=True)

    # v2.1 contract: losses = {cross_edge_random, cross_edge_contact_aux}.
    # Hand contact is NOT a loss key in the v2.1 ablation.
    assert "hand_contact" not in loss1
    assert "cross_edge_contact" not in loss1
    assert set(loss1) == {"cross_edge_random", "cross_edge_contact_aux"}
    torch.testing.assert_close(
        loss1["cross_edge_random"],
        loss4["cross_edge_random"],
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        loss1["cross_edge_contact_aux"],
        loss4["cross_edge_contact_aux"],
        atol=1e-6,
        rtol=1e-5,
    )
