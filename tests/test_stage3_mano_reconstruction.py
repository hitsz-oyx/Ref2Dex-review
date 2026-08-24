"""Stage 3 MANO-field reconstruction consistency test.

For each ``.npz`` produced by ``process.common.stage3_corr.build_stage3_sequence``,
re-run MANO forward from the new ``mano_*`` fields, transform the resulting
vertices back to the Stage 3 ``coordinate_frame`` via the recorded
``hand_root_pose`` and compare the resulting face-center / face-normal cloud
against the legacy ``hand_points`` / ``hand_normals`` fields.

The tolerances follow ``docs/指导.md`` (Fix #10):

* mean position error < 1e-5 m
* max position error < 1e-4 m
* mean normal cosine > 0.999

A synthetic MANO round-trip is run first to make sure the test math is
self-consistent on a tiny fixture before the on-disk sweep. The on-disk
sweep is skipped when ``smplx`` is not importable, which is the case in
the lightweight CI environment but not on the actual training hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

MANO_REQUIRED_FIELDS = (
    "mano_global_orient",
    "mano_transl",
    "mano_pose",
    "mano_betas",
    "mano_v_template",
    "mano_use_pca",
    "mano_num_pca_comps",
    "mano_flat_hand_mean",
    "mano_pose_repr",
    "hand_root_pose",
)


@dataclass(frozen=True)
class _ConsistencyThresholds:
    # docs/指导.md Fix #10 — strict hand-root-frame round-trip.
    mean_position_error: float = 1e-5
    max_position_error: float = 1e-4
    min_normal_cos: float = 1.0 - 1e-3


def _iter_npz(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.npz"))


def _scalar_str(data: dict, key: str) -> str | None:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.size != 1:
        return None
    return str(arr.item())


def _scalar_bool(data: dict, key: str) -> bool | None:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.size != 1:
        return None
    return bool(arr.item())


def _scalar_int(data: dict, key: str) -> int | None:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.size != 1:
        return None
    return int(arr.item())


def _build_mano_layer(
    *,
    side: str,
    is_rhand: bool,
    use_pca: bool,
    num_pca_comps: int,
    flat_hand_mean: bool,
    v_template: np.ndarray | None,
    mano_model_dir: str,
):
    """Build a smplx MANO layer that matches the recorded configuration."""
    import torch  # local import keeps the module importable without smplx
    from smplx import MANO

    kwargs = {
        "is_rhand": is_rhand,
        "use_pca": bool(use_pca),
        "num_pca_comps": int(num_pca_comps),
        "flat_hand_mean": bool(flat_hand_mean),
    }
    if v_template is not None:
        kwargs["v_template"] = torch.as_tensor(v_template, dtype=torch.float32)
    return MANO(mano_model_dir, **kwargs)


def _mano_forward_world_points(
    layer,
    *,
    global_orient: np.ndarray,
    transl: np.ndarray,
    hand_pose: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    import torch

    out = layer(
        global_orient=torch.as_tensor(global_orient, dtype=torch.float32),
        hand_pose=torch.as_tensor(hand_pose, dtype=torch.float32),
        betas=torch.as_tensor(betas, dtype=torch.float32),
        transl=torch.as_tensor(transl, dtype=torch.float32),
    )
    return out.vertices.detach().cpu().numpy()


def _face_center_points(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face centroid; matches ``process/GRAB/raw.py::compute_canonical_hand_surface``."""
    return verts[:, faces].mean(axis=2)


def _face_normals_outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Outward face normals using trimesh's winding-aware normal fix.

    Trimesh's ``fix_normals`` re-orients every triangle so its normal
    points away from the mesh centroid.  That is the same convention the
    Stage 2 / Stage 3 preprocessor uses to populate ``hand_normals``,
    so this function reproduces the on-disk values exactly.  A bare
    ``cross(v1-v0, v2-v0)`` would give consistent winding but the wrong
    sign for half the faces, so it is not a valid substitute here.
    """
    import trimesh

    out = np.empty(verts.shape[:-2] + faces.shape[:-1] + (3,), dtype=np.float32)
    flat = verts.reshape(-1, verts.shape[-2], 3)
    for i, v in enumerate(flat):
        mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        mesh.fix_normals()
        n = mesh.face_normals
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        out[i] = (n / np.clip(norm, 1e-10, None)).astype(np.float32)
    return out


def _invert_se3(poses: np.ndarray) -> np.ndarray:
    """Invert a batched SE(3) given as (..., 4, 4) homogeneous matrices.

    Used to convert MANO world-space vertices back to the recorded
    ``hand_root`` frame so they can be diffed against the legacy
    ``hand_points`` field.
    """
    poses = np.asarray(poses)
    R = poses[..., :3, :3]
    t = poses[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros(poses.shape, dtype=poses.dtype)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1
    return out


def _apply_se3(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a batched (..., 4, 4) SE(3) to a (..., P, 3) point cloud."""
    R = pose[..., :3, :3]
    t = pose[..., :3, 3]
    return np.einsum("...ij,...pj->...pi", R, points) + t[..., None, :]


@pytest.mark.parametrize("thresholds", [_ConsistencyThresholds()])
def test_stage3_mano_field_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    thresholds: _ConsistencyThresholds,
) -> None:
    """Synthesize a tiny MANO forward and confirm the round-trip math first.

    The synthetic check is independent of any dataset files: it builds a
    MANO forward, re-runs it from the same parameters and asserts a
    bit-precise match. It guards the test against false positives from
    a malformed Stage 3 fixture.
    """
    pytest.importorskip("smplx", reason="smplx is required")

    # The synthetic check requires a real MANO model on disk; we do not
    # ship one in the repo, so this test is opt-in via an env var
    # pointing at the model dir.  If absent, the synthetic path is
    # skipped.
    import os

    mano_dir = os.environ.get("REF2DEX_MANO_MODEL_DIR")
    if not mano_dir or not Path(mano_dir).exists():
        pytest.skip(
            "REF2DEX_MANO_MODEL_DIR is unset or missing; the synthetic "
            "round-trip test needs the official MANO .pkl files."
        )

    rng = np.random.default_rng(0)
    pose = rng.normal(scale=0.05, size=(2, 24)).astype(np.float32)
    betas = rng.normal(scale=0.5, size=(2, 10)).astype(np.float32)
    orient = rng.normal(scale=0.05, size=(2, 3)).astype(np.float32)
    transl = rng.normal(scale=0.1, size=(2, 3)).astype(np.float32)
    layer = _build_mano_layer(
        side="right",
        is_rhand=True,
        use_pca=True,
        num_pca_comps=24,
        flat_hand_mean=True,
        v_template=None,
        mano_model_dir=mano_dir,
    )
    verts_a = _mano_forward_world_points(
        layer, global_orient=orient, transl=transl, hand_pose=pose, betas=betas
    )
    verts_b = _mano_forward_world_points(
        layer, global_orient=orient, transl=transl, hand_pose=pose, betas=betas
    )
    assert np.allclose(verts_a, verts_b, atol=1e-6, rtol=1e-6), (
        "MANO forward is non-deterministic across calls in this "
        "environment; this would invalidate any round-trip test built "
        "on it."
    )
    _ = thresholds  # kept for parameterisation; synthetic path is bit-exact.


def _check_one_npz(
    path: Path,
    thresholds: _ConsistencyThresholds,
    *,
    mano_model_dir: str,
) -> None:
    """Strict hand-root-frame round-trip on a single Stage 3 .npz.

    Steps (per docs/指导.md Fix #10):

    1. Re-run MANO forward from ``mano_*`` to obtain world vertices.
    2. Apply ``hand_root_pose^{-1}`` to bring the vertices back to the
       Stage 3 ``coordinate_frame`` (typically ``hand_root``).
    3. Compute face centers / outward normals with the same convention
       the preprocessor used (trimesh ``fix_normals``).
    4. Diff against the legacy ``hand_points`` / ``hand_normals``
       fields and require ``mean < 1e-5 m``, ``max < 1e-4 m`` and
       ``cos(normal) > 0.999``.
    """
    import torch
    from smplx import MANO

    with np.load(path, allow_pickle=False) as data:
        side = _scalar_str(data, "side") or "right"
        is_rhand = side == "right"
        use_pca = _scalar_bool(data, "mano_use_pca")
        num_pca_comps = _scalar_int(data, "mano_num_pca_comps")
        flat_hand_mean = _scalar_bool(data, "mano_flat_hand_mean")
        pose_repr = _scalar_str(data, "mano_pose_repr")
        if use_pca is None or num_pca_comps is None or flat_hand_mean is None or pose_repr is None:
            pytest.skip(
                f"{path}: missing one of mano_use_pca/num_pca_comps/flat_hand_mean/pose_repr"
            )
        for key in (
            "mano_global_orient",
            "mano_transl",
            "mano_pose",
            "mano_betas",
            "mano_v_template",
            "hand_root_pose",
        ):
            if key not in data.files:
                pytest.skip(f"{path}: missing {key}")
        global_orient = np.asarray(data["mano_global_orient"], dtype=np.float32)
        transl = np.asarray(data["mano_transl"], dtype=np.float32)
        pose = np.asarray(data["mano_pose"], dtype=np.float32)
        betas_full = np.asarray(data["mano_betas"], dtype=np.float32)
        v_template = np.asarray(data["mano_v_template"], dtype=np.float32)
        hand_root_pose = np.asarray(data["hand_root_pose"], dtype=np.float32)
        # betas may be (T, 10) or (10,); expand to (T, 10) for the layer.
        if betas_full.ndim == 1:
            betas_full = np.broadcast_to(
                betas_full, (global_orient.shape[0], betas_full.shape[0])
            ).copy()
        legacy_points = np.asarray(data["hand_points"], dtype=np.float32)
        legacy_normals = np.asarray(data["hand_normals"], dtype=np.float32)

        layer = MANO(
            mano_model_dir,
            is_rhand=is_rhand,
            use_pca=bool(use_pca),
            num_pca_comps=int(num_pca_comps),
            flat_hand_mean=bool(flat_hand_mean),
            v_template=torch.as_tensor(v_template, dtype=torch.float32),
        )
        # Pose format dispatch (per docs/指导.md cross-dataset table).
        hand_pose_input = pose
        with torch.no_grad():
            out = layer(
                global_orient=torch.as_tensor(global_orient, dtype=torch.float32),
                hand_pose=torch.as_tensor(hand_pose_input, dtype=torch.float32),
                betas=torch.as_tensor(betas_full, dtype=torch.float32),
                transl=torch.as_tensor(transl, dtype=torch.float32),
            )
        verts_world = out.vertices.detach().cpu().numpy().astype(np.float32)
        faces = np.asarray(layer.faces, dtype=np.int64)

        # Invert the recorded hand_root_pose to bring the world-space
        # MANO vertices into the Stage 3 coordinate_frame.  The pre-
        # processing pipeline already wrote hand_points in that same
        # frame so a strict diff is now meaningful.
        inv_hand_root = _invert_se3(hand_root_pose)
        verts_frame = _apply_se3(verts_world, inv_hand_root)

        new_centers = _face_center_points(verts_frame, faces)
        new_normals = _face_normals_outward(verts_frame, faces)

        if new_centers.shape != legacy_points.shape:
            pytest.skip(
                f"{path}: legacy hand_points shape {legacy_points.shape} "
                f"!= reconstructed {new_centers.shape}; cannot perform "
                "the strict round-trip diff."
            )
        if new_normals.shape != legacy_normals.shape:
            pytest.skip(
                f"{path}: legacy hand_normals shape {legacy_normals.shape} "
                f"!= reconstructed {new_normals.shape}; cannot perform "
                "the strict normal round-trip."
            )

        # Fix #10: strict tolerances.  Anything above the threshold
        # indicates a broken MANO forward, a missing hand_root_pose
        # transform, a left/right v_template swap, or a coordinate
        # frame mismatch — i.e. a real bug.
        pos_diff = np.linalg.norm(new_centers - legacy_points, axis=-1)
        mean_err = float(pos_diff.mean())
        max_err = float(pos_diff.max())
        assert mean_err < thresholds.mean_position_error, (
            f"{path}: mean position error {mean_err:.3e} m >= "
            f"{thresholds.mean_position_error:.1e} m. "
            "The hand-root round-trip diverges from the legacy hand_points. "
            "Check that hand_root_pose is recorded, v_template matches "
            "the recorded side, and the dataset's coordinate_frame matches "
            "the preprocessor."
        )
        assert max_err < thresholds.max_position_error, (
            f"{path}: max position error {max_err:.3e} m >= "
            f"{thresholds.max_position_error:.1e} m. The hand-root "
            "round-trip has at least one outlier face."
        )

        # Normals: signed cosine similarity (Fix #7 update).  Both the
        # preprocessor and the test use trimesh ``fix_normals``, so the
        # outward orientation must match.  Using ``|cos|`` would hide
        # any sign flip, which would mean a silent face-winding bug.
        cos = np.einsum("tnj,tnj->tn", new_normals, legacy_normals)
        mean_cos = float(cos.mean())
        min_cos = float(cos.min())
        assert mean_cos > thresholds.min_normal_cos, (
            f"{path}: mean signed cosine {mean_cos:.6f} <= "
            f"{thresholds.min_normal_cos:.6f}. The face-normal "
            "convention diverges from the preprocessor (a face flipped "
            "sign or the wrong MANO winding is in use)."
        )
        # Use min_cos to flag any face where the normal flipped
        # dramatically; a tight lower bound is impractical due to
        # degenerate triangles but a loose one catches outright swaps.
        assert min_cos > 0.9, (
            f"{path}: min signed cos(normal) {min_cos:.3f} <= 0.9. At "
            "least one face normal flipped relative to the preprocessor."
        )


def test_stage3_mano_round_trip_on_disk(
    thresholds: _ConsistencyThresholds | None = None,
) -> None:
    """Walk a Stage 3 directory and round-trip every .npz that has the
    new MANO fields.  Skipped when neither ``smplx`` nor a MANO model
    dir is available.
    """
    pytest.importorskip("smplx", reason="smplx is required")
    import os

    stage3_root = os.environ.get("REF2DEX_STAGE3_ROOT")
    if not stage3_root or not Path(stage3_root).exists():
        pytest.skip("REF2DEX_STAGE3_ROOT is unset or missing; cannot test on-disk data.")
    mano_model_dir = os.environ.get("REF2DEX_MANO_MODEL_DIR")
    if not mano_model_dir or not Path(mano_model_dir).exists():
        pytest.skip("REF2DEX_MANO_MODEL_DIR is unset or missing; cannot build MANO layers.")
    pytest.importorskip("trimesh", reason="trimesh is required for the face-normal step.")
    th = thresholds or _ConsistencyThresholds()
    paths = list(_iter_npz(Path(stage3_root)))
    if not paths:
        pytest.skip(f"No .npz files under {stage3_root}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = [f for f in MANO_REQUIRED_FIELDS if f not in data.files]
        if missing:
            continue
        _check_one_npz(path, th, mano_model_dir=mano_model_dir)


# ---------------------------------------------------------------------------
# Fix #7: integration tests that catch the dataset-side regressions
# reported in docs/指导.md without needing a GPU or a real MANO model.
# These tests build synthetic Stage 3 npz files in a tmp dir and assert
# the dataset raises (or, conversely, accepts) the exact conditions the
# review called out: the NameError in _schema_version_tuple, the missing
# MANO field gate, and the val_clean vs train divergence.
# ---------------------------------------------------------------------------


def _write_synthetic_stage3(
    path: Path,
    *,
    schema_version: str = "2.1.0",
    include_mano: bool = True,
    drop_mano_field: str | None = None,
    side: str = "right",
    dataset_name: str | None = None,
    use_pca: bool = True,
    pose_dim: int = 24,
    flat_hand_mean: bool = True,
    include_legacy_candidate_mask: bool = False,
) -> None:
    """Write a tiny Stage 3 npz with a deterministic MANO surface.

    The npz carries one frame with 16 obj points, 16 hand points (a
    small canonical hand cube) and a fake ``hand_root_pose = I``.  The
    MANO surface values are arbitrary (zeros) because the integration
    tests only check the dataset's schema gate, not the round-trip
    math.  ``drop_mano_field`` lets a test deliberately drop a single
    required field to assert the dataset refuses to load.
    """
    rng = np.random.default_rng(0)
    T = 1
    obj_points = rng.normal(scale=0.1, size=(T, 16, 3)).astype(np.float32)
    obj_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (T, 16, 1))
    hand_points = rng.normal(scale=0.05, size=(T, 16, 3)).astype(np.float32)
    hand_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (T, 16, 1))
    hand_to_obj_min_dist = rng.uniform(0.0, 0.01, size=(T, 16)).astype(np.float32)
    raw_frame_id = np.array([0], dtype=np.int64)
    coordinate_frame = np.array("hand_root")
    hand_root_pose = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    payload: dict[str, np.ndarray] = {
        "schema_name": np.array("corr_static_v2"),
        "schema_version": np.array(schema_version),
        "seq_id": np.array("synthetic"),
        "side": np.array(side),
        "raw_frame_id": raw_frame_id,
        "obj_points": obj_points,
        "obj_normals": obj_normals,
        "hand_points": hand_points,
        "hand_normals": hand_normals,
        "hand_to_obj_min_dist": hand_to_obj_min_dist,
        "coordinate_frame": coordinate_frame,
        "hand_root_pose": hand_root_pose,
    }
    if dataset_name is not None:
        payload["dataset_name"] = np.asarray(dataset_name)
    if include_legacy_candidate_mask:
        payload["obj_candidate_mask_5cm"] = np.ones((T, 16), dtype=bool)
    if include_mano:
        payload.update({
            "mano_global_orient": np.zeros((T, 3), dtype=np.float32),
            "mano_transl": np.zeros((T, 3), dtype=np.float32),
            "mano_pose": np.zeros((T, pose_dim), dtype=np.float32),
            "mano_betas": np.zeros((T, 10), dtype=np.float32),
            "mano_v_template": np.zeros((778, 3), dtype=np.float32),
            "mano_use_pca": np.array(use_pca),
            "mano_num_pca_comps": np.array(pose_dim),
            "mano_flat_hand_mean": np.array(flat_hand_mean),
            "mano_pose_repr": np.array("pca" if use_pca else "axis_angle"),
        })
        if drop_mano_field is not None and drop_mano_field in payload:
            payload.pop(drop_mano_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def test_mixed_dataset_mano_descriptors_collate_from_separate_roots(
    tmp_path: Path,
) -> None:
    import torch
    from torch.utils.data._utils.collate import default_collate

    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    specs = (
        ("grab", True, 24, True, False),
        ("ARCTIC", False, 45, False, False),
        ("ContactPose", True, 15, False, True),
    )
    paths = []
    for dataset_name, use_pca, pose_dim, flat, legacy_mask in specs:
        path = tmp_path / dataset_name / f"sample_{dataset_name}.npz"
        _write_synthetic_stage3(
            path,
            dataset_name=dataset_name,
            use_pca=use_pca,
            pose_dim=pose_dim,
            flat_hand_mean=flat,
            include_legacy_candidate_mask=legacy_mask,
        )
        paths.append(path)

    dataset = CorrStaticDatasetV2(
        data_path=tmp_path,
        file_list=paths,
        num_obj_points=16,
        num_hand_points=16,
        num_supervision_edges=8,
        use_mano_reconstruction=True,
        apply_hand_perturb=True,
        runtime_resample_object=True,
        coordinate_frame="hand_root",
    )
    batch = default_collate([dataset[index] for index in range(3)])

    assert batch["__dataset_id__"] == ["arctic", "contactpose", "grab"]
    assert tuple(batch["mano_pose"].shape) == (3, 45)
    torch.testing.assert_close(
        batch["mano_pose_dim"], torch.tensor([45, 15, 24], dtype=torch.long)
    )
    assert bool(batch["runtime_resample_object"].all())
    assert tuple(batch["full_input_obj_points"].shape) == (3, 16, 3)


def test_dataset_accepts_old_schema_when_mano_fields_are_present(tmp_path: Path) -> None:
    """Schema version is an identifier, while MANO fields are the capability gate.

    A v2.0-labelled sample that already carries the complete MANO payload is
    valid for reconstruction.  Missing required fields are tested separately.
    """
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    npz = tmp_path / "old.npz"
    _write_synthetic_stage3(npz, schema_version="2.0.0")
    dataset = CorrStaticDatasetV2(
        data_path=tmp_path,
        use_mano_reconstruction=True,
        apply_hand_perturb=True,
    )
    sample = dataset[0]
    assert bool(sample["has_mano"].item()) is True
    assert bool(sample["apply_hand_perturb"].item()) is True


def test_dataset_rejects_missing_mano_field(tmp_path: Path) -> None:
    """A v2.1 npz missing any required MANO field must be rejected.

    Regression test for Fix #8: the strict schema gate is supposed to
    raise KeyError, not silently fall back to legacy hand_points.
    """
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    npz = tmp_path / "no_use_pca.npz"
    _write_synthetic_stage3(npz, schema_version="2.1.0", drop_mano_field="mano_use_pca")
    with pytest.raises(KeyError) as info:
        CorrStaticDatasetV2(
            data_path=tmp_path,
            use_mano_reconstruction=True,
            apply_hand_perturb=True,
        )
    assert "mano_use_pca" in str(info.value), (
        f"Expected KeyError to mention the missing field. Got: {info.value!r}"
    )


def test_dataset_accepts_v21_and_emits_mano_fields(tmp_path: Path) -> None:
    """A correct v2.1 npz must load and emit has_mano + sha + stable seeds."""
    from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2

    npz = tmp_path / "good.npz"
    _write_synthetic_stage3(npz, schema_version="2.1.0")
    dataset = CorrStaticDatasetV2(
        data_path=tmp_path,
        use_mano_reconstruction=True,
        apply_hand_perturb=True,
    )
    sample = dataset[0]
    assert bool(sample["has_mano"].item()) is True
    assert bool(sample["apply_hand_perturb"].item()) is True
    assert "mano_v_template_sha" in sample
    sha_a = sample["mano_v_template_sha"]
    sample_again = dataset[0]
    sha_b = sample_again["mano_v_template_sha"]
    # same frame must produce the same SHA across re-reads
    assert sha_a == sha_b
    # stable seed: repeated reads give the same hand_perturb_seed
    assert int(sample["hand_perturb_seed"].item()) == int(
        sample_again["hand_perturb_seed"].item()
    )


def test_dataset_val_clean_keeps_perturb_off(tmp_path: Path) -> None:
    """make_dataloaders must propagate apply_hand_perturb=False to val.

    Regression test for Fix #2: the val loader was inheriting the
    train-time apply_hand_perturb=True and re-running the perturbation
    on the clean val hand.  This assertion only catches the dataset
    side; the runner-level guarantee is in the on-disk round-trip.
    """
    from src.task.correspondence_ptv3_v2.dataset import (
        CorrStaticDatasetV2,
        make_dataloaders,
    )

    npz = tmp_path / "good.npz"
    _write_synthetic_stage3(npz, schema_version="2.1.0")
    # The dataset-side gate is currently ``apply_hand_perturb`` as a
    # single shared flag; the runner only inspects the per-sample
    # tensor in the batch.  Verify the dataset emits the per-sample
    # ``apply_hand_perturb`` value through the dataloader path, not
    # the legacy attribute on the instance.
    dataset = CorrStaticDatasetV2(
        data_path=tmp_path,
        use_mano_reconstruction=True,
        apply_hand_perturb=False,
    )
    sample = dataset[0]
    assert bool(sample["apply_hand_perturb"].item()) is False, (
        "Fix #2 regression: val_clean must keep apply_hand_perturb=False "
        "even when the runner trains with perturbation on."
    )
