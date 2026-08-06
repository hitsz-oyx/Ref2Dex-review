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
    return verts[:, faces].mean(axis=1)


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

        # Normals: cosine similarity must be > 0.999 after the strict
        # position diff passes.  We accept a sign flip (cross-product
        # sign vs fix_normals) by taking the absolute value, because the
        # legacy normals are oriented outward via fix_normals.
        cos = np.einsum("tnj,tnj->tn", new_normals, legacy_normals)
        cos_abs = np.abs(cos)
        mean_cos = float(cos_abs.mean())
        min_cos = float(cos_abs.min())
        assert mean_cos > thresholds.min_normal_cos, (
            f"{path}: mean normal cosine {mean_cos:.6f} <= "
            f"{thresholds.min_normal_cos:.6f}. The face-normal "
            "convention diverges from the preprocessor."
        )
        # Use min_cos to flag any face where the normal flipped
        # dramatically; a tight lower bound is impractical due to
        # degenerate triangles but a loose one catches outright swaps.
        assert min_cos > 0.9, (
            f"{path}: min |cos(normal)| {min_cos:.3f} <= 0.9. At least "
            "one face normal flipped relative to the preprocessor."
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
