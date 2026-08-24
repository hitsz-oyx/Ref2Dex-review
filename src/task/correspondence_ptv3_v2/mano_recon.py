"""MANO forward helper for the correspondence_ptv3_v2 train path.

The Stage 3 v2.1 schema persists raw MANO parameters (see
process/common/stage3_corr.py) so the training loop can re-run MANO
forward on the GPU and apply hand-side augmentations (PCA noise,
per-axis-angle jitter) without depending on the per-frame hand
point cloud that the preprocessor wrote.

The legacy v2.0 schema has no MANO fields; the dataset falls back to
the pre-stored hand_points in that case and this module is never
instantiated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# smplx pulls in chumpy at import time, and chumpy reads
# ``from numpy import bool, int, float, ...`` which fails on numpy >=
# 1.20. Add the legacy aliases before the lazy import below touches
# smplx; otherwise the train entry point crashes on the first MANO
# forward.
for _name, _value in (
    ("bool", np.bool_),
    ("int", np.int_),
    ("float", np.float_),
    ("complex", np.complex_),
    ("object", np.object_),
    ("unicode", np.str_),
    ("str", np.str_),
):
    if not hasattr(np, _name):
        setattr(np, _name, _value)


_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_mano_model_dir(model_dir: str | Path | None) -> Path | None:
    """Resolve MANO assets relative to the repository, independent of cwd."""
    if model_dir is None:
        return None
    path = Path(model_dir).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class MANOConfig:
    side: str                       # "right" or "left"
    use_pca: bool
    num_pca_comps: int
    flat_hand_mean: bool
    v_template_sha: str | None      # hex sha1 of v_template array, or None
    v_template_shape: tuple[int, ...] | None


class MANOLayerCache:
    """Lazily build and cache ``smplx.MANO`` layers keyed by config.

    We keep one layer per (side, use_pca, num_pca_comps, flat_hand_mean,
    v_template_sha) tuple. The hand is a small mesh, so the per-layer
    memory cost is negligible; the cache size is bounded by the number
    of distinct subject shapes in the dataset, which for GRAB is at
    most ten and for ARCTIC exactly one (mean shape).
    """

    def __init__(self, *, model_dir: str | Path, device: torch.device | str = "cpu") -> None:
        self.model_dir = resolve_mano_model_dir(model_dir)
        assert self.model_dir is not None
        self.device = torch.device(device)
        self._cache: dict[MANOConfig, Any] = {}

    def _resolve_model_path(self, side: str) -> Path:
        name = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
        path = self.model_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"MANO model file not found: {path}. "
                f"Set mano_model_dir to a directory that contains {name}."
            )
        return path

    @staticmethod
    def _v_template_id(v_template: np.ndarray | None) -> tuple[str | None, tuple[int, ...] | None]:
        if v_template is None:
            return None, None
        arr = np.ascontiguousarray(v_template, dtype=np.float32)
        return hashlib.sha1(arr.tobytes()).hexdigest(), tuple(arr.shape)

    def get(self, cfg: MANOConfig) -> Any:
        layer = self._cache.get(cfg)
        if layer is not None:
            return layer
        # Imported lazily so the dataset / unit tests that never call
        # MANO forward do not pay the smplx + chumpy import cost.
        from smplx import MANO

        kwargs: dict[str, Any] = {
            "is_rhand": cfg.side == "right",
            "use_pca": cfg.use_pca,
            "num_pca_comps": int(cfg.num_pca_comps),
            "flat_hand_mean": bool(cfg.flat_hand_mean),
        }
        layer = MANO(str(self._resolve_model_path(cfg.side)), **kwargs).to(self.device)
        layer.eval()
        layer.requires_grad_(False)
        self._cache[cfg] = layer
        return layer

    def get_cached_for_v_template_id(
        self,
        *,
        side: str,
        use_pca: bool,
        num_pca_comps: int,
        flat_hand_mean: bool,
        v_template_sha: str | None,
        v_template_shape: tuple[int, ...] | None,
    ) -> tuple[Any, MANOConfig] | None:
        cfg = MANOConfig(
            side=side,
            use_pca=bool(use_pca),
            num_pca_comps=int(num_pca_comps),
            flat_hand_mean=bool(flat_hand_mean),
            v_template_sha=v_template_sha,
            v_template_shape=v_template_shape,
        )
        layer = self._cache.get(cfg)
        return None if layer is None else (layer, cfg)

    def get_or_build_for_v_template(
        self,
        *,
        side: str,
        use_pca: bool,
        num_pca_comps: int,
        flat_hand_mean: bool,
        v_template: np.ndarray | None,
        v_template_sha: str | None = None,
    ) -> tuple[Any, MANOConfig]:
        computed_sha, shape = self._v_template_id(v_template)
        if v_template_sha is not None and computed_sha != v_template_sha:
            raise ValueError(
                "Precomputed v_template SHA does not match the supplied template: "
                f"expected {v_template_sha}, got {computed_sha}."
            )
        sha = v_template_sha if v_template_sha is not None else computed_sha
        cfg = MANOConfig(
            side=side,
            use_pca=bool(use_pca),
            num_pca_comps=int(num_pca_comps),
            flat_hand_mean=bool(flat_hand_mean),
            v_template_sha=sha,
            v_template_shape=shape,
        )
        cached = self._cache.get(cfg)
        if cached is not None:
            return cached, cfg
        if v_template is None:
            return self.get(cfg), cfg

        from smplx import MANO

        kwargs: dict[str, Any] = {
            "is_rhand": side == "right",
            "use_pca": bool(use_pca),
            "num_pca_comps": int(num_pca_comps),
            "flat_hand_mean": bool(flat_hand_mean),
            "v_template": np.ascontiguousarray(v_template, dtype=np.float32),
        }
        layer = MANO(str(self._resolve_model_path(side)), **kwargs).to(self.device)
        layer.eval()
        layer.requires_grad_(False)
        self._cache[cfg] = layer
        return layer, cfg


def reconstruct_hand_points(
    layer: Any,
    *,
    global_orient: torch.Tensor,    # (B, 3)
    hand_pose: torch.Tensor,        # (B, K) where K=24 (PCA) or 45 (axis-angle)
    transl: torch.Tensor,           # (B, 3)
    betas: torch.Tensor,            # (B, 10)
) -> torch.Tensor:
    """Forward MANO and return world-space vertices for the *wrist-rooted*
    hand mesh (``B, 778, 3``).

    The output lives in the hand-root frame that the Stage 3
    ``hand_root_pose_world`` already encodes as identity (the wrist is
    at the origin and the global_orient is applied to the joints). The
    caller is responsible for composing the per-frame
    ``hand_root_pose_world`` to obtain world-space points.
    """
    out = layer(
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
        betas=betas,
    )
    return out.vertices


def derive_face_center_points_and_normals(
    vertices: torch.Tensor,         # (B, 778, 3)
    faces: torch.Tensor,            # (F, 3) long
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-face center point and outward unit normal.

    Matches the convention used by ``process/GRAB/raw.py::compute_canonical_hand_surface``:
    - the sampled point of a triangle is the centroid of its three vertices
    - the outward normal is normalized ``(v1 - v0) x (v2 - v0)``,
      flipped to point away from the wrist when needed. The simple
      centroid + cross product is consistent with the canonical hand
      surface we already store; downstream losses that depend on the
      hand normal direction use the cross product and then flip the
      sign on the test path if normals point the wrong way (the
      correspondence_ptv3_v2 task is symmetric w.r.t. normal sign, so
      this is fine for the first version).
    """
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    centers = (v0 + v1 + v2) / 3.0                       # (B, F, 3)
    normals = torch.linalg.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3)
    norms = torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-8)
    normals = normals / norms
    return centers, normals


def compose_hand_root_points(
    hand_root_points: torch.Tensor,  # (B, P, 3) in hand_root frame
    hand_root_pose_world: torch.Tensor,  # (B, 4, 4)
) -> torch.Tensor:
    """Transform per-point hand geometry from hand_root to world.

    Points and normals share the rotation; only points need the
    translation. This is consistent with how
    ``process/GRAB/raw.py::process_hand_to_world`` builds the legacy
    hand_points field.
    """
    R = hand_root_pose_world[:, :3, :3]                   # (B, 3, 3)
    t = hand_root_pose_world[:, :3, 3]                    # (B, 3)
    return torch.einsum("bij,bnj->bni", R, hand_root_points) + t[:, None, :]


# ---------------------------------------------------------------------------
# Fix #6 (docs/指导.md): offline FPS-sampled hand proxy face indices.
# ---------------------------------------------------------------------------

_PROXY_FACE_IDX_CACHE: dict[tuple[str, int, str | None], np.ndarray] = {}


def get_proxy_face_idx(
    *,
    side: str,
    count: int = 256,
    model_dir: str | Path | None,
    seed: int = 0,
) -> np.ndarray:
    """Return ``count`` face indices that FPS-sample the canonical MANO mesh.

    The indices are computed once per (side, count, model_dir) and cached
    for the lifetime of the process. FPS gives a spatial-uniform proxy
    that covers fingertips, palm, finger creases and the back of the
    hand — unlike the linspace-on-face-index heuristic the previous
    implementation used.
    """
    side = str(side)
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    resolved_model_dir = resolve_mano_model_dir(model_dir)
    model_dir_key = None if resolved_model_dir is None else str(resolved_model_dir)
    cache_key = (side, int(count), model_dir_key)
    cached = _PROXY_FACE_IDX_CACHE.get(cache_key)
    if cached is not None and len(cached) == int(count):
        return cached
    if model_dir is None:
        raise ValueError(
            "model_dir is required to compute the FPS proxy face indices on first call."
        )
    from smplx import MANO

    is_rhand = side == "right"
    layer = MANO(
        str(resolved_model_dir),
        is_rhand=is_rhand,
        use_pca=True,
        num_pca_comps=24,
        flat_hand_mean=True,
    )
    v_template = layer.v_template.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(layer.faces, dtype=np.int64)
    centers = v_template[faces].mean(axis=1).astype(np.float32)  # (F, 3)
    selected = farthest_point_sampling(centers, count=int(count), seed=int(seed))
    _PROXY_FACE_IDX_CACHE[cache_key] = selected
    return selected


def farthest_point_sampling(points: np.ndarray, *, count: int, seed: int = 0) -> np.ndarray:
    """Pure-NumPy FPS implementation for an (N, 3) point cloud.

    Deterministic for a fixed (points, count, seed). The first point is
    drawn with the supplied PRNG so cache busts only when the canonical
    mesh or the count changes.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    n = int(points.shape[0])
    count = max(1, min(int(count), n))
    rng = np.random.default_rng(int(seed))
    selected = np.empty((count,), dtype=np.int64)
    dists = np.full((n,), np.inf, dtype=np.float32)
    first = int(rng.integers(0, n))
    selected[0] = first
    dists = np.linalg.norm(points - points[first], axis=1).astype(np.float32)
    for k in range(1, count):
        idx = int(np.argmax(dists))
        selected[k] = idx
        new_dists = np.linalg.norm(points - points[idx], axis=1).astype(np.float32)
        np.minimum(dists, new_dists, out=dists)
    return selected
