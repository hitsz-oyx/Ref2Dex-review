#!/usr/bin/env python3
"""Generate a 2048-point Region-weighted MANO/Inspire preview.

This is a preview-only experiment.  It deliberately does not modify the
ObjectInteractionCm cache or its 1538-point production sampler.  Both hands
are sampled with fixed per-Region quotas and area-weighted triangle draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# smplx/chumpy in the existing preprocessing stack still imports the removed
# NumPy legacy aliases.  Keep the shim local to this preview entrypoint, as in
# process/GRAB/raw.py; it does not alter any production module.
for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)

import torch
from smplx import MANO


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (  # noqa: E402
    InspireUrdfModel,
)


NUM_POINTS = 2048
INSPIRE_RATIO_POINTS = 10135
SURFACE_SEED = 2024
TIP_FRACTION = 0.20
JOINT_TO_FINGER = np.asarray(
    [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    dtype=np.int64,
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
REGION_NAMES = (
    "palm",
    "thumb_body", "thumb_tip",
    "index_body", "index_tip",
    "middle_body", "middle_tip",
    "ring_body", "ring_tip",
    "pinky_body", "pinky_tip",
)
# The balanced profile is the V1.2.6 baseline: palm=512, body=1024, tip=512.
# The one-point differences are deterministic and avoid any dependence on
# floating-point rounding when the preview is regenerated.
REGION_QUOTA_PROFILES = {
    "balanced": {
        0: 512,
        1: 205, 2: 103,
        3: 205, 4: 103,
        5: 205, 6: 102,
        7: 205, 8: 102,
        9: 204, 10: 102,
    },
    # Diagnostic variant: keep 2048 total and swap the body/tip budgets.
    "tip_heavy": {
        0: 512,
        1: 103, 2: 205,
        3: 103, 4: 205,
        5: 102, 6: 205,
        7: 102, 8: 205,
        9: 102, 10: 204,
    },
}
# Strict link-stratified Inspire profile.  The hand base gets 512 points, the
# five terminal parent links get 1024 points in total, and the remaining
# seven non-terminal finger links get the other 512 points.  Every non-empty
# visual link therefore has an explicit quota and is sampled independently.
INSPIRE_LINK_QUOTAS = {
    "hand_base_link": 512,
    "thumb_proximal_base": 74,
    "thumb_proximal": 73,
    "thumb_intermediate": 73,
    "thumb_distal": 205,
    "index_proximal": 73,
    "index_intermediate": 205,
    "middle_proximal": 73,
    "middle_intermediate": 205,
    "ring_proximal": 73,
    "ring_intermediate": 205,
    "pinky_proximal": 73,
    "pinky_intermediate": 204,
}
INSPIRE_LINK_REGION_IDS = {
    "hand_base_link": 0,
    "thumb_proximal_base": 1,
    "thumb_proximal": 1,
    "thumb_intermediate": 1,
    "thumb_distal": 2,
    "index_proximal": 3,
    "index_intermediate": 4,
    "middle_proximal": 5,
    "middle_intermediate": 6,
    "ring_proximal": 7,
    "ring_intermediate": 8,
    "pinky_proximal": 9,
    "pinky_intermediate": 10,
}
# MANO has no URDF links.  Its five finger body/tip joint segments are the
# corresponding semantic strata, with the same palm/body/tip budget as the
# strict Inspire link profile.
MANO_SEGMENT_QUOTAS = {
    0: 512,
    1: 103, 2: 205,
    3: 103, 4: 205,
    5: 102, 6: 205,
    7: 102, 8: 205,
    9: 102, 10: 204,
}
REGION_COLORS = np.asarray(
    [
        (190, 190, 190),
        (231, 76, 60), (192, 57, 43),
        (243, 156, 18), (211, 122, 10),
        (241, 196, 15), (194, 153, 0),
        (46, 204, 113), (30, 150, 80),
        (52, 152, 219), (36, 103, 166),
    ],
    dtype=np.uint8,
)


@dataclass
class TrianglePool:
    triangles: np.ndarray       # [F, 3, 3], canonical coordinates
    normals: np.ndarray         # [F, 3], flat face normals
    areas: np.ndarray           # [F], m^2
    region_ids: np.ndarray      # [F], 0..10
    source_face_ids: np.ndarray # [F]
    source_visual_ids: np.ndarray # [F], -1 for MANO
    stratum_ids: np.ndarray      # [F], link ID for Inspire or segment ID for MANO
    source_name: str


@dataclass
class SampledCloud:
    points: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    region_ids: np.ndarray
    source_face_ids: np.ndarray
    source_visual_ids: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _face_geometry(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    areas = 0.5 * lengths
    normals = cross / np.clip(lengths[:, None], 1e-12, None)
    return areas, normals


def _axis_tip_mask(points: np.ndarray, palm_center: np.ndarray, fraction: float) -> np.ndarray:
    """Return the terminal fraction of one finger's canonical triangles."""
    if len(points) < 2:
        return np.zeros(len(points), dtype=bool)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if float(np.dot(axis, points.mean(axis=0) - palm_center)) < 0.0:
        axis = -axis
    projection = points @ axis
    span = float(projection.max() - projection.min())
    if span <= 1e-12:
        return np.zeros(len(points), dtype=bool)
    threshold = float(projection.max() - fraction * span)
    return projection >= threshold


def _make_region_ids(face_centers: np.ndarray, finger_groups: np.ndarray, tip_fraction: float) -> np.ndarray:
    if face_centers.ndim != 2 or face_centers.shape[1] != 3:
        raise ValueError(f"invalid face centers: {face_centers.shape}")
    if finger_groups.shape != (len(face_centers),):
        raise ValueError("finger group shape does not match face centers")
    palm = face_centers[finger_groups == 0]
    if len(palm) == 0:
        raise ValueError("no palm triangles were assigned")
    palm_center = palm.mean(axis=0)
    region_ids = np.full(len(face_centers), -1, dtype=np.int32)
    region_ids[finger_groups == 0] = 0
    for finger_id in range(1, 6):
        mask = finger_groups == finger_id
        if not np.any(mask):
            raise ValueError(f"no triangles assigned to {FINGER_NAMES[finger_id - 1]}")
        tip_mask = _axis_tip_mask(face_centers[mask], palm_center, tip_fraction)
        body_region = 1 + 2 * (finger_id - 1)
        region_ids[np.flatnonzero(mask)[~tip_mask]] = body_region
        region_ids[np.flatnonzero(mask)[tip_mask]] = body_region + 1
    if np.any(region_ids < 0):
        raise AssertionError("unassigned triangle Region")
    return region_ids


def _build_mano_pool(model_dir: Path, tip_fraction: float) -> Tuple[TrianglePool, Dict[str, object]]:
    mano = MANO(
        str(model_dir),
        is_rhand=True,
        use_pca=True,
        num_pca_comps=24,
        flat_hand_mean=True,
    ).to("cpu")
    with torch.no_grad():
        output = mano()
    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float64)
    joints = output.joints[0].detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(mano.faces, dtype=np.int64)
    triangles = vertices[faces]
    areas, normals = _face_geometry(triangles)
    valid = areas > 1e-12
    triangles = triangles[valid]
    areas = areas[valid]
    normals = normals[valid]
    face_ids = np.flatnonzero(valid).astype(np.int64)
    centers = triangles.mean(axis=1)
    nearest_joint = ((centers[:, None, :] - joints[None, :, :]) ** 2).sum(axis=-1).argmin(axis=1)
    finger_groups = JOINT_TO_FINGER[nearest_joint]
    region_ids = _make_region_ids(centers, finger_groups, tip_fraction)
    info = {
        "asset": str((model_dir / "MANO_RIGHT.pkl").resolve()),
        "asset_sha256": _sha256(model_dir / "MANO_RIGHT.pkl"),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "valid_faces": int(len(triangles)),
        "sampling_space": "canonical_mano_flat_hand_mean",
    }
    return TrianglePool(
        triangles=triangles,
        normals=normals,
        areas=areas,
        region_ids=region_ids,
        source_face_ids=face_ids,
        source_visual_ids=np.full(len(triangles), -1, dtype=np.int64),
        stratum_ids=region_ids.copy(),
        source_name="mano",
    ), info


def _build_inspire_pool(urdf_path: Path, tip_fraction: float) -> Tuple[TrianglePool, Dict[str, object]]:
    model = InspireUrdfModel(urdf_path)
    zero_q = np.zeros(18, dtype=np.float64)
    links = model.link_transforms(model.qpos_to_urdf_order(zero_q))
    triangles_all: List[np.ndarray] = []
    normals_all: List[np.ndarray] = []
    areas_all: List[np.ndarray] = []
    centers_all: List[np.ndarray] = []
    groups_all: List[np.ndarray] = []
    face_ids_all: List[np.ndarray] = []
    visual_ids_all: List[np.ndarray] = []
    link_to_finger = {
        "hand_base_link": 0,
        "thumb_proximal_base": 1, "thumb_proximal": 1,
        "thumb_intermediate": 1, "thumb_distal": 1,
        "index_proximal": 2, "index_intermediate": 2,
        "middle_proximal": 3, "middle_intermediate": 3,
        "ring_proximal": 4, "ring_intermediate": 4,
        "pinky_proximal": 5, "pinky_intermediate": 5,
    }
    for visual_id, visual in enumerate(model.visuals):
        if visual.link not in link_to_finger:
            raise ValueError(f"unmapped Inspire visual link: {visual.link}")
        tri_local = visual.vertices[visual.faces]
        transform = links[visual.link] @ visual.local_transform
        tri = tri_local @ transform[:3, :3].T + transform[:3, 3]
        areas, normals = _face_geometry(tri)
        valid = areas > 1e-12
        triangles_all.append(tri[valid])
        normals_all.append(normals[valid])
        areas_all.append(areas[valid])
        centers_all.append(tri[valid].mean(axis=1))
        groups_all.append(np.full(int(valid.sum()), link_to_finger[visual.link], dtype=np.int64))
        face_ids_all.append(np.flatnonzero(valid).astype(np.int64))
        visual_ids_all.append(np.full(int(valid.sum()), visual_id, dtype=np.int64))
    triangles = np.concatenate(triangles_all, axis=0)
    normals = np.concatenate(normals_all, axis=0)
    areas = np.concatenate(areas_all, axis=0)
    centers = np.concatenate(centers_all, axis=0)
    finger_groups = np.concatenate(groups_all, axis=0)
    face_ids = np.concatenate(face_ids_all, axis=0)
    visual_ids = np.concatenate(visual_ids_all, axis=0)
    region_ids = _make_region_ids(centers, finger_groups, tip_fraction)
    info = {
        "asset": str(urdf_path.resolve()),
        "asset_sha256": _sha256(urdf_path),
        "visual_meshes": int(len(model.visuals)),
        "valid_triangles": int(len(triangles)),
        "sampling_space": "canonical_inspire_urdf_zero_q",
        "visual_links": [visual.link for visual in model.visuals],
    }
    return TrianglePool(
        triangles=triangles,
        normals=normals,
        areas=areas,
        region_ids=region_ids,
        source_face_ids=face_ids,
        source_visual_ids=visual_ids,
        stratum_ids=visual_ids.copy(),
        source_name="inspire",
    ), info


def _sample_pool(pool: TrianglePool, seed: int, quotas: Dict[int, int]) -> SampledCloud:
    rng = np.random.default_rng(int(seed))
    points: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    region_ids: List[np.ndarray] = []
    face_ids: List[np.ndarray] = []
    visual_ids: List[np.ndarray] = []
    for region_id in range(len(REGION_NAMES)):
        candidate = np.flatnonzero(pool.region_ids == region_id)
        if len(candidate) == 0:
            raise ValueError(f"Region {REGION_NAMES[region_id]} has no candidate triangles")
        quota = quotas[region_id]
        probabilities = pool.areas[candidate].astype(np.float64)
        probabilities /= probabilities.sum()
        chosen = rng.choice(candidate, size=quota, replace=True, p=probabilities)
        tri = pool.triangles[chosen]
        u = np.sqrt(rng.random(quota))
        v = rng.random(quota)
        bary0 = 1.0 - u
        bary1 = u * (1.0 - v)
        bary2 = u * v
        points.append(
            bary0[:, None] * tri[:, 0]
            + bary1[:, None] * tri[:, 1]
            + bary2[:, None] * tri[:, 2]
        )
        normals.append(pool.normals[chosen])
        region_ids.append(np.full(quota, region_id, dtype=np.int32))
        face_ids.append(pool.source_face_ids[chosen])
        visual_ids.append(pool.source_visual_ids[chosen])
    out_regions = np.concatenate(region_ids)
    return SampledCloud(
        points=np.concatenate(points).astype(np.float32),
        normals=np.concatenate(normals).astype(np.float32),
        colors=REGION_COLORS[out_regions],
        region_ids=out_regions,
        source_face_ids=np.concatenate(face_ids).astype(np.int64),
        source_visual_ids=np.concatenate(visual_ids).astype(np.int64),
    )


def _sample_strata(
    pool: TrianglePool,
    seed: int,
    quotas: Dict[int, int],
    region_by_stratum: Dict[int, int] | None = None,
) -> SampledCloud:
    """Sample each link/segment independently, while retaining Region labels."""
    if sum(quotas.values()) != NUM_POINTS:
        raise AssertionError(f"stratum quotas do not sum to {NUM_POINTS}")
    rng = np.random.default_rng(int(seed))
    points: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    region_ids: List[np.ndarray] = []
    face_ids: List[np.ndarray] = []
    visual_ids: List[np.ndarray] = []
    for stratum_id in sorted(quotas):
        candidate = np.flatnonzero(pool.stratum_ids == stratum_id)
        if len(candidate) == 0:
            raise ValueError(f"stratum {stratum_id} has no candidate triangles")
        quota = int(quotas[stratum_id])
        if quota <= 0:
            raise ValueError(f"stratum {stratum_id} has non-positive quota {quota}")
        probabilities = pool.areas[candidate].astype(np.float64)
        probabilities /= probabilities.sum()
        chosen = rng.choice(candidate, size=quota, replace=True, p=probabilities)
        tri = pool.triangles[chosen]
        u = np.sqrt(rng.random(quota))
        v = rng.random(quota)
        bary0 = 1.0 - u
        bary1 = u * (1.0 - v)
        bary2 = u * v
        points.append(
            bary0[:, None] * tri[:, 0]
            + bary1[:, None] * tri[:, 1]
            + bary2[:, None] * tri[:, 2]
        )
        normals.append(pool.normals[chosen])
        if region_by_stratum is None:
            region_ids.append(pool.region_ids[chosen])
        else:
            region_ids.append(np.full(quota, region_by_stratum[stratum_id], dtype=np.int32))
        face_ids.append(pool.source_face_ids[chosen])
        visual_ids.append(pool.source_visual_ids[chosen])
    out_regions = np.concatenate(region_ids)
    return SampledCloud(
        points=np.concatenate(points).astype(np.float32),
        normals=np.concatenate(normals).astype(np.float32),
        colors=REGION_COLORS[out_regions],
        region_ids=out_regions,
        source_face_ids=np.concatenate(face_ids).astype(np.int64),
        source_visual_ids=np.concatenate(visual_ids).astype(np.int64),
    )


def _sample_uniform_surface(pool: TrianglePool, seed: int, count: int) -> SampledCloud:
    """Sample the complete mesh uniformly with respect to surface area."""
    if count <= 0:
        raise ValueError(f"sample count must be positive, got {count}")
    rng = np.random.default_rng(int(seed))
    probabilities = pool.areas.astype(np.float64)
    probabilities /= probabilities.sum()
    chosen = rng.choice(len(pool.triangles), size=count, replace=True, p=probabilities)
    tri = pool.triangles[chosen]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    bary0 = 1.0 - u
    bary1 = u * (1.0 - v)
    bary2 = u * v
    points = (
        bary0[:, None] * tri[:, 0]
        + bary1[:, None] * tri[:, 1]
        + bary2[:, None] * tri[:, 2]
    )
    return SampledCloud(
        points=points.astype(np.float32),
        normals=pool.normals[chosen].astype(np.float32),
        colors=np.full((count, 3), 190, dtype=np.uint8),
        region_ids=pool.region_ids[chosen].astype(np.int32),
        source_face_ids=pool.source_face_ids[chosen].astype(np.int64),
        source_visual_ids=pool.source_visual_ids[chosen].astype(np.int64),
    )


def _write_ply(path: Path, cloud: SampledCloud, sampling_profile: str, expected_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.points.shape != (expected_count, 3):
        raise ValueError(f"expected {expected_count} points, got {cloud.points.shape}")
    if not np.isfinite(cloud.points).all() or not np.isfinite(cloud.normals).all():
        raise ValueError("PLY contains non-finite point or normal")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        if sampling_profile in ("uniform_surface", "uniform_surface_ratio"):
            handle.write(f"comment uniform_surface_hand_sampling seed={SURFACE_SEED}\n")
        else:
            handle.write(f"comment region_weighted_hand_sampling seed={SURFACE_SEED}\n")
        handle.write(f"element vertex {len(cloud.points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int region_id\nproperty int source_face_id\nproperty int source_visual_id\n")
        handle.write("end_header\n")
        for point, normal, color, region_id, face_id, visual_id in zip(
            cloud.points,
            cloud.normals,
            cloud.colors,
            cloud.region_ids,
            cloud.source_face_ids,
            cloud.source_visual_ids,
        ):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(region_id)} {int(face_id)} {int(visual_id)}\n"
            )


def _pool_summary(pool: TrianglePool, cloud: SampledCloud, quotas: Dict[int, int]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for region_id, name in enumerate(REGION_NAMES):
        mask = pool.region_ids == region_id
        sample_mask = cloud.region_ids == region_id
        result[name] = {
            "region_id": region_id,
            "quota": quotas[region_id],
            "sampled_points": int(sample_mask.sum()),
            "candidate_triangles": int(mask.sum()),
            "candidate_area_m2": float(pool.areas[mask].sum()),
            "candidate_area_fraction": float(pool.areas[mask].sum() / pool.areas.sum()),
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="research output directory")
    parser.add_argument(
        "--urdf",
        default="/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf",
    )
    parser.add_argument(
        "--mano-model-dir",
        default="dataset/arctic/data/body_models/mano",
    )
    parser.add_argument("--surface-seed", type=int, default=SURFACE_SEED)
    parser.add_argument("--tip-fraction", type=float, default=TIP_FRACTION)
    parser.add_argument(
        "--quota-profile",
        choices=tuple(REGION_QUOTA_PROFILES) + ("link_stratified", "uniform_surface", "uniform_surface_ratio"),
        default="balanced",
        help="sampling profile; uniform_surface applies no Region or link quota, uniform_surface_ratio uses area-ratio counts",
    )
    parser.add_argument("--modification-version", default="V1.2.6")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.surface_seed != SURFACE_SEED:
        raise ValueError(f"V1.2.6 preview is pinned to surface_seed={SURFACE_SEED}")
    if not 0.0 < args.tip_fraction < 0.5:
        raise ValueError("tip fraction must be in (0, 0.5)")
    quotas = REGION_QUOTA_PROFILES.get(args.quota_profile)
    if quotas is not None and sum(quotas.values()) != NUM_POINTS:
        raise AssertionError(f"quota profile {args.quota_profile} does not sum to {NUM_POINTS}")
    output_dir = Path(args.output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = Path(args.urdf).resolve()
    mano_dir = Path(args.mano_model_dir).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if not (mano_dir / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(mano_dir / "MANO_RIGHT.pkl")

    mano_pool, mano_info = _build_mano_pool(mano_dir, args.tip_fraction)
    inspire_pool, inspire_info = _build_inspire_pool(urdf_path, args.tip_fraction)
    point_counts = {"mano": NUM_POINTS, "inspire": NUM_POINTS}
    if args.quota_profile in ("uniform_surface", "uniform_surface_ratio"):
        inspire_count = INSPIRE_RATIO_POINTS if args.quota_profile == "uniform_surface_ratio" else NUM_POINTS
        mano_cloud = _sample_uniform_surface(mano_pool, args.surface_seed, NUM_POINTS)
        inspire_cloud = _sample_uniform_surface(inspire_pool, args.surface_seed, inspire_count)
        summary_region_quotas = None
        stratum_quotas = None
        sampling_method = "global_surface_area_uniform_triangle_barycentric"
        point_counts = {"mano": NUM_POINTS, "inspire": inspire_count}
    elif args.quota_profile == "link_stratified":
        mano_cloud = _sample_strata(
            mano_pool,
            args.surface_seed,
            MANO_SEGMENT_QUOTAS,
            region_by_stratum={
                segment_id: segment_id for segment_id in MANO_SEGMENT_QUOTAS
            },
        )
        inspire_link_quotas = {
            index: INSPIRE_LINK_QUOTAS[link_name]
            for index, link_name in enumerate(inspire_info["visual_links"])
        }
        inspire_link_regions = {
            index: INSPIRE_LINK_REGION_IDS[link_name]
            for index, link_name in enumerate(inspire_info["visual_links"])
        }
        inspire_cloud = _sample_strata(
            inspire_pool,
            args.surface_seed,
            inspire_link_quotas,
            region_by_stratum=inspire_link_regions,
        )
        summary_region_quotas = None
        stratum_quotas = {
            "mano_segments": {REGION_NAMES[k]: int(v) for k, v in MANO_SEGMENT_QUOTAS.items()},
            "inspire_links": {str(k): int(v) for k, v in INSPIRE_LINK_QUOTAS.items()},
        }
        sampling_method = "stratum_surface_area_weighted_triangle_barycentric"
    else:
        assert quotas is not None
        mano_cloud = _sample_pool(mano_pool, args.surface_seed, quotas)
        inspire_cloud = _sample_pool(inspire_pool, args.surface_seed, quotas)
        summary_region_quotas = {REGION_NAMES[k]: int(v) for k, v in quotas.items()}
        stratum_quotas = None
        sampling_method = "region_surface_area_weighted_triangle_barycentric"
    if args.quota_profile == "uniform_surface":
        mano_ply = output_dir / "mano_uniform_2048.ply"
        inspire_ply = output_dir / "inspire_uniform_2048.ply"
    elif args.quota_profile == "uniform_surface_ratio":
        mano_ply = output_dir / "mano_uniform_2048.ply"
        inspire_ply = output_dir / f"inspire_uniform_{INSPIRE_RATIO_POINTS}.ply"
    else:
        mano_ply = output_dir / "mano_weighted_2048.ply"
        inspire_ply = output_dir / "inspire_weighted_2048.ply"
    _write_ply(mano_ply, mano_cloud, args.quota_profile, point_counts["mano"])
    _write_ply(inspire_ply, inspire_cloud, args.quota_profile, point_counts["inspire"])

    summary = {
        "schema_name": "ref2dex_hand_region_sampling_preview_v1",
        "modification_version": args.modification_version,
        "total_points": NUM_POINTS if len(set(point_counts.values())) == 1 else None,
        "point_counts": point_counts,
        "surface_seed": args.surface_seed,
        "tip_fraction": args.tip_fraction,
        "quota_profile": args.quota_profile,
        "region_names": list(REGION_NAMES),
        "region_quotas": summary_region_quotas,
        "stratum_quotas": stratum_quotas,
        "sampling_method": sampling_method,
        "region_colors_rgb": {
            REGION_NAMES[i]: [int(x) for x in REGION_COLORS[i]] for i in range(len(REGION_NAMES))
        },
        "within_region_sampling": (
            None if args.quota_profile in ("uniform_surface", "uniform_surface_ratio")
            else "area_weighted_triangle_with_replacement_then_barycentric"
        ),
        "mano": {**mano_info, "regions": _pool_summary(mano_pool, mano_cloud, quotas or {
            region_id: int(np.sum(mano_cloud.region_ids == region_id))
            for region_id in range(len(REGION_NAMES))
        })},
        "inspire": {**inspire_info, "regions": _pool_summary(inspire_pool, inspire_cloud, quotas or {
            region_id: int(np.sum(inspire_cloud.region_ids == region_id))
            for region_id in range(len(REGION_NAMES))
        })},
        "outputs": {
            "mano_ply": str(mano_ply),
            "inspire_ply": str(inspire_ply),
        },
    }
    summary_path = output_dir / "sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "modification_version": args.modification_version,
        "operation": "hand_region_sampling_preview",
        "run_id": output_dir.name,
        "run_status": "COMPLETED",
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "urdf": str(urdf_path),
            "urdf_sha256": _sha256(urdf_path),
            "mano_model": str((mano_dir / "MANO_RIGHT.pkl").resolve()),
            "mano_model_sha256": _sha256(mano_dir / "MANO_RIGHT.pkl"),
        },
        "parameters": {
            "total_points": summary["total_points"],
            "point_counts": summary["point_counts"],
            "surface_seed": args.surface_seed,
            "tip_fraction": args.tip_fraction,
            "quota_profile": args.quota_profile,
            "region_quotas": summary["region_quotas"],
            "stratum_quotas": summary["stratum_quotas"],
            "sampling_method": summary["sampling_method"],
            "within_region_sampling": summary["within_region_sampling"],
        },
        "outputs": {
            "mano_ply": str(mano_ply),
            "inspire_ply": str(inspire_ply),
            "sampling_summary": str(summary_path),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), "point_counts": point_counts, "files": [mano_ply.name, inspire_ply.name]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
