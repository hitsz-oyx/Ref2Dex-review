#!/usr/bin/env python3
"""Mouse-driven Viser viewer for the ObjectInteractionCm training cache.

The default index is the same Dexplore-RL cache used by the recent
ObjectInteractionCm training configuration.  Every trajectory is loaded from
its index entry, so the displayed split, source, variant, point clouds, object
pose, and optional mesh all refer to the same training sample provenance.

All interaction is exposed as Viser GUI controls.  The distance threshold is
cumulative: for example, ``3 cm`` colors every hand point whose nearest object
point is no farther than 3 cm.  The optional object-to-hand KNN control colors
the union of hand points selected by each object point's nearest 1/4/8/16/32/64
hand neighbors; no connection lines are drawn.
Point-cloud and mesh visibility are controlled independently from mouse-only
dropdowns, including object-only, hand-only, combined, and hidden modes.
The future-frame slider can overlay the hand and object at ``t + delta`` for
``delta`` from 1 to 30 cache frames; ``delta = 0`` hides the future overlay.

Examples::

    python -m src.task.ObjectInteractionCm.visualize_grab
    python -m src.task.ObjectInteractionCm.visualize_grab \
        --split val --sequence s1/banana_lift --mesh-display both
    python -m src.task.ObjectInteractionCm.visualize_grab \
        --check-only --split train --sequence s1/airplane_fly_1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .dataset import _nearest_distances
    from .tools.data.build_dexplore_rl_cache import (
        InspireUrdfModel,
        NATIVE_Q_START,
        NUM_DOFS,
        _load_tensor,
    )
except ImportError:  # Allow direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.task.ObjectInteractionCm.dataset import _nearest_distances
    from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (
        InspireUrdfModel,
        NATIVE_Q_START,
        NUM_DOFS,
        _load_tensor,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
MODIFICATION_VERSION = "V1.4.22"
INDEX_SCHEMA = "ref2dex_object_interaction_cm_index_v1_1"
INDEX_SCHEMAS = {INDEX_SCHEMA, "ref2dex_object_interaction_cm_index_v1_2"}
SEQUENCE_SCHEMA = "ref2dex_object_interaction_cm_dexplore_rl_v1"
BILATERAL_SEQUENCE_SCHEMA = "ref2dex_object_interaction_cm_bilateral_geometry_v1"
EXPECTED_OBJECT_POINTS = 4096
DEFAULT_HAND_POINTS = 1538
MODEL_RUNTIME_OBJECT_POINTS = 1024

SPLIT_LABELS = {"全部": None, "训练集": "train", "验证集": "val", "测试集": "test"}
SPLIT_FROM_CLI = {"all": "全部", "train": "训练集", "val": "验证集", "test": "测试集"}
VARIANT_LABELS = {
    "mano": "MANO",
    "mano_bilateral_raw": "MANO-bilateral",
    "inspire_rl": "Inspire-RL",
    "inspire_geometric": "Inspire-geometric",
}
THRESHOLD_LABELS = ("关闭", "1 cm", "2 cm", "3 cm", "4 cm", "5 cm")
THRESHOLD_METERS = {
    "关闭": 0.0,
    "1 cm": 0.01,
    "2 cm": 0.02,
    "3 cm": 0.03,
    "4 cm": 0.04,
    "5 cm": 0.05,
}
KNN_LABELS = ("关闭", "n = 1", "n = 4", "n = 8", "n = 16", "n = 32", "n = 64")
KNN_VALUES = {"关闭": None, "n = 1": 1, "n = 4": 4, "n = 8": 8, "n = 16": 16, "n = 32": 32, "n = 64": 64}
POINT_DISPLAY_LABELS = ("关闭", "仅物体点", "仅手点", "物体点 + 手点")
POINT_DISPLAY_FROM_CLI = {
    "off": "关闭",
    "object": "仅物体点",
    "hand": "仅手点",
    "both": "物体点 + 手点",
}
MESH_DISPLAY_LABELS = ("关闭", "仅物体 mesh", "仅手 mesh", "物体 + 手 mesh")
MESH_DISPLAY_FROM_CLI = {
    "off": "关闭",
    "object": "仅物体 mesh",
    "hand": "仅手 mesh",
    "both": "物体 + 手 mesh",
}

OBJECT_COLOR = np.asarray([170, 175, 185], dtype=np.uint8)
MANO_COLOR = np.asarray([70, 150, 245], dtype=np.uint8)
INSPIRE_COLOR = np.asarray([245, 150, 60], dtype=np.uint8)
HIGHLIGHT_COLOR = np.asarray([238, 64, 64], dtype=np.uint8)
KNN_HIGHLIGHT_COLOR = np.asarray([250, 205, 55], dtype=np.uint8)
OBJECT_MESH_COLOR = (190, 195, 205)
MANO_MESH_COLOR = (80, 155, 245)
INSPIRE_MESH_COLOR = (245, 155, 70)
FUTURE_OBJECT_COLOR = np.asarray([60, 210, 145], dtype=np.uint8)
FUTURE_HAND_COLOR = np.asarray([190, 90, 235], dtype=np.uint8)
CONTEXT_OBJECT_COLOR = np.asarray([105, 110, 120], dtype=np.uint8)
FUTURE_OBJECT_MESH_COLOR = (60, 210, 145)
FUTURE_HAND_MESH_COLOR = (190, 90, 235)
FUTURE_DELTA_MAX = 30


@dataclass(frozen=True)
class SequenceRecord:
    """One exact trajectory entry from the training index."""

    sequence_id: str
    split: str
    source: str
    variant: str
    object_name: str
    dataset: str
    path: Path

    @property
    def label(self) -> str:
        variant = VARIANT_LABELS.get(self.variant, self.variant)
        return f"[{self.split} | {variant}] {self.sequence_id}"


@dataclass(frozen=True)
class MeshPart:
    """A mesh whose vertices are local to ``pose``."""

    key: str
    vertices: np.ndarray
    faces: np.ndarray
    pose: np.ndarray
    dynamic_vertices: bool = False


def _resolve_index_entry(index_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (index_root / path).resolve()


def _is_mano_variant(variant: str) -> bool:
    return variant in {"mano", "mano_bilateral_raw"}


def discover_sequences(index_path: Path) -> Tuple[Dict[str, Any], List[SequenceRecord]]:
    """Read sequence order and provenance strictly from a training index."""

    index_path = Path(index_path).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"training index does not exist: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_name") not in INDEX_SCHEMAS:
        raise ValueError(
            f"unsupported index schema {payload.get('schema_name')!r}; "
            f"expected one of {sorted(INDEX_SCHEMAS)!r}"
        )
    records: List[SequenceRecord] = []
    for split in ("train", "val", "test"):
        for entry in payload.get("sequences", {}).get(split, []):
            entry_split = str(entry.get("split", split))
            if entry_split != split:
                raise ValueError(f"index split mismatch for {entry.get('id')}: {entry_split} != {split}")
            path = _resolve_index_entry(index_path.parent, str(entry["path"]))
            sequence_id = str(entry["id"])
            source = str(entry["source"])
            dataset = str(entry.get("dataset") or source)
            native_id = sequence_id[len(dataset) + 1:] if sequence_id.startswith(dataset + "/") else sequence_id
            variant = str(
                entry.get("variant")
                or ("inspire_geometric" if source == "inspire_f1" else source)
            )
            if variant == "mano_bilateral_raw":
                missing = [
                    name for name in ("left.npz", "right.npz", "shared.npz")
                    if not (path / name).is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"indexed bilateral MANO sequence is missing {missing}: {path}"
                    )
            elif not (path / "geometry" / "manifest.json").is_file():
                raise FileNotFoundError(f"indexed sequence geometry is unavailable: {path}")
            object_name = str(entry.get("object_name") or Path(native_id).name.split("_", 1)[0])
            records.append(
                SequenceRecord(
                    sequence_id=sequence_id,
                    split=split,
                    source=source,
                    variant=variant,
                    object_name=object_name,
                    dataset=dataset,
                    path=path,
                )
            )
    if not records:
        raise ValueError(f"training index contains no sequences: {index_path}")
    return payload, records


def _filter_records(records: Sequence[SequenceRecord], split: Optional[str]) -> List[SequenceRecord]:
    selected = [record for record in records if split is None or record.split == split]
    if not selected:
        raise ValueError(f"no trajectories are available for split={split!r}")
    return selected


def choose_sequence(
    records: Sequence[SequenceRecord], sequence_id: Optional[str], sequence_index: int
) -> SequenceRecord:
    if sequence_id is not None:
        matches = [record for record in records if record.sequence_id == sequence_id]
        if not matches:
            raise ValueError(f"sequence {sequence_id!r} is absent from the selected split")
        if len(matches) > 1:
            raise ValueError(f"sequence {sequence_id!r} is ambiguous in the selected split")
        return matches[0]
    return records[int(sequence_index) % len(records)]


class TrainingSequence:
    """Read-only view of an ObjectInteractionCm cache or raw bilateral MANO triplet."""

    def __init__(self, record: SequenceRecord) -> None:
        self.record = record
        if record.variant == "mano_bilateral_raw":
            self._load_raw_bilateral_mano()
        else:
            self._load_geometry_cache()

        manifest_path = self.metadata_path
        manifest_hand_points = self.manifest.get("hand_points")
        if manifest_hand_points is None:
            manifest_hand_points = DEFAULT_HAND_POINTS
        try:
            self.hand_points_count = int(manifest_hand_points)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid manifest hand_points at {manifest_path}: {manifest_hand_points!r}") from exc
        if self.hand_points_count <= 0:
            raise ValueError(f"manifest hand_points must be positive at {manifest_path}")
        if self.hand_points_all.ndim != 3 or self.hand_points_all.shape[1:] != (
            self.hand_points_count,
            3,
        ):
            raise ValueError(
                f"{record.path}: manifest hand_points={self.hand_points_count} disagrees with "
                f"hand geometry shape {self.hand_points_all.shape}"
            )

        if self.object_points_all.ndim != 3 or self.object_points_all.shape[1:] != (
            EXPECTED_OBJECT_POINTS,
            3,
        ):
            raise ValueError(
                f"{record.path}: expected object pool [T,{EXPECTED_OBJECT_POINTS},3], "
                f"got {self.object_points_all.shape}"
            )
        self.frame_count = int(self.object_points_all.shape[0])
        expected_shapes = {
            "object normals": (self.frame_count, EXPECTED_OBJECT_POINTS, 3),
            "object pose": (self.frame_count, 4, 4),
            "source frame": (self.frame_count,),
            "hand points": (self.frame_count, self.hand_points_count, 3),
            "hand normals": (self.frame_count, self.hand_points_count, 3),
            "5 cm activity": (self.frame_count,),
        }
        actual = {
            "object normals": self.object_normals_all.shape,
            "object pose": self.object_pose_all.shape,
            "source frame": self.source_frame_all.shape,
            "hand points": self.hand_points_all.shape,
            "hand normals": self.hand_normals_all.shape,
            "5 cm activity": self.active_5cm_all.shape,
        }
        invalid = {name: (actual[name], shape) for name, shape in expected_shapes.items() if actual[name] != shape}
        if invalid:
            raise ValueError(f"geometry shape mismatch at {record.path}: {invalid}")
        if self.context_points_all is not None:
            if (
                self.context_points_all.ndim != 3
                or self.context_points_all.shape[0] != self.frame_count
                or self.context_points_all.shape[2] != 3
                or not np.isfinite(self.context_points_all).all()
            ):
                raise ValueError(
                    f"invalid viewer-only context geometry at {record.path}: "
                    f"{self.context_points_all.shape}"
                )
        self.effective_fps = float(self.manifest.get("effective_fps", 30.0) or 30.0)

    def _load_raw_bilateral_mano(self) -> None:
        """Adapt the immutable V1.4 GRAB MANO NPZ triplet in memory."""

        self.geometry = self.record.path
        left_path = self.record.path / "left.npz"
        right_path = self.record.path / "right.npz"
        shared_path = self.record.path / "shared.npz"
        self.metadata_path = shared_path
        with np.load(left_path, allow_pickle=False) as left, np.load(
            right_path, allow_pickle=False
        ) as right, np.load(shared_path, allow_pickle=False) as shared:
            schemas = {
                "left": str(np.asarray(left["schema_name"]).item()),
                "right": str(np.asarray(right["schema_name"]).item()),
                "shared": str(np.asarray(shared["schema_name"]).item()),
            }
            expected_schemas = {
                "left": "ref2dex_cm_sequence_hand",
                "right": "ref2dex_cm_sequence_hand",
                "shared": "ref2dex_cm_sequence_shared",
            }
            if schemas != expected_schemas:
                raise ValueError(f"unsupported bilateral MANO schemas at {self.record.path}: {schemas}")
            left_side = str(np.asarray(left["side"]).item())
            right_side = str(np.asarray(right["side"]).item())
            if (left_side, right_side) != ("left", "right"):
                raise ValueError(
                    f"bilateral MANO side order must be left then right, got {(left_side, right_side)}"
                )
            native_id = str(np.asarray(shared["seq_id"]).item())
            expected_id = self.record.sequence_id
            if expected_id.startswith(self.record.dataset + "/"):
                expected_id = expected_id[len(self.record.dataset) + 1 :]
            if native_id != expected_id:
                raise ValueError(
                    f"index/raw MANO sequence mismatch at {shared_path}: {native_id!r} != {expected_id!r}"
                )
            source_fps = float(np.asarray(shared["source_fps"]).item())
            ds_rate = int(np.asarray(shared["ds_rate"]).item())
            if source_fps <= 0.0 or ds_rate <= 0:
                raise ValueError(
                    f"invalid bilateral MANO timing at {shared_path}: "
                    f"source_fps={source_fps}, ds_rate={ds_rate}"
                )
            coordinate_frame = str(np.asarray(shared["coordinate_frame"]).item())
            self.object_points_all = np.asarray(shared["obj_points_world"], dtype=np.float32)
            self.object_normals_all = np.asarray(shared["obj_normals_world"], dtype=np.float32)
            self.context_points_all = (
                np.asarray(shared["context_points_world"], dtype=np.float32)
                if "context_points_world" in shared.files else None
            )
            self.object_pose_available = "obj_pose_world" in shared.files
            if self.object_pose_available:
                self.object_pose_all = np.asarray(shared["obj_pose_world"], dtype=np.float32)
            else:
                self.object_pose_all = np.broadcast_to(
                    np.eye(4, dtype=np.float32),
                    (len(shared["raw_frame_id"]), 4, 4),
                ).copy()
            self.source_frame_all = np.asarray(shared["raw_frame_id"], dtype=np.int32)
            self.hand_points_all = np.concatenate(
                [np.asarray(left["hand_points_world"]), np.asarray(right["hand_points_world"])],
                axis=1,
            ).astype(np.float32, copy=False)
            self.hand_normals_all = np.concatenate(
                [np.asarray(left["hand_normals_world"]), np.asarray(right["hand_normals_world"])],
                axis=1,
            ).astype(np.float32, copy=False)
            candidate = np.logical_or(
                np.asarray(left["obj_candidate_mask_5cm"], dtype=bool),
                np.asarray(right["obj_candidate_mask_5cm"], dtype=bool),
            )
            self.active_5cm_all = candidate.any(axis=1)
            mesh_keys = ("hand_mesh_vertices_world", "hand_mesh_faces")
            left_has_mesh = all(key in left.files for key in mesh_keys)
            right_has_mesh = all(key in right.files for key in mesh_keys)
            if left_has_mesh != right_has_mesh:
                raise ValueError(f"bilateral MANO mesh availability differs at {self.record.path}")
            if left_has_mesh:
                self.raw_mano_vertices = (
                    np.asarray(left["hand_mesh_vertices_world"], dtype=np.float32),
                    np.asarray(right["hand_mesh_vertices_world"], dtype=np.float32),
                )
                self.raw_mano_faces = (
                    np.asarray(left["hand_mesh_faces"], dtype=np.int32),
                    np.asarray(right["hand_mesh_faces"], dtype=np.int32),
                )
            else:
                self.raw_mano_vertices = None
                self.raw_mano_faces = None
        self.world_frame = f"{self.record.dataset}_native_world"
        self.manifest = {
            "schema_name": "ref2dex_viewer_raw_bilateral_mano_v1",
            "sequence_id": native_id,
            "source": self.record.source,
            "source_dataset": self.record.dataset,
            "coordinate_frame": coordinate_frame,
            "world_frame": self.world_frame,
            "hand_side": "bilateral_merged_left_then_right",
            "merged_hand_sides": True,
            "hand_points": int(self.hand_points_all.shape[1]),
            "effective_fps": source_fps / ds_rate,
            "viewer_only_adapter": True,
            "object_pose_available": self.object_pose_available,
            "hand_mesh_available": left_has_mesh,
            "context_points_available": self.context_points_all is not None,
            "source_files": [str(left_path), str(right_path), str(shared_path)],
        }

    def _load_geometry_cache(self) -> None:
        self.geometry = self.record.path / "geometry"
        manifest_path = self.geometry / "manifest.json"
        self.metadata_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_schema = self.manifest.get("schema_name")
        if manifest_schema == SEQUENCE_SCHEMA:
            expected = {
                "schema_name": SEQUENCE_SCHEMA,
                "sequence_id": self.record.sequence_id,
                "split": self.record.split,
                "source": self.record.source,
                "variant": self.record.variant,
                "world_frame": "dexplore_native_object_pose_world",
                "coordinate_frame": "object_pose_t",
                "hand_side": "right",
            }
            self.world_frame = str(self.manifest["world_frame"])
        elif manifest_schema == BILATERAL_SEQUENCE_SCHEMA:
            native_id = (
                self.record.sequence_id[len(self.record.dataset) + 1:]
                if self.record.sequence_id.startswith(self.record.dataset + "/")
                else self.record.sequence_id
            )
            expected = {
                "schema_name": BILATERAL_SEQUENCE_SCHEMA,
                "sequence_id": native_id,
                "source": self.record.source,
                "source_dataset": self.record.dataset,
                "coordinate_frame": "object_pose_t",
                "hand_side": "bilateral_merged_left_then_right",
                "merged_hand_sides": True,
            }
            self.world_frame = str(
                self.manifest.get("world_frame") or f"{self.record.dataset}_native_world"
            )
        else:
            raise ValueError(f"unsupported sequence schema at {manifest_path}: {manifest_schema!r}")
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"index/sequence manifest mismatch at {manifest_path}: {mismatches}")

        self.object_points_all = self._load("obj_points_pool_world.npy")
        self.context_points_all = None
        self.object_normals_all = self._load("obj_normals_pool_world.npy")
        self.object_pose_all = self._load("obj_pose_world.npy")
        self.object_pose_available = True
        self.source_frame_all = self._load("source_frame_id.npy")
        self.hand_points_all = self._load("hand_points_world.npy")
        self.hand_normals_all = self._load("hand_normals_world.npy")
        candidate_path = self.geometry / "obj_candidate_mask_5cm.npy"
        if not candidate_path.is_file():
            candidate_path = self.geometry / "obj_candidate_mask_2cm.npy"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"indexed geometry candidate mask is unavailable: {self.geometry}")
        candidate = np.load(candidate_path, mmap_mode="r")
        if candidate.ndim == 2:
            candidate = np.asarray(candidate, dtype=bool).any(axis=1)
        self.active_5cm_all = np.asarray(candidate, dtype=bool)

    def _load(self, filename: str) -> np.ndarray:
        path = self.geometry / filename
        if not path.is_file():
            raise FileNotFoundError(f"indexed geometry field is unavailable: {path}")
        return np.load(path, mmap_mode="r")

    def object_points(self, frame: int) -> np.ndarray:
        return np.asarray(self.object_points_all[int(frame)], dtype=np.float32)

    def hand_points(self, frame: int) -> np.ndarray:
        return np.asarray(self.hand_points_all[int(frame)], dtype=np.float32)

    def context_points(self, frame: int) -> np.ndarray:
        if self.context_points_all is None:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(self.context_points_all[int(frame)], dtype=np.float32)

    def object_pose(self, frame: int) -> np.ndarray:
        return np.asarray(self.object_pose_all[int(frame)], dtype=np.float32)

    def source_frame(self, frame: int) -> int:
        return int(self.source_frame_all[int(frame)])

    def active_5cm(self, frame: int) -> bool:
        return bool(self.active_5cm_all[int(frame)])


class DistanceCache:
    """Cache nearest distances and object-to-hand KNN masks for visited frames."""

    def __init__(self, sequence: TrainingSequence) -> None:
        self.sequence = sequence
        self._values: Dict[int, np.ndarray] = {}
        self._knn_masks: Dict[Tuple[int, int], np.ndarray] = {}

    def get(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if frame not in self._values:
            self._values[frame] = _nearest_distances(
                self.sequence.hand_points(frame), self.sequence.object_points(frame)
            )
        return self._values[frame]

    def knn_hand_mask(self, frame: int, k: int) -> np.ndarray:
        """Return hand points selected by any object's nearest ``k`` hand points.

        This is the same direction as the model's local interaction KNN:
        every object point queries its nearest hand points.  The viewer only
        colors the union of selected hand points and deliberately does not
        draw object-hand connection lines.
        """

        frame = int(frame)
        k = min(max(int(k), 1), self.sequence.hand_points_count)
        cache_key = (frame, k)
        if cache_key in self._knn_masks:
            return self._knn_masks[cache_key]
        object_points = self.sequence.object_points(frame)
        hand_points = self.sequence.hand_points(frame)
        try:
            from scipy.spatial import cKDTree

            indices = cKDTree(hand_points).query(object_points, k=k, workers=1)[1]
            indices = np.asarray(indices, dtype=np.int64)
            if indices.ndim == 1:
                indices = indices[:, None]
        except (ImportError, TypeError):
            indices_chunks = []
            for start in range(0, len(object_points), 512):
                chunk = object_points[start : start + 512]
                delta = chunk[:, None, :] - hand_points[None, :, :]
                distances = np.einsum("ijk,ijk->ij", delta, delta)
                indices_chunks.append(np.argpartition(distances, kth=k - 1, axis=1)[:, :k])
            indices = np.concatenate(indices_chunks, axis=0).astype(np.int64, copy=False)
        mask = np.zeros(len(hand_points), dtype=bool)
        mask[indices.reshape(-1)] = True
        self._knn_masks[cache_key] = mask
        return mask


class ObjectMeshLibrary:
    """Lazy loader for canonical GRAB object meshes."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._meshes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def load(self, object_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if object_name in self._meshes:
            return self._meshes[object_name]
        candidates = [self.root / f"{object_name}{suffix}" for suffix in (".ply", ".obj")]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"object mesh not found for {object_name!r} below {self.root}")
        try:
            import trimesh
        except ImportError as exc:
            raise RuntimeError("trimesh is required to load GRAB object meshes") from exc
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            geometries = tuple(mesh.geometry.values())
            if not geometries:
                raise ValueError(f"object mesh scene is empty: {path}")
            mesh = trimesh.util.concatenate(geometries)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"invalid object mesh geometry: {path}")
        self._meshes[object_name] = (vertices, faces)
        return vertices, faces


class InspireMeshLibrary:
    """Exact Inspire visual meshes posed by the cache builder's own FK."""

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self._model: Optional[InspireUrdfModel] = None

    @property
    def model(self) -> InspireUrdfModel:
        if self._model is None:
            if not self.urdf_path.is_file():
                raise FileNotFoundError(f"Inspire URDF does not exist: {self.urdf_path}")
            self._model = InspireUrdfModel(self.urdf_path)
        return self._model

    def parts(self, native_q: np.ndarray) -> List[MeshPart]:
        model = self.model
        links = model.link_transforms(model.qpos_to_urdf_order(native_q))
        parts: List[MeshPart] = []
        for index, visual in enumerate(model.visuals):
            parts.append(
                MeshPart(
                    key=f"inspire_{index:02d}_{visual.link}",
                    vertices=np.asarray(visual.vertices, dtype=np.float32),
                    faces=np.asarray(visual.faces, dtype=np.int32),
                    pose=np.asarray(links[visual.link] @ visual.local_transform, dtype=np.float32),
                )
            )
        return parts


class TrainingMeshProvider:
    """Build optional meshes in the same world frame as one training sequence."""

    def __init__(
        self,
        sequence: TrainingSequence,
        object_library: ObjectMeshLibrary,
        inspire_library: InspireMeshLibrary,
    ) -> None:
        self.sequence = sequence
        self.object_library = object_library
        self.inspire_library = inspire_library
        self._mano_vertices: Optional[np.ndarray] = None
        self._mano_faces: Optional[np.ndarray] = None
        self._parent_pose: Optional[np.ndarray] = None
        self._parent_raw_to_index: Optional[Dict[int, int]] = None
        self._native_q: Optional[np.ndarray] = None

    def object_parts(self, frame: int) -> List[MeshPart]:
        if not getattr(self.sequence, "object_pose_available", True):
            raise FileNotFoundError(
                "object mesh is disabled because this articulated sequence has no single obj_pose_world"
            )
        vertices, faces = self.object_library.load(self.sequence.record.object_name)
        return [
            MeshPart(
                key=f"object_{self.sequence.record.object_name}",
                vertices=vertices,
                faces=faces,
                pose=self.sequence.object_pose(frame),
            )
        ]

    def _load_mano(self) -> None:
        if self._mano_vertices is not None:
            return
        parent_root_value = self.sequence.manifest.get("input_parent_cache")
        if not parent_root_value:
            raise FileNotFoundError("MANO sequence manifest does not declare input_parent_cache")
        parent_root = Path(str(parent_root_value)).expanduser().resolve()
        vertices_path = parent_root / "right" / "hand_mesh_vertices_world.npy"
        faces_path = parent_root / "right" / "hand_mesh_faces.npy"
        pose_path = parent_root / "shared" / "obj_pose_world.npy"
        raw_path = parent_root / "shared" / "raw_frame_id.npy"
        for path in (vertices_path, faces_path, pose_path, raw_path):
            if not path.is_file():
                raise FileNotFoundError(f"MANO mesh provenance file is unavailable: {path}")
        vertices = np.load(vertices_path, mmap_mode="r")
        faces = np.load(faces_path, mmap_mode="r")
        pose = np.load(pose_path, mmap_mode="r")
        raw = np.load(raw_path, mmap_mode="r")
        if vertices.ndim != 3 or vertices.shape[0] != len(raw) or vertices.shape[2] != 3:
            raise ValueError(f"invalid MANO mesh vertices: {vertices_path} {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"invalid MANO mesh faces: {faces_path} {faces.shape}")
        if pose.shape != (len(raw), 4, 4):
            raise ValueError(f"invalid parent object poses: {pose_path} {pose.shape}")
        self._mano_vertices = vertices
        self._mano_faces = faces
        self._parent_pose = pose
        self._parent_raw_to_index = {int(value): index for index, value in enumerate(np.asarray(raw))}

    def _mano_parts(self, frame: int) -> List[MeshPart]:
        self._load_mano()
        assert self._mano_vertices is not None
        assert self._mano_faces is not None
        assert self._parent_pose is not None
        assert self._parent_raw_to_index is not None
        source_frame = self.sequence.source_frame(frame)
        parent_frame = self._parent_raw_to_index.get(source_frame)
        if parent_frame is None:
            raise KeyError(f"source frame {source_frame} is absent from the MANO parent cache")
        old_pose = np.asarray(self._parent_pose[parent_frame], dtype=np.float32)
        new_pose = self.sequence.object_pose(frame)
        old_world = np.asarray(self._mano_vertices[parent_frame], dtype=np.float32)
        local = (old_world - old_pose[:3, 3]) @ old_pose[:3, :3]
        new_world = local @ new_pose[:3, :3].T + new_pose[:3, 3]
        return [
            MeshPart(
                key="mano_right",
                vertices=np.asarray(new_world, dtype=np.float32),
                faces=np.asarray(self._mano_faces, dtype=np.int32),
                pose=np.eye(4, dtype=np.float32),
                dynamic_vertices=True,
            )
        ]

    def _raw_bilateral_mano_parts(self, frame: int) -> List[MeshPart]:
        vertices = getattr(self.sequence, "raw_mano_vertices", None)
        faces = getattr(self.sequence, "raw_mano_faces", None)
        if vertices is None or faces is None:
            raise FileNotFoundError("raw bilateral MANO mesh arrays are unavailable")
        return [
            MeshPart(
                key=f"mano_{side}",
                vertices=np.asarray(vertices[index][int(frame)], dtype=np.float32),
                faces=np.asarray(faces[index], dtype=np.int32),
                pose=np.eye(4, dtype=np.float32),
                dynamic_vertices=True,
            )
            for index, side in enumerate(("left", "right"))
        ]

    def _load_inspire_q(self) -> None:
        if self._native_q is not None:
            return
        rl_q = self.sequence.manifest.get("rl_q")
        if not isinstance(rl_q, dict) or not rl_q.get("tensor"):
            raise FileNotFoundError("Inspire-RL sequence manifest does not declare rl_q.tensor")
        tensor_path = Path(str(rl_q["tensor"])).expanduser().resolve()
        tensor = _load_tensor(tensor_path)
        native_q = tensor[:, NATIVE_Q_START:NATIVE_Q_START + NUM_DOFS]
        if native_q.shape != (self.sequence.frame_count, NUM_DOFS):
            raise ValueError(
                f"Inspire q/geometry frame mismatch: {native_q.shape} vs "
                f"({self.sequence.frame_count},{NUM_DOFS})"
            )
        self._native_q = np.asarray(native_q, dtype=np.float32)

    def _inspire_parts(self, frame: int) -> List[MeshPart]:
        self._load_inspire_q()
        assert self._native_q is not None
        return self.inspire_library.parts(self._native_q[int(frame)])

    def hand_parts(self, frame: int) -> List[MeshPart]:
        if self.sequence.record.variant == "mano":
            return self._mano_parts(frame)
        if self.sequence.record.variant == "mano_bilateral_raw":
            return self._raw_bilateral_mano_parts(frame)
        if self.sequence.record.variant == "inspire_rl":
            return self._inspire_parts(frame)
        raise ValueError(f"unsupported hand mesh variant: {self.sequence.record.variant!r}")


def _pose_components(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 pose to Viser's wxyz quaternion and position."""

    from scipy.spatial.transform import Rotation

    matrix = np.asarray(pose, dtype=np.float64)
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    return wxyz, np.asarray(matrix[:3, 3], dtype=np.float32)


def _world_vertices(parts: Sequence[MeshPart]) -> np.ndarray:
    values = [part.vertices @ part.pose[:3, :3].T + part.pose[:3, 3] for part in parts]
    return np.concatenate(values, axis=0) if values else np.empty((0, 3), dtype=np.float32)


def _bbox_center(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (points.min(axis=0) + points.max(axis=0)) * 0.5


def _future_frame(frame: int, delta: int, frame_count: int) -> int:
    """Return ``frame + delta``, clamped to the final cache frame."""

    frame = int(frame)
    delta = int(delta)
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame < 0 or frame >= frame_count:
        raise ValueError(f"frame {frame} is outside [0, {frame_count - 1}]")
    if delta < 0:
        raise ValueError("future delta must be non-negative")
    return min(frame + delta, frame_count - 1)


def _flow_summary(current: np.ndarray, future: np.ndarray) -> Dict[str, float]:
    """Summarize correspondence-preserving point displacement in millimetres."""

    current = np.asarray(current, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if current.shape != future.shape or current.ndim != 2 or current.shape[1] != 3:
        raise ValueError(f"flow endpoints must share [N,3], got {current.shape} and {future.shape}")
    values = np.linalg.norm(future - current, axis=1) * 1000.0
    return {
        "median_mm": float(np.median(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(values.max(initial=0.0)),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_state() -> Tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    return commit, dirty


def _prepare_run(
    args: argparse.Namespace,
    index: Dict[str, Any],
    record: SequenceRecord,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if args.output is None:
        return None, None
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_state()
    config = {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    _write_json(output / "config.json", config)
    manifest: Dict[str, Any] = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "grab_stride_flow_visualization",
        "run_id": output.name,
        "run_status": "RUNNING",
        "started_at": _now(),
        "work_version": args.work_version,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": "config.json",
        "metadata_snapshot": str(TrainingSequence(record).metadata_path.resolve()),
        "seed": None,
        "checkpoint": None,
        "inputs": {
            "index": str(args.index.expanduser().resolve()),
            "index_schema": index.get("schema_name"),
            "initial_sequence": record.sequence_id,
            "sequence_path": str(record.path.resolve()),
        },
        "parameters": {
            "future_delta_cache_frames": int(args.future_delta),
            "effective_fps": 30.0,
            "visualization_only": True,
        },
        "viewer": f"http://{args.host}:{args.port}",
        "output": str(output),
        "log": "viewer.log",
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(output / "run_manifest.json", manifest)
    (output / "viewer.log").write_text("", encoding="utf-8")
    return output, manifest


def _mesh_check(
    label: str, loader: Any, reference: np.ndarray
) -> Dict[str, Any]:
    try:
        parts = loader()
        vertices = _world_vertices(parts)
        return {
            "available": True,
            "parts": len(parts),
            "vertices": int(sum(len(part.vertices) for part in parts)),
            "faces": int(sum(len(part.faces) for part in parts)),
            "bbox_center_error_mm": float(np.linalg.norm(_bbox_center(vertices) - _bbox_center(reference)) * 1000.0),
        }
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"available": False, "label": label, "error": str(exc)}


def run_check(args: argparse.Namespace) -> None:
    index, all_records = discover_sequences(args.index)
    records = _filter_records(all_records, None if args.split == "all" else args.split)
    record = choose_sequence(records, args.sequence, args.sequence_index)
    sequence = TrainingSequence(record)
    frame = int(args.frame) % sequence.frame_count
    future_delta = int(args.future_delta)
    future_frame = _future_frame(frame, future_delta, sequence.frame_count)
    object_flow = _flow_summary(sequence.object_points(frame), sequence.object_points(future_frame))
    hand_flow = _flow_summary(sequence.hand_points(frame), sequence.hand_points(future_frame))
    distance_cache = DistanceCache(sequence)
    distances = distance_cache.get(frame)
    knn_counts = {
        str(k): int(distance_cache.knn_hand_mask(frame, k).sum())
        for k in (1, 4, 8, 16, 32, 64)
    }
    provider = TrainingMeshProvider(
        sequence,
        ObjectMeshLibrary(args.object_mesh_root),
        InspireMeshLibrary(args.inspire_urdf),
    )
    result = {
        "index": str(Path(args.index).expanduser().resolve()),
        "index_schema": index.get("schema_name"),
        "sequence": record.sequence_id,
        "split": record.split,
        "source": record.source,
        "variant": record.variant,
        "world_frame": sequence.world_frame,
        "model_coordinate_frame": sequence.manifest["coordinate_frame"],
        "frames": sequence.frame_count,
        "object_pool_points": int(sequence.object_points_all.shape[1]),
        "model_runtime_object_points": int(index.get("model_object_points", MODEL_RUNTIME_OBJECT_POINTS)),
        "hand_side": sequence.manifest["hand_side"],
        "hand_points": int(sequence.hand_points_all.shape[1]),
        "context_points": int(
            sequence.context_points_all.shape[1]
            if sequence.context_points_all is not None else 0
        ),
        "frame": frame,
        "source_frame": sequence.source_frame(frame),
        "future_delta_cache_frames": future_delta,
        "future_visible": future_delta > 0,
        "future_frame": future_frame if future_delta > 0 else None,
        "future_source_frame": sequence.source_frame(future_frame) if future_delta > 0 else None,
        "future_source_frame_delta": (
            sequence.source_frame(future_frame) - sequence.source_frame(frame)
            if future_delta > 0 else 0
        ),
        "future_delta_seconds": (
            (future_frame - frame) / sequence.effective_fps if future_delta > 0 else 0.0
        ),
        "future_clamped": bool(future_delta > 0 and frame + future_delta >= sequence.frame_count),
        "object_point_flow": object_flow,
        "hand_point_flow": hand_flow,
        "candidate_active_5cm": sequence.active_5cm(frame),
        "nearest_distance_mm": float(distances.min() * 1000.0),
        "knn_hand_counts_object_to_hand": knn_counts,
        "object_mesh": _mesh_check(
            "object", lambda: provider.object_parts(frame), sequence.object_points(frame)
        ),
        "hand_mesh": _mesh_check(
            "hand", lambda: provider.hand_parts(frame), sequence.hand_points(frame)
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_server(args: argparse.Namespace) -> None:
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is required to start the interactive viewer") from exc

    index, all_records = discover_sequences(args.index)
    initial_split = None if args.split == "all" else args.split
    initial_records = _filter_records(all_records, initial_split)
    initial_record = choose_sequence(initial_records, args.sequence, args.sequence_index)
    initial_sequence = TrainingSequence(initial_record)
    output, run_manifest = _prepare_run(args, index, initial_record)

    def emit(message: str) -> None:
        line = f"{_now()} {message}"
        if output is not None:
            with (output / "viewer.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        print(line, flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    object_library = ObjectMeshLibrary(args.object_mesh_root)
    inspire_library = InspireMeshLibrary(args.inspire_urdf)
    state: Dict[str, Any] = {
        "records": initial_records,
        "record": initial_record,
        "sequence": initial_sequence,
        "distance": None,
        "mesh_provider": None,
        "playing": False,
        "suppress": False,
        "point_handles": {},
        "mesh_handles": {"object": [], "hand": [], "future_object": [], "future_hand": []},
        "mesh_keys": {"object": (), "hand": (), "future_object": (), "future_hand": ()},
        "mesh_errors": {
            "object": None,
            "hand": None,
            "future_object": None,
            "future_hand": None,
        },
    }
    state["distance"] = DistanceCache(state["sequence"])
    state["mesh_provider"] = TrainingMeshProvider(
        state["sequence"], object_library, inspire_library
    )
    lock = threading.RLock()

    with server.gui.add_folder("数据与轨迹"):
        split_gui = server.gui.add_dropdown(
            "数据 split", options=tuple(SPLIT_LABELS), initial_value=SPLIT_FROM_CLI[args.split]
        )
        trajectory_gui = server.gui.add_dropdown(
            "轨迹", options=tuple(record.label for record in initial_records), initial_value=initial_record.label
        )
        previous_trajectory_gui = server.gui.add_button("上一条轨迹")
        next_trajectory_gui = server.gui.add_button("下一条轨迹")

    with server.gui.add_folder("播放"):
        frame_gui = server.gui.add_slider(
            "帧", min=0, max=state["sequence"].frame_count - 1, step=1, initial_value=int(args.frame) % state["sequence"].frame_count
        )
        future_delta_gui = server.gui.add_slider(
            "未来 Δ（cache 帧）",
            min=0,
            max=FUTURE_DELTA_MAX,
            step=1,
            initial_value=int(args.future_delta),
        )
        previous_frame_gui = server.gui.add_button("上一帧")
        next_frame_gui = server.gui.add_button("下一帧")
        play_gui = server.gui.add_button("播放 / 暂停")
        fps_gui = server.gui.add_slider("播放 FPS", min=1.0, max=60.0, step=1.0, initial_value=float(args.fps))

    with server.gui.add_folder("距离着色"):
        threshold_gui = server.gui.add_dropdown(
            "累计距离阈值", options=THRESHOLD_LABELS, initial_value="5 cm"
        )
        knn_gui = server.gui.add_dropdown(
            "物体→手 KNN", options=KNN_LABELS, initial_value="关闭"
        )
        hand_size_gui = server.gui.add_slider(
            "手点大小", min=0.0001, max=0.02, step=0.0001, initial_value=float(args.point_size)
        )
        object_size_gui = server.gui.add_slider(
            "物体点大小", min=0.0001, max=0.02, step=0.0001, initial_value=float(args.point_size)
        )

    with server.gui.add_folder("点云与 Mesh 显示"):
        point_display_gui = server.gui.add_dropdown(
            "点云显示",
            options=POINT_DISPLAY_LABELS,
            initial_value=POINT_DISPLAY_FROM_CLI[args.point_display],
        )
        context_gui = server.gui.add_checkbox(
            "场景其他对象（灰色）",
            initial_value=initial_sequence.context_points_all is not None,
        )
        mesh_display_gui = server.gui.add_dropdown(
            "Mesh 显示", options=MESH_DISPLAY_LABELS, initial_value=MESH_DISPLAY_FROM_CLI[args.mesh_display]
        )
        mesh_opacity_gui = server.gui.add_slider(
            "Mesh 透明度", min=0.05, max=1.0, step=0.05, initial_value=float(args.mesh_opacity)
        )

    status_gui = server.gui.add_markdown("")
    help_gui = server.gui.add_markdown(
        "全部操作均使用鼠标。点云直接来自索引序列的训练几何；界面显示完整 4096 点物体池，"
        "训练运行时从中确定性采样 1024 点。`物体→手 KNN` 会把每个物体点的最近 n 个手点取并集后着色，"
        "不绘制连线；KNN 黄色高亮会覆盖同一点的距离阈值红色。点云使用 float32 圆点平面着色，"
        f"点大小最低可调到 0.1 mm。未来 Δ 为 0 时隐藏未来帧，1–{FUTURE_DELTA_MAX} 时叠加 "
        "t+Δ（末尾夹到最后一帧）；"
        "未来物体为绿色、未来手为紫色。点云和 Mesh 可独立切换，Mesh 按需加载。"
    )
    del help_gui

    def clear_mesh_group(group: str) -> None:
        for handle in state["mesh_handles"][group]:
            handle.remove()
        state["mesh_handles"][group] = []
        state["mesh_keys"][group] = ()

    def update_mesh_group(group: str, parts: Sequence[MeshPart], visible: bool) -> None:
        keys = tuple(part.key for part in parts)
        if keys != state["mesh_keys"][group]:
            clear_mesh_group(group)
            color = OBJECT_MESH_COLOR
            if group == "hand":
                color = MANO_MESH_COLOR if _is_mano_variant(state["record"].variant) else INSPIRE_MESH_COLOR
            elif group == "future_object":
                color = FUTURE_OBJECT_MESH_COLOR
            elif group == "future_hand":
                color = FUTURE_HAND_MESH_COLOR
            handles = []
            for index_part, part in enumerate(parts):
                wxyz, position = _pose_components(part.pose)
                handles.append(
                    server.scene.add_mesh_simple(
                        f"/meshes/{group}/{index_part:02d}_{part.key}",
                        vertices=part.vertices,
                        faces=part.faces,
                        color=color,
                        opacity=float(mesh_opacity_gui.value),
                        wxyz=wxyz,
                        position=position,
                    )
                )
            state["mesh_handles"][group] = handles
            state["mesh_keys"][group] = keys
        for handle, part in zip(state["mesh_handles"][group], parts):
            if part.dynamic_vertices:
                handle.vertices = part.vertices
            wxyz, position = _pose_components(part.pose)
            handle.wxyz = wxyz
            handle.position = position
            handle.opacity = float(mesh_opacity_gui.value)
            handle.visible = bool(visible)

    def render() -> None:
        with lock:
            sequence: TrainingSequence = state["sequence"]
            frame = int(frame_gui.value) % sequence.frame_count
            future_delta = int(future_delta_gui.value)
            future_visible = future_delta > 0
            future_frame = _future_frame(frame, future_delta, sequence.frame_count)
            object_points = sequence.object_points(frame)
            hand_points = sequence.hand_points(frame)
            context_points = sequence.context_points(frame)
            future_object_points = sequence.object_points(future_frame)
            future_hand_points = sequence.hand_points(future_frame)
            distances = state["distance"].get(frame)
            threshold = THRESHOLD_METERS[str(threshold_gui.value)]
            hand_color = MANO_COLOR if _is_mano_variant(state["record"].variant) else INSPIRE_COLOR
            hand_colors = np.broadcast_to(hand_color, (len(hand_points), 3)).copy()
            distance_highlighted = (
                distances <= threshold if threshold > 0.0 else np.zeros(len(distances), dtype=bool)
            )
            knn_k = KNN_VALUES[str(knn_gui.value)]
            knn_highlighted = (
                state["distance"].knn_hand_mask(frame, int(knn_k))
                if knn_k is not None
                else np.zeros(len(hand_points), dtype=bool)
            )
            hand_colors[distance_highlighted] = HIGHLIGHT_COLOR
            hand_colors[knn_highlighted] = KNN_HIGHLIGHT_COLOR
            # Keep the legacy status field for the distance threshold; KNN
            # count is reported separately below.
            highlighted = distance_highlighted
            object_colors = np.broadcast_to(OBJECT_COLOR, (len(object_points), 3)).copy()
            future_object_colors = np.broadcast_to(
                FUTURE_OBJECT_COLOR, (len(future_object_points), 3)
            ).copy()
            future_hand_colors = np.broadcast_to(
                FUTURE_HAND_COLOR, (len(future_hand_points), 3)
            ).copy()
            context_render_points = (
                context_points if len(context_points) else np.zeros((1, 3), dtype=np.float32)
            )
            context_colors = np.broadcast_to(
                CONTEXT_OBJECT_COLOR, (len(context_render_points), 3)
            ).copy()
            point_display = str(point_display_gui.value)
            show_object_points = point_display in ("仅物体点", "物体点 + 手点")
            show_hand_points = point_display in ("仅手点", "物体点 + 手点")

            if not state["point_handles"]:
                state["point_handles"]["object"] = server.scene.add_point_cloud(
                    "/points/object",
                    points=object_points,
                    colors=object_colors,
                    point_size=float(object_size_gui.value),
                    point_shape="circle",
                    precision="float32",
                    point_shading="flat",
                )
                state["point_handles"]["hand"] = server.scene.add_point_cloud(
                    "/points/hand_right",
                    points=hand_points,
                    colors=hand_colors,
                    point_size=float(hand_size_gui.value),
                    point_shape="circle",
                    precision="float32",
                    point_shading="flat",
                )
                state["point_handles"]["context"] = server.scene.add_point_cloud(
                    "/points/context_objects",
                    points=context_render_points,
                    colors=context_colors,
                    point_size=float(object_size_gui.value),
                    point_shape="circle",
                    precision="float32",
                    point_shading="flat",
                )
                state["point_handles"]["future_object"] = server.scene.add_point_cloud(
                    "/points/future/object",
                    points=future_object_points,
                    colors=future_object_colors,
                    point_size=float(object_size_gui.value),
                    point_shape="circle",
                    precision="float32",
                    point_shading="flat",
                )
                state["point_handles"]["future_hand"] = server.scene.add_point_cloud(
                    "/points/future/hand_right",
                    points=future_hand_points,
                    colors=future_hand_colors,
                    point_size=float(hand_size_gui.value),
                    point_shape="circle",
                    precision="float32",
                    point_shading="flat",
                )
            else:
                object_handle = state["point_handles"]["object"]
                object_handle.points = object_points
                object_handle.colors = object_colors
                object_handle.point_size = float(object_size_gui.value)
                hand_handle = state["point_handles"]["hand"]
                hand_handle.points = hand_points
                hand_handle.colors = hand_colors
                hand_handle.point_size = float(hand_size_gui.value)
                context_handle = state["point_handles"]["context"]
                context_handle.points = context_render_points
                context_handle.colors = context_colors
                context_handle.point_size = float(object_size_gui.value)
                future_object_handle = state["point_handles"]["future_object"]
                future_object_handle.points = future_object_points
                future_object_handle.colors = future_object_colors
                future_object_handle.point_size = float(object_size_gui.value)
                future_hand_handle = state["point_handles"]["future_hand"]
                future_hand_handle.points = future_hand_points
                future_hand_handle.colors = future_hand_colors
                future_hand_handle.point_size = float(hand_size_gui.value)
            state["point_handles"]["object"].visible = show_object_points
            state["point_handles"]["hand"].visible = show_hand_points
            state["point_handles"]["context"].visible = bool(context_gui.value) and bool(
                len(context_points)
            )
            state["point_handles"]["future_object"].visible = future_visible and show_object_points
            state["point_handles"]["future_hand"].visible = future_visible and show_hand_points

            display = str(mesh_display_gui.value)
            requested = {
                "object": display in ("仅物体 mesh", "物体 + 手 mesh"),
                "hand": display in ("仅手 mesh", "物体 + 手 mesh"),
                "future_object": future_visible
                and display in ("仅物体 mesh", "物体 + 手 mesh"),
                "future_hand": future_visible
                and display in ("仅手 mesh", "物体 + 手 mesh"),
            }
            provider: TrainingMeshProvider = state["mesh_provider"]
            for group, loader, target_frame in (
                ("object", provider.object_parts, frame),
                ("hand", provider.hand_parts, frame),
                ("future_object", provider.object_parts, future_frame),
                ("future_hand", provider.hand_parts, future_frame),
            ):
                if not requested[group]:
                    for handle in state["mesh_handles"][group]:
                        handle.visible = False
                    state["mesh_errors"][group] = None
                    continue
                try:
                    parts = loader(target_frame)
                    update_mesh_group(group, parts, True)
                    state["mesh_errors"][group] = None
                except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    for handle in state["mesh_handles"][group]:
                        handle.visible = False
                    state["mesh_errors"][group] = str(exc)

            record: SequenceRecord = state["record"]
            variant_label = VARIANT_LABELS.get(record.variant, record.variant)
            mesh_status = []
            for group, label in (
                ("object", "当前物体"),
                ("hand", "当前手"),
                ("future_object", "未来物体"),
                ("future_hand", "未来手"),
            ):
                if state["mesh_errors"][group]:
                    mesh_status.append(f"{label} mesh 不可用：{state['mesh_errors'][group]}")
            mesh_line = "；".join(mesh_status) if mesh_status else "mesh 状态正常"
            if future_visible:
                clamp_label = "（已夹到末帧）" if frame + future_delta >= sequence.frame_count else ""
                object_flow = _flow_summary(object_points, future_object_points)
                hand_flow = _flow_summary(hand_points, future_hand_points)
                future_line = (
                    f"**未来 Δ** `{future_delta}` cache 帧  |  "
                    f"**未来帧** `{future_frame + 1}/{sequence.frame_count}`  |  "
                    f"**未来原始帧** `{sequence.source_frame(future_frame)}`  |  "
                    f"**原始帧差** `{sequence.source_frame(future_frame) - sequence.source_frame(frame)}`  |  "
                    f"**时间跨度** `{(future_frame - frame) / sequence.effective_fps:.3f} s` {clamp_label}  \n"
                    f"**物体点流 mm（median/P95/max）** "
                    f"`{object_flow['median_mm']:.2f}/{object_flow['p95_mm']:.2f}/{object_flow['max_mm']:.2f}`  |  "
                    f"**手点流 mm（median/P95/max）** "
                    f"`{hand_flow['median_mm']:.2f}/{hand_flow['p95_mm']:.2f}/{hand_flow['max_mm']:.2f}`"
                )
            else:
                future_line = "**未来 Δ** `0`（不显示未来）"
            status_gui.content = (
                f"**轨迹** `{record.sequence_id}`  |  **split** `{record.split}`  |  "
                f"**source** `{record.source}`  |  **variant** `{variant_label}`  \n"
                f"**帧** `{frame + 1}/{sequence.frame_count}`  |  **原始帧** `{sequence.source_frame(frame)}`  |  "
                f"**5 cm candidate** `{sequence.active_5cm(frame)}`  \n"
                f"**最近手物距离** `{float(distances.min()) * 1000.0:.3f} mm`  |  "
                f"**阈值内手点** `{int(highlighted.sum())}/{len(highlighted)}`  \n"
                f"**坐标系** `{sequence.world_frame}`  |  **{mesh_line}**"
            )
            status_gui.content += (
                "  " + chr(10)
                + future_line
                + "  " + chr(10)
                + f"**KNN（物体→手）** `{('关闭' if knn_k is None else 'n=' + str(knn_k))}`  |  "
                + f"**KNN 着色手点** `{int(knn_highlighted.sum())}/{len(knn_highlighted)}`（黄色）"
                + "  " + chr(10)
                + f"**点云显示** `{point_display}`  |  **Mesh 显示** `{display}`"
                + "  |  "
                + f"**场景上下文点** `{len(context_points)}`"
            )

    def load_record(record: SequenceRecord, *, reset_frame: bool = True) -> None:
        with lock:
            sequence = TrainingSequence(record)
            state["record"] = record
            state["sequence"] = sequence
            state["distance"] = DistanceCache(sequence)
            state["mesh_provider"] = TrainingMeshProvider(sequence, object_library, inspire_library)
            state["mesh_errors"] = {
                "object": None,
                "hand": None,
                "future_object": None,
                "future_hand": None,
            }
            for group in ("object", "hand", "future_object", "future_hand"):
                clear_mesh_group(group)
            state["suppress"] = True
            try:
                trajectory_gui.value = record.label
                frame_gui.max = sequence.frame_count - 1
                frame_gui.value = 0 if reset_frame else min(int(frame_gui.value), sequence.frame_count - 1)
            finally:
                state["suppress"] = False
            render()

    def shift_trajectory(delta: int) -> None:
        with lock:
            records: List[SequenceRecord] = state["records"]
            index_record = records.index(state["record"])
            load_record(records[(index_record + delta) % len(records)])

    @split_gui.on_update
    def _(_: Any) -> None:
        if state["suppress"]:
            return
        with lock:
            selected = _filter_records(all_records, SPLIT_LABELS[str(split_gui.value)])
            state["records"] = selected
            state["suppress"] = True
            try:
                trajectory_gui.options = tuple(record.label for record in selected)
            finally:
                state["suppress"] = False
            load_record(selected[0])

    @trajectory_gui.on_update
    def _(_: Any) -> None:
        if state["suppress"]:
            return
        match = next(record for record in state["records"] if record.label == trajectory_gui.value)
        load_record(match)

    @previous_trajectory_gui.on_click
    def _(_: Any) -> None:
        shift_trajectory(-1)

    @next_trajectory_gui.on_click
    def _(_: Any) -> None:
        shift_trajectory(1)

    @frame_gui.on_update
    def _(_: Any) -> None:
        if not state["suppress"]:
            render()

    @previous_frame_gui.on_click
    def _(_: Any) -> None:
        sequence: TrainingSequence = state["sequence"]
        frame_gui.value = (int(frame_gui.value) - 1) % sequence.frame_count

    @next_frame_gui.on_click
    def _(_: Any) -> None:
        sequence: TrainingSequence = state["sequence"]
        frame_gui.value = (int(frame_gui.value) + 1) % sequence.frame_count

    @play_gui.on_click
    def _(_: Any) -> None:
        state["playing"] = not bool(state["playing"])
        render()

    for gui in (
        threshold_gui,
        knn_gui,
        future_delta_gui,
        point_display_gui,
        context_gui,
        hand_size_gui,
        object_size_gui,
        mesh_display_gui,
        mesh_opacity_gui,
    ):
        gui.on_update(lambda _: render())

    render()
    emit(json.dumps(
        {
            "viewer": f"http://{args.host}:{args.port}",
            "index": str(Path(args.index).expanduser().resolve()),
            "index_schema": index.get("schema_name"),
            "sequence_count": len(all_records),
            "initial_sequence": initial_record.sequence_id,
            "initial_variant": initial_record.variant,
            "future_delta_cache_frames": int(args.future_delta),
        },
        ensure_ascii=False,
    ))

    def stop_handler(_: int, __: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    last_tick = time.monotonic()
    try:
        while True:
            time.sleep(0.01)
            if not state["playing"]:
                last_tick = time.monotonic()
                continue
            interval = 1.0 / max(float(fps_gui.value), 1.0)
            now = time.monotonic()
            if now - last_tick < interval:
                continue
            last_tick = now
            with lock:
                sequence = state["sequence"]
                state["suppress"] = True
                try:
                    frame_gui.value = (int(frame_gui.value) + 1) % sequence.frame_count
                finally:
                    state["suppress"] = False
                render()
    except KeyboardInterrupt:
        if output is not None and run_manifest is not None:
            run_manifest["run_status"] = "STOPPED"
            run_manifest["completed_at"] = _now()
            _write_json(output / "run_manifest.json", run_manifest)
            emit("viewer stopped")
        return
    except Exception:
        if output is not None and run_manifest is not None:
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            run_manifest["run_status"] = "FAILED"
            run_manifest["completed_at"] = _now()
            run_manifest["error"] = "error.txt"
            run_manifest["conclusion"] = "INVALID_IMPLEMENTATION"
            _write_json(output / "run_manifest.json", run_manifest)
            emit("viewer failed; see error.txt")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json"),
        help="Recent ObjectInteractionCm training index.",
    )
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--sequence", help="Exact sequence id from the selected split.")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--future-delta",
        type=int,
        choices=range(FUTURE_DELTA_MAX + 1),
        default=0,
        help=f"Overlay t+delta for 1..{FUTURE_DELTA_MAX} cache frames; 0 hides the future overlay.",
    )
    parser.add_argument(
        "--object-mesh-root",
        type=Path,
        default=Path("data/raw_data/GRAB/tools/object_meshes/contact_meshes"),
        help="Canonical GRAB object mesh directory; loaded read-only and on demand.",
    )
    parser.add_argument(
        "--inspire-urdf",
        type=Path,
        default=Path("src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"),
        help="The same Inspire visual URDF used by the Dexplore-RL cache builder.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--point-display", choices=tuple(POINT_DISPLAY_FROM_CLI), default="both")
    parser.add_argument("--mesh-display", choices=tuple(MESH_DISPLAY_FROM_CLI), default="off")
    parser.add_argument("--mesh-opacity", type=float, default=0.45)
    parser.add_argument(
        "--output", type=Path,
        help="Optional independent diagnostic run directory for config, manifest, and viewer log.",
    )
    parser.add_argument(
        "--work-version", default=MODIFICATION_VERSION,
        help="Work version recorded in this visualization run's manifest.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_only:
        run_check(args)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
