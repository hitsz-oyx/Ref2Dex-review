from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.task.correspondence_ptv3.contracts import EdgeSample, Stage3Frame
from src.task.correspondence_ptv3.data.runtime import (
    AugmentedGeometry,
    augment_geometry,
    compute_runtime_context_neighbors,
    sample_object_indices,
    stable_frame_seed,
)
from src.task.correspondence_ptv3.data.stage3 import Stage3Store
from src.utils.correspondence import soft_contact_label


@dataclass(frozen=True)
class ObjectPointSample:
    selected_idx: np.ndarray
    safe_idx: np.ndarray
    valid_mask: np.ndarray
    obj_points: np.ndarray
    obj_normals: np.ndarray
    obj_point_id: np.ndarray
    obj_min_dist: np.ndarray
    clean_knn_idx: np.ndarray


class CorrStaticDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        k_cross: int = 32,
        k_ctx: int = 32,
        ctx_radius: float = 0.04,
        base_seed: int = 42,
        augment: bool = True,
        apply_hand_perturb: bool = True,
        augment_rotation: bool = True,
        augment_translation: bool = False,
        augment_scale: bool = False,
        rotation_range: float = 180.0,
        translation_range: float = 0.1,
        scale_range: tuple[float, float] = (0.9, 1.1),
        d_pos: float = 0.005,
        d_neg: float = 0.03,
        gamma: float = 2.0,
        hand_rot_std_deg: float = 10.0,
        hand_trans_std: float = 0.01,
        hand_perturb_prob: float = 1.0,
        fix_overfit_seed: bool = False,
        blacklist_path: str | None = None,
        edge_sampler: object | None = None,
    ) -> None:
        super().__init__()
        if edge_sampler is None:
            raise ValueError("CorrStaticDataset requires an instantiated edge_sampler.")
        if int(k_ctx) <= 0:
            raise ValueError("k_ctx must be positive.")

        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.k_gt = int(k_cross)
        self.k_ctx = int(k_ctx)
        self.ctx_radius = float(ctx_radius)
        self.base_seed = int(base_seed)
        self.augment = bool(augment)
        self.apply_hand_perturb = bool(apply_hand_perturb)
        self.augment_rotation = bool(augment_rotation)
        self.augment_translation = bool(augment_translation)
        self.augment_scale = bool(augment_scale)
        self.rotation_range = float(rotation_range)
        self.translation_range = float(translation_range)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.d_pos = float(d_pos)
        self.d_neg = float(d_neg)
        self.gamma = float(gamma)
        self.hand_rot_std_deg = float(hand_rot_std_deg)
        self.hand_trans_std = float(hand_trans_std)
        self.hand_perturb_prob = float(hand_perturb_prob)
        self.fix_overfit_seed = bool(fix_overfit_seed)
        self.edge_sampler = edge_sampler

        self._epoch = mp.Value("q", 0, lock=True)
        self.frame_store = Stage3Store(
            data_path,
            file_list=file_list,
            blacklist_path=blacklist_path,
        )
        self.file_paths = self.frame_store.file_paths
        self.file_sample_ranges = self.frame_store.file_sample_ranges
        self.data_root = self.frame_store.data_root

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.frame_store)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.frame_store.load_frame(index)
        obj_sample = self._sample_object_points(frame)
        geometry = self._build_geometry(frame, obj_sample)
        context_edges = self._build_context_edges(geometry, obj_sample.valid_mask)
        supervision_edges = self.edge_sampler.sample(
            gt_obj_points=geometry.gt_obj_points,
            gt_hand_points=geometry.gt_hand_points,
            obj_valid=obj_sample.valid_mask,
            seed=self._sample_seed(frame, namespace="logit-neighbors"),
        )
        return self._pack_sample(
            frame=frame,
            obj_sample=obj_sample,
            geometry=geometry,
            context_edges=context_edges,
            supervision_edges=supervision_edges,
        )

    def _sample_object_points(self, frame: Stage3Frame) -> ObjectPointSample:
        selected_idx, valid_mask = sample_object_indices(
            frame.obj_candidate_mask_5cm,
            num_samples=self.num_obj_points,
            seed=self._sample_seed(frame, namespace="object-sampling"),
        )
        safe_idx = np.maximum(selected_idx, 0)
        obj_points = np.asarray(frame.obj_points[safe_idx], dtype=np.float32).copy()
        obj_normals = np.asarray(frame.obj_normals[safe_idx], dtype=np.float32).copy()
        obj_point_id = np.asarray(frame.obj_point_id[safe_idx], dtype=np.int64).copy()
        obj_min_dist = np.asarray(frame.obj_to_hand_min_dist[safe_idx], dtype=np.float32).copy()
        clean_knn_idx = np.asarray(frame.gt_obj_to_hand_knn_idx[safe_idx], dtype=np.int64).copy()

        obj_points[~valid_mask] = 0
        obj_normals[~valid_mask] = 0
        obj_point_id[~valid_mask] = -1
        obj_min_dist[~valid_mask] = 0
        clean_knn_idx[~valid_mask] = -1
        return ObjectPointSample(
            selected_idx=selected_idx,
            safe_idx=safe_idx,
            valid_mask=valid_mask,
            obj_points=obj_points,
            obj_normals=obj_normals,
            obj_point_id=obj_point_id,
            obj_min_dist=obj_min_dist,
            clean_knn_idx=clean_knn_idx,
        )

    def _build_geometry(
        self,
        frame: Stage3Frame,
        obj_sample: ObjectPointSample,
    ) -> AugmentedGeometry:
        return augment_geometry(
            obj_points=obj_sample.obj_points,
            obj_normals=obj_sample.obj_normals,
            hand_points=frame.hand_points,
            hand_normals=frame.hand_normals,
            seed=self._sample_seed(frame, namespace="augmentation"),
            apply_hand_perturb=self.apply_hand_perturb,
            hand_rot_std_deg=self.hand_rot_std_deg,
            hand_trans_std=self.hand_trans_std,
            hand_perturb_prob=self.hand_perturb_prob,
            apply_global_aug=self.augment,
            augment_rotation=self.augment_rotation,
            rotation_range_deg=self.rotation_range,
            augment_translation=self.augment_translation,
            translation_range=self.translation_range,
            augment_scale=self.augment_scale,
            scale_range=self.scale_range,
        )

    def _build_context_edges(
        self,
        geometry: AugmentedGeometry,
        obj_valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_runtime_context_neighbors(
            geometry.input_obj_points,
            geometry.input_hand_points,
            obj_valid_mask,
            k_ctx=self.k_ctx,
            ctx_radius=self.ctx_radius,
        )

    def _pack_sample(
        self,
        *,
        frame: Stage3Frame,
        obj_sample: ObjectPointSample,
        geometry: AugmentedGeometry,
        context_edges: tuple[np.ndarray, np.ndarray],
        supervision_edges: EdgeSample,
    ) -> dict[str, torch.Tensor]:
        input_ctx_idx, input_ctx_valid = context_edges
        obj_min_dist = obj_sample.obj_min_dist.copy()
        obj_min_dist *= float(geometry.distance_scale)

        input_points = np.concatenate(
            [geometry.input_obj_points, geometry.input_hand_points],
            axis=0,
        )
        input_normals = np.concatenate(
            [geometry.input_obj_normals, geometry.input_hand_normals],
            axis=0,
        )
        gt_points = np.concatenate(
            [geometry.gt_obj_points, geometry.gt_hand_points],
            axis=0,
        )
        gt_normals = np.concatenate(
            [geometry.gt_obj_normals, geometry.gt_hand_normals],
            axis=0,
        )
        point_valid_mask = np.concatenate(
            [obj_sample.valid_mask, np.ones((frame.hand_points.shape[0],), dtype=bool)],
            axis=0,
        )
        contact_label = soft_contact_label(
            torch.from_numpy(obj_min_dist),
            d_pos=self.d_pos,
            d_neg=self.d_neg,
            gamma=self.gamma,
        )
        return {
            "points": torch.from_numpy(input_points).float(),
            "normals": torch.from_numpy(input_normals).float(),
            "gt_points": torch.from_numpy(gt_points).float(),
            "gt_normals": torch.from_numpy(gt_normals).float(),
            "point_valid_mask": torch.from_numpy(point_valid_mask),
            "runtime_obj_valid_mask": torch.from_numpy(obj_sample.valid_mask),
            "selected_obj_idx": torch.from_numpy(obj_sample.selected_idx).long(),
            "selected_obj_point_id": torch.from_numpy(obj_sample.obj_point_id).long(),
            "selected_obj_min_dist": torch.from_numpy(obj_min_dist).float(),
            "obj_contact_label": contact_label.float(),
            "gt_obj_to_hand_knn_idx": torch.from_numpy(obj_sample.clean_knn_idx).long(),
            "input_obj_to_hand_ctx_idx": torch.from_numpy(input_ctx_idx).long(),
            "input_obj_to_hand_ctx_valid_mask": torch.from_numpy(input_ctx_valid),
            "input_obj_to_hand_logit_idx": torch.from_numpy(supervision_edges.idx).long(),
            "input_obj_to_hand_logit_valid_mask": torch.from_numpy(supervision_edges.valid_mask),
            "input_obj_to_hand_logit_loss_weight": torch.from_numpy(supervision_edges.loss_weight).float(),
            "hand_cano_points": torch.from_numpy(frame.hand_cano_points).float(),
            "hand_finger_id": torch.from_numpy(frame.hand_finger_id).long(),
            "hand_region_id": torch.from_numpy(frame.hand_region_id).long(),
            "num_obj_points": torch.tensor(self.num_obj_points, dtype=torch.long),
            "num_hand_points": torch.tensor(frame.hand_points.shape[0], dtype=torch.long),
        }

    def _sample_seed(self, frame: Stage3Frame, *, namespace: str) -> int:
        epoch = 0 if self.fix_overfit_seed else self.epoch
        return stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=frame.seq_id,
            side=frame.side,
            raw_frame_id=frame.raw_frame_id,
            epoch=epoch,
            namespace=namespace,
        )
