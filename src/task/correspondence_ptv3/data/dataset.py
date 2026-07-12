from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base import make_file_split_dataloaders
from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, shard_sampler_for_distributed
from src.task.correspondence_ptv3.composition import (
    EdgeSamplerConfig,
    ResolvedCorrespondenceComponents,
    resolve_correspondence_components,
    resolve_edge_sampler_config,
    resolve_logit_far_min_radius,
    resolve_logit_near_radius,
)
from src.task.correspondence_ptv3.contracts import EdgeSample, Stage3Frame
from src.task.correspondence_ptv3.data.split import SequenceLocalitySampler, sequence_group_key
from src.task.correspondence_ptv3.data.stage3 import Stage3Store
from src.task.correspondence_ptv3.sampling import (
    AugmentedGeometry,
    BalancedEdgeSampler,
    DenseEdgeSampler,
    StratifiedEdgeSampler,
    augment_geometry,
    build_edge_sampler,
    sample_object_indices,
    stable_frame_seed,
    validate_stratified_edge_sampler_config,
)
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
        k_near_logit: int = 32,
        k_far_logit: int = 32,
        logit_sampling_mode: str = "balanced",
        logit_stratified_distance_edges: tuple[float, ...] = (0.005, 0.015, 0.03, 0.06),
        logit_stratified_quotas: tuple[int, ...] = (16, 32, 32, 32, 16),
        ctx_radius: float = 0.04,
        logit_near_radius: float | None = None,
        logit_far_min_radius: float | None = None,
        logit_pos_radius: float | None = None,
        logit_neg_min_radius: float | None = None,
        logit_neg_radius: float = 0.06,
        logit_far_weight: float = 0.5,
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
        edge_sampler_config: EdgeSamplerConfig | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.k_gt = int(k_cross)
        self.k_ctx = int(k_ctx)
        self.k_near_logit = int(k_near_logit)
        self.k_far_logit = int(k_far_logit)
        self.ctx_radius = float(ctx_radius)

        resolved_logit_near_radius = (
            logit_near_radius if logit_near_radius is not None else logit_pos_radius
        )
        resolved_logit_far_min_radius = (
            logit_far_min_radius
            if logit_far_min_radius is not None
            else logit_neg_min_radius
        )
        if resolved_logit_near_radius is None:
            resolved_logit_near_radius = resolve_logit_near_radius(None)
        if resolved_logit_far_min_radius is None:
            resolved_logit_far_min_radius = resolve_logit_far_min_radius(None)
        self.logit_near_radius = float(resolved_logit_near_radius)
        self.logit_far_min_radius = float(resolved_logit_far_min_radius)
        self.logit_pos_radius = self.logit_near_radius
        self.logit_neg_min_radius = self.logit_far_min_radius
        self.logit_neg_radius = float(logit_neg_radius)
        self.logit_far_weight = float(logit_far_weight)
        self.logit_sampling_mode = str(logit_sampling_mode).lower()
        if self.logit_sampling_mode not in {"balanced", "dense", "stratified"}:
            raise ValueError(
                "logit_sampling_mode must be 'balanced', 'dense', or 'stratified', got "
                f"{self.logit_sampling_mode!r}."
            )
        (
            self.logit_stratified_distance_edges,
            self.logit_stratified_quotas,
        ) = validate_stratified_edge_sampler_config(
            logit_stratified_distance_edges,
            logit_stratified_quotas,
        )
        if self.k_ctx <= 0:
            raise ValueError("k_ctx must be positive.")

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

        self._epoch = mp.Value("q", 0, lock=True)

        self.frame_store = Stage3Store(
            data_path,
            file_list=file_list,
            blacklist_path=blacklist_path,
        )
        self.file_paths = self.frame_store.file_paths
        self.file_sample_ranges = self.frame_store.file_sample_ranges
        self.data_root = self.frame_store.data_root

        if edge_sampler is None:
            sampler_cfg = edge_sampler_config or self._legacy_edge_sampler_config()
            edge_sampler = build_edge_sampler(sampler_cfg)
            self._apply_edge_sampler_config(sampler_cfg)
        self.edge_sampler = edge_sampler

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

    def _legacy_edge_sampler_config(self) -> EdgeSamplerConfig:
        if self.logit_sampling_mode == "balanced":
            return EdgeSamplerConfig(
                name="balanced",
                params={
                    "k_near_logit": self.k_near_logit,
                    "k_far_logit": self.k_far_logit,
                    "logit_near_radius": self.logit_near_radius,
                    "logit_far_min_radius": self.logit_far_min_radius,
                },
            )
        if self.logit_sampling_mode == "stratified":
            return EdgeSamplerConfig(
                name="stratified",
                params={
                    "distance_edges": tuple(float(v) for v in self.logit_stratified_distance_edges.tolist()),
                    "quotas": tuple(int(v) for v in self.logit_stratified_quotas.tolist()),
                    "logit_near_radius": self.logit_near_radius,
                    "logit_far_min_radius": self.logit_far_min_radius,
                },
            )
        if self.logit_sampling_mode == "dense":
            return EdgeSamplerConfig(
                name="dense",
                params={
                    "logit_near_radius": self.logit_near_radius,
                    "logit_far_min_radius": self.logit_far_min_radius,
                },
            )
        raise ValueError(
            "logit_sampling_mode must be 'balanced', 'dense', or 'stratified', got "
            f"{self.logit_sampling_mode!r}."
        )

    def _apply_edge_sampler_config(self, sampler_cfg: EdgeSamplerConfig) -> None:
        self.logit_sampling_mode = sampler_cfg.name
        self.logit_near_radius = float(sampler_cfg.params["logit_near_radius"])
        self.logit_far_min_radius = float(sampler_cfg.params["logit_far_min_radius"])
        self.logit_pos_radius = self.logit_near_radius
        self.logit_neg_min_radius = self.logit_far_min_radius
        if sampler_cfg.name == "balanced":
            self.k_near_logit = int(sampler_cfg.params["k_near_logit"])
            self.k_far_logit = int(sampler_cfg.params["k_far_logit"])
            return
        if sampler_cfg.name == "stratified":
            self.logit_stratified_distance_edges = np.asarray(
                sampler_cfg.params["distance_edges"],
                dtype=np.float32,
            )
            self.logit_stratified_quotas = np.asarray(
                sampler_cfg.params["quotas"],
                dtype=np.int64,
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
        return _compute_runtime_context_neighbors(
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
            "input_obj_to_hand_logit_near_count": torch.from_numpy(supervision_edges.stats["near_count"]).long(),
            "input_obj_to_hand_logit_far_count": torch.from_numpy(supervision_edges.stats["far_count"]).long(),
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


def _compute_runtime_context_neighbors(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_ctx: int,
    ctx_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_obj = obj_points.shape[0]
    ctx_idx = np.full((num_obj, k_ctx), -1, dtype=np.int64)
    ctx_valid = np.zeros((num_obj, k_ctx), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return ctx_idx, ctx_valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    distance = torch.cdist(obj, hand).numpy()
    for row, obj_idx in enumerate(valid_obj_idx.tolist()):
        dist_row = distance[row]
        ctx_candidates = np.flatnonzero(dist_row <= float(ctx_radius))
        if ctx_candidates.size > 0:
            order = np.argsort(dist_row[ctx_candidates], kind="stable")
            chosen_ctx = ctx_candidates[order[:k_ctx]]
            ctx_count = int(chosen_ctx.size)
            ctx_idx[obj_idx, :ctx_count] = chosen_ctx
            ctx_valid[obj_idx, :ctx_count] = True
    return ctx_idx, ctx_valid


def _compute_runtime_logit_neighbors(
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_near_logit: int,
    k_far_logit: int,
    logit_near_radius: float,
    logit_far_min_radius: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = BalancedEdgeSampler(
        k_near_logit=k_near_logit,
        k_far_logit=k_far_logit,
        logit_near_radius=logit_near_radius,
        logit_far_min_radius=logit_far_min_radius,
    ).sample(
        gt_obj_points=gt_obj_points,
        gt_hand_points=gt_hand_points,
        obj_valid=obj_valid,
        seed=0 if seed is None else int(seed),
    )
    return (
        sample.idx,
        sample.valid_mask,
        sample.loss_weight,
        sample.stats["near_count"],
        sample.stats["far_count"],
    )


def _compute_dense_logit_neighbors(
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    logit_near_radius: float,
    logit_far_min_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = DenseEdgeSampler(
        logit_near_radius=logit_near_radius,
        logit_far_min_radius=logit_far_min_radius,
    ).sample(
        gt_obj_points=gt_obj_points,
        gt_hand_points=gt_hand_points,
        obj_valid=obj_valid,
        seed=0,
    )
    return (
        sample.idx,
        sample.valid_mask,
        sample.loss_weight,
        sample.stats["near_count"],
        sample.stats["far_count"],
    )


def _compute_stratified_logit_neighbors(
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    distance_edges: np.ndarray | tuple[float, ...] | list[float],
    quotas: np.ndarray | tuple[int, ...] | list[int],
    logit_near_radius: float,
    logit_far_min_radius: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = StratifiedEdgeSampler(
        distance_edges=tuple(float(value) for value in np.asarray(distance_edges).tolist()),
        quotas=tuple(int(value) for value in np.asarray(quotas).tolist()),
        logit_near_radius=logit_near_radius,
        logit_far_min_radius=logit_far_min_radius,
    ).sample(
        gt_obj_points=gt_obj_points,
        gt_hand_points=gt_hand_points,
        obj_valid=obj_valid,
        seed=0 if seed is None else int(seed),
    )
    return (
        sample.idx,
        sample.valid_mask,
        sample.loss_weight,
        sample.stats["near_count"],
        sample.stats["far_count"],
    )


def _compute_input_knn(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_cross: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_obj = obj_points.shape[0]
    result = np.full((num_obj, k_cross), -1, dtype=np.int64)
    valid = np.zeros((num_obj, k_cross), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return result, valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    topk = min(int(k_cross), int(hand.shape[0]))
    idx = torch.topk(torch.cdist(obj, hand), k=topk, dim=-1, largest=False).indices.numpy()
    result[valid_obj_idx, :topk] = idx
    valid[valid_obj_idx, :topk] = True
    return result, valid


def _resolved_components_metadata(
    components: ResolvedCorrespondenceComponents,
) -> dict[str, dict[str, Any]]:
    return {
        "model": {
            "name": components.model.name,
            "params": dict(components.model.params),
        },
        "contact_supervision": {
            "name": components.contact_supervision.name,
            "params": dict(components.contact_supervision.params),
        },
        "edge_sampler": {
            "name": components.edge_sampler.name,
            "params": dict(components.edge_sampler.params),
        },
    }


def make_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    components: ResolvedCorrespondenceComponents | None = None,
    distributed: Any | None = None,
    explicit_override_keys: set[str] | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any], dict[str, DataLoader]]:
    if distributed is None:
        distributed = SimpleNamespace(enabled=False, rank=0)
    if components is None:
        wrapper = type("CompositionWrapper", (), {})()
        wrapper.meta = meta_cfg
        wrapper.model = None
        setattr(
            wrapper,
            "_explicit_override_keys",
            set(explicit_override_keys or getattr(meta_cfg, "_explicit_override_keys", set()) or set()),
        )
        components = resolve_correspondence_components(
            wrapper,
            explicit_override_keys=explicit_override_keys,
        )
    edge_sampler_cfg = components.edge_sampler
    edge_sampler_params = edge_sampler_cfg.params
    train_kwargs = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "k_cross": int(meta_cfg.k_cross),
        "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
        "k_near_logit": int(edge_sampler_params.get("k_near_logit", getattr(meta_cfg, "k_near_logit", 32))),
        "k_far_logit": int(edge_sampler_params.get("k_far_logit", getattr(meta_cfg, "k_far_logit", 32))),
        "logit_sampling_mode": edge_sampler_cfg.name,
        "logit_stratified_distance_edges": tuple(
            edge_sampler_params.get("distance_edges", (0.005, 0.015, 0.03, 0.06))
        ),
        "logit_stratified_quotas": tuple(
            edge_sampler_params.get("quotas", (16, 32, 32, 32, 16))
        ),
        "ctx_radius": float(getattr(meta_cfg, "ctx_radius", 0.04)),
        "logit_near_radius": float(edge_sampler_params["logit_near_radius"]),
        "logit_far_min_radius": float(edge_sampler_params["logit_far_min_radius"]),
        "logit_neg_radius": float(getattr(meta_cfg, "logit_neg_radius", 0.06)),
        "logit_far_weight": float(getattr(meta_cfg, "loss_cross_edge_far_weight", 0.5)),
        "base_seed": int(seed),
        "augment": bool(getattr(meta_cfg, "augment", True)),
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", True)),
        "augment_rotation": bool(meta_cfg.augment_rotation),
        "augment_translation": bool(meta_cfg.augment_translation),
        "augment_scale": bool(meta_cfg.augment_scale),
        "rotation_range": float(meta_cfg.rotation_range),
        "translation_range": float(meta_cfg.translation_range),
        "scale_range": tuple(meta_cfg.scale_range),
        "d_pos": float(meta_cfg.d_pos),
        "d_neg": float(meta_cfg.d_neg),
        "gamma": float(meta_cfg.gamma),
        "hand_rot_std_deg": float(meta_cfg.hand_rot_std_deg),
        "hand_trans_std": float(meta_cfg.hand_trans_std),
        "hand_perturb_prob": float(meta_cfg.hand_perturb_prob),
        "fix_overfit_seed": bool(getattr(meta_cfg, "fix_overfit_seed", False)),
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
        "edge_sampler_config": edge_sampler_cfg,
    }
    val_clean_kwargs = {
        **train_kwargs,
        "augment": False,
        "apply_hand_perturb": False,
        "hand_perturb_prob": 0.0,
    }
    val_perturbed_kwargs = {
        **train_kwargs,
        "augment": False,
        "apply_hand_perturb": bool(getattr(meta_cfg, "val_augment", False)),
        "hand_perturb_prob": float(getattr(meta_cfg, "val_hand_perturb_prob", 1.0)),
    }
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg,
        seed,
        dataset_cls=CorrStaticDataset,
        file_pattern="**/*.npz",
        train_dataset_kwargs=train_kwargs,
        val_dataset_kwargs=val_clean_kwargs,
        split_group_fn=(
            sequence_group_key
            if bool(getattr(data_cfg, "group_val_by_sequence", True))
            else None
        ),
        distributed=distributed,
    )
    if bool(data_cfg.shuffle) and bool(
        getattr(data_cfg, "sequence_locality_shuffle", True)
    ):
        train_sampler = SequenceLocalitySampler(train_loader.dataset, seed)
        train_sampler = shard_sampler_for_distributed(
            train_sampler,
            distributed=distributed,
            drop_last=bool(data_cfg.drop_last),
            pad=True,
        )
        loader_seed = int(seed) + int(
            getattr(distributed, "rank", 0)
            if getattr(distributed, "enabled", False)
            else 0
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(data_cfg.batch_size),
            shuffle=False,
            sampler=train_sampler,
            **make_dataloader_kwargs(data_cfg, loader_seed),
        )
    val_loaders: dict[str, DataLoader] = {}
    if val_loader is not None:
        val_loader.dataset.set_epoch(0)
        val_loaders["val_clean/"] = val_loader
        if bool(getattr(meta_cfg, "val_augment", False)):
            val_perturbed_dataset = CorrStaticDataset(
                val_loader.dataset.data_root,
                file_list=val_loader.dataset.file_paths,
                **val_perturbed_kwargs,
            )
            val_perturbed_dataset.set_epoch(0)
            loader_seed = int(seed) + int(
                getattr(distributed, "rank", 0)
                if getattr(distributed, "enabled", False)
                else 0
            )
            val_perturbed_loader = DataLoader(
                val_perturbed_dataset,
                batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                shuffle=False,
                sampler=make_default_eval_sampler(
                    val_perturbed_dataset,
                    distributed=distributed,
                ),
                **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
            )
            val_loaders["val_perturbed/"] = val_perturbed_loader

    first_path = train_loader.dataset.file_paths[0]
    first_store = train_loader.dataset.frame_store
    data = first_store.load_raw_file(first_path)
    hand_finger_id = np.asarray(
        data.get("hand_finger_id", np.full((int(data["hand_points"].shape[1]),), -1, dtype=np.int64))
    )
    hand_region_id = np.asarray(
        data.get("hand_region_id", np.full((int(data["hand_points"].shape[1]),), -1, dtype=np.int64))
    )
    metadata.update(
        {
            "resolved_components": _resolved_components_metadata(components),
            "num_obj_pool": int(data["obj_points"].shape[1]),
            "num_obj_points": int(meta_cfg.num_obj_points),
            "num_hand_points": int(data["hand_points"].shape[1]),
            "k_cross": int(data["gt_obj_to_hand_knn_idx"].shape[2]),
            "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
            "k_near_logit": int(edge_sampler_params.get("k_near_logit", getattr(meta_cfg, "k_near_logit", 32))),
            "k_far_logit": int(edge_sampler_params.get("k_far_logit", getattr(meta_cfg, "k_far_logit", 32))),
            "logit_sampling_mode": edge_sampler_cfg.name,
            "logit_stratified_distance_edges": tuple(
                edge_sampler_params.get("distance_edges", (0.005, 0.015, 0.03, 0.06))
            ),
            "logit_stratified_quotas": tuple(
                edge_sampler_params.get("quotas", (16, 32, 32, 32, 16))
            ),
            "k_logit": (
                int(sum(edge_sampler_params["quotas"]))
                if edge_sampler_cfg.name == "stratified"
                else (
                    int(data["hand_points"].shape[1])
                    if edge_sampler_cfg.name == "dense"
                    else int(edge_sampler_params["k_near_logit"]) + int(edge_sampler_params["k_far_logit"])
                )
            ),
            "logit_near_radius": float(edge_sampler_params["logit_near_radius"]),
            "logit_far_min_radius": float(edge_sampler_params["logit_far_min_radius"]),
            "k_logit_hard_neg": int(getattr(meta_cfg, "k_logit_hard_neg", 16)),
            "num_fingers": int(np.max(hand_finger_id)) + 1 if hand_finger_id.size > 0 else 0,
            "num_regions": int(np.max(hand_region_id)) + 1 if hand_region_id.size > 0 else 0,
            "fix_overfit_seed": bool(getattr(meta_cfg, "fix_overfit_seed", False)),
            "val_loader_names": sorted(val_loaders),
        }
    )
    return train_loader, val_loader, metadata, val_loaders
