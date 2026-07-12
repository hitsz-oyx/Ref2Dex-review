from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from src.base import make_file_split_dataloaders
from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, shard_sampler_for_distributed
from src.task.correspondence_ptv3.data.dataset import CorrStaticDataset
from src.task.correspondence_ptv3.data.split import SequenceLocalitySampler, sequence_group_key


def make_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    edge_sampler: object,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any], dict[str, DataLoader]]:
    if distributed is None:
        distributed = SimpleNamespace(enabled=False, rank=0)

    common_kwargs = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "k_cross": int(meta_cfg.k_cross),
        "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
        "ctx_radius": float(getattr(meta_cfg, "ctx_radius", 0.04)),
        "base_seed": int(seed),
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
        "fix_overfit_seed": bool(getattr(meta_cfg, "fix_overfit_seed", False)),
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
        "edge_sampler": edge_sampler,
    }
    train_kwargs = {
        **common_kwargs,
        "augment": bool(getattr(meta_cfg, "augment", True)),
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", True)),
        "hand_perturb_prob": float(getattr(meta_cfg, "hand_perturb_prob", 1.0)),
    }
    val_clean_kwargs = {
        **common_kwargs,
        "augment": False,
        "apply_hand_perturb": False,
        "hand_perturb_prob": 0.0,
    }
    val_perturbed_kwargs = {
        **common_kwargs,
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
    if bool(data_cfg.shuffle) and bool(getattr(data_cfg, "sequence_locality_shuffle", True)):
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
            "num_obj_pool": int(data["obj_points"].shape[1]),
            "num_hand_points": int(data["hand_points"].shape[1]),
            "k_cross": int(data["gt_obj_to_hand_knn_idx"].shape[2]),
            "num_fingers": int(np.max(hand_finger_id)) + 1 if hand_finger_id.size > 0 else 0,
            "num_regions": int(np.max(hand_region_id)) + 1 if hand_region_id.size > 0 else 0,
            "val_loader_names": sorted(val_loaders),
        }
    )
    return train_loader, val_loader, metadata, val_loaders


__all__ = ["make_dataloaders"]
