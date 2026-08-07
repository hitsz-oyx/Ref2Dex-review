from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.base import BaseRunner, MetricStat, RunnerOutput, TaskConfig, set_config_default_if_not_explicit
from src.task.correspondence_ptv3_v2.dataset import make_dataloaders
from src.task.correspondence_ptv3_v2.checkpoint_compat import adapt_checkpoint_payload
from src.task.correspondence_ptv3_v2.losses import (
    contact_target_from_distance,
    binary_cross_entropy_with_logits_map,
    binary_entropy_floor_map,
    quality_focal_loss_map,
    reduce_loss_map,
    reduce_loss_map_per_object,
    target_strength_bin_mask,
    zero_predictor_bce_map,
    zero_predictor_mae_map,
    zero_predictor_qfl_map,
)
from src.task.correspondence_ptv3_v2.hand_noise_profiles import (
    HandGeometryNoiseProfiles,
)
from src.utils.correspondence import gather_batched_knn_features


_TARGET_STRENGTH_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.25, "edge_y_0_025"),
    (0.25, 0.50, "edge_y_025_050"),
    (0.50, 0.75, "edge_y_050_075"),
    (0.75, 1.00, "edge_y_075_100"),
)


class CorrespondencePTV3V2Runner(BaseRunner):
    _CONTACT_POSITIVE_EPS = 1e-4
    _CONTACT_SAMPLE_CHUNK = 64

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fix #4 (docs/指导.md): legacy mode (``use_mano_reconstruction=False``)
        # is documented as "no MANO forward, just pre-stored hand_points".
        # But ``runtime_resample_object`` independently calls
        # ``get_proxy_face_idx`` to score the object pool against a hand
        # proxy, which still requires smplx + MANO .pkl on disk. Make the
        # conflict loud at construction time instead of crashing deep in
        # the training loop.
        if (
            not bool(getattr(self.cfg.meta, "use_mano_reconstruction", False))
            and bool(getattr(self.cfg.meta, "runtime_resample_object", True))
        ):
            raise ValueError(
                "cfg.meta.use_mano_reconstruction=False but "
                "cfg.meta.runtime_resample_object=True. The runtime object "
                "resampler still needs smplx + MANO .pkl to build a hand "
                "proxy (see hand_noise_profiles.get_proxy_face_idx), which "
                "contradicts the legacy 'no MANO forward' contract. Pick "
                "one of:\n"
                "  - use_mano_reconstruction: true  (full MANO path, current default)\n"
                "  - runtime_resample_object: false (pure pre-stored hand_points, "
                "no runtime obj resampling)\n"
                "If you want a true zero-smplx legacy smoke test, set BOTH "
                "explicitly in your YAML."
            )
        # Lazily-built MANO layer cache. Only populated when MANO forward
        # is actually needed (i.e. the batch has hand MANO parameters and
        # cfg.meta.use_mano_reconstruction is True).
        self._mano_cache = None
        self._mano_cache_root_face = None
        profile_paths = getattr(self.cfg.meta, "hand_geometry_calibration_paths", {})
        if hasattr(profile_paths, "to_dict"):
            profile_paths = profile_paths.to_dict()
        # Fix #3 (docs/指导.md): resolve ``hand_geometry_calibration_paths``
        # relative to the repo root, NOT the current working directory. This
        # way ``python -m src.task...`` works the same whether you launch
        # from the repo root, from src/, or from inside an IDE.
        from src.task.correspondence_ptv3_v2.config import ROOT as _REPO_ROOT
        self._hand_geometry_profiles = HandGeometryNoiseProfiles.from_paths(
            profile_paths,
            base_dir=_REPO_ROOT,
        )

    @classmethod
    def configure_overfit_mode(
        cls,
        cfg: TaskConfig,
        explicit_override_keys: set[str],
    ) -> None:
        super().configure_overfit_mode(cfg, explicit_override_keys)
        for key, value in (
            ("meta.apply_obj_perturb", False),
            ("meta.obj_perturb_prob", 0.0),
            ("meta.ptv3_drop_path", 0.0),
            ("meta.ptv3_shuffle_orders", False),
        ):
            set_config_default_if_not_explicit(
                cfg,
                key=key,
                value=value,
                explicit_override_keys=explicit_override_keys,
            )

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(
            data_cfg,
            meta_cfg=self.cfg.meta,
            seed=seed,
            distributed=self.distributed,
        )

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        # Validate coordinate_frame BEFORE we let metadata overwrite the cfg
        # value; the data is the source of truth and a mismatch must raise
        # rather than be silently coerced by the update loop below.
        if "coordinate_frame" not in metadata:
            raise ValueError(
                "Stage 3 metadata is missing 'coordinate_frame'. Re-run Stage 3 "
                "with the current prepare_corr_static.py."
            )
        stage3_frame = metadata["coordinate_frame"]
        if stage3_frame != self.cfg.meta.coordinate_frame:
            raise ValueError(
                f"Stage 3 data was generated in coordinate_frame={stage3_frame!r} "
                f"but cfg.meta.coordinate_frame={self.cfg.meta.coordinate_frame!r}. "
                f"Re-run Stage 3 with --coordinate-frame {self.cfg.meta.coordinate_frame} "
                f"on the same Stage 2 root, or pass "
                f"--set meta.coordinate_frame={stage3_frame} to this training run."
            )
        for field in (
            "num_obj_pool",
            "num_obj_points",
            "num_hand_points",
            "num_supervision_edges",
            "contact_radius",
            "coordinate_frame",
        ):
            if field in metadata:
                setattr(self.cfg.meta, field, metadata[field])

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def adapt_checkpoint_payload(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        return adapt_checkpoint_payload(checkpoint)

    def dataset_epoch_for_train(self, epoch: int) -> int:
        # Keep the sampled geometry fixed during overfit diagnosis while the
        # base runner continues to advance a distributed sampler normally.
        return 0 if bool(getattr(self.cfg.train, "overfit_mode", False)) else epoch

    def prepare_batch(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return super().prepare_batch(batch)

        # Pull the per-sample seeds and the contact-supervision seed out of
        # the batch BEFORE super().prepare_batch so we can hand the GPU-side
        # MANO forward the right layer. ``__mano_side__`` and
        # ``hand_perturb_seed`` stay in the batch: the resample step later
        # in ``prepare_batch`` also needs the side to pick a per-side hand
        # proxy index list.
        mano_side = batch.get("__mano_side__")
        contact_seed = batch.pop("contact_seed", None)
        batch = super().prepare_batch(batch)
        if torch.is_tensor(contact_seed):
            contact_seed_list = [int(value) for value in contact_seed.reshape(-1).tolist()]
        elif contact_seed is not None:
            contact_seed_list = [int(contact_seed)]
        else:
            contact_seed_list = None
        with torch.no_grad():
            if mano_side is not None and self._should_reconstruct_mano(batch):
                self._reconstruct_hand_from_mano(batch, side=mano_side)
            if contact_seed_list is not None:
                self._resample_object_from_perturbed_hand(batch)
                self._build_supervision_gpu(batch, contact_seed_list=contact_seed_list)
        return batch

    # ---------- MANO reconstruction on the GPU ----------

    def _should_reconstruct_mano(self, batch: dict[str, torch.Tensor]) -> bool:
        if not bool(getattr(self.cfg.meta, "use_mano_reconstruction", False)):
            return False
        has_mano = batch.get("has_mano")
        if has_mano is None:
            return False
        if torch.is_tensor(has_mano):
            # DataLoader stacks the per-sample booleans into a (B,) tensor.
            # Reconstruction is only triggered if every sample in the
            # batch carries MANO parameters; mixed batches fall back to
            # the legacy hand_points for safety.
            if has_mano.numel() == 0:
                return False
            if not bool(has_mano.all().item()):
                return False
        elif not bool(has_mano):
            return False
        required = ("mano_global_orient", "mano_transl", "mano_pose", "mano_betas")
        return all(batch.get(k) is not None for k in required)

    def _get_mano_cache(self, device: torch.device):
        if self._mano_cache is None:
            from src.task.correspondence_ptv3_v2.mano_recon import MANOLayerCache
            self._mano_cache = MANOLayerCache(
                model_dir=str(self.cfg.meta.mano_model_dir),
                device=device,
            )
        return self._mano_cache

    def _lookup_mano_descriptor(self, side: str) -> tuple[bool, int, bool]:
        """Return (use_pca, num_pca_comps, flat_hand_mean) for the side.

        Looks at the active train dataset's descriptive fields first;
        falls back to GRAB-style defaults (PCA24, flat_hand_mean=True)
        so a misconfigured pipeline still produces a valid forward pass.
        """
        train_ds = getattr(self, "train_dataset", None) or getattr(self, "dataset", None)
        for candidate in (train_ds, getattr(self, "val_dataset", None)):
            use_pca = getattr(candidate, "mano_use_pca", None)
            num_pca_comps = getattr(candidate, "mano_num_pca_comps", None)
            flat_hand_mean = getattr(candidate, "mano_flat_hand_mean", None)
            if use_pca is not None:
                return bool(use_pca), int(num_pca_comps or 24), bool(flat_hand_mean)
        # GRAB-style fallback.
        del side
        return True, 24, True

    @staticmethod
    def _normalize_side_batch(side: Any, batch_size: int) -> list[str]:
        if isinstance(side, str):
            return [side] * batch_size
        if isinstance(side, (list, tuple)):
            values = [str(x) for x in side]
            if len(values) == batch_size:
                return values
            if len(values) == 1:
                return values * batch_size
        return [str(side)] * batch_size

    @staticmethod
    def _v_template_for_index(v_template: torch.Tensor | None, index: int) -> np.ndarray | None:
        if v_template is None:
            return None
        arr = v_template[index] if v_template.ndim >= 3 else v_template
        return arr.detach().cpu().numpy().astype(np.float32, copy=True)

    @staticmethod
    def _v_template_sha(v_template: torch.Tensor | None, index: int) -> str | None:
        arr = CorrespondencePTV3V2Runner._v_template_for_index(v_template, index)
        if arr is None:
            return None
        import hashlib

        return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()

    @staticmethod
    def _normalize_v_template_sha(value: Any, batch_size: int) -> list[str | None]:
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, (list, tuple)):
            return [None if item is None else str(item) for item in value]
        if torch.is_tensor(value):
            return [str(item) for item in value.reshape(-1).tolist()]
        return [None] * batch_size

    @staticmethod
    def _sample_passed_gate(
        *,
        i_global: int,
        seed: int,
        per_sample_apply: list[bool],
        hand_perturb_prob: float,
    ) -> bool:
        """Return True iff sample ``i_global`` should be perturbed this step.

        Centralised so the PCA and axis-angle paths agree on the
        per-sample gating (Fix #3): a sample is perturbed only when
        ``per_sample_apply[i]`` is True and the per-step Bernoulli
        gate (with seed ``seed ^ 0xC0FFEE``) is below
        ``hand_perturb_prob``.
        """
        if not per_sample_apply[i_global]:
            return False
        if hand_perturb_prob <= 0.0:
            return False
        if hand_perturb_prob >= 1.0:
            return True
        gate_rng = torch.Generator(device="cpu")
        gate_rng.manual_seed(int(seed) ^ 0xC0FFEE)
        return float(torch.rand((), generator=gate_rng).item()) <= hand_perturb_prob

    def _apply_pca_noise(
        self,
        *,
        hand_pose_g: torch.Tensor,
        num_pca_comps: int,
        indices: list[int],
        seeds_list: list[int],
        per_sample_apply: list[bool],
        hand_pca_std: float,
        hand_pca_scale: float,
        hand_pca_clip: float,
        per_dim_std_t: torch.Tensor | None,
        hand_perturb_prob: float,
        geometry_std_t: torch.Tensor | None = None,
        geometry_clip_sigma: float | None = None,
    ) -> torch.Tensor:
        """Apply PCA-coefficient noise to the active dims of ``hand_pose_g``.

        ``hand_pose_g`` is padded to MANO_POSE_MAX_DIM (=45) by the
        dataset.  The active PCA coefficients live in the first
        ``num_pca_comps`` slots; the remaining slots are zero padding
        and must NOT receive any noise (Fix #4).
        """
        if num_pca_comps <= 0:
            return hand_pose_g
        if geometry_std_t is None and per_dim_std_t is None and hand_pca_std <= 0:
            return hand_pose_g
        if geometry_std_t is None and per_dim_std_t is not None and hand_pca_scale <= 0:
            return hand_pose_g
        device = hand_pose_g.device
        dtype = hand_pose_g.dtype
        group_size = int(hand_pose_g.shape[0])
        if group_size == 0:
            return hand_pose_g
        noise = torch.zeros((group_size, num_pca_comps), device=device, dtype=dtype)
        any_perturbed = False
        for j_local, i_global in enumerate(indices):
            if not self._sample_passed_gate(
                i_global=i_global,
                seed=int(seeds_list[i_global]),
                per_sample_apply=per_sample_apply,
                hand_perturb_prob=hand_perturb_prob,
            ):
                continue
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seeds_list[i_global]))
            sample_noise = torch.empty((num_pca_comps,), device=device, dtype=dtype)
            sample_noise.normal_(generator=gen)
            noise[j_local] = sample_noise
            any_perturbed = True
        if not any_perturbed:
            return hand_pose_g
        # Per-dim std must match the active PCA dim exactly (Fix #4).
        # The pre-grouping shape is padded to 45 to make DataLoader
        # collate work, but the descriptor we want to scale here is
        # only the leading ``num_pca_comps`` dims.
        if geometry_std_t is not None:
            if geometry_std_t.numel() != int(num_pca_comps):
                raise ValueError(
                    f"Geometry profile must have {num_pca_comps} entries, got "
                    f"{geometry_std_t.numel()}."
                )
            effective_std = geometry_std_t.to(device=device, dtype=dtype)
        elif per_dim_std_t is not None:
            if per_dim_std_t.numel() != int(num_pca_comps):
                raise ValueError(
                    f"hand_pca_train_std_per_dim must have {num_pca_comps} entries "
                    f"to match the active PCA dim of this group, got {per_dim_std_t.numel()}."
                )
            effective_std = per_dim_std_t.to(device=device, dtype=dtype) * hand_pca_scale
        else:
            effective_std = torch.full(
                (num_pca_comps,), hand_pca_std, device=device, dtype=dtype
            )
        clip_sigma = (
            float(geometry_clip_sigma)
            if geometry_std_t is not None and geometry_clip_sigma is not None
            else hand_pca_clip
        )
        if clip_sigma > 0:
            noise = noise.clamp(min=-clip_sigma, max=clip_sigma)
        delta = noise * effective_std
        hand_pose_g = hand_pose_g.clone()
        hand_pose_g[:, :num_pca_comps] = hand_pose_g[:, :num_pca_comps] + delta
        return hand_pose_g

    def _apply_axis_angle_noise(
        self,
        *,
        hand_pose_g: torch.Tensor,
        indices: list[int],
        seeds_list: list[int],
        per_sample_apply: list[bool],
        hand_aa_std: float,
        hand_aa_clip: float,
        hand_perturb_prob: float,
        geometry_std_t: torch.Tensor | None = None,
        geometry_clip_sigma: float | None = None,
    ) -> torch.Tensor:
        """Apply axis-angle (rad) noise to ``hand_pose_g`` (Fix #3).

        ARCTIC uses axis-angle45, so the active pose dim is 45
        (which is also MANO_POSE_MAX_DIM — no padding).  PCA noise
        must never be added to these samples; this method is only
        called from the ``use_pca=False`` branch.
        """
        if geometry_std_t is None and hand_aa_std <= 0:
            return hand_pose_g
        device = hand_pose_g.device
        dtype = hand_pose_g.dtype
        group_size = int(hand_pose_g.shape[0])
        if group_size == 0:
            return hand_pose_g
        pose_dim = int(hand_pose_g.shape[1])
        noise = torch.zeros((group_size, pose_dim), device=device, dtype=dtype)
        any_perturbed = False
        for j_local, i_global in enumerate(indices):
            if not self._sample_passed_gate(
                i_global=i_global,
                seed=int(seeds_list[i_global]) ^ 0xAA,
                per_sample_apply=per_sample_apply,
                hand_perturb_prob=hand_perturb_prob,
            ):
                continue
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seeds_list[i_global]) ^ 0xAA)
            sample_noise = torch.empty((pose_dim,), device=device, dtype=dtype)
            sample_noise.normal_(generator=gen)
            noise[j_local] = sample_noise
            any_perturbed = True
        if not any_perturbed:
            return hand_pose_g
        if geometry_std_t is not None:
            if geometry_std_t.numel() != pose_dim:
                raise ValueError(
                    f"Geometry profile must have {pose_dim} entries, got "
                    f"{geometry_std_t.numel()}."
                )
            if geometry_clip_sigma is not None and geometry_clip_sigma > 0:
                noise = noise.clamp(
                    min=-float(geometry_clip_sigma), max=float(geometry_clip_sigma)
                )
            delta = noise * geometry_std_t.to(device=device, dtype=dtype)
        else:
            delta = noise * hand_aa_std
            if hand_aa_clip > 0:
                delta = torch.clamp(delta, min=-hand_aa_clip, max=hand_aa_clip)
        return hand_pose_g + delta

    def _reconstruct_hand_from_mano(
        self,
        batch: dict[str, torch.Tensor],
        *,
        side: Any,
    ) -> None:
        """Re-build hand_points / hand_normals from MANO forward on the GPU.

        Only called for batches that carry MANO parameters. The
        preprocessor's hand_points (already used as GT and as the
        ``hand_min_dist`` cache) is left untouched; this method replaces
        the *input* hand points/normals the same way
        ``perturb_object_geometry`` replaces the input obj geometry.
        """
        from src.task.correspondence_ptv3_v2.mano_recon import (
            derive_face_center_points_and_normals,
            get_proxy_face_idx,
            reconstruct_hand_points,
        )

        device = batch["points"].device
        cache = self._get_mano_cache(device)

        global_orient = batch["mano_global_orient"].to(device=device, dtype=torch.float32)
        transl = batch["mano_transl"].to(device=device, dtype=torch.float32)
        hand_pose = batch["mano_pose"].to(device=device, dtype=torch.float32)
        betas = batch["mano_betas"].to(device=device, dtype=torch.float32)
        v_template = batch.get("mano_v_template")
        if v_template is not None:
            v_template = v_template.to(device=device, dtype=torch.float32)

        # Subject-aware MANO config selection. We trust the npz's
        # descriptive fields (carried on the train dataset) over the cfg
        # defaults so a mixed-dataset batch (e.g. ARCTIC + GRAB) can
        # never desync. The values are read once in
        # ``CorrStaticDatasetV2.__init__``.
        batch_size = int(global_orient.shape[0])
        side_list = self._normalize_side_batch(side, batch_size)

        # ---- Per-sample, per-side hand perturbation (Fix #4, #5) ----
        # Per-sample toggles that come from the dataset:
        #   apply_hand_perturb[b]  – per-sample on/off (val_clean = False)
        #   hand_perturb_seed[b]   – stable per-frame seed for the noise
        batch_apply = batch.get("apply_hand_perturb")
        apply_hand_perturb_cfg = bool(getattr(self.cfg.meta, "apply_hand_perturb", False))
        per_sample_apply = torch.ones(
            (batch_size,), dtype=torch.bool, device=device
        )
        if torch.is_tensor(batch_apply):
            per_sample_apply = batch_apply.to(device=device, dtype=torch.bool).reshape(-1)
        elif batch_apply is not None:
            per_sample_apply = torch.as_tensor(
                bool(batch_apply), dtype=torch.bool, device=device
            ).expand(batch_size).clone()
        per_sample_apply_list = [
            bool(value) for value in per_sample_apply.detach().cpu().tolist()
        ]
        hand_pca_std = float(getattr(self.cfg.meta, "hand_pca_std", 0.0))
        hand_pca_scale = float(getattr(self.cfg.meta, "hand_pca_noise_scale", 0.0))
        hand_pca_clip = float(getattr(self.cfg.meta, "hand_pca_noise_clip", 0.0))
        hand_aa_std = float(getattr(self.cfg.meta, "hand_axis_angle_std_rad", 0.0))
        hand_aa_clip = float(getattr(self.cfg.meta, "hand_axis_angle_clip_rad", 0.0))
        hand_perturb_prob = float(getattr(self.cfg.meta, "hand_perturb_prob", 1.0))
        geometry_noise_scale = float(
            getattr(self.cfg.meta, "hand_geometry_noise_scale", 1.0)
        )
        geometry_noise_required = bool(
            getattr(self.cfg.meta, "hand_geometry_noise_required", False)
        )
        per_dim_std = getattr(self.cfg.meta, "hand_pca_train_std_per_dim", None)
        per_dim_std_t: torch.Tensor | None = None
        if per_dim_std is not None:
            per_dim_std_t = torch.as_tensor(
                list(per_dim_std), dtype=torch.float32, device=device
            )

        # Per-sample seeds: prefer the dataset-supplied one (Fix #4) so the
        # noise is stable across runs; fall back to a positional seed.
        seeds_t = batch.get("hand_perturb_seed")
        if torch.is_tensor(seeds_t):
            seeds_list = [int(v) for v in seeds_t.reshape(-1).tolist()]
        else:
            seeds_list = list(range(batch_size))

        # Pre-computed per-sample v_template SHA (Fix #6).  Computing
        # SHA1 from a GPU tensor on every batch would force a per-sample
        # GPU→CPU copy and an SHA1 evaluation per sample, so the
        # dataset already pre-computes the digest on CPU when loading
        # the npz.  We group on the string here.
        sha_list = self._normalize_v_template_sha(
            batch.get("mano_v_template_sha"), batch_size
        )
        descriptor_lists: dict[str, list[Any]] = {}
        for key in ("mano_use_pca", "mano_num_pca_comps", "mano_flat_hand_mean"):
            value = batch.get(key)
            if torch.is_tensor(value):
                descriptor_lists[key] = value.detach().cpu().reshape(-1).tolist()
        dataset_id_list = self._normalize_side_batch(
            batch.get("__dataset_id__", "unknown"), batch_size
        )

        # Reuse one MANO layer per (side, config, subject v_template). This
        # handles DataLoader-collated side lists and mixed GRAB/ARCTIC batches.
        face_centers: torch.Tensor | None = None
        face_normals: torch.Tensor | None = None
        groups: dict[tuple[str, str, bool, int, bool, str | None], list[int]] = {}
        for i in range(batch_size):
            default_use_pca, default_num, default_flat = self._lookup_mano_descriptor(side_list[i])
            use_pca_i = bool(
                descriptor_lists.get("mano_use_pca", [default_use_pca] * batch_size)[i]
            )
            num_i = int(
                descriptor_lists.get("mano_num_pca_comps", [default_num] * batch_size)[i]
            )
            flat_i = bool(
                descriptor_lists.get("mano_flat_hand_mean", [default_flat] * batch_size)[i]
            )
            # Prefer the pre-computed digest (Fix #6); fall back to a
            # per-sample GPU→CPU SHA if the dataset is older.
            sha_i = sha_list[i] if i < len(sha_list) else None
            if sha_i is None:
                sha_i = self._v_template_sha(v_template, i)
            key = (
                dataset_id_list[i],
                side_list[i],
                use_pca_i,
                num_i,
                flat_i,
                sha_i,
            )
            groups.setdefault(key, []).append(i)

        for (
            dataset_id_i,
            side_i,
            use_pca_i,
            num_i,
            flat_i,
            _sha,
        ), indices in groups.items():
            idx = torch.as_tensor(indices, dtype=torch.long, device=device)
            template_shape = (
                tuple(int(dim) for dim in v_template.shape[-2:])
                if v_template is not None
                else None
            )
            cached = cache.get_cached_for_v_template_id(
                side=side_i,
                use_pca=use_pca_i,
                num_pca_comps=num_i,
                flat_hand_mean=flat_i,
                v_template_sha=_sha,
                v_template_shape=template_shape,
            )
            if cached is None:
                layer, _cfg = cache.get_or_build_for_v_template(
                    side=side_i,
                    use_pca=use_pca_i,
                    num_pca_comps=num_i,
                    flat_hand_mean=flat_i,
                    v_template=self._v_template_for_index(v_template, indices[0]),
                    v_template_sha=_sha,
                )
            else:
                layer, _cfg = cached
            pose_dim = int(num_i) if bool(use_pca_i) else 45
            hand_pose_g = hand_pose.index_select(0, idx).clone()
            geometry_profile = self._hand_geometry_profiles.get(
                dataset_id=dataset_id_i,
                side=side_i,
                use_pca=bool(use_pca_i),
                num_components=pose_dim,
                flat_hand_mean=flat_i,
            )
            if (
                apply_hand_perturb_cfg
                and geometry_noise_required
                and geometry_profile is None
            ):
                raise ValueError(
                    "Missing required hand geometry-noise profile for "
                    f"dataset={dataset_id_i!r}, side={side_i!r}, "
                    f"use_pca={bool(use_pca_i)}, dims={pose_dim}, "
                    f"flat_hand_mean={flat_i}."
                )
            geometry_std_t = None
            geometry_clip_sigma = None
            if geometry_profile is not None:
                geometry_std_t = geometry_profile.std_tensor(
                    device=device, dtype=hand_pose_g.dtype
                ) * geometry_noise_scale
                geometry_clip_sigma = geometry_profile.clip_sigma

            # Fix #3 + #4: apply per-group noise.  Doing it per group
            # (instead of once on the whole 45-dim padded pose) keeps
            # the PCA path and the axis-angle path strictly
            # disjoint — ARCTIC axis-angle samples must NEVER receive
            # the PCA scalar noise, and the PCA per-dim std must
            # multiply only the active PCA dims, not the 21 padding
            # zeros used to bring axis-angle45 up to MANO_POSE_MAX_DIM.
            if apply_hand_perturb_cfg:
                if bool(use_pca_i):
                    hand_pose_g = self._apply_pca_noise(
                        hand_pose_g=hand_pose_g,
                        num_pca_comps=int(num_i),
                        indices=indices,
                        seeds_list=seeds_list,
                        per_sample_apply=per_sample_apply_list,
                        hand_pca_std=hand_pca_std,
                        hand_pca_scale=hand_pca_scale,
                        hand_pca_clip=hand_pca_clip,
                        per_dim_std_t=per_dim_std_t,
                        hand_perturb_prob=hand_perturb_prob,
                        geometry_std_t=geometry_std_t,
                        geometry_clip_sigma=geometry_clip_sigma,
                    )
                else:
                    hand_pose_g = self._apply_axis_angle_noise(
                        hand_pose_g=hand_pose_g,
                        indices=indices,
                        seeds_list=seeds_list,
                        per_sample_apply=per_sample_apply_list,
                        hand_aa_std=hand_aa_std,
                        hand_aa_clip=hand_aa_clip,
                        hand_perturb_prob=hand_perturb_prob,
                        geometry_std_t=geometry_std_t,
                        geometry_clip_sigma=geometry_clip_sigma,
                    )

            vertices_i = reconstruct_hand_points(
                layer,
                global_orient=global_orient.index_select(0, idx),
                hand_pose=hand_pose_g[..., :pose_dim],
                transl=transl.index_select(0, idx),
                betas=betas.index_select(0, idx),
            )
            faces_i = torch.as_tensor(
                np.asarray(layer.faces).astype(np.int64), dtype=torch.long, device=device
            )
            centers_i, normals_i = derive_face_center_points_and_normals(vertices_i, faces_i)
            if face_centers is None:
                face_centers = torch.empty(
                    (batch_size, centers_i.shape[1], 3), dtype=centers_i.dtype, device=device
                )
                face_normals = torch.empty_like(face_centers)
            face_centers.index_copy_(0, idx, centers_i)
            face_normals.index_copy_(0, idx, normals_i)
        assert face_centers is not None and face_normals is not None

        # The MANO forward output `vertices` lives in the **world** frame
        # (smplx applies global_orient + transl internally). To splice
        # the rebuilt hand into the batch we must bring the face
        # centers back to the stored ``hand_root`` frame, exactly the
        # inverse of ``_points_world_to_hand_root`` in
        # ``process/common/stage3_corr.py``:
        #     hand_root_point = R_inv @ (world_point - t)
        # where (R, t) is ``hand_root_pose_world``. ``hand_root_pose`` is
        # optional in v2.0; if absent we assume identity (i.e. the npz
        # was built with world-frame points, which matches the v2.0
        # legacy "object" frame convention).
        hand_root_pose = batch.get("hand_root_pose_world")
        if hand_root_pose is None:
            hand_root_points = face_centers
            hand_root_normals = face_normals
        else:
            R = hand_root_pose[:, :3, :3]
            t = hand_root_pose[:, :3, 3]
            R_inv = R.transpose(-1, -2)
            hand_root_points = torch.einsum(
                "bij,bnj->bni", R_inv, face_centers - t[:, None, :]
            )
            hand_root_normals = torch.einsum("bij,bnj->bni", R_inv, face_normals)

        # Splice the rebuilt hand points into the input batch.
        num_hand_points_tensor = batch["num_hand_points"]
        if num_hand_points_tensor.numel() > 1:
            num_hand_points = int(num_hand_points_tensor.flatten()[0].item())
        else:
            num_hand_points = int(num_hand_points_tensor.item())
        if hand_root_points.shape[1] != num_hand_points:
            raise RuntimeError(
                f"MANO forward produced {hand_root_points.shape[1]} face centers "
                f"but the dataset expects {num_hand_points} hand points. "
                f"This usually means the npz is in an older convention."
            )
        offset = batch["points"].shape[1] - num_hand_points
        batch["points"][:, offset:] = hand_root_points.to(batch["points"].dtype)
        batch["normals"][:, offset:] = hand_root_normals.to(batch["normals"].dtype)

    def _resample_object_from_perturbed_hand(self, batch: dict[str, torch.Tensor]) -> None:
        """Runtime v2.1 object sampling from perturbed geometry."""
        flag = batch.get("runtime_resample_object")
        if flag is None or "full_input_obj_points" not in batch:
            return
        if torch.is_tensor(flag) and not bool(flag.all().item()):
            return

        from src.task.correspondence_ptv3_v2.mano_recon import get_proxy_face_idx

        num_obj = int(self.cfg.meta.num_obj_points)
        num_hand = int(self.cfg.meta.num_hand_points)
        near_quota = min(int(getattr(self.cfg.meta, "runtime_near_obj_points", 384)), num_obj)
        global_quota = max(0, num_obj - near_quota)
        proxy_count_cfg = int(getattr(self.cfg.meta, "hand_proxy_face_count", num_hand))
        proxy_count = min(proxy_count_cfg, num_hand)
        cdist_chunk = max(1, int(getattr(self.cfg.meta, "runtime_cdist_chunk_size", 512)))

        full_input_obj = batch["full_input_obj_points"].float()
        full_input_normals = batch["full_input_obj_normals"].float()
        full_gt_obj = batch["full_gt_obj_points"].float()
        full_gt_normals = batch["full_gt_obj_normals"].float()
        hand_input = batch["points"][:, -num_hand:].float()
        hand_input_normals = batch["normals"][:, -num_hand:].float()
        hand_gt = batch["gt_points"][:, -num_hand:].float()
        hand_gt_normals = batch["gt_normals"][:, -num_hand:].float()

        device = full_input_obj.device
        batch_size, pool_size, _ = full_input_obj.shape

        # Fix #5: per-sample side list. The dataset emits one string per
        # sample via ``__mano_side__``; PyTorch's default collate keeps
        # them as a list of strings. We also accept a single string
        # (single-side npz) for backwards compatibility.
        side_hint = batch.get("__mano_side__")
        if isinstance(side_hint, (list, tuple)):
            if len(side_hint) == batch_size:
                per_sample_sides = [str(s) for s in side_hint]
            elif len(side_hint) == 1:
                per_sample_sides = [str(side_hint[0])] * batch_size
            else:
                per_sample_sides = [str(side_hint[0])] * batch_size
        elif isinstance(side_hint, str):
            per_sample_sides = [side_hint] * batch_size
        else:
            per_sample_sides = ["right"] * batch_size

        # Build per-side proxy indices. The MANO left/right hand meshes
        # share the face topology but their face indices are NOT in the
        # same spatial order, so a single shared index would mix
        # fingertip points with palm points when the batch contains
        # both sides. We build one proxy per side and select by sample
        # later. The over-broad ``except`` is gone (Fix #5): a
        # missing model dir is now a hard failure, only an explicit
        # linspace fallback runs on a controlled exception.
        mano_model_dir = str(self.cfg.meta.mano_model_dir)

        def _proxy_for(side: str) -> torch.Tensor:
            try:
                proxy_idx_np = get_proxy_face_idx(
                    side=side,
                    count=int(proxy_count),
                    model_dir=mano_model_dir,
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"MANO model dir {mano_model_dir!r} is required to build "
                    f"the FPS proxy for side={side!r}. Set meta.mano_model_dir "
                    "to the directory that contains the .pkl files."
                ) from e
            except Exception as e:
                # Any other failure (e.g. smplx import error) bubbles
                # up: the runner must not silently degrade to linspace.
                raise RuntimeError(
                    f"get_proxy_face_idx(side={side!r}) failed: {e}. "
                    "This is a hard failure; do not silently fall back."
                ) from e
            return torch.as_tensor(
                np.unique(proxy_idx_np), dtype=torch.long, device=device
            )

        proxy_per_side: dict[str, torch.Tensor] = {
            "right": _proxy_for("right"),
            "left": _proxy_for("left"),
        }

        # Per-side min-distance accumulator. We do not rely on a single
        # shared proxy; each sample's hand is indexed with the proxy
        # matching its own side. The cdist is still chunked (Fix #7).
        obj_min_dist = torch.full(
            (batch_size, pool_size), float("inf"), device=device, dtype=hand_input.dtype
        )
        for side_name in set(per_sample_sides):
            side_idx_list = [i for i, s in enumerate(per_sample_sides) if s == side_name]
            if not side_idx_list:
                continue
            proxy = proxy_per_side.get(side_name, proxy_per_side["right"])
            side_idx_t = torch.as_tensor(side_idx_list, dtype=torch.long, device=device)
            side_hand_proxy = hand_input.index_select(0, side_idx_t).index_select(1, proxy)
            side_obj = full_input_obj.index_select(0, side_idx_t)
            for start in range(0, pool_size, cdist_chunk):
                end = min(start + cdist_chunk, pool_size)
                dist = torch.cdist(side_obj[:, start:end], side_hand_proxy)  # (b, chunk, P)
                obj_min_dist[side_idx_t, start:end] = dist.amin(dim=-1)
        # Any sample whose side was missing gets the right-hand proxy
        # as a defensive fallback (the locality sampler keeps mixed
        # batches rare, so this branch almost never fires).
        if not torch.isfinite(obj_min_dist).all():
            fallback_idx = torch.nonzero(
                ~torch.isfinite(obj_min_dist).all(dim=-1), as_tuple=False
            ).squeeze(-1)
            if fallback_idx.numel() > 0:
                proxy = proxy_per_side["right"]
                side_hand_proxy = hand_input.index_select(0, fallback_idx).index_select(1, proxy)
                for start in range(0, pool_size, cdist_chunk):
                    end = min(start + cdist_chunk, pool_size)
                    dist = torch.cdist(
                        full_input_obj[fallback_idx, start:end], side_hand_proxy
                    )
                    obj_min_dist[fallback_idx, start:end] = dist.amin(dim=-1)

        near_pool = min(int(getattr(self.cfg.meta, "runtime_near_pool_points", 1024)), pool_size)
        # Fix #7: batched topk instead of per-sample ``topk(...).cpu()``.
        near_idx = torch.topk(
            obj_min_dist, k=near_pool, dim=1, largest=False
        ).indices  # (B, near_pool) on GPU

        # Fix #2: single batch-level sync of the topk indices. The
        # per-sample loop below runs entirely on CPU using a stable
        # per-frame seed, so the GPU→CPU transition happens exactly
        # once per batch instead of once per sample.
        near_idx_cpu = near_idx.cpu()

        seeds = batch.get("object_seed")
        if torch.is_tensor(seeds):
            seeds_list = [int(v) for v in seeds.reshape(-1).tolist()]
        else:
            seeds_list = list(range(batch_size))
        selected_all_cpu: list[torch.Tensor] = []
        valid_all_cpu: list[torch.Tensor] = []
        for b in range(batch_size):
            seed = seeds_list[b]
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            near_take = min(near_quota, int(near_idx_cpu.shape[1]))
            near_perm = torch.randperm(int(near_idx_cpu.shape[1]), generator=gen)[:near_take]
            parts = [near_idx_cpu[b, near_perm]]
            valid_count = near_take
            if global_quota > 0:
                mask = torch.ones(pool_size, dtype=torch.bool)
                mask[near_idx_cpu[b]] = False
                remaining = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if remaining.numel() == 0:
                    remaining = torch.arange(pool_size, dtype=torch.long)
                global_take = min(global_quota, int(remaining.numel()))
                global_perm = torch.randperm(int(remaining.numel()), generator=gen)[:global_take]
                parts.append(remaining[global_perm])
                valid_count += global_take
            chosen_idx = torch.cat(parts, dim=0)
            if chosen_idx.numel() < num_obj:
                chosen_idx = torch.cat([chosen_idx, chosen_idx.new_zeros(num_obj - chosen_idx.numel())], dim=0)
            selected_all_cpu.append(chosen_idx[:num_obj])
            valid = torch.zeros(num_obj, dtype=torch.bool)
            valid[: min(valid_count, num_obj)] = True
            valid_all_cpu.append(valid)

        selected = torch.stack(selected_all_cpu, dim=0).to(device=device)
        valid_mask = torch.stack(valid_all_cpu, dim=0).to(device=device)

        def gather_pool(pool: torch.Tensor) -> torch.Tensor:
            idx = selected.unsqueeze(-1).expand(-1, -1, pool.shape[-1])
            return torch.gather(pool, dim=1, index=idx)

        obj_points = gather_pool(full_input_obj).to(batch["points"].dtype)
        obj_normals = gather_pool(full_input_normals).to(batch["normals"].dtype)
        gt_obj_points = gather_pool(full_gt_obj).to(batch["gt_points"].dtype)
        gt_obj_normals = gather_pool(full_gt_normals).to(batch["gt_normals"].dtype)

        batch["points"] = torch.cat([obj_points, hand_input.to(batch["points"].dtype)], dim=1)
        batch["normals"] = torch.cat([obj_normals, hand_input_normals.to(batch["normals"].dtype)], dim=1)
        batch["gt_points"] = torch.cat([gt_obj_points, hand_gt.to(batch["gt_points"].dtype)], dim=1)
        batch["gt_normals"] = torch.cat([gt_obj_normals, hand_gt_normals.to(batch["gt_normals"].dtype)], dim=1)
        batch["runtime_obj_valid_mask"] = valid_mask
        batch["runtime_obj_selected_idx"] = selected
        batch["point_valid_mask"] = torch.cat(
            [valid_mask, torch.ones((batch_size, num_hand), dtype=torch.bool, device=device)],
            dim=1,
        )

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        preds = model(batch)
        diagnostic_every = max(int(getattr(self.cfg.train, "diagnostic_every_steps", 20)), 1)
        compute_diagnostics = mode == "eval" or ((self.global_step + 1) % diagnostic_every == 0)
        # The pseudo target is a nuisance baseline made from the perturbed
        # input geometry. It is built only for diagnostics: every validation
        # batch and the sparse training diagnostic steps. It never becomes a
        # loss or an implicit training signal.
        record_pseudo_recovery = compute_diagnostics
        if record_pseudo_recovery:
            with torch.no_grad():
                self._build_random_edge_pseudo_target(batch)
        losses, aux_metrics = self._compute_losses(
            preds,
            batch,
            compute_diagnostics=compute_diagnostics,
            record_pseudo_recovery=record_pseudo_recovery,
        )
        total_loss = sum(losses.values())
        metrics: dict[str, float | torch.Tensor | MetricStat] = {**losses, **aux_metrics}
        metrics["loss"] = total_loss
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(batch["points"].shape[0]))

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        compute_diagnostics: bool,
        record_pseudo_recovery: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor | MetricStat]]:
        meta = self.cfg.meta
        obj_valid_mask = batch["runtime_obj_valid_mask"].bool()
        beta = float(meta.quality_focal_beta)

        # ---- Random-edge stream (L_r) ----
        random_edge_valid_mask = batch["random_edge_valid_mask"].bool() & obj_valid_mask.unsqueeze(-1)
        random_target = batch["random_edge_contact_target"].float()
        random_logits = preds["pred_cross_random_logits"]
        random_prob = preds["pred_cross_random_prob"]

        random_qfl_map = quality_focal_loss_map(random_logits, random_target, beta=beta)
        cross_edge_random_qfl = reduce_loss_map_per_object(
            random_qfl_map, random_edge_valid_mask, obj_valid_mask
        )
        recovery_terms: dict[str, torch.Tensor] | None = None
        if record_pseudo_recovery:
            recovery_terms = self._compute_pseudo_recovery_terms(
                pred_prob=random_prob,
                clean_target=random_target,
                pseudo_target=batch["random_edge_pseudo_target"].float(),
                valid_mask=random_edge_valid_mask,
                change_threshold=float(getattr(meta, "pseudo_recovery_change_threshold", 0.05)),
            )

        # ---- Contact-aware auxiliary stream (L_c) ----
        contact_edge_valid_mask = batch["contact_edge_valid_mask"].bool() & obj_valid_mask.unsqueeze(-1)
        contact_edge_target = batch["contact_edge_contact_target"].float()
        contact_logits = preds["pred_cross_contact_aux_logits"]
        contact_prob = preds["pred_cross_contact_aux_prob"]

        contact_qfl_map = quality_focal_loss_map(contact_logits, contact_edge_target, beta=beta)
        cross_edge_contact_aux_qfl = reduce_loss_map_per_object(
            contact_qfl_map, contact_edge_valid_mask, obj_valid_mask
        )

        # ---- Dense hand-contact heatmap (L_h) ----
        hand_contact_weight = float(getattr(meta, "loss_hand_contact_weight", 0.005))
        hand_target: torch.Tensor | None = None
        hand_prob: torch.Tensor | None = None
        hand_contact_bce: torch.Tensor | None = None
        if hand_contact_weight > 0.0:
            hand_target = batch["hand_contact_target"].float()
            hand_logits = preds["pred_hand_contact_logits"]
            hand_prob = preds["pred_hand_contact_prob"]
            if hand_logits.shape != hand_target.shape:
                raise ValueError(
                    "pred_hand_contact_logits and hand_contact_target must have the same "
                    f"shape, got {tuple(hand_logits.shape)} and {tuple(hand_target.shape)}."
                )
            hand_contact_bce = reduce_loss_map(
                binary_cross_entropy_with_logits_map(hand_logits, hand_target),
                torch.ones_like(hand_target, dtype=torch.bool),
            )

        aux_metrics: dict[str, float | torch.Tensor | MetricStat] = {
            "cross_edge_random_qfl": cross_edge_random_qfl,
            "contact_aux_qfl": cross_edge_contact_aux_qfl,
        }
        if hand_contact_bce is not None:
            aux_metrics["hand_contact_bce"] = hand_contact_bce
        if recovery_terms is not None:
            aux_metrics.update(self._pseudo_recovery_metrics(recovery_terms))
        if compute_diagnostics:
            random_bce_map = binary_cross_entropy_with_logits_map(random_logits, random_target)
            random_mae_map = torch.abs(random_prob - random_target)
            cross_edge_random_bce = reduce_loss_map_per_object(
                random_bce_map, random_edge_valid_mask, obj_valid_mask
            )
            cross_edge_random_mae = reduce_loss_map_per_object(
                random_mae_map, random_edge_valid_mask, obj_valid_mask
            )
            contact_soft_bce_map = binary_cross_entropy_with_logits_map(contact_logits, contact_edge_target)
            contact_mae_map = torch.abs(contact_prob - contact_edge_target)
            cross_edge_contact_aux_soft_bce = reduce_loss_map_per_object(
                contact_soft_bce_map, contact_edge_valid_mask, obj_valid_mask
            )
            cross_edge_contact_aux_bce = reduce_loss_map_per_object(
                contact_soft_bce_map, contact_edge_valid_mask, obj_valid_mask
            )
            cross_edge_contact_aux_mae = reduce_loss_map_per_object(
                contact_mae_map, contact_edge_valid_mask, obj_valid_mask
            )
            random_oracle_bce_map = binary_entropy_floor_map(random_target)
            cross_edge_random_oracle_bce = reduce_loss_map_per_object(
                random_oracle_bce_map, random_edge_valid_mask, obj_valid_mask
            )
            cross_edge_random_oracle_bce_global = reduce_loss_map(
                random_oracle_bce_map, random_edge_valid_mask
            )
            cross_edge_random_bce_global = reduce_loss_map(
                random_bce_map, random_edge_valid_mask
            )
            cross_edge_random_excess_bce = (cross_edge_random_bce - cross_edge_random_oracle_bce).detach()
            cross_edge_random_excess_bce_global = (
                cross_edge_random_bce_global - cross_edge_random_oracle_bce_global
            ).detach()
            random_sampled_nonzero_mask = (random_target > 0) & random_edge_valid_mask
            random_sampled_nonzero_edge_count = random_sampled_nonzero_mask.sum()
            random_valid_edge_count = random_edge_valid_mask.sum()
            random_per_obj_has_nonzero = random_sampled_nonzero_mask.any(dim=-1) & obj_valid_mask
            random_num_valid_obj = obj_valid_mask.sum()
            random_num_obj_with_nonzero = random_per_obj_has_nonzero.sum()
            aux_metrics.update({
                "cross_edge_random_bce": cross_edge_random_bce,
                "cross_edge_random_mae": cross_edge_random_mae,
                "cross_edge_random_oracle_bce": cross_edge_random_oracle_bce,
                "cross_edge_random_oracle_bce_global": cross_edge_random_oracle_bce_global,
                "cross_edge_random_excess_bce": cross_edge_random_excess_bce,
                "cross_edge_random_excess_bce_global": cross_edge_random_excess_bce_global,
                "random_num_valid_obj": random_num_valid_obj.float(),
                "random_num_valid_edges": random_valid_edge_count.float(),
                "random_sampled_nonzero_edge_count": random_sampled_nonzero_edge_count.float(),
                "contact_aux_bce": cross_edge_contact_aux_bce,
                "contact_aux_soft_bce": cross_edge_contact_aux_soft_bce,
                "contact_aux_mae": cross_edge_contact_aux_mae,
                "random_sampled_nonzero_edge_fraction": MetricStat(
                    total=float(random_sampled_nonzero_edge_count.detach().cpu()),
                    count=float(random_valid_edge_count.detach().cpu()),
                ),
                "random_object_nonzero_edge_coverage": MetricStat(
                    total=float(random_num_obj_with_nonzero.detach().cpu()),
                    count=float(random_num_valid_obj.detach().cpu()),
                ),
            })
            aux_metrics.update(self._compute_random_diagnostic_metrics(
                edge_target=random_target,
                edge_prob=random_prob,
                edge_valid_mask=random_edge_valid_mask,
                obj_valid_mask=obj_valid_mask,
                beta=beta,
            ))
            aux_metrics.update(self._compute_contact_aux_diagnostic_metrics(
                edge_target=contact_edge_target,
                edge_prob=contact_prob,
                edge_valid_mask=contact_edge_valid_mask,
                obj_valid_mask=obj_valid_mask,
            ))
            if hand_target is not None and hand_prob is not None:
                aux_metrics.update(self._compute_hand_contact_gt_metrics(hand_target=hand_target))
                aux_metrics.update(
                    self._compute_hand_contact_diagnostic_metrics(
                        hand_target=hand_target,
                        hand_prob=hand_prob,
                    )
                )

        losses = {
            "cross_edge_random": float(meta.loss_cross_edge_weight) * cross_edge_random_qfl,
            "cross_edge_contact_aux": float(meta.loss_contact_aux_weight) * cross_edge_contact_aux_qfl,
        }
        if hand_contact_bce is not None:
            losses["hand_contact"] = hand_contact_weight * hand_contact_bce
        return losses, aux_metrics

    def _build_supervision_gpu(
        self,
        batch: dict[str, torch.Tensor],
        *,
        contact_seed_list: list[int],
    ) -> None:
        num_obj = int(self.cfg.meta.num_obj_points)
        num_hand = int(self.cfg.meta.num_hand_points)
        gt_points = batch["gt_points"].float()
        gt_obj = gt_points[:, :num_obj]
        gt_hand = gt_points[:, num_obj : num_obj + num_hand]
        obj_valid_mask = batch["runtime_obj_valid_mask"].bool()
        random_edge_idx = batch["random_edge_idx"].long()
        random_edge_valid_mask = batch["random_edge_valid_mask"].bool()

        batch["random_edge_contact_target"] = self._build_random_edge_target(
            gt_obj=gt_obj,
            gt_hand=gt_hand,
            random_edge_idx=random_edge_idx,
            random_edge_valid_mask=random_edge_valid_mask,
        )
        contact_edge_idx, contact_edge_valid_mask, contact_edge_target = self._build_contact_supervision_edges(
            gt_obj=gt_obj,
            gt_hand=gt_hand,
            obj_valid_mask=obj_valid_mask,
            contact_seed_list=contact_seed_list,
        )
        batch["contact_edge_idx"] = contact_edge_idx
        batch["contact_edge_valid_mask"] = contact_edge_valid_mask
        batch["contact_edge_contact_target"] = contact_edge_target
        batch["hand_contact_target"] = contact_target_from_distance(
            batch["hand_min_dist"].float(),
            contact_radius=float(self.cfg.meta.contact_radius),
        ).float()

    def _build_random_edge_pseudo_target(self, batch: dict[str, torch.Tensor]) -> None:
        """Build the perturbed-geometry baseline used only by eval metrics."""
        num_obj = int(self.cfg.meta.num_obj_points)
        num_hand = int(self.cfg.meta.num_hand_points)
        input_points = batch["points"].float()
        input_obj = input_points[:, :num_obj]
        input_hand = input_points[:, num_obj : num_obj + num_hand]
        batch["random_edge_pseudo_target"] = self._build_random_edge_target(
            gt_obj=input_obj,
            gt_hand=input_hand,
            random_edge_idx=batch["random_edge_idx"].long(),
            random_edge_valid_mask=batch["random_edge_valid_mask"].bool(),
        )

    def _build_random_edge_target(
        self,
        *,
        gt_obj: torch.Tensor,
        gt_hand: torch.Tensor,
        random_edge_idx: torch.Tensor,
        random_edge_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_hand = gather_batched_knn_features(
            gt_hand,
            random_edge_idx,
            random_edge_valid_mask,
        )
        distance = torch.norm(neighbor_hand - gt_obj.unsqueeze(2), dim=-1)
        target = contact_target_from_distance(
            distance,
            contact_radius=float(self.cfg.meta.contact_radius),
        )
        return target * random_edge_valid_mask.float()

    @staticmethod
    def _compute_pseudo_recovery_terms(
        *,
        pred_prob: torch.Tensor,
        clean_target: torch.Tensor,
        pseudo_target: torch.Tensor,
        valid_mask: torch.Tensor,
        change_threshold: float,
    ) -> dict[str, torch.Tensor]:
        """Return globally reducible recovery statistics from ``docs/指导.md``.

        ``pseudo_target`` comes from directly trusting the perturbed input
        geometry, whereas ``clean_target`` comes from clean GT geometry.
        These are evaluation statistics, never a loss.
        """
        if change_threshold < 0.0:
            raise ValueError(f"pseudo_recovery_change_threshold must be non-negative, got {change_threshold}.")
        if pred_prob.shape != clean_target.shape or pseudo_target.shape != clean_target.shape:
            raise ValueError(
                "pred_prob, clean_target and pseudo_target must have identical shapes, got "
                f"{tuple(pred_prob.shape)}, {tuple(clean_target.shape)}, and {tuple(pseudo_target.shape)}."
            )
        if valid_mask.shape != clean_target.shape:
            raise ValueError(
                "valid_mask must match the target shape, got "
                f"{tuple(valid_mask.shape)} and {tuple(clean_target.shape)}."
            )

        p = pred_prob.float()
        y = clean_target.float()
        y_pseudo = pseudo_target.float()
        valid = valid_mask.bool()
        direction = y - y_pseudo
        correction = p - y_pseudo

        def sums(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            mask_f = mask.float()
            pseudo_error = (direction.square() * mask_f).sum()
            model_error = ((p - y).square() * mask_f).sum()
            projection = (correction * direction * mask_f).sum()
            return pseudo_error, model_error, projection, mask_f.sum()

        changed_terms = sums(valid & (direction.abs() > change_threshold))
        fake_terms = sums(valid & (-direction > change_threshold))
        missed_terms = sums(valid & (direction > change_threshold))
        return {
            "changed_pseudo_error": changed_terms[0],
            "changed_model_error": changed_terms[1],
            "changed_projection": changed_terms[2],
            "changed_count": changed_terms[3],
            "fake_pseudo_error": fake_terms[0],
            "fake_model_error": fake_terms[1],
            "fake_projection": fake_terms[2],
            "fake_count": fake_terms[3],
            "missed_pseudo_error": missed_terms[0],
            "missed_model_error": missed_terms[1],
            "missed_projection": missed_terms[2],
            "missed_count": missed_terms[3],
            "valid_count": valid.float().sum(),
        }

    @staticmethod
    def _pseudo_recovery_metrics(terms: dict[str, torch.Tensor]) -> dict[str, MetricStat]:
        """Convert recovery sufficient statistics into global validation ratios."""
        def stat(numerator: torch.Tensor, denominator: torch.Tensor) -> MetricStat:
            denominator_value = float(denominator.detach().cpu())
            if denominator_value <= 0.0:
                return MetricStat.invalid(expose_validity=True)
            return MetricStat(
                total=float(numerator.detach().cpu()),
                count=denominator_value,
                expose_validity=True,
            )

        return {
            # Weighting batch ratios by E_pseudo recovers the exact global
            # ratio, rather than an unstable mean of per-batch ratios.
            "pseudo_recovery_brier": stat(
                terms["changed_pseudo_error"] - terms["changed_model_error"],
                terms["changed_pseudo_error"],
            ),
            "pseudo_recovery_projection": stat(
                terms["changed_projection"], terms["changed_pseudo_error"]
            ),
            "pseudo_fake_contact_recovery_brier": stat(
                terms["fake_pseudo_error"] - terms["fake_model_error"],
                terms["fake_pseudo_error"],
            ),
            "pseudo_fake_contact_recovery_projection": stat(
                terms["fake_projection"], terms["fake_pseudo_error"]
            ),
            "pseudo_missed_contact_recovery_brier": stat(
                terms["missed_pseudo_error"] - terms["missed_model_error"],
                terms["missed_pseudo_error"],
            ),
            "pseudo_missed_contact_recovery_projection": stat(
                terms["missed_projection"], terms["missed_pseudo_error"]
            ),
            "pseudo_changed_edge_fraction": stat(terms["changed_count"], terms["valid_count"]),
        }

    def _build_contact_supervision_edges(
        self,
        *,
        gt_obj: torch.Tensor,
        gt_hand: torch.Tensor,
        obj_valid_mask: torch.Tensor,
        contact_seed_list: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_obj, _ = gt_obj.shape
        num_hand = gt_hand.shape[1]
        quotas = tuple(int(value) for value in getattr(self.cfg.meta, "contact_supervision_quotas", (16, 16, 16, 16)))
        hard_negative_quota = int(getattr(self.cfg.meta, "contact_supervision_hard_negative_quota", 16))
        neg_min, neg_max = tuple(
            float(value)
            for value in getattr(self.cfg.meta, "contact_supervision_hard_negative_distance_range", (0.02, 0.03))
        )
        total_quota = sum(quotas) + hard_negative_quota
        edge_idx = torch.full((batch_size, num_obj, total_quota), -1, device=gt_obj.device, dtype=torch.long)
        edge_valid = torch.zeros((batch_size, num_obj, total_quota), device=gt_obj.device, dtype=torch.bool)
        edge_target = torch.zeros((batch_size, num_obj, total_quota), device=gt_obj.device, dtype=torch.float32)
        seed_tensor = torch.as_tensor(contact_seed_list, device=gt_obj.device, dtype=torch.long)
        contact_radius = float(self.cfg.meta.contact_radius)

        for start in range(0, num_obj, self._CONTACT_SAMPLE_CHUNK):
            end = min(start + self._CONTACT_SAMPLE_CHUNK, num_obj)
            dist = torch.cdist(gt_obj[:, start:end], gt_hand)
            target = contact_target_from_distance(dist, contact_radius=contact_radius)
            row_valid = obj_valid_mask[:, start:end].unsqueeze(-1)
            score = self._deterministic_contact_scores(
                seed_tensor=seed_tensor,
                obj_start=start,
                obj_count=end - start,
                num_hand=num_hand,
                device=gt_obj.device,
            )
            masks = (
                row_valid & (target > self._CONTACT_POSITIVE_EPS) & (target <= 0.25),
                row_valid & (target > 0.25) & (target <= 0.50),
                row_valid & (target > 0.50) & (target <= 0.75),
                row_valid & (target > 0.75),
                row_valid & (dist >= neg_min) & (dist < neg_max) & (target <= 0.0),
            )
            widths = (*quotas, hard_negative_quota)
            write_col = 0
            for mask, width in zip(masks, widths):
                sampled_idx, sampled_valid, sampled_target = self._sample_contact_edges_from_mask(
                    mask=mask,
                    score=score,
                    target=target,
                    quota=width,
                )
                next_col = write_col + width
                edge_idx[:, start:end, write_col:next_col] = sampled_idx
                edge_valid[:, start:end, write_col:next_col] = sampled_valid
                edge_target[:, start:end, write_col:next_col] = sampled_target
                write_col = next_col
        return edge_idx, edge_valid, edge_target

    @staticmethod
    def _sample_contact_edges_from_mask(
        *,
        mask: torch.Tensor,
        score: torch.Tensor,
        target: torch.Tensor,
        quota: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_obj_chunk, _ = mask.shape
        if quota <= 0:
            empty_idx = torch.empty((batch_size, num_obj_chunk, 0), device=mask.device, dtype=torch.long)
            empty_valid = torch.empty((batch_size, num_obj_chunk, 0), device=mask.device, dtype=torch.bool)
            empty_target = torch.empty((batch_size, num_obj_chunk, 0), device=mask.device, dtype=target.dtype)
            return empty_idx, empty_valid, empty_target
        masked_score = score.masked_fill(~mask, float("inf"))
        values, idx = torch.topk(masked_score, k=quota, dim=-1, largest=False)
        valid = torch.isfinite(values)
        sampled_target = torch.gather(target, dim=-1, index=idx) * valid.float()
        return idx.masked_fill(~valid, -1), valid, sampled_target

    @staticmethod
    def _deterministic_contact_scores(
        *,
        seed_tensor: torch.Tensor,
        obj_start: int,
        obj_count: int,
        num_hand: int,
        device: torch.device,
    ) -> torch.Tensor:
        obj_idx = torch.arange(obj_start, obj_start + obj_count, device=device, dtype=torch.long).view(1, obj_count, 1)
        hand_idx = torch.arange(num_hand, device=device, dtype=torch.long).view(1, 1, num_hand)
        seed = seed_tensor.view(-1, 1, 1)
        hashed = seed ^ (obj_idx * 1000003) ^ (hand_idx * 9176)
        hashed = (hashed * 1103515245 + 12345) & 0x7FFFFFFF
        return hashed.to(torch.float32) / 2147483648.0

    def _compute_random_diagnostic_metrics(
        self,
        *,
        edge_target: torch.Tensor,
        edge_prob: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        obj_valid_mask: torch.Tensor,
        beta: float,
    ) -> dict[str, float | MetricStat]:
        """Diagnostic metrics for the random128 stream.

        These are the metrics that must remain comparable to the
        ``random128 baseline`` runs, so they MUST NOT mix in any
        contact-aux edges.
        """
        metrics: dict[str, float | MetricStat] = {}

        # ---- Nonzero edges ----
        nonzero_mask = (edge_target > 0) & edge_valid_mask
        nonzero_count = nonzero_mask.sum()
        if bool(nonzero_count > 0):
            metrics["cross_edge_random_nonzero_mae"] = (
                (edge_prob - edge_target).abs()[nonzero_mask].sum() / nonzero_count.float()
            )
            metrics["cross_edge_random_nonzero_bce"] = (
                -(
                    edge_target[nonzero_mask] * torch.log(edge_prob[nonzero_mask].clamp(min=1e-6))
                    + (1.0 - edge_target[nonzero_mask])
                    * torch.log((1.0 - edge_prob[nonzero_mask]).clamp(min=1e-6))
                ).mean()
            )
            metrics["cross_edge_random_nonzero_pred_mean"] = edge_prob[nonzero_mask].mean()
            metrics["cross_edge_random_nonzero_target_mean"] = edge_target[nonzero_mask].mean()
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["cross_edge_random_nonzero_mae"] = zero_tensor
            metrics["cross_edge_random_nonzero_bce"] = zero_tensor
            metrics["cross_edge_random_nonzero_pred_mean"] = zero_tensor
            metrics["cross_edge_random_nonzero_target_mean"] = zero_tensor
        metrics["cross_edge_random_nonzero_count"] = MetricStat(
            total=float(nonzero_count.detach().cpu()),
            count=float(1),
            expose_validity=True,
        )

        # ---- Target strength bins (count + MAE) ----
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(edge_target, lower=lower, upper=upper) & edge_valid_mask
            bin_count = bin_mask.sum()
            metrics[f"random_{name}_count"] = MetricStat(
                total=float(bin_count.detach().cpu()),
                count=float(1),
                expose_validity=True,
            )
            if bool(bin_count > 0):
                metrics[f"random_{name}_mae"] = (
                    (edge_prob - edge_target).abs()[bin_mask].sum() / bin_count.float()
                )
            else:
                metrics[f"random_{name}_mae"] = edge_target.sum() * 0.0

        # ---- Zero edge prediction distribution ----
        zero_edge_mask = (edge_target == 0) & edge_valid_mask
        zero_edge_count = zero_edge_mask.sum()
        if bool(zero_edge_count > 0):
            zero_pred = edge_prob[zero_edge_mask].float()
            metrics["cross_edge_random_zero_pred_mean"] = zero_pred.mean()
            metrics["cross_edge_random_zero_pred_p95"] = torch.quantile(zero_pred, 0.95)
            metrics["cross_edge_random_zero_pred_p99"] = torch.quantile(zero_pred, 0.99)
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["cross_edge_random_zero_pred_mean"] = zero_tensor
            metrics["cross_edge_random_zero_pred_p95"] = zero_tensor
            metrics["cross_edge_random_zero_pred_p99"] = zero_tensor

        # ---- Zero predictor baseline ----
        zero_qfl_map = zero_predictor_qfl_map(edge_target, beta=beta)
        zero_bce_map = zero_predictor_bce_map(edge_target)
        zero_mae_map = zero_predictor_mae_map(edge_target)
        metrics["zero_baseline_qfl"] = reduce_loss_map_per_object(
            zero_qfl_map, edge_valid_mask, obj_valid_mask
        )
        metrics["zero_baseline_bce"] = reduce_loss_map_per_object(
            zero_bce_map, edge_valid_mask, obj_valid_mask
        )
        metrics["zero_baseline_mae"] = reduce_loss_map_per_object(
            zero_mae_map, edge_valid_mask, obj_valid_mask
        )
        return metrics

    def _compute_contact_aux_diagnostic_metrics(
        self,
        *,
        edge_target: torch.Tensor,
        edge_prob: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> dict[str, float | MetricStat]:
        """Diagnostic metrics for the contact auxiliary stream only.

        These must NEVER be mixed with the random stream metrics: the
        contact stream is biased by construction (stratified positives
        plus a narrow hard-negative band), so diagnostic quantities must
        be interpreted on that auxiliary distribution only.
        """
        metrics: dict[str, float | MetricStat] = {}

        # Per-stream means and MAE
        if bool(edge_valid_mask.sum() > 0):
            metrics["contact_aux_pred_mean"] = edge_prob[edge_valid_mask].mean()
            metrics["contact_aux_target_mean"] = edge_target[edge_valid_mask].mean()
            metrics["contact_aux_nonzero_mae"] = (
                (edge_prob - edge_target).abs()[edge_valid_mask].sum()
                / edge_valid_mask.sum().float()
            )
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["contact_aux_pred_mean"] = zero_tensor
            metrics["contact_aux_target_mean"] = zero_tensor
            metrics["contact_aux_nonzero_mae"] = zero_tensor

        metrics["contact_aux_valid_edge_count"] = MetricStat(
            total=float(edge_valid_mask.sum().detach().cpu()),
            count=float(1),
            expose_validity=True,
        )
        hard_negative_mask = (edge_target <= 0) & edge_valid_mask
        hard_negative_count = hard_negative_mask.sum()
        metrics["contact_aux_hard_neg_count"] = MetricStat(
            total=float(hard_negative_count.detach().cpu()),
            count=float(1),
            expose_validity=True,
        )
        if bool(hard_negative_count > 0):
            hard_negative_pred = edge_prob[hard_negative_mask].float()
            metrics["contact_aux_hard_neg_pred_mean"] = hard_negative_pred.mean()
            metrics["contact_aux_hard_neg_pred_p95"] = torch.quantile(hard_negative_pred, 0.95)
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["contact_aux_hard_neg_pred_mean"] = zero_tensor
            metrics["contact_aux_hard_neg_pred_p95"] = zero_tensor
        per_obj_has_edge = edge_valid_mask.any(dim=-1) & obj_valid_mask
        num_valid_obj = obj_valid_mask.sum()
        num_obj_with_edge = per_obj_has_edge.sum()
        metrics["contact_aux_object_coverage"] = MetricStat(
            total=float(num_obj_with_edge.detach().cpu()),
            count=float(num_valid_obj.detach().cpu()),
        )

        # Per-bin count and MAE (matches the sampler order weak/medium/strong/very_strong).
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(edge_target, lower=lower, upper=upper) & edge_valid_mask
            bin_count = bin_mask.sum()
            metrics[f"contact_aux_{name}_count"] = MetricStat(
                total=float(bin_count.detach().cpu()),
                count=float(1),
                expose_validity=True,
            )
            if bool(bin_count > 0):
                metrics[f"contact_aux_{name}_mae"] = (
                    (edge_prob - edge_target).abs()[bin_mask].sum() / bin_count.float()
                )
            else:
                metrics[f"contact_aux_{name}_mae"] = edge_target.sum() * 0.0
        return metrics

    def _compute_hand_contact_gt_metrics(
        self,
        *,
        hand_target: torch.Tensor,
    ) -> dict[str, float | MetricStat]:
        """GT-only hand contact observation.

        The hand head is NOT supervised in v2.1, so any prediction-side
        metric (MAE, QFL, prob mean) would be meaningless. We only log
        the GT distribution to detect train/val domain shift and to keep
        the option open for a future detached probe on z_hand.
        """
        metrics: dict[str, float | MetricStat] = {}
        nonzero_mask = hand_target > 0
        nonzero_count = nonzero_mask.sum()
        total = float(hand_target.numel())
        if bool(nonzero_count > 0):
            metrics["hand_contact_target_nonzero_fraction"] = float(nonzero_count) / max(total, 1.0)
            metrics["hand_contact_target_nonzero_mean"] = hand_target[nonzero_mask].mean()
        else:
            zero_tensor = hand_target.sum() * 0.0
            metrics["hand_contact_target_nonzero_fraction"] = zero_tensor
            metrics["hand_contact_target_nonzero_mean"] = zero_tensor
        metrics["hand_contact_target_mean"] = hand_target.mean()
        metrics["hand_contact_target_count"] = MetricStat(
            total=total,
            count=float(1),
            expose_validity=True,
        )
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(hand_target, lower=lower, upper=upper)
            bin_count = bin_mask.sum()
            metrics[f"hand_target_{name}_fraction"] = (
                float(bin_count) / max(total, 1.0)
            )
        return metrics

    @staticmethod
    def _compute_hand_contact_diagnostic_metrics(
        *,
        hand_target: torch.Tensor,
        hand_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Prediction-side diagnostics for the object-conditioned hand heatmap."""
        if hand_prob.shape != hand_target.shape:
            raise ValueError(
                "pred_hand_contact_prob and hand_contact_target must have the same "
                f"shape, got {tuple(hand_prob.shape)} and {tuple(hand_target.shape)}."
            )
        nonzero_mask = hand_target > 0
        if bool(nonzero_mask.any()):
            nonzero_mae = (hand_prob - hand_target).abs()[nonzero_mask].mean()
            nonzero_pred_mean = hand_prob[nonzero_mask].mean()
        else:
            nonzero_mae = hand_target.sum() * 0.0
            nonzero_pred_mean = hand_target.sum() * 0.0
        return {
            "hand_contact_mae": (hand_prob - hand_target).abs().mean(),
            "hand_contact_pred_mean": hand_prob.mean(),
            "hand_contact_nonzero_mae": nonzero_mae,
            "hand_contact_nonzero_pred_mean": nonzero_pred_mean,
        }

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
