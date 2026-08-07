from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


_DATASET_ALIASES = {
    "grab": "grab",
    "arctic": "arctic",
    "contactpose": "contactpose",
    "contact_pose": "contactpose",
    "contact-pose": "contactpose",
}


def normalize_dataset_id(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return _DATASET_ALIASES.get(text, text)


def infer_stage3_dataset_id(data: Any) -> str:
    """Resolve a stable dataset identifier, including legacy Stage 3 files."""
    files = data.files if hasattr(data, "files") else data.keys()
    for key in ("dataset_id", "dataset_name"):
        if key in files:
            return normalize_dataset_id(np.asarray(data[key]).item())
    if "seq_id" in files:
        seq_id = str(np.asarray(data["seq_id"]).item())
        prefix = seq_id.split(":", 1)[0].lower() if ":" in seq_id else ""
        if prefix in _DATASET_ALIASES:
            return normalize_dataset_id(prefix)

    use_pca = bool(np.asarray(data.get("mano_use_pca", True)).item())
    pose = np.asarray(data.get("mano_pose", np.empty((0, 0))))
    pose_dim = int(np.asarray(data.get("mano_num_pca_comps", pose.shape[-1])).item())
    flat = bool(np.asarray(data.get("mano_flat_hand_mean", True)).item())
    signature = (use_pca, pose_dim, flat)
    inferred = {
        (True, 24, True): "grab",
        (False, 45, False): "arctic",
        (True, 15, False): "contactpose",
    }.get(signature)
    if inferred is None:
        raise ValueError(
            "Stage 3 sample has no dataset_id/dataset_name and its MANO signature "
            f"cannot be inferred: use_pca={use_pca}, pose_dim={pose_dim}, flat={flat}."
        )
    return inferred


def profile_descriptor(
    *, side: str, use_pca: bool, num_components: int, flat_hand_mean: bool
) -> str:
    representation = "pca" if use_pca else "axis_angle"
    mean_name = "flat" if flat_hand_mean else "mean"
    return f"{side}_{representation}{int(num_components)}_{mean_name}"


@dataclass(frozen=True)
class HandGeometryNoiseProfile:
    dataset_id: str
    descriptor: str
    std_per_dimension: tuple[float, ...]
    clip_sigma: float
    target_median_rms_mm: float
    source_path: str

    def std_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(self.std_per_dimension, device=device, dtype=dtype)


class HandGeometryNoiseProfiles:
    def __init__(self, profiles: Mapping[tuple[str, str], HandGeometryNoiseProfile]):
        self._profiles = dict(profiles)

    @classmethod
    def from_paths(
        cls, paths: Mapping[str, str | Path] | None
    ) -> "HandGeometryNoiseProfiles":
        profiles: dict[tuple[str, str], HandGeometryNoiseProfile] = {}
        for raw_dataset_id, raw_path in dict(paths or {}).items():
            dataset_id = normalize_dataset_id(raw_dataset_id)
            path = Path(raw_path).expanduser().resolve()
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "ref2dex_mano_geometry_noise_calibration_v1":
                raise ValueError(f"Unsupported hand-noise calibration schema: {path}")
            for result in payload.get("results", []):
                descriptor = str(result["descriptor"])
                std = tuple(
                    float(value)
                    for value in result["balanced_coefficient_std_per_dimension"]
                )
                num_components = int(result["num_components"])
                if len(std) != num_components:
                    raise ValueError(
                        f"{path}: {descriptor} has {len(std)} std values, expected "
                        f"{num_components}"
                    )
                key = (dataset_id, descriptor)
                if key in profiles:
                    raise ValueError(f"Duplicate hand-noise profile {key} in {path}")
                profiles[key] = HandGeometryNoiseProfile(
                    dataset_id=dataset_id,
                    descriptor=descriptor,
                    std_per_dimension=std,
                    clip_sigma=float(result["clip_sigma"]),
                    target_median_rms_mm=float(result["target_median_rms_mm"]),
                    source_path=str(path),
                )
        return cls(profiles)

    def get(
        self,
        *,
        dataset_id: str,
        side: str,
        use_pca: bool,
        num_components: int,
        flat_hand_mean: bool,
    ) -> HandGeometryNoiseProfile | None:
        descriptor = profile_descriptor(
            side=side,
            use_pca=use_pca,
            num_components=num_components,
            flat_hand_mean=flat_hand_mean,
        )
        return self._profiles.get((normalize_dataset_id(dataset_id), descriptor))

    def __bool__(self) -> bool:
        return bool(self._profiles)
