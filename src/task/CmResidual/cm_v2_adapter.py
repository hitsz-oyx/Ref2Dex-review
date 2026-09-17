"""Frozen ObjectInteractionCmv2 V1.3 adapter and structured effect contract."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model


SCHEMA = "cmv2_v13_rigid_16x40_v1"
CONTEXT_DIM = 640
MODEL_CONFIG = dict(hidden_width=128, num_tokens=16, use_residual=False,
                    knn_k=32, interaction_radius_m=0.02,
                    interaction_mode="swept", feature_scale_m=0.02,
                    frame_dt_s=1 / 30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_tokens(output: dict, object_points: torch.Tensor) -> torch.Tensor:
    """Encode ordered 16×40 tokens; every invalid token is exactly zero."""
    if object_points.ndim != 3 or object_points.shape[1:] != (1024, 3):
        raise ValueError("Cmv2 object_points must be [B,1024,3]")
    batch = object_points.shape[0]
    shapes = {"cm_tokens": (batch, 16, 32), "token_anchors": (batch, 16, 3),
              "token_normals": (batch, 16, 3), "token_mass": (batch, 16),
              "token_mask": (batch, 16)}
    for key, shape in shapes.items():
        if key not in output or tuple(output[key].shape) != shape:
            raise ValueError(f"Cmv2 {key} must be {shape}")
    if output["token_mask"].dtype != torch.bool:
        raise ValueError("Cmv2 token_mask must be boolean")
    if not torch.isfinite(object_points).all():
        raise ValueError("Non-finite Cmv2 object geometry")
    center = object_points.mean(1)
    radius = (object_points - center[:, None]).square().sum(-1).mean(1).sqrt()
    if not torch.isfinite(radius).all() or (radius <= 1e-8).any():
        raise ValueError("Degenerate Cmv2 object scale")
    mask = output["token_mask"]
    for key in ("cm_tokens", "token_anchors", "token_normals", "token_mass"):
        value = output[key]
        valid = mask if value.ndim == 2 else mask[..., None].expand_as(value)
        if not torch.isfinite(value[valid]).all():
            raise ValueError(f"Non-finite valid Cmv2 {key}")
    mass = output["token_mass"]
    if (mass[mask] < 0).any():
        raise ValueError("Negative Cmv2 token_mass")
    relative_anchor = (output["token_anchors"] - center[:, None]) / radius[:, None, None]
    normal = F.normalize(output["token_normals"], dim=-1, eps=1e-8)
    log_mass = torch.log1p(mass.clamp_min(0)) / math.log1p(1024)
    features = torch.cat((output["cm_tokens"], relative_anchor, normal,
                          log_mass[..., None], mask[..., None].to(object_points.dtype)), -1)
    features = torch.where(mask[..., None], features, torch.zeros_like(features))
    if features.shape != (batch, 16, 40) or not torch.isfinite(features).all():
        raise ValueError("Invalid Cmv2 token encoding")
    return features


def encode_context(output: dict, object_points: torch.Tensor) -> torch.Tensor:
    """Flatten the canonical 16×40 token encoding for legacy callers."""
    context = encode_tokens(output, object_points).flatten(1)
    batch = object_points.shape[0]
    if context.shape != (batch, CONTEXT_DIM) or not torch.isfinite(context).all():
        raise ValueError("Invalid Cmv2 context")
    return context


class FrozenCmv2Adapter:
    def __init__(self, checkpoint: str | Path, expected_sha256: str, device: str | torch.device):
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cmv2 checkpoint missing: {path}")
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("Cmv2 checkpoint requires explicit SHA256")
        self.checkpoint_sha256 = _sha256(path)
        if self.checkpoint_sha256 != expected_sha256:
            raise ValueError("Cmv2 checkpoint SHA256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("architecture_version") != "v1_3_rigid_only":
            raise ValueError("Cmv2 checkpoint architecture_version mismatch")
        if not isinstance(payload.get("model"), dict):
            raise ValueError("Cmv2 checkpoint has no model state_dict")
        self.model = ObjectInteractionCmv2V13Model(SimpleNamespace(**MODEL_CONFIG))
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = torch.device(device)

    def _validate_inputs(self, object_points: torch.Tensor, object_normals: torch.Tensor,
                         hand_points: torch.Tensor, hand_normals: torch.Tensor,
                         hand_flow: torch.Tensor, delta_time_s: float,
                         hand_valid_mask: torch.Tensor | None) -> torch.Tensor:
        batch = object_points.shape[0]
        tensors = ((object_points, (batch, 1024, 3)),
                   (object_normals, (batch, 1024, 3)),
                   (hand_points, (batch, 1538, 3)),
                   (hand_normals, (batch, 1538, 3)),
                   (hand_flow, (batch, 1538, 3)))
        for value, shape in tensors:
            if tuple(value.shape) != shape or value.device != self.device or not torch.isfinite(value).all():
                raise ValueError(f"Cmv2 input must be finite {shape} on {self.device}")
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones((batch, 1538), dtype=torch.bool, device=self.device)
        if tuple(hand_valid_mask.shape) != (batch, 1538) or hand_valid_mask.device != self.device:
            raise ValueError(f"hand_valid_mask must be {(batch, 1538)} on {self.device}")
        if hand_valid_mask.dtype != torch.bool:
            raise ValueError("hand_valid_mask must be boolean")
        if not math.isfinite(delta_time_s) or abs(delta_time_s - 1 / 30) > 1e-6:
            raise ValueError("Cmv2 delta_time_s must equal 1/30 s")
        return hand_valid_mask

    @torch.inference_mode()
    def predict(self, object_points: torch.Tensor, object_normals: torch.Tensor,
                hand_points: torch.Tensor, hand_normals: torch.Tensor,
                hand_flow: torch.Tensor, delta_time_s: float,
                hand_valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return frozen structured Cmv2 outputs for one-step effect evaluation."""
        hand_valid_mask = self._validate_inputs(
            object_points, object_normals, hand_points, hand_normals, hand_flow,
            delta_time_s, hand_valid_mask)
        batch = object_points.shape[0]
        output = self.model({"obj_points": object_points, "obj_normals": object_normals,
                             "hand_points": hand_points, "hand_normals": hand_normals,
                             "hand_flow": hand_flow,
                             "hand_valid_mask": hand_valid_mask,
                             "delta_time_s": torch.full((batch,), delta_time_s, device=self.device,
                                                        dtype=object_points.dtype)})
        output = dict(output)
        output["cm_context"] = encode_context(output, object_points)
        if not torch.isfinite(output["delta_xi_root"]).all() or not torch.isfinite(output["obj_flow_pred"]).all():
            raise FloatingPointError("Non-finite frozen Cmv2 effect output")
        return output

    @torch.inference_mode()
    def __call__(self, object_points: torch.Tensor, object_normals: torch.Tensor,
                 hand_points: torch.Tensor, hand_normals: torch.Tensor,
                 hand_flow: torch.Tensor, delta_time_s: float,
                 hand_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compatibility call returning only the deprecated diagnostic context."""
        return self.predict(object_points, object_normals, hand_points, hand_normals,
                            hand_flow, delta_time_s, hand_valid_mask)["cm_context"]
