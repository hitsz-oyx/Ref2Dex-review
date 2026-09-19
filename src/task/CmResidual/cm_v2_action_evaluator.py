"""Action-conditioned Cmv2 effect evaluation for frozen DExplore residuals.

The evaluator deliberately keeps action realization outside the learned model:
controller targets are converted to nominal Inspire link poses with the pinned
URDF, surface points are swept over one 1/30 s step, and only then is Cmv2
queried.  It never consumes a post-step simulator state or a future reference
frame as an online Cmv2 input.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import types
from typing import Any

import torch

from src.task.CmResidual.cm_v2_adapter import encode_tokens

from src.task.CmDecoderv2.kinematics import QUERY_LINKS, InspireKinematics


ACTION_DIM = 18
OBJECT_POINTS = 1024
HAND_POINTS = 1538
DELTA_TIME_S = 1.0 / 30.0
EVALUATOR_SCHEMA = "cmv2_action_effect_v1"


def _compose_physical_residual(*args, **kwargs):
    """Load the vendor mapping without importing IsaacGym's task registry."""
    root = Path(__file__).resolve().parents[3] / "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual"
    package_name = "cmresidual_action_evaluator_vendor"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
        contract_spec = importlib.util.spec_from_file_location(
            f"{package_name}.contract", root / "contract.py")
        contract = importlib.util.module_from_spec(contract_spec)
        sys.modules[contract_spec.name] = contract
        contract_spec.loader.exec_module(contract)
        mapping_spec = importlib.util.spec_from_file_location(
            f"{package_name}.action_mapping", root / "action_mapping.py")
        mapping = importlib.util.module_from_spec(mapping_spec)
        sys.modules[mapping_spec.name] = mapping
        mapping_spec.loader.exec_module(mapping)
    return sys.modules[f"{package_name}.action_mapping"].compose_physical_residual(*args, **kwargs)


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}")


def _axis_angle_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta2 = axis_angle.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-16).sqrt()
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120,
                    torch.sin(theta) / theta.clamp_min(1e-8))
    b = torch.where(small, 0.5 - theta2 / 24 + theta2.square() / 720,
                    (1 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(skew)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def _rotation_angle(relative_rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    return torch.acos(cosine.clamp(-1.0, 1.0))


def object_pose_delta_to_xi(current_pose: torch.Tensor,
                            next_pose: torch.Tensor) -> torch.Tensor:
    """Convert adjacent world object poses to the local `[translation, rotvec]` effect."""
    if current_pose.shape != next_pose.shape or current_pose.shape[-2:] != (4, 4):
        raise ValueError("object poses must have matching [...,4,4] shapes")
    _finite("current_object_pose", current_pose)
    _finite("next_object_pose", next_pose)
    current_rotation = current_pose[..., :3, :3]
    current_translation = current_pose[..., :3, 3]
    inverse = torch.zeros_like(current_pose)
    inverse[..., :3, :3] = current_rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -(current_rotation.transpose(-1, -2) @ current_translation[..., None]).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    relative = inverse @ next_pose
    rotation = relative[..., :3, :3]
    skew = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                        rotation[..., 0, 2] - rotation[..., 2, 0],
                        rotation[..., 1, 0] - rotation[..., 0, 1]), -1)
    angle = _rotation_angle(rotation)
    sine = torch.linalg.vector_norm(skew, dim=-1)
    rotvec = skew * (angle / sine.clamp_min(1e-8))[..., None]
    rotvec = torch.where(angle[..., None].lt(1e-6), skew * 0.5, rotvec)
    return torch.cat((relative[..., :3, 3], rotvec), -1)


def effect_metrics(predicted_delta_xi: torch.Tensor,
                   target_delta_xi: torch.Tensor,
                   predicted_obj_flow: torch.Tensor | None = None,
                   target_obj_flow: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Measure one-step translation, rotation and optional point-flow errors."""
    if predicted_delta_xi.shape != target_delta_xi.shape or predicted_delta_xi.shape[-1] != 6:
        raise ValueError("effect transforms must have matching [...,6] shapes")
    _finite("predicted_delta_xi", predicted_delta_xi)
    _finite("target_delta_xi", target_delta_xi)
    predicted_rotation = _axis_angle_matrix(predicted_delta_xi[..., 3:])
    target_rotation = _axis_angle_matrix(target_delta_xi[..., 3:])
    relative = target_rotation.transpose(-1, -2) @ predicted_rotation
    metrics = {
        "translation_error_m": torch.linalg.vector_norm(
            predicted_delta_xi[..., :3] - target_delta_xi[..., :3], dim=-1),
        "rotation_error_rad": _rotation_angle(relative),
    }
    if (predicted_obj_flow is None) != (target_obj_flow is None):
        raise ValueError("predicted and target object flow must be provided together")
    if predicted_obj_flow is not None:
        if predicted_obj_flow.shape != target_obj_flow.shape or predicted_obj_flow.shape[-1] != 3:
            raise ValueError("object flow tensors must have matching [...,N,3] shapes")
        _finite("predicted_obj_flow", predicted_obj_flow)
        _finite("target_obj_flow", target_obj_flow)
        metrics["point_flow_error_m"] = torch.linalg.vector_norm(
            predicted_obj_flow - target_obj_flow, dim=-1).mean(-1)
    return metrics


def pairwise_rank_agreement(predicted_score: torch.Tensor,
                            target_error: torch.Tensor,
                            valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return per-batch pairwise agreement, where lower score/error is better."""
    if predicted_score.shape != target_error.shape or predicted_score.ndim != 2:
        raise ValueError("scores and target_error must have matching [B,K] shapes")
    if valid_mask is None:
        valid_mask = torch.ones_like(predicted_score, dtype=torch.bool)
    if valid_mask.shape != predicted_score.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B,K]")
    result = torch.full((predicted_score.shape[0],), float("nan"),
                        dtype=predicted_score.dtype, device=predicted_score.device)
    for batch_index in range(predicted_score.shape[0]):
        indices = torch.where(valid_mask[batch_index])[0].tolist()
        pairs = list(itertools.combinations(indices, 2))
        if not pairs:
            continue
        agreements = []
        for first, second in pairs:
            score_delta = predicted_score[batch_index, first] - predicted_score[batch_index, second]
            target_delta = target_error[batch_index, first] - target_error[batch_index, second]
            if abs(float(score_delta)) <= 1e-8 or abs(float(target_delta)) <= 1e-8:
                continue
            agreements.append((score_delta * target_delta).gt(0))
        if agreements:
            result[batch_index] = torch.stack(agreements).to(result.dtype).mean()
    return result


@dataclass(frozen=True)
class NominalHandSweep:
    """Current and candidate next hand surfaces produced by controller/FK."""

    current_points: torch.Tensor
    current_normals: torch.Tensor
    next_points: torch.Tensor
    next_normals: torch.Tensor
    source: str = "controller_fk"
    next_valid_mask: torch.Tensor | None = None

    def validate(self) -> None:
        if self.source != "controller_fk":
            raise ValueError("Cmv2 accepts only controller_fk nominal hand sweeps")
        if self.current_points.ndim != 3 or self.current_points.shape[1:] != (HAND_POINTS, 3):
            raise ValueError("current_points must be [B,1538,3]")
        if self.current_normals.shape != self.current_points.shape:
            raise ValueError("current_normals shape mismatch")
        if self.next_points.ndim != 4 or self.next_points.shape[0] != self.current_points.shape[0]:
            raise ValueError("next_points must be [B,K,1538,3]")
        if self.next_points.shape[2:] != (HAND_POINTS, 3) or self.next_normals.shape != self.next_points.shape:
            raise ValueError("next hand surface shape mismatch")
        for name, value in (("current_points", self.current_points),
                            ("current_normals", self.current_normals),
                            ("next_points", self.next_points),
                            ("next_normals", self.next_normals)):
            _finite(name, value)
        if self.next_valid_mask is not None:
            expected = self.next_points.shape[:3]
            if self.next_valid_mask.shape != expected or self.next_valid_mask.dtype != torch.bool:
                raise ValueError(f"next_valid_mask must be boolean {expected}")

    @property
    def batch_size(self) -> int:
        return int(self.current_points.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.next_points.shape[1])

    @property
    def hand_flow(self) -> torch.Tensor:
        return self.next_points - self.current_points[:, None]


def compose_nominal_targets(base_action: torch.Tensor, residual_actions: torch.Tensor,
                            current_native: torch.Tensor, lower: torch.Tensor,
                            upper: torch.Tensor, *, translation_scale_m: float,
                            rotation_scale_rad: float, finger_scale_rad: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply existing residual semantics to a [B,K,18] candidate batch."""
    if base_action.ndim != 2 or base_action.shape[-1] != ACTION_DIM:
        raise ValueError("base_action must be [B,18]")
    if residual_actions.ndim != 3 or residual_actions.shape[0] != base_action.shape[0] or residual_actions.shape[-1] != ACTION_DIM:
        raise ValueError("residual_actions must be [B,K,18]")
    if current_native.shape != base_action.shape:
        raise ValueError("current_native must match base_action [B,18]")
    batch, candidates = residual_actions.shape[:2]
    expanded_base = base_action[:, None].expand(batch, candidates, ACTION_DIM).reshape(-1, ACTION_DIM)
    expanded_current = current_native[:, None].expand(batch, candidates, ACTION_DIM).reshape(-1, ACTION_DIM)
    expanded_residual = residual_actions.reshape(-1, ACTION_DIM)
    targets, details = _compose_physical_residual(
        expanded_base, expanded_residual, expanded_current, lower, upper,
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
        finger_scale_rad=finger_scale_rad)
    return targets.view(batch, candidates, ACTION_DIM), {
        name: value.view(batch, candidates, ACTION_DIM) for name, value in details.items()
    }


def build_nominal_link_poses(kinematics: InspireKinematics,
                             native_targets: torch.Tensor,
                             query_links: tuple[str, ...] = QUERY_LINKS) -> torch.Tensor:
    """Evaluate pinned URDF FK and return [B,K,L,4,4] world link poses."""
    if native_targets.ndim not in (2, 3) or native_targets.shape[-1] != ACTION_DIM:
        raise ValueError("native_targets must be [N,18] or [B,K,18]")
    original_shape = native_targets.shape[:-1]
    flat = native_targets.reshape(-1, ACTION_DIM).detach().cpu()
    _finite("native_targets", native_targets)
    poses = []
    for row in flat:
        links = kinematics.link_transforms_native(row.numpy())
        poses.append([links[name] for name in query_links])
    result = torch.as_tensor(poses, dtype=native_targets.dtype, device=native_targets.device)
    return result.view(*original_shape, len(query_links), 4, 4)


def build_nominal_hand_sweep(surface_geometry: Any, kinematics: InspireKinematics,
                             current_native: torch.Tensor, native_targets: torch.Tensor) -> NominalHandSweep:
    """Transform the fixed hand surface through current and nominal FK poses."""
    if current_native.ndim != 2 or native_targets.ndim != 3 or current_native.shape[0] != native_targets.shape[0]:
        raise ValueError("current_native must be [B,18] and native_targets [B,K,18]")
    current_links = build_nominal_link_poses(kinematics, current_native)
    target_links = build_nominal_link_poses(kinematics, native_targets)
    current_points, current_normals = surface_geometry.hand(current_links)
    batch, candidates = native_targets.shape[:2]
    next_points, next_normals = surface_geometry.hand(
        target_links.reshape(batch * candidates, target_links.shape[2], 4, 4))
    next_points = next_points.view(batch, candidates, HAND_POINTS, 3)
    next_normals = next_normals.view(batch, candidates, HAND_POINTS, 3)
    sweep = NominalHandSweep(current_points, current_normals, next_points, next_normals)
    sweep.validate()
    return sweep


class Cmv2ActionEvaluator:
    """Evaluate K nominal action sweeps with one frozen Cmv2 model."""

    def __init__(self, adapter: Any, *, rotation_weight: float = 1.0,
                 flow_weight: float = 0.0) -> None:
        if not math.isfinite(rotation_weight) or rotation_weight < 0:
            raise ValueError("rotation_weight must be finite and non-negative")
        if not math.isfinite(flow_weight) or flow_weight < 0:
            raise ValueError("flow_weight must be finite and non-negative")
        self.adapter = adapter
        self.rotation_weight = float(rotation_weight)
        self.flow_weight = float(flow_weight)

    @staticmethod
    def _expand_target(value: torch.Tensor, batch: int, candidates: int,
                       tail_shape: tuple[int, ...]) -> torch.Tensor:
        if value.shape == (batch, *tail_shape):
            return value[:, None].expand(batch, candidates, *tail_shape)
        if value.shape == (batch, candidates, *tail_shape):
            return value
        raise ValueError(f"target must be [{batch},{','.join(map(str, tail_shape))}] or [B,K,...]")

    @torch.inference_mode()
    def evaluate(self, object_points: torch.Tensor, object_normals: torch.Tensor,
                 sweep: NominalHandSweep, *, candidate_actions: torch.Tensor | None = None,
                 candidate_valid_mask: torch.Tensor | None = None,
                 desired_delta_xi: torch.Tensor | None = None,
                 desired_obj_flow: torch.Tensor | None = None) -> dict[str, Any]:
        sweep.validate()
        batch, candidates = sweep.batch_size, sweep.candidate_count
        if object_points.shape != (batch, OBJECT_POINTS, 3) or object_normals.shape != object_points.shape:
            raise ValueError("object geometry must be [B,1024,3]")
        if object_points.device != sweep.current_points.device:
            raise ValueError("object and hand geometry must share a device")
        _finite("object_points", object_points)
        _finite("object_normals", object_normals)
        if candidate_actions is not None:
            if candidate_actions.shape != (batch, candidates, ACTION_DIM):
                raise ValueError("candidate_actions must be [B,K,18]")
            _finite("candidate_actions", candidate_actions)
        if candidate_valid_mask is None:
            candidate_valid_mask = torch.ones((batch, candidates), dtype=torch.bool,
                                              device=object_points.device)
        elif (candidate_valid_mask.shape != (batch, candidates) or
              candidate_valid_mask.dtype != torch.bool or
              candidate_valid_mask.device != object_points.device):
            raise ValueError("candidate_valid_mask must be boolean [B,K] on the geometry device")
        object_batch = object_points[:, None].expand(batch, candidates, OBJECT_POINTS, 3).reshape(-1, OBJECT_POINTS, 3)
        normal_batch = object_normals[:, None].expand(
            batch, candidates, OBJECT_POINTS, 3).reshape(-1, OBJECT_POINTS, 3)
        current = sweep.current_points[:, None].expand(batch, candidates, HAND_POINTS, 3).reshape(-1, HAND_POINTS, 3)
        current_normals = sweep.current_normals[:, None].expand(batch, candidates, HAND_POINTS, 3).reshape(-1, HAND_POINTS, 3)
        next_points = sweep.next_points.reshape(-1, HAND_POINTS, 3)
        next_normals = sweep.next_normals.reshape(-1, HAND_POINTS, 3)
        flow = next_points - current
        valid_hand = (torch.ones((batch, candidates, HAND_POINTS), dtype=torch.bool,
                                 device=flow.device) if sweep.next_valid_mask is None
                      else sweep.next_valid_mask).reshape(-1, HAND_POINTS)
        output = self.adapter.predict(object_batch, normal_batch, current, current_normals,
                                      flow, DELTA_TIME_S, valid_hand)
        result: dict[str, Any] = {
            "schema": EVALUATOR_SCHEMA,
            "candidate_valid_mask": candidate_valid_mask,
        }
        if candidate_actions is not None:
            result["candidate_actions"] = candidate_actions
        for key in ("delta_xi_root", "obj_flow_pred", "cm_tokens", "token_anchors",
                    "token_normals", "token_mass", "token_mask", "p_effect", "cm_context"):
            if key in output:
                value = output[key]
                result[key] = value.view(batch, candidates, *value.shape[1:])
        # This is the one canonical 16×40 representation consumed by the V1.15
        # actor.  It remains an inference-only Cmv2 output; no Cmv2 tensor enters
        # the critic or an optimizer through this evaluator.
        result["encoded_tokens"] = encode_tokens(output, object_batch).view(
            batch, candidates, 16, 40)
        result["predicted_delta_xi"] = result["delta_xi_root"]
        result["predicted_obj_flow"] = result["obj_flow_pred"]
        if desired_delta_xi is not None:
            target = self._expand_target(desired_delta_xi, batch, candidates, (6,))
            target_flow = (self._expand_target(desired_obj_flow, batch, candidates,
                                                (OBJECT_POINTS, 3))
                           if desired_obj_flow is not None else None)
            metrics = effect_metrics(result["delta_xi_root"], target,
                                     result.get("obj_flow_pred") if target_flow is not None else None,
                                     target_flow)
            result.update(metrics)
            result["effect_score"] = (metrics["translation_error_m"] +
                                       self.rotation_weight * metrics["rotation_error_rad"])
            if "point_flow_error_m" in metrics:
                result["effect_score"] = result["effect_score"] + self.flow_weight * metrics["point_flow_error_m"]
            result["effect_score"] = result["effect_score"].masked_fill(
                ~candidate_valid_mask, float("inf"))
        elif desired_obj_flow is not None:
            raise ValueError("desired_obj_flow requires desired_delta_xi")
        return result

    @staticmethod
    def select_best(result: dict[str, torch.Tensor]) -> torch.Tensor:
        if "effect_score" not in result:
            raise ValueError("evaluate with desired_delta_xi before selecting an action")
        scores = result["effect_score"].masked_fill(~result["candidate_valid_mask"], float("inf"))
        best = scores.argmin(dim=1)
        return torch.where(result["candidate_valid_mask"].any(1), best,
                           torch.full_like(best, -1))
