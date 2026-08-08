"""Uni3D-S-shaped world encoder for InteractionDynamics V1.

The module owns deterministic patchification and exposes a strict, auditable
checkpoint loader. Training code must call ``load_pretrained`` before formal
experiments; tests may construct the encoder from random initialization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def deterministic_fps(points: torch.Tensor, count: int) -> torch.Tensor:
    """Farthest-point sampling with a deterministic centroid-farthest start."""
    if points.ndim != 3 or points.shape[-1] != 3 or count > points.shape[1]:
        raise ValueError(f"Cannot FPS {tuple(points.shape)} into {count} centers")
    centroid = points.mean(dim=1, keepdim=True)
    farthest = (points - centroid).square().sum(-1).argmax(-1)
    indices = torch.empty(points.shape[0], count, dtype=torch.long, device=points.device)
    distance = torch.full(points.shape[:2], float("inf"), device=points.device)
    batch = torch.arange(points.shape[0], device=points.device)
    for step in range(count):
        indices[:, step] = farthest
        center = points[batch, farthest]
        distance = torch.minimum(distance, (points - center[:, None]).square().sum(-1))
        farthest = distance.argmax(-1)
    return indices


def gather_points(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch[:, None, None], indices]


def patchify(points: torch.Tensor, num_patches: int, patch_size: int,
             fps_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fps_idx = deterministic_fps(points, num_patches) if fps_idx is None else fps_idx
    batch = torch.arange(points.shape[0], device=points.device)
    centers = points[batch[:, None], fps_idx]
    knn_idx = torch.cdist(centers.float(), points.float()).topk(patch_size, largest=False).indices
    return centers, fps_idx, knn_idx


class PatchEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.local = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 256),
                                   nn.GELU(), nn.Linear(256, dim))
        # RGB checkpoint channels must have no initial effect when interpreted as normals.
        nn.init.zeros_(self.local[0].weight[:, 3:])
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mean_normal = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.entity = nn.Embedding(2, dim)

    def forward(self, points: torch.Tensor, normals: torch.Tensor, centers: torch.Tensor,
                knn_idx: torch.Tensor, entity_id: int) -> torch.Tensor:
        local_points = gather_points(points, knn_idx) - centers[:, :, None]
        local_normals = gather_points(normals, knn_idx)
        feature = self.local(torch.cat([local_points, local_normals], -1)).amax(2)
        entity = self.entity.weight[entity_id][None, None]
        return feature + self.position(centers) + self.mean_normal(local_normals.mean(2)) + entity


class Uni3DPointPatchEncoder(nn.Module):
    """Official Uni3D two-stage PointNet patch encoder, with normals replacing RGB."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
        )
        self.encoder2trans = nn.Linear(512, dim)
        self.position = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dim))
        self.mean_normal = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.entity = nn.Embedding(2, dim)
        nn.init.zeros_(self.first_conv[0].weight[:, 3:])

    def forward(self, points: torch.Tensor, normals: torch.Tensor, centers: torch.Tensor,
                knn_idx: torch.Tensor, entity_id: int) -> torch.Tensor:
        local_points = gather_points(points, knn_idx) - centers[:, :, None]
        local_normals = gather_points(normals, knn_idx)
        batch, patches, size, _ = local_points.shape
        grouped = torch.cat([local_points, local_normals], -1).reshape(batch * patches, size, 6).transpose(1, 2)
        feature = self.first_conv(grouped)
        global_feature = feature.amax(2, keepdim=True).expand(-1, -1, size)
        feature = self.second_conv(torch.cat([feature, global_feature], 1)).amax(2)
        feature = self.encoder2trans(feature).reshape(batch, patches, -1)
        return (feature + self.position(centers) + self.mean_normal(local_normals.mean(2))
                + self.entity.weight[entity_id][None, None])


class Uni3DWorldEncoder(nn.Module):
    """Joint current hand/object patch transformer with stable patch indices."""
    def __init__(self, dim: int = 384, num_hand_patches: int = 64, num_obj_patches: int = 64,
                 patch_size: int = 32, depth: int = 12, heads: int = 6) -> None:
        super().__init__()
        self.dim, self.num_hand_patches, self.num_obj_patches = dim, num_hand_patches, num_obj_patches
        self.patch_size = patch_size
        self.official_uni3d = dim == 384 and depth == 12 and heads == 6
        self.patch_encoder = Uni3DPointPatchEncoder(dim) if self.official_uni3d else PatchEncoder(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))
        if self.official_uni3d:
            try:
                import timm
            except ImportError as exc:
                raise ImportError("Uni3D-S requires timm in the training environment") from exc
            visual = timm.create_model("eva02_small_patch14_224", pretrained=False, num_classes=0)
            self.transformer = visual.blocks
            self.transformer_norm = visual.fc_norm
        else:
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim))
            self.transformer_norm = nn.Identity()
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.cls_pos, std=.02)

    def forward(self, hand_points: torch.Tensor, hand_normals: torch.Tensor,
                obj_points: torch.Tensor, obj_normals: torch.Tensor) -> dict[str, torch.Tensor]:
        hand_centers, hand_fps, hand_knn = patchify(
            hand_points, self.num_hand_patches, self.patch_size)
        obj_centers, obj_fps, obj_knn = patchify(obj_points, self.num_obj_patches, self.patch_size)
        hand = self.patch_encoder(hand_points, hand_normals, hand_centers, hand_knn, 0)
        obj = self.patch_encoder(obj_points, obj_normals, obj_centers, obj_knn, 1)
        sequence = torch.cat([self.cls_token.expand(len(hand), -1, -1) + self.cls_pos, hand, obj], 1)
        if self.official_uni3d:
            for block in self.transformer:
                sequence = block(sequence)
            encoded = self.transformer_norm(sequence)[:, 1:]
        else:
            encoded = self.transformer(sequence)[:, 1:]
        return {
            "world_hand_tokens": encoded[:, :self.num_hand_patches],
            "world_obj_tokens": encoded[:, self.num_hand_patches:], "world_tokens": encoded,
            "hand_patch_centers_object": hand_centers, "obj_patch_centers_object": obj_centers,
            "hand_fps_idx": hand_fps, "hand_knn_idx": hand_knn,
            "obj_fps_idx": obj_fps, "obj_knn_idx": obj_knn,
        }

    def load_pretrained(self, checkpoint: str | Path) -> dict[str, Any]:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Required Uni3D-S checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("module", payload.get("model", payload.get("state_dict", payload)))
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported Uni3D checkpoint payload: {path}")
        if not self.official_uni3d:
            raise ValueError("Official Uni3D-S weights require dim=384, depth=12, heads=6")
        translated: dict[str, torch.Tensor] = {}
        handled_source: set[str] = set()
        direct = {
            "point_encoder.cls_token": "cls_token", "point_encoder.cls_pos": "cls_pos",
            "point_encoder.encoder2trans.weight": "patch_encoder.encoder2trans.weight",
            "point_encoder.encoder2trans.bias": "patch_encoder.encoder2trans.bias",
        }
        for source, target in direct.items():
            if source in state:
                translated[target] = state[source]; handled_source.add(source)
        for source, value in state.items():
            target = None
            if source.startswith("point_encoder.encoder.first_conv."):
                target = "patch_encoder.first_conv." + source[len("point_encoder.encoder.first_conv."):]
            elif source.startswith("point_encoder.encoder.second_conv."):
                target = "patch_encoder.second_conv." + source[len("point_encoder.encoder.second_conv."):]
            elif source.startswith("point_encoder.pos_embed."):
                target = "patch_encoder.position." + source[len("point_encoder.pos_embed."):]
            elif source.startswith("point_encoder.visual.blocks."):
                target = "transformer." + source[len("point_encoder.visual.blocks."):]
            elif source.startswith("point_encoder.visual.fc_norm."):
                target = "transformer_norm." + source[len("point_encoder.visual.fc_norm."):]
            if target is not None:
                translated[target] = value; handled_source.add(source)
        own = self.state_dict()
        compatible = {key: value for key, value in translated.items()
                      if key in own and own[key].shape == value.shape}
        unexpected = sorted(set(state) - handled_source)
        mismatched = sorted(key for key in translated if key in own and own[key].shape != translated[key].shape)
        missing = sorted(set(own) - set(compatible))
        # These belong to Uni3D contrastive/image output paths unused by the patch-token world encoder.
        allowed = ("logit_scale", "point_encoder.trans2embed.", "point_encoder.visual.pos_embed",
                   "point_encoder.visual.cls_token", "point_encoder.visual.patch_embed.")
        invalid_unexpected = [key for key in unexpected if not key.startswith(allowed)]
        if invalid_unexpected:
            raise ValueError(f"Unexpected Uni3D checkpoint keys: {invalid_unexpected[:20]}")
        self.load_state_dict(compatible, strict=False)
        # Preserve pretrained xyz channels while making the former RGB/normal channels initially zero.
        nn.init.zeros_(self.patch_encoder.first_conv[0].weight[:, 3:])
        report = {"loaded_pretrained_parameter_count": sum(v.numel() for v in compatible.values()),
                  "new_parameter_count": sum(own[k].numel() for k in missing),
                  "missing_keys": missing, "unexpected_keys": unexpected, "mismatched_keys": mismatched}
        print(f"[Uni3D-S] loaded pretrained parameter count: {report['loaded_pretrained_parameter_count']}")
        print(f"[Uni3D-S] new parameter count: {report['new_parameter_count']}")
        print(f"[Uni3D-S] missing keys: {missing}")
        print(f"[Uni3D-S] unexpected keys: {unexpected}")
        return report
