"""Cm 的最小 inference Component adapter。

该 adapter 只包装已经训练好的 ``CmFlowModel``，不接管 CmActionRunner 的
训练、dataloader、DDP、wandb 或 checkpoint 保存流程。真实执行必须显式调用
``from_checkpoint``，普通 registry discovery 不会加载模型权重。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.base.artifact import Artifact
from src.base.base_config import load_config, task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.component import Component, ComponentSpec, load_manifest
from src.base.context import ExecutionContext


_REQUIRED_INPUTS = (
    "object_points",
    "object_normals",
    "hand_points",
    "hand_normals",
    "hand_flow",
    "obj_valid_mask",
)


class CmInferenceComponent(Component):
    """将现有 Cm 模型包装为单步、无副作用的 Component。"""

    def __init__(self, model: torch.nn.Module, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        config: str | Path | Mapping[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> "CmInferenceComponent":
        """显式从 Cm checkpoint 构造 adapter。

        未传 config 时使用 checkpoint 中保存的 config；传入 config 只用于
        明确覆盖模型构造配置，checkpoint 本身不会被自动搜索或替换。
        """
        payload = load_checkpoint(checkpoint, map_location="cpu")
        if config is None:
            raw_config = payload.get("config")
            if not isinstance(raw_config, Mapping):
                raise ValueError("Cm checkpoint must contain a mapping under 'config'.")
            cfg = task_config_from_dict(dict(raw_config))
        else:
            cfg = load_config(config)

        from src.task.Cm.src.model import CmFlowModel

        model = CmFlowModel(cfg)
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, device=device)

    def spec(self) -> ComponentSpec:
        return load_manifest(Path(__file__).with_name("inference") / "component.yaml")

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        context: ExecutionContext,
    ) -> Mapping[str, Artifact]:
        del context
        missing = [name for name in _REQUIRED_INPUTS if name not in inputs]
        if missing:
            raise ValueError(f"Cm inference is missing required inputs: {', '.join(missing)}")

        batch: dict[str, torch.Tensor] = {}
        for name, artifact in inputs.items():
            if not isinstance(artifact, Artifact):
                raise TypeError(f"Cm input {name!r} must be an Artifact.")
            expected_type = self.spec().inputs.get(name)
            if expected_type is not None and artifact.type != expected_type.type:
                raise ValueError(
                    f"Cm input {name!r} has type {artifact.type!r}; "
                    f"expected {expected_type.type!r}."
                )
            if not isinstance(artifact.value, torch.Tensor):
                raise TypeError(f"Cm input {name!r} must carry a torch.Tensor value.")
            batch[name] = artifact.value.to(self.device)

        self._validate_shapes(batch)
        with torch.inference_mode():
            prediction = self.model(batch)
        try:
            representation = prediction["cm_tokens"]
            object_flow = prediction["pred_obj_flow"]
        except KeyError as exc:
            raise RuntimeError(f"Cm model output is missing {exc.args[0]!r}.") from exc

        spec = self.spec()
        return {
            "representation": Artifact(
                type="latent",
                value=representation,
                producer=spec.id,
                producer_version=spec.version,
                metadata={"coordinate_frame": "hand_root_t"},
            ),
            "object_flow": Artifact(
                type="point_flow",
                value=object_flow,
                producer=spec.id,
                producer_version=spec.version,
                metadata={
                    "unit": "meter",
                    "coordinate_frame": "hand_root_t",
                    "temporal_semantics": "t_to_t_plus_stride",
                },
            ),
        }

    @staticmethod
    def _validate_shapes(batch: Mapping[str, torch.Tensor]) -> None:
        for name in ("object_points", "object_normals", "hand_points", "hand_normals", "hand_flow"):
            value = batch[name]
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"Cm input {name!r} must have shape [B,N,3], got {tuple(value.shape)}")
        if batch["object_points"].shape[1] != 512:
            raise ValueError("Cm object_points must contain exactly 512 runtime points.")
        if batch["hand_points"].shape[1] != 1538:
            raise ValueError("Cm hand_points must contain exactly 1538 points.")
        mask = batch["obj_valid_mask"]
        if mask.ndim != 2 or mask.shape[1] != 512:
            raise ValueError(f"Cm obj_valid_mask must have shape [B,512], got {tuple(mask.shape)}")
        batch_size = batch["object_points"].shape[0]
        for name, value in batch.items():
            if value.shape[0] != batch_size:
                raise ValueError(f"Cm input {name!r} has inconsistent batch dimension.")
