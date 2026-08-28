from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.base.artifact import Artifact
from src.base.component import Component, ComponentSpec, load_manifest
from src.base.context import ExecutionContext


class ScaleComponent(Component):
    def spec(self) -> ComponentSpec:
        return load_manifest(Path(__file__).with_name("scale") / "component.yaml")

    def execute(self, inputs: Mapping[str, Artifact], context: ExecutionContext) -> Mapping[str, Artifact]:
        factor = float(context.config.get("factor", 1.0))
        value = inputs["value"]
        return {"value": Artifact(type="scalar", value=value.value * factor, producer=self.spec().id, producer_version=self.spec().version)}
