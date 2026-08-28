from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.base.artifact import Artifact
from src.base.component import Component, ComponentSpec, load_manifest
from src.base.context import ExecutionContext


class IdentityComponent(Component):
    def spec(self) -> ComponentSpec:
        return load_manifest(Path(__file__).with_name("identity") / "component.yaml")

    def execute(self, inputs: Mapping[str, Artifact], context: ExecutionContext) -> Mapping[str, Artifact]:
        del context
        value = inputs["value"]
        return {"value": Artifact(type="scalar", value=value.value, producer=self.spec().id, producer_version=self.spec().version)}
