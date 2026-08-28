"""Pipeline 声明和拓扑工具。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml


class PipelineError(ValueError):
    """Pipeline YAML 不合法。"""


@dataclass(frozen=True)
class PipelineNode:
    id: str
    component: str


@dataclass(frozen=True)
class PipelineEdge:
    source: str
    target: str


@dataclass(frozen=True)
class PipelineSpec:
    id: str
    version: str
    nodes: Tuple[PipelineNode, ...]
    edges: Tuple[PipelineEdge, ...]
    path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: Path | None = None) -> "PipelineSpec":
        pipeline_id = raw.get("id", path.stem if path is not None else "pipeline")
        version = raw.get("version", "0.1.0")
        if not isinstance(pipeline_id, str) or not pipeline_id.strip():
            raise PipelineError("Pipeline id must be a non-empty string.")
        if not isinstance(version, str) or not version.strip():
            raise PipelineError("Pipeline version must be a non-empty string.")
        raw_nodes = raw.get("nodes", [])
        raw_edges = raw.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise PipelineError("Pipeline nodes and edges must be lists.")
        nodes: List[PipelineNode] = []
        seen = set()
        for node in raw_nodes:
            if not isinstance(node, Mapping) or not isinstance(node.get("id"), str) or not isinstance(node.get("component"), str):
                raise PipelineError("Each pipeline node requires string id and component fields.")
            if node["id"] in seen:
                raise PipelineError(f"Duplicate pipeline node id: {node['id']}")
            seen.add(node["id"])
            nodes.append(PipelineNode(node["id"], node["component"]))
        edges: List[PipelineEdge] = []
        for edge in raw_edges:
            if not isinstance(edge, Mapping):
                raise PipelineError("Each pipeline edge must be a mapping.")
            source = edge.get("from")
            target = edge.get("to")
            if not isinstance(source, str) or not isinstance(target, str):
                raise PipelineError("Pipeline edge references must use 'node.port' strings.")
            if source.count(".") != 1 or target.count(".") != 1:
                raise PipelineError("Pipeline edge references must use 'node.port' strings.")
            edges.append(PipelineEdge(source, target))
        return cls(pipeline_id, version, tuple(nodes), tuple(edges), path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineSpec":
        pipeline_path = Path(path)
        if not pipeline_path.is_file():
            raise FileNotFoundError(pipeline_path)
        raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise PipelineError(f"Pipeline must contain a mapping: {pipeline_path}")
        return cls.from_mapping(raw, pipeline_path.resolve())

    def topological_order(self) -> Tuple[str, ...]:
        node_ids = {node.id for node in self.nodes}
        adjacency: Dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        indegree = {node_id: 0 for node_id in node_ids}
        for edge in self.edges:
            source, _ = edge.source.split(".")
            target, _ = edge.target.split(".")
            if source not in node_ids or target not in node_ids:
                raise PipelineError(f"Edge references unknown node: {edge.source} -> {edge.target}")
            if target not in adjacency[source]:
                adjacency[source].add(target)
                indegree[target] += 1
        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: List[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for target in sorted(adjacency[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(node_ids):
            raise PipelineError("Pipeline graph contains a cycle.")
        return tuple(order)
