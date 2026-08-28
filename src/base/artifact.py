"""任务无关的 Artifact 描述。

Artifact 可以指内存中的值，也可以指已经落盘的文件。框架只保存 provenance
元数据，不主动解释具体科学领域的内容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class Artifact:
    type: str
    value: Any = None
    schema_version: Optional[str] = None
    path: Optional[Path] = None
    producer: Optional[str] = None
    producer_version: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("Artifact.type must be a non-empty string.")
        if self.path is not None:
            self.path = Path(self.path)

    @property
    def is_materialized(self) -> bool:
        return self.value is not None or (self.path is not None and self.path.exists())

    def describe(self) -> dict[str, Any]:
        """返回可写入 run manifest 的 JSON 友好摘要。"""
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "path": str(self.path) if self.path is not None else None,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactRef:
    """Pipeline 中引用 Artifact 的轻量值对象。"""

    type: str
    schema_version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
