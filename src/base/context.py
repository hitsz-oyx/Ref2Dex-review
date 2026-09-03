"""Component 执行上下文。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class ExecutionContext:
    """组件共享的运行时信息；不绑定 PyTorch。"""

    run_id: str = "local"
    output_dir: Path = Path("outputs")
    seed: int = 42
    device: str = "cpu"
    config: Mapping[str, Any] = field(default_factory=dict)
    logger: Any = None
    artifact_store: Any = None
    code_revision: Optional[str] = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
