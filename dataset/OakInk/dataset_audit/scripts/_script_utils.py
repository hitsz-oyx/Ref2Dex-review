from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def bootstrap(script_file: str) -> Path:
    repo_root = Path(script_file).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_adapter(repo_root: Path) -> Any:
    adapter_dir = repo_root / "dataset_audit" / "adapters"
    local_adapters = sorted(
        path.stem for path in adapter_dir.glob("*_adapter.py") if not path.name.startswith("_")
    )
    if len(local_adapters) == 1:
        return importlib.import_module(f"dataset_audit.adapters.{local_adapters[0]}")
    mapping = {
        "arctic": "arctic_adapter",
        "OakInk": "oakink_adapter",
        "OakInk2": "oakink2_adapter",
        "TACO-Instructions": "taco_adapter",
        "GRAB": "grab_adapter",
    }
    module_name = mapping.get(repo_root.name)
    if module_name is None:
        lowered = repo_root.name.lower()
        for key, value in mapping.items():
            if lowered == key.lower():
                module_name = value
                break
    if module_name is None:
        raise SystemExit(f"No dataset_audit adapter mapping for repository: {repo_root.name}")
    return importlib.import_module(f"dataset_audit.adapters.{module_name}")


def add_common_args(parser: Any) -> None:
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--unit-scale-to-mm", type=float, default=1000.0)
