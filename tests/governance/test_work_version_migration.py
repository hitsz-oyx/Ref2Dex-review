from __future__ import annotations

import subprocess
from pathlib import Path

from src.base.base_config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_tracked_task_sources_and_configs_use_work_version() -> None:
    suffixes = {".py", ".yaml", ".yml", ".json", ".toml"}
    offenders = []
    tracked_paths = subprocess.run(
        ["git", "ls-files", "-z", "src/task"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for relative in tracked_paths:
        if not relative:
            continue
        path = ROOT / relative
        if path.suffix not in suffixes or not path.is_file() or "output" in path.relative_to(ROOT).parts:
            continue
        if "modification_version" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders


def test_legacy_config_field_is_read_only_compatible() -> None:
    config = load_config({"modification_version": "V0.9"})
    serialized = config.to_dict()
    assert serialized["work_version"] == "V0.9"
    assert "modification_version" not in serialized
