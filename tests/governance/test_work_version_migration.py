from __future__ import annotations

from pathlib import Path

from src.base.base_config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_active_task_sources_and_configs_use_work_version() -> None:
    suffixes = {".py", ".yaml", ".yml", ".json", ".toml"}
    offenders = []
    for path in (ROOT / "src/task").rglob("*"):
        if path.suffix not in suffixes or not path.is_file():
            continue
        if "modification_version" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders


def test_legacy_config_field_is_read_only_compatible() -> None:
    config = load_config({"modification_version": "V0.9"})
    serialized = config.to_dict()
    assert serialized["work_version"] == "V0.9"
    assert "modification_version" not in serialized
