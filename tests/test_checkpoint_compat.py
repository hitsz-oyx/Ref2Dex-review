from __future__ import annotations

from pathlib import Path

from src.base.checkpoint import resolve_checkpoint_file
from src.task.correspondence_ptv3.checkpoint_compat import (
    adapt_checkpoint_payload as adapt_old_checkpoint,
    resolve_checkpoint_path,
)
from src.task.correspondence_ptv3_v2.checkpoint_compat import (
    adapt_checkpoint_payload as adapt_v2_checkpoint,
)


def test_base_resolves_only_the_current_checkpoint_layout(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "checkpoint.pt").touch()
    (legacy_root / "last").touch()

    # Base must not infer task history from generic directory names.
    assert resolve_checkpoint_file(legacy_root) == legacy_root

    current_root = tmp_path / "current"
    current_root.mkdir()
    latest = current_root / "latest.pt"
    latest.touch()
    assert resolve_checkpoint_file(current_root) == latest


def test_correspondence_task_owns_its_legacy_checkpoint_layout(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    epoch_dir = root / "000007"
    epoch_dir.mkdir(parents=True)
    legacy_checkpoint = epoch_dir / "checkpoint.pt"
    legacy_checkpoint.touch()

    assert resolve_checkpoint_path(root) == legacy_checkpoint


def test_correspondence_tasks_upgrade_old_runner_state_without_base_fallback() -> None:
    old_payload = {"best_metric": 0.25, "runner_state": {"evals_without_improvement": 3}}
    for adapt in (adapt_old_checkpoint, adapt_v2_checkpoint):
        upgraded = adapt(old_payload)
        assert upgraded is not old_payload
        assert upgraded["runner_state"]["early_stopping_metric"] == 0.25
        assert old_payload["runner_state"] == {"evals_without_improvement": 3}
