from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ref2dex_verify", ROOT / "tools/verify.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_feature_branch_scope_contains_branch_diff_and_staged_paths(monkeypatch) -> None:
    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "ai/governance/test\n"
        if args == ("merge-base", "HEAD", "origin/oyx"):
            return "base\n"
        if "--cached" in args:
            return "docs/plan/V1.2a.md\0"
        return "tools/verify.py\0"

    monkeypatch.setattr(VERIFY, "_git", fake_git)
    monkeypatch.setattr(VERIFY, "_stable_branch", lambda: "origin/oyx")
    assert VERIFY._changed_paths() == {"tools/verify.py", "docs/plan/V1.2a.md"}


def test_guidance_addendum_requires_numeric_base(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    guide = tmp_path / "docs/指导"
    guide.mkdir(parents=True)
    (guide / "V1.2.md").write_text("base", encoding="utf-8")
    failures: list[str] = []
    VERIFY._check_guidance({"docs/指导/V1.2a.md"}, failures)
    assert not failures


def test_markdown_skips_historical_logs() -> None:
    assert VERIFY._is_historical("docs/logs/activity_log.md")
    assert VERIFY._is_historical("docs/activities/archive/old.md")
    assert not VERIFY._is_historical("docs/activities/ACT-001.md")


def test_governance_change_selects_governance_tests() -> None:
    assert "tests/governance/test_verify.py" in VERIFY._select_tests({"AGENTS.md"})
