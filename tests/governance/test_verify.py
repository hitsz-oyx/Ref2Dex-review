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


def test_base_change_selects_only_hermetic_shared_tests() -> None:
    selected = VERIFY._select_tests({"src/base/run_manifest.py"})

    assert "tests/test_run_manifest.py" in selected
    assert "tests/test_framework_contracts.py" not in selected


def test_unresolved_directory_path_contract_is_discoverable() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    directory_skill = (ROOT / ".agents/skills/directory-and-artifacts/SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "目录合同不能决定新内容的位置" in agents
    assert "向用户确认位置" in directory_skill
    assert "是否应把该例外提升为本" in directory_skill


def test_worktree_lifecycle_is_discoverable() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    change_control = (ROOT / ".agents/skills/research-change-control/SKILL.md").read_text(
        encoding="utf-8"
    )

    for document in (agents, change_control):
        assert "复用一个干净的 AI worktree" in document
        assert "无未跟踪内容且用户明确授权后" in document
        assert "删除分支" in document
        assert "确认" in document
