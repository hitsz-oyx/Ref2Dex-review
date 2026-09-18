from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ref2dex_verify", ROOT / "tools/verify.py"
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_forbidden_artifact_paths_are_rejected() -> None:
    assert VERIFY._is_forbidden_artifact("outputs/demo/result.json")
    assert VERIFY._is_forbidden_artifact("runs/demo/train.log")
    assert VERIFY._is_forbidden_artifact("weights/model.pt")
    assert VERIFY._is_forbidden_artifact("data/processed_data/example.npz")


def test_task_changes_select_only_the_task_tests() -> None:
    selected = VERIFY._select_tests(
        {"src/task/CmResidual/v118_planner.py"}
    )
    assert "src/task/CmResidual/tests/test_v118_planner.py" in selected
    assert "src/task/ObjectInteractionCmv2/tests/test_v1_3_spatial.py" not in selected


def test_governance_changes_select_governance_tests() -> None:
    selected = VERIFY._select_tests({"AGENTS.md"})
    assert "tests/governance/test_verify.py" in selected


def test_local_changed_scope_uses_staged_paths_only(monkeypatch) -> None:
    def fake_git(*args):
        if args == ("branch", "--show-current"):
            return "oyx\n"
        if "--cached" in args:
            return "tools/verify.py\0"
        return "src/task/CmResidual/v118_planner.py\0"

    monkeypatch.setattr(VERIFY, "_git", fake_git)
    assert VERIFY._changed_paths(None) == {"tools/verify.py"}


def test_activity_format_requires_traceability_and_sections() -> None:
    failures = []
    VERIFY._check_activity_format(
        "docs/activities/V1.1-governance.md",
        "\n".join(
            [
                "timestamp: 2026-09-18 19:39:43 +0800",
                "base_commit: abc123",
                "branch: ai/governance/phase1",
                *VERIFY.ACTIVITY_REQUIRED_HEADINGS,
            ]
        ),
        failures,
    )
    assert not failures


def test_activity_format_rejects_missing_sections() -> None:
    failures = []
    VERIFY._check_activity_format("docs/activities/bad.md", "# bad", failures)
    assert failures


def test_guidance_suffix_requires_base_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    guidance = tmp_path / "docs" / "指导"
    guidance.mkdir(parents=True)
    (guidance / "V1.1.md").write_text("base", encoding="utf-8")
    (guidance / "V1.1a.md").write_text("supplement", encoding="utf-8")
    failures = []
    VERIFY._check_guidance_paths({"docs/指导/V1.1a.md"}, failures)
    assert not failures


def test_guidance_suffix_without_base_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    failures = []
    VERIFY._check_guidance_paths({"docs/指导/V1.1a.md"}, failures)
    assert failures


def test_feature_branch_scope_includes_branch_diff_and_staged_paths(monkeypatch) -> None:
    def fake_git(*args):
        if args == ("branch", "--show-current"):
            return "ai/governance/V1.2\n"
        if args == ("merge-base", "HEAD", "oyx"):
            return "base\n"
        if "--cached" in args:
            return "docs/plan/V1.2.md\0"
        return "tools/verify.py\0"

    monkeypatch.setattr(VERIFY, "_git", fake_git)
    assert VERIFY._changed_paths(None) == {"tools/verify.py", "docs/plan/V1.2.md"}
