import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/research-change-control/scripts/audit_diff.py"
)
SPEC = importlib.util.spec_from_file_location("research_change_audit_diff", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT_DIFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_DIFF)


def _entry(timestamp: str, body: str) -> str:
    return f"""## {timestamp} — test

- activity_id: ACT-{timestamp}
- timestamp: {timestamp} +0800
- run_status: COMPLETED

{body}
"""


def test_latest_entry_uses_timestamp_instead_of_document_order():
    older = _entry("2026-09-01 10:00:00", "older")
    newer = _entry("2026-09-02 10:00:00", "newer")

    assert "newer" in AUDIT_DIFF._latest_entry(older + newer)
    assert "older" not in AUDIT_DIFF._latest_entry(older + newer)


def test_link_check_resolves_document_relative_targets(tmp_path):
    log_dir = tmp_path / "docs/logs"
    log_dir.mkdir(parents=True)
    target = tmp_path / "outputs/task/run/run_manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    log_path = log_dir / "activity_log.md"
    entry = _entry(
        "2026-09-02 10:00:00",
        "[outputs/task/run/run_manifest.json](../../outputs/task/run/run_manifest.json)",
    )

    count, issues = AUDIT_DIFF._link_issues(entry, log_path)

    assert count == 1
    assert issues == []


def test_link_check_reports_missing_target(tmp_path):
    log_path = tmp_path / "docs/logs/activity_log.md"
    log_path.parent.mkdir(parents=True)
    entry = _entry(
        "2026-09-02 10:00:00",
        "[outputs/task/missing.json](../../outputs/task/missing.json)",
    )

    count, issues = AUDIT_DIFF._link_issues(entry, log_path)

    assert count == 1
    assert len(issues) == 1
    assert "目标不存在" in issues[0]


def test_running_entry_can_mark_future_target_pending(tmp_path):
    log_path = tmp_path / "docs/logs/activity_log.md"
    log_path.parent.mkdir(parents=True)
    entry = _entry(
        "2026-09-02 10:00:00",
        "PENDING: [outputs/task/summary.json](../../outputs/task/summary.json)",
    ).replace("run_status: COMPLETED", "run_status: RUNNING")

    count, issues = AUDIT_DIFF._link_issues(entry, log_path)

    assert count == 1
    assert issues == []


def test_external_and_anchor_links_are_not_local_targets(tmp_path):
    log_path = tmp_path / "activity_log.md"
    entry = _entry(
        "2026-09-02 10:00:00",
        "[web](https://example.com) [section](#section)",
    )

    count, issues = AUDIT_DIFF._link_issues(entry, log_path)

    assert count == 0
    assert issues == ["最新 activity 条目没有本地 Markdown 链接"]


def test_directory_link_groups_changed_paths(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "new.md").write_text("new", encoding="utf-8")
    log_path = docs_dir / "logs/activity_log.md"
    log_path.parent.mkdir(parents=True)
    entry = _entry(
        "2026-09-02 10:00:00",
        "[docs/](../)",
    )
    monkeypatch.setattr(AUDIT_DIFF, "_repo_root", lambda: root)

    listed = AUDIT_DIFF._listed_paths(entry, log_path)

    assert "docs/" in listed
    assert AUDIT_DIFF._path_is_listed("docs/new.md", listed)
