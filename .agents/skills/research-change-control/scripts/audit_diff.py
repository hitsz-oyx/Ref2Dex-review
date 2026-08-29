#!/usr/bin/env python3
"""Check that a modification-log entry matches a selected Git diff.

This script audits documentation consistency only; it does not infer scientific risk.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def _git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _latest_entry(text: str) -> str:
    matches = list(re.finditer(r"^##\s+", text, flags=re.MULTILINE))
    if not matches:
        raise ValueError("modification log has no level-2 entry heading")
    return text[matches[0].start() :]


def _field(entry: str, name: str) -> str | None:
    match = re.search(rf"^-\s+{re.escape(name)}:\s*(.+?)\s*$", entry, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _listed_paths(entry: str) -> set[str]:
    paths: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", entry):
        value = match.group(1).strip()
        if "/" in value or value.endswith((".py", ".yaml", ".yml", ".json", ".md")):
            paths.add(value)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Project modification log path")
    parser.add_argument("--staged", action="store_true", help="Audit staged paths only")
    parser.add_argument("--worktree", action="store_true", help="Audit unstaged worktree paths")
    args = parser.parse_args()
    if args.staged and args.worktree:
        parser.error("choose only one of --staged or --worktree")

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"ERROR modification log not found: {log_path}", file=sys.stderr)
        return 1
    diff_args = ["--cached"] if args.staged else []
    changed = set(_git_paths(*diff_args))
    if args.worktree:
        changed |= set(_git_paths())
    changed.discard(log_path.as_posix())
    entry = _latest_entry(log_path.read_text(encoding="utf-8"))
    required = ("change_level", "approval", "branch", "scope")
    missing = [name for name in required if not _field(entry, name)]
    if not re.search(r"\*\*(?:Validation|验证)\*\*", entry, flags=re.IGNORECASE):
        missing.append("Validation section")
    listed = _listed_paths(entry)
    unlisted = sorted(path for path in changed if path not in listed)
    if missing:
        print("ERROR modification log is missing: " + ", ".join(missing), file=sys.stderr)
    if unlisted:
        print("ERROR changed paths absent from latest modification entry:", file=sys.stderr)
        for path in unlisted:
            print(f"  {path}", file=sys.stderr)
    if missing or unlisted:
        return 1
    print(f"OK modification log matches {len(changed)} changed path(s); level is recorded, not inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
