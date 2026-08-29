#!/usr/bin/env python3
"""检查 modification log 条目是否与选定的 Git 差异一致。

本脚本只审计文档一致性，不推断科研风险。
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
        raise ValueError("modification log 中没有二级标题条目")
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
    parser.add_argument("--log", required=True, help="项目 modification log 路径")
    parser.add_argument("--staged", action="store_true", help="只审计已暂存路径")
    parser.add_argument("--worktree", action="store_true", help="审计未暂存工作树路径")
    args = parser.parse_args()
    if args.staged and args.worktree:
        parser.error("--staged 和 --worktree 只能选择一个")

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"错误：找不到 modification log：{log_path}", file=sys.stderr)
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
        missing.append("验证段")
    listed = _listed_paths(entry)
    unlisted = sorted(path for path in changed if path not in listed)
    if missing:
        print("错误：modification log 缺少：" + ", ".join(missing), file=sys.stderr)
    if unlisted:
        print("错误：最新 modification 条目未列出以下变更路径：", file=sys.stderr)
        for path in unlisted:
            print(f"  {path}", file=sys.stderr)
    if missing or unlisted:
        return 1
    print(f"通过：modification log 与 {len(changed)} 个变更路径一致；等级只记录，不由脚本推断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
