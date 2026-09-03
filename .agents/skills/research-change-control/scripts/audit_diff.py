#!/usr/bin/env python3
"""检查 activity log 条目是否与选定的 Git 差异一致。

本脚本只审计文档一致性，不推断科研风险。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote


_MARKDOWN_LINK_RE = re.compile(
    r"!?\[([^\]\n]*)\]\(\s*(?:<([^>\n]+)>|([^\s)\n]+))"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def _untracked_paths() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def _entries(text: str) -> list[str]:
    matches = list(re.finditer(r"^##\s+", text, flags=re.MULTILINE))
    if not matches:
        raise ValueError("activity log 中没有二级标题条目")
    return [
        text[match.start() : matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        for index, match in enumerate(matches)
    ]


def _field(entry: str, name: str) -> str | None:
    match = re.search(rf"^-\s+{re.escape(name)}:\s*(.+?)\s*$", entry, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _latest_entry(text: str) -> str:
    """按字段时间选择最新事件，而不是假定日志已经倒序排列。"""
    entries = _entries(text)
    timestamped: list[tuple[datetime, str]] = []
    for entry in entries:
        value = _field(entry, "timestamp")
        if value is None:
            continue
        try:
            timestamp = datetime.strptime(value.strip("` "), "%Y-%m-%d %H:%M:%S %z")
        except ValueError:
            continue
        timestamped.append((timestamp, entry))
    if timestamped:
        return max(timestamped, key=lambda item: item[0])[1]
    return entries[0]


def _markdown_links(text: str) -> list[tuple[str, str, str]]:
    """返回 ``(label, target, source_line)``，兼容尖括号包裹的本地路径。"""
    links: list[tuple[str, str, str]] = []
    for match in _MARKDOWN_LINK_RE.finditer(text):
        target = match.group(2) or match.group(3)
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end < 0:
            line_end = len(text)
        links.append((match.group(1).strip(), target.strip(), text[line_start:line_end]))
    return links


def _local_link_path(target: str, log_path: Path) -> Path | None:
    """将文档相对链接解析为本地路径；外部 URL 和纯锚点返回 ``None``。"""
    decoded = unquote(target.strip())
    if not decoded or decoded.startswith(("#", "//")) or _URI_SCHEME_RE.match(decoded):
        return None
    path_text = decoded.split("#", 1)[0].split("?", 1)[0]
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else log_path.resolve().parent / path


def _link_issues(entry: str, log_path: Path) -> tuple[int, list[str]]:
    """检查最新条目中的本地 Markdown 链接是否可导航。"""
    local_count = 0
    issues: list[str] = []
    run_status = (_field(entry, "run_status") or "").strip("` ").upper()
    for label, target, source_line in _markdown_links(entry):
        resolved = _local_link_path(target, log_path)
        if resolved is None:
            continue
        local_count += 1
        if Path(unquote(target.split("#", 1)[0].split("?", 1)[0])).is_absolute():
            issues.append(f"不允许绝对本地链接：[{label}]({target})")
            continue
        if resolved.exists():
            continue
        pending = run_status in {"STARTED", "RUNNING"} and "PENDING" in source_line.upper()
        if not pending:
            issues.append(f"目标不存在：[{label}]({target}) -> {resolved.resolve()}")
    if local_count == 0:
        issues.append("最新 activity 条目没有本地 Markdown 链接")
    return local_count, issues


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _repo_relative_path(path: Path, root: Path) -> str:
    """Normalize a repository path, retaining a suffix for directory grouping."""
    resolved = path.resolve()
    relative = resolved.relative_to(root).as_posix()
    return relative + "/" if resolved.is_dir() and not relative.endswith("/") else relative


def _listed_paths(entry: str, log_path: Path) -> set[str]:
    """Normalize both repository-root paths and Markdown-relative paths."""
    root = _repo_root()
    log_parent = log_path.resolve().parent
    paths: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", entry):
        value = match.group(1).strip()
        if "/" in value or value.endswith((".py", ".yaml", ".yml", ".json", ".md")):
            candidate = Path(value).expanduser()
            candidates = [candidate] if candidate.is_absolute() else [root / candidate, log_parent / candidate]
            for path in candidates:
                try:
                    paths.add(_repo_relative_path(path, root))
                except ValueError:
                    continue
    for _, target, _ in _markdown_links(entry):
        path = _local_link_path(target, log_path)
        if path is None:
            continue
        try:
            paths.add(_repo_relative_path(path, root))
        except ValueError:
            continue
    return paths


def _path_is_listed(path: str, listed: set[str]) -> bool:
    """支持用带尾斜杠的目录前缀归组记录中的多个文件。"""
    return path in listed or any(
        value.endswith("/") and path.startswith(value) for value in listed
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="项目 activity log 路径")
    parser.add_argument("--staged", action="store_true", help="只审计已暂存路径")
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="审计整个工作树：已暂存、未暂存和未跟踪路径",
    )
    parser.add_argument(
        "--scope-prefix",
        action="append",
        default=[],
        help="只审计指定路径前缀；可重复传入，适合 dirty worktree 中的局部任务",
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="校验最新 activity 条目中的本地 Markdown 链接目标",
    )
    args = parser.parse_args()
    if args.staged and args.worktree:
        parser.error("--staged 和 --worktree 只能选择一个")

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"错误：找不到 activity log：{log_path}", file=sys.stderr)
        return 1
    repo_root = _repo_root()
    if args.staged:
        changed = set(_git_paths("--cached"))
    elif args.worktree:
        changed = set(_git_paths("HEAD"))
        changed.update(_untracked_paths())
    else:
        changed = set(_git_paths())
    if args.scope_prefix:
        prefixes = tuple(value.rstrip("/") + "/" for value in args.scope_prefix)
        changed = {
            path for path in changed
            if any(path == prefix[:-1] or path.startswith(prefix) for prefix in prefixes)
        }
    try:
        changed.discard(log_path.resolve().relative_to(repo_root).as_posix())
    except ValueError:
        changed.discard(log_path.as_posix())
    try:
        entry = _latest_entry(log_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    required = ("activity_id", "modification_version", "timestamp", "base_commit", "change_level", "approval", "branch", "scope")
    missing = [name for name in required if not _field(entry, name)]
    if not re.search(r"\*\*(?:Validation|验证)\*\*", entry, flags=re.IGNORECASE):
        missing.append("验证段")
    if not re.search(r"\*\*(?:Reason|原因)\*\*", entry, flags=re.IGNORECASE):
        missing.append("原因段")
    listed = _listed_paths(entry, log_path)
    unlisted = sorted(path for path in changed if not _path_is_listed(path, listed))
    link_count = 0
    link_issues: list[str] = []
    if args.check_links:
        link_count, link_issues = _link_issues(entry, log_path)
    if missing:
        print("错误：activity log 缺少：" + ", ".join(missing), file=sys.stderr)
    if unlisted:
        print("错误：最新 activity 条目未列出以下变更路径：", file=sys.stderr)
        for path in unlisted:
            print(f"  {path}", file=sys.stderr)
    if link_issues:
        print("错误：最新 activity 条目的本地 Markdown 链接不可导航：", file=sys.stderr)
        for issue in link_issues:
            print(f"  {issue}", file=sys.stderr)
    if missing or unlisted or link_issues:
        return 1
    link_result = f"；本地链接 {link_count} 个可导航" if args.check_links else ""
    print(
        f"通过：activity log 与 {len(changed)} 个变更路径一致{link_result}；"
        "等级只记录，不由脚本推断。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
