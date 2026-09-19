#!/usr/bin/env python3
"""Verify the current branch's changed Ref2Dex files.

The gate intentionally checks the complete feature-branch diff relative to the
stable integration branch plus explicitly staged paths. It never treats an
empty index as a successful verification of an already committed branch. It
runs only hermetic tests available in a clean checkout; data/asset-dependent
Task tests belong to the run workflow with their declared manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|([^\s)\n]+))")
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
GUIDANCE_FILENAME_RE = re.compile(r"^V\d+(?:\.\d+)*[a-z]?\.md$")
HISTORICAL_LOG_PARTS = {"logs", "archive"}
RETIRED_CURRENT_FILES = (
    "modification_policy.md",
    "目录规范.md",
    "ai_task_checklist.md",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _nul_paths(value: str) -> set[str]:
    return {path for path in value.split("\0") if path}


def _stable_branch() -> str:
    for candidate in ("origin/oyx", "oyx"):
        if subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0:
            return candidate
    raise RuntimeError("无法找到稳定集成分支 origin/oyx 或 oyx")


def _changed_paths(base: str | None = None) -> set[str]:
    branch = _git("branch", "--show-current").strip()
    changed: set[str] = set()
    if base:
        changed.update(_nul_paths(_git("diff", "--name-only", "-z", "--diff-filter=ACMR", f"{base}...HEAD")))
    elif branch and branch != "oyx":
        stable = _stable_branch()
        merge_base = _git("merge-base", "HEAD", stable).strip()
        changed.update(_nul_paths(_git("diff", "--name-only", "-z", "--diff-filter=ACMR", f"{merge_base}...HEAD")))
    changed.update(_nul_paths(_git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")))
    return changed


def _repo_path(path: str) -> Path:
    return (ROOT / path).resolve()


def _is_historical(path: str) -> bool:
    return bool(HISTORICAL_LOG_PARTS.intersection(Path(path).parts))


def _is_tracked(path: Path) -> bool:
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path.as_posix()],
        cwd=ROOT,
        capture_output=True,
    ).returncode == 0


def _markdown_targets(text: str) -> Iterable[str]:
    for match in MARKDOWN_LINK_RE.finditer(text):
        yield (match.group(1) or match.group(2) or "").strip()


def _resolve_target(target: str, source: Path) -> Path | None:
    decoded = unquote(target).strip()
    if not decoded or decoded.startswith(("#", "//")) or URI_SCHEME_RE.match(decoded):
        return None
    location = decoded.split("#", 1)[0].split("?", 1)[0]
    if not location:
        return None
    candidate = Path(location)
    resolved = candidate if candidate.is_absolute() else (source.parent / candidate)
    try:
        return resolved.resolve().relative_to(ROOT)
    except ValueError:
        return Path("__outside_repository__")


def _check_python(paths: Iterable[str], failures: list[str]) -> None:
    for relative in sorted(paths):
        if not relative.endswith(".py"):
            continue
        candidate = _repo_path(relative)
        if not candidate.is_file():
            continue
        try:
            compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec")
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append(f"{relative}: Python 编译失败: {exc}")


def _check_structured(paths: Iterable[str], failures: list[str]) -> None:
    yaml_module = None
    for relative in sorted(paths):
        candidate = _repo_path(relative)
        if not candidate.is_file():
            continue
        try:
            if relative.endswith(".json"):
                json.loads(candidate.read_text(encoding="utf-8"))
            elif relative.endswith((".yaml", ".yml")):
                if yaml_module is None:
                    import yaml as yaml_module
                yaml_module.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception as exc:  # Parsing diagnostics are user-facing gate output.
            failures.append(f"{relative}: 结构化文件解析失败: {exc}")


def _check_guidance(paths: Iterable[str], failures: list[str]) -> None:
    for relative in sorted(paths):
        path = Path(relative)
        if path.suffix != ".md" or "指导" not in path.parts:
            continue
        if not GUIDANCE_FILENAME_RE.fullmatch(path.name):
            failures.append(f"{relative}: 指导文件名必须是 V<数字>[a-z].md")
            continue
        suffix = path.stem[-1]
        if suffix.isalpha() and not (ROOT / path.parent / f"{path.stem[:-1]}.md").is_file():
            failures.append(f"{relative}: 补充指导缺少数值基线文件")


def _check_markdown(paths: Iterable[str], failures: list[str]) -> None:
    for relative in sorted(paths):
        if not relative.endswith(".md") or _is_historical(relative):
            continue
        source = _repo_path(relative)
        if not source.is_file():
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{relative}: Markdown 读取失败: {exc}")
            continue
        for target in _markdown_targets(text):
            resolved = _resolve_target(target, source)
            if resolved is None:
                continue
            if resolved == Path("__outside_repository__"):
                failures.append(f"{relative}: Markdown 链接越出仓库: {target}")
            elif not (ROOT / resolved).exists():
                failures.append(f"{relative}: Markdown 链接目标不存在: {target}")
            elif resolved.suffix == ".md" and not _is_tracked(resolved):
                failures.append(f"{relative}: Markdown 文档链接目标未跟踪: {target}")


def _check_current_governance(failures: list[str]) -> None:
    import yaml

    versions = ROOT / "docs/current_versions.yaml"
    try:
        data = yaml.safe_load(versions.read_text(encoding="utf-8"))
        scopes = [data["repository"], *data["tasks"].values()]
        if not all(set(scope) == {"work_version"} and scope["work_version"] for scope in scopes):
            failures.append("docs/current_versions.yaml: 每个当前作用域只能包含非空 work_version")
    except Exception as exc:
        failures.append(f"docs/current_versions.yaml: work_version 合同无效: {exc}")

    current_documents = [
        ROOT / "AGENTS.md",
        ROOT / "docs/README.md",
        ROOT / "docs/项目总览.md",
        *ROOT.glob(".agents/skills/**/*.md"),
    ]
    for document in current_documents:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        for retired in RETIRED_CURRENT_FILES:
            if retired in text:
                failures.append(f"{document.relative_to(ROOT)}: 仍引用已删除的 {retired}")


def _test_files(directory: Path) -> list[str]:
    return sorted(path.relative_to(ROOT).as_posix() for path in directory.glob("test_*.py") if path.is_file()) if directory.is_dir() else []


def _select_tests(paths: Iterable[str]) -> list[str]:
    selected: set[str] = set()
    needs_governance = False
    needs_shared = False
    for path in paths:
        parts = Path(path).parts
        if path.startswith("tests/") and path.endswith(".py"):
            selected.add(path)
        if path.startswith("src/base/"):
            needs_shared = True
        if path == "AGENTS.md" or path.startswith(("docs/", ".agents/", ".github/")) or path == "tools/verify.py":
            needs_governance = True
    if needs_shared:
        selected.update(path for path in ("tests/test_run_manifest.py", "tests/test_framework_contracts.py") if (ROOT / path).is_file())
    if needs_governance:
        selected.update(_test_files(ROOT / "tests/governance"))
    return sorted(selected)


def _run_tests(paths: Sequence[str], failures: list[str]) -> None:
    if not paths:
        return
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        failures.append(f"pytest 失败，退出码 {result.returncode}: {(result.stdout + result.stderr)[-4000:]}")


def verify_changed(paths: set[str]) -> int:
    failures: list[str] = []
    _check_python(paths, failures)
    _check_structured(paths, failures)
    _check_guidance(paths, failures)
    _check_markdown(paths, failures)
    _check_current_governance(failures)
    if not failures:
        _run_tests(_select_tests(paths), failures)
    if failures:
        print("VERIFY FAIL")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("VERIFY PASS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed", action="store_true", required=True)
    args = parser.parse_args(argv)
    base = os.environ.get("VERIFY_BASE", "").strip() or None
    try:
        paths = _changed_paths(base)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"VERIFY FAIL: 无法读取 Git 变更: {exc}")
        return 1
    print("VERIFY PATHS:", *sorted(paths), sep="\n  ")
    return verify_changed(paths)


if __name__ == "__main__":
    raise SystemExit(main())
