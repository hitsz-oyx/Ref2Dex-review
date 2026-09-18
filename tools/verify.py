#!/usr/bin/env python3
"""Run the repository change gate.

The gate is intentionally small and deterministic. Locally it checks the
current branch's staged paths; CI supplies VERIFY_BASE and checks the branch
diff against that base. It selects the tests that own those paths and reports
one machine-readable VERIFY PASS or VERIFY FAIL result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_FILE_BYTES = 50 * 1024 * 1024
MARKDOWN_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|([^\s)\n]+))"
)
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
ACTIVITY_REQUIRED_HEADINGS = (
    "## Goal",
    "## Changes",
    "## Protected semantics",
    "## Verification",
    "## References",
    "## Result",
)
GUIDANCE_FILENAME_RE = re.compile(r"^V\d+(?:\.\d+)*([a-z]?)\.md$")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _nul_paths(text: str) -> Set[str]:
    return {value for value in text.split("\0") if value}


def _changed_paths(base: Optional[str]) -> Set[str]:
    changed: Set[str] = set()
    if base:
        changed.update(
            _nul_paths(
                _git(
                    "diff",
                    "--name-only",
                    "-z",
                    "--diff-filter=ACMR",
                    f"{base}...HEAD",
                )
            )
        )
    elif _git("branch", "--show-current").strip() != "oyx":
        merge_base = _git("merge-base", "HEAD", "oyx").strip()
        changed.update(
            _nul_paths(
                _git(
                    "diff", "--name-only", "-z", "--diff-filter=ACMR", f"{merge_base}...HEAD"
                )
            )
        )
    # Unstaged and untracked files may belong to another active task. They are
    # deliberately excluded from the local branch gate; the owner stages the
    # paths that belong to this branch before running verify.
    changed.update(
        _nul_paths(
            _git(
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--diff-filter=ACMR",
            )
        )
    )
    return changed


def _repo_path(relative_path: str) -> Path:
    return (ROOT / relative_path).resolve()


def _is_forbidden_artifact(path: str) -> Optional[str]:
    normalized = path.replace("\\", "/")
    parts = Path(normalized).parts
    lower = normalized.lower()
    if parts and parts[0] in {"output", "outputs", "runs"}:
        return "运行输出目录不得进入变更"
    if lower.startswith("data/raw_data/") or lower.startswith("data/processed_data/"):
        return "原始或处理数据目录不得进入变更"
    if "/cache/" in f"/{lower}" or "/checkpoints/" in f"/{lower}":
        return "cache/checkpoint 目录不得进入变更"
    if Path(normalized).suffix.lower() in {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".bin",
        ".log",
        ".out",
    }:
        return "checkpoint 或大型运行日志不得进入变更"
    candidate = _repo_path(path)
    if candidate.is_file() and candidate.stat().st_size > MAX_TRACKED_FILE_BYTES:
        return f"文件超过 {MAX_TRACKED_FILE_BYTES // (1024 * 1024)} MiB"
    return None


def _check_forbidden_artifacts(paths: Iterable[str], failures: List[str]) -> None:
    for path in sorted(paths):
        reason = _is_forbidden_artifact(path)
        if reason:
            failures.append(f"{path}: {reason}")


def _check_python(paths: Iterable[str], failures: List[str]) -> None:
    for path in sorted(paths):
        if not path.endswith(".py"):
            continue
        candidate = _repo_path(path)
        if not candidate.is_file():
            continue
        try:
            source = candidate.read_text(encoding="utf-8")
            compile(source, str(candidate), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append(f"{path}: Python 编译失败: {exc}")


def _check_structured_files(paths: Iterable[str], failures: List[str]) -> None:
    yaml_module = None
    for path in sorted(paths):
        candidate = _repo_path(path)
        if not candidate.is_file():
            continue
        try:
            if path.endswith(".json"):
                json.loads(candidate.read_text(encoding="utf-8"))
            elif path.endswith((".yaml", ".yml")):
                if yaml_module is None:
                    try:
                        import yaml as yaml_module
                    except ImportError as exc:
                        failures.append(
                            f"{path}: YAML 检查需要 PyYAML: {exc}"
                        )
                        continue
                yaml_module.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            failures.append(f"{path}: 结构化文件解析失败: {exc}")
        except Exception as exc:
            failures.append(f"{path}: 结构化文件解析失败: {exc}")


def _check_guidance_paths(paths: Iterable[str], failures: List[str]) -> None:
    for relative_path in sorted(paths):
        path = Path(relative_path)
        if path.suffix != ".md" or "指导" not in path.parts:
            continue
        match = GUIDANCE_FILENAME_RE.fullmatch(path.name)
        if not match:
            failures.append(f"{relative_path}: 指导文件名必须是 V<数字>[a-z].md")
            continue
        suffix = match.group(1)
        if suffix:
            base_name = f"{path.stem[:-1]}.md"
            if not (ROOT / path.parent / base_name).is_file():
                failures.append(
                    f"{relative_path}: 补充指导缺少数值基线文件 {base_name}"
                )


def _local_markdown_target(target: str, source: Path) -> Optional[Path]:
    decoded = unquote(target.strip())
    if not decoded or decoded.startswith(("#", "//")):
        return None
    if URI_SCHEME_RE.match(decoded):
        return None
    path_text = decoded.split("#", 1)[0].split("?", 1)[0]
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    resolved = path if path.is_absolute() else source.parent / path
    try:
        return resolved.resolve().relative_to(ROOT)
    except ValueError:
        return Path("__outside_repository__")


def _is_index_tracked(relative_path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_path.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _markdown_targets(text: str) -> Iterable[str]:
    for match in MARKDOWN_LINK_RE.finditer(text):
        yield (match.group(1) or match.group(2) or "").strip()


def _audit_latest_activity(path: Path, failures: List[str]) -> None:
    audit_path = ROOT / ".agents/skills/research-change-control/scripts/audit_diff.py"
    try:
        spec = importlib.util.spec_from_file_location("ref2dex_audit_diff", audit_path)
        if spec is None or spec.loader is None:
            raise ImportError("无法加载 audit_diff.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = module._latest_entry(path.read_text(encoding="utf-8"))
        _, issues = module._link_issues(entry, path)
        failures.extend(f"{path}: {issue}" for issue in issues)
    except (OSError, UnicodeError, ValueError, ImportError, AttributeError) as exc:
        failures.append(f"{path}: activity 链接审计失败: {exc}")


def _check_markdown(paths: Iterable[str], failures: List[str]) -> None:
    for relative_path in sorted(paths):
        if not relative_path.endswith(".md"):
            continue
        path = _repo_path(relative_path)
        if not path.is_file():
            continue
        if path.name == "activity_log.md":
            _audit_latest_activity(path, failures)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{relative_path}: Markdown 读取失败: {exc}")
            continue
        if _is_activity_file(relative_path):
            _check_activity_format(relative_path, text, failures)
        for target in _markdown_targets(text):
            resolved = _local_markdown_target(target, path)
            if resolved is None:
                continue
            if str(resolved) == "__outside_repository__":
                failures.append(
                    f"{relative_path}: Markdown 链接越出仓库: {target}"
                )
            elif not (ROOT / resolved).exists():
                failures.append(
                    f"{relative_path}: Markdown 链接目标不存在: {target}"
                )
            elif path.parts and path.parts[0] == "docs" and resolved.suffix == ".md" and not _is_index_tracked(resolved):
                failures.append(
                    f"{relative_path}: Git 文档链接目标未跟踪: {target}"
                )


def _is_activity_file(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    return "activities" in parts and Path(relative_path).name != "README.md"


def _check_activity_format(
    relative_path: str, text: str, failures: List[str]
) -> None:
    missing = [heading for heading in ACTIVITY_REQUIRED_HEADINGS if heading not in text]
    if missing:
        failures.append(
            f"{relative_path}: Activity 缺少章节: {', '.join(missing)}"
        )
    required_fields = ("timestamp:", "base_commit:", "branch:")
    missing_fields = [field for field in required_fields if field not in text]
    if missing_fields:
        failures.append(
            f"{relative_path}: Activity 缺少字段: {', '.join(missing_fields)}"
        )


def _task_name(path: str) -> Optional[str]:
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "task":
        if len(parts) >= 4 and parts[3] == "docs":
            return None
        return parts[2]
    for candidate in (
        "ObjectInteractionCmv2",
        "ObjectInteractionCm",
        "CmResidual",
        "CmDecoderv2",
        "CmDecoder",
        "correspondence_ptv3_v2",
        "Actiontoken",
        "Posetoken",
        "InteractionDynamics",
    ):
        if candidate in parts:
            return candidate
    return None


def _test_files(directory: Path) -> List[str]:
    if not directory.is_dir():
        return []
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in directory.glob("test_*.py")
        if path.is_file()
    )


def _select_tests(paths: Iterable[str]) -> List[str]:
    selected: Set[str] = set()
    task_names: Set[str] = set()
    needs_governance = False
    needs_process = False
    needs_shared = False

    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            selected.add(path)
        task = _task_name(path)
        if task:
            task_names.add(task)
        if path.startswith("process/"):
            needs_process = True
        if path.startswith("src/base/"):
            needs_shared = True
        if (
            path == "AGENTS.md"
            or path.startswith(".agents/")
            or path.startswith(".github/")
            or path.startswith("docs/")
            or path == "tools/verify.py"
        ):
            needs_governance = True

    for task in task_names:
        selected.update(_test_files(ROOT / "src/task" / task / "tests"))
    if needs_process:
        selected.update(_test_files(ROOT / "process/tests"))
    if needs_shared:
        shared = _test_files(ROOT / "tests/shared")
        if shared:
            selected.update(shared)
        else:
            legacy_shared = {
                "tests/test_base_runner_max_steps.py",
                "tests/test_base_runner_validation.py",
                "tests/test_component_registry.py",
                "tests/test_framework_contracts.py",
                "tests/test_run_manifest.py",
            }
            selected.update(
                path
                for path in legacy_shared
                if (ROOT / path).is_file()
            )
    if needs_governance:
        selected.update(_test_files(ROOT / "tests/governance"))
    return sorted(selected)


def _run_tests(test_paths: Sequence[str], failures: List[str]) -> None:
    if not test_paths:
        print("VERIFY TESTS: 未选择到对应测试")
        return
    print("VERIFY TESTS:")
    for path in test_paths:
        print(f"  {path}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output[-12000:])
    if result.returncode:
        failures.append(f"pytest 失败，退出码 {result.returncode}")


def _verify_changed(paths: Set[str]) -> int:
    if not paths:
        print("VERIFY: 没有检测到变更")
        print("VERIFY PASS")
        return 0
    print("VERIFY PATHS:")
    for path in sorted(paths):
        print(f"  {path}")

    failures: List[str] = []
    _check_forbidden_artifacts(paths, failures)
    _check_python(paths, failures)
    _check_structured_files(paths, failures)
    _check_guidance_paths(paths, failures)
    _check_markdown(paths, failures)
    if not failures:
        _run_tests(_select_tests(paths), failures)

    if failures:
        print("VERIFY FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("VERIFY PASS")
    return 0


def _verify_all() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
    )
    if result.returncode:
        print(f"VERIFY FAIL: pytest 失败，退出码 {result.returncode}")
        return result.returncode
    print("VERIFY PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed", action="store_true", required=True)
    args = parser.parse_args(argv)

    base = os.environ.get("VERIFY_BASE", "").strip() or None
    try:
        paths = _changed_paths(base)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        print(f"VERIFY FAIL: 无法读取 Git 变更: {detail}")
        return 1
    return _verify_changed(paths)


if __name__ == "__main__":
    raise SystemExit(main())
