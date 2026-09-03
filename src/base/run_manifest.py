"""Small, task-agnostic provenance manifests for research runs.

The manifest records references and lightweight file metadata rather than
copying configs, caches, checkpoints, or generated results into the run
directory.  It deliberately does not calculate cryptographic hashes.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .utils import to_jsonable


_PATH_KEY_MARKERS = ("manifest", "split", "cache", "checkpoint", "root")
_PATH_KEY_NAMES = {
    "component_registry",
    "data_paths_registry",
    "pipeline_spec",
    "initial_checkpoint",
}
_EXCLUDED_PATH_KEYS = {
    "class_path",
    "path_encoder",
    "entrypoint",
    # These are provenance pointers written into the run directory itself;
    # treating them as input data creates a false self-reference.
    "run_manifest_path",
    "config_snapshot_path",
    "metadata_snapshot_path",
}


def _git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    cwd = str(repo_root or Path.cwd())

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"repository": None, "commit": None, "branch": None, "dirty": None}
    root = run("rev-parse", "--show-toplevel")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "repository": root,
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
    }


def _iter_references(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield from _iter_references(child, path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_references(child, f"{prefix}[{index}]")
        return
    if not isinstance(value, (str, Path)) or not prefix:
        return
    key = prefix.rsplit(".", 1)[-1].split("[", 1)[0].lower()
    if key in _EXCLUDED_PATH_KEYS:
        return
    if not (
        key.endswith("_path")
        or key == "path"
        or key in _PATH_KEY_NAMES
        or any(marker in key for marker in _PATH_KEY_MARKERS)
    ):
        return
    text = str(value).strip()
    if text and text.lower() not in {"none", "null", "auto"}:
        yield prefix, text


def _reference_records(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    initial_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    sources = [("config", config), ("metadata", metadata)]
    if initial_checkpoint is not None:
        sources.append(("run", {"initial_checkpoint": initial_checkpoint}))
    for source_name, source in sources:
        for key, raw in _iter_references(source):
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()
            identifier = str(candidate)
            record = records.setdefault(
                identifier,
                {"path": raw, "resolved_path": identifier, "sources": [], "exists": candidate.exists()},
            )
            record["sources"].append(f"{source_name}:{key}")
            if candidate.is_file():
                record["kind"] = "file"
                stat = candidate.stat()
                record["size_bytes"] = int(stat.st_size)
                record["mtime_ns"] = int(stat.st_mtime_ns)
            elif candidate.is_dir():
                record["kind"] = "directory"
            else:
                record["kind"] = "missing"
    return [records[key] for key in sorted(records)]


def _component_key(entry: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return a stable identity for a selected component entry."""
    return tuple(
        str(entry.get(field, ""))
        for field in ("role", "id", "version", "manifest")
    )


def _component_tree(components: Any) -> list[dict[str, Any]]:
    """Build a nested view while retaining the historical flat component list.

    The selected roots are the canonical config view.  For compatibility with
    older callers, a flat list may still be supplied alongside a parent's
    ``children`` field; duplicate child selections are removed from the tree
    roots by identity so the manifest exposes one unambiguous tree.
    """
    if not isinstance(components, (list, tuple)):
        return []
    entries = [entry for entry in components if isinstance(entry, Mapping)]
    child_keys: set[tuple[str, str, str, str]] = set()

    def collect_children(entry: Mapping[str, Any]) -> None:
        children = entry.get("children", ())
        if children is None or children == "" or children == () or children == []:
            return
        if not isinstance(children, (list, tuple)):
            raise ValueError("Component children must be a list.")
        for child in children:
            if not isinstance(child, Mapping):
                raise ValueError("Each nested Component selection must be a mapping.")
            child_keys.add(_component_key(child))
            collect_children(child)

    for entry in entries:
        collect_children(entry)

    def build_node(entry: Mapping[str, Any], stack: set[tuple[str, str, str, str]]) -> dict[str, Any]:
        key = _component_key(entry)
        if key in stack:
            raise ValueError(f"Component selection contains a cycle at {key!r}.")
        node = {name: value for name, value in entry.items() if name != "children"}
        children = entry.get("children", ())
        if children is not None and children != "" and children != () and children != []:
            node["children"] = [
                build_node(child, stack | {key})
                for child in children
            ]
        return to_jsonable(node)

    roots = [entry for entry in entries if _component_key(entry) not in child_keys]
    if not roots:
        roots = entries
    return [build_node(entry, set()) for entry in roots]


def _flatten_component_tree(tree: Any) -> list[dict[str, Any]]:
    """Return a deterministic child-first compatibility view of a component tree.

    The nested tree is the canonical provenance representation.  Older
    consumers still expect ``manifest["components"]`` to be a flat list, so
    this view is generated rather than maintained as a second config source.
    A component is emitted once by its ``role/id/version/manifest`` identity.
    """
    if not isinstance(tree, (list, tuple)):
        return []
    flattened: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        children = node.get("children", ())
        if children not in (None, ""):
            if not isinstance(children, (list, tuple)):
                raise ValueError("Component children must be a list.")
            for child in children:
                visit(child)
        key = _component_key(node)
        if key in seen:
            return
        seen.add(key)
        flattened.append({name: value for name, value in node.items() if name != "children"})

    for root in tree:
        visit(root)
    return [to_jsonable(entry) for entry in flattened]


def _first_value(*sources: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def build_run_manifest(
    *,
    task: str,
    run_name: str,
    output_dir: str | Path,
    mode: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config_source: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    config_snapshot: str | Path | None = None,
    metadata_snapshot: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact provenance record without task-specific assumptions."""
    output = Path(output_dir).resolve()
    config_data = dict(config)
    metadata_data = dict(metadata)
    modification_version = _first_value(
        config_data, metadata_data, keys=("modification_version", "revision")
    )
    operation_category = _first_value(
        config_data, metadata_data, keys=("operation_category", "category")
    )
    component_registry = _first_value(
        config_data, metadata_data, keys=("component_registry",)
    )
    components = _first_value(
        config_data,
        metadata_data,
        keys=("component_versions", "components"),
    )
    component_tree = _component_tree(components or [])
    return {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode),
        "task": str(task),
        "run_name": str(run_name),
        "modification_version": modification_version,
        "operation_category": to_jsonable(operation_category or []),
        "component_registry": component_registry,
        "output_dir": str(output),
        "config_source": None if config_source is None else str(Path(config_source).resolve()),
        "config_snapshot": str(Path(config_snapshot).resolve() if config_snapshot is not None else output / "config.json"),
        "metadata_snapshot": str(Path(metadata_snapshot).resolve() if metadata_snapshot is not None else output / "metadata.json"),
        "git": _git_provenance(None if repo_root is None else Path(repo_root).resolve()),
        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
        # ``component_tree`` is canonical; ``components`` is generated for
        # compatibility with existing tooling and is never read from a
        # duplicated top-level config list.
        "components": _flatten_component_tree(component_tree),
        "component_tree": component_tree,
        "contract": {
            key: to_jsonable(metadata_data[key])
            for key in (
                "schema_name",
                "schema_version",
                "coordinate_frame",
                "num_obj_pool",
                "num_obj_points",
                "num_hand_points",
                "dataset_split",
            )
            if key in metadata_data
        },
        "dataset_metadata": to_jsonable(metadata_data),
        "input_references": _reference_records(
            config_data, metadata_data, initial_checkpoint=initial_checkpoint
        ),
        "seed": config_data.get("train", {}).get("seed")
        if isinstance(config_data.get("train"), Mapping)
        else None,
    }


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a JSON manifest atomically enough for a single primary writer."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(to_jsonable(dict(manifest)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def write_run_summary(
    path: str | Path,
    *,
    task: str,
    run_name: str,
    output_dir: str | Path,
    mode: str,
    run_status: str,
    conclusion: str = "N/A",
    modification_version: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    global_step: int | None = None,
    epoch: int | None = None,
    best_metric: float | None = None,
    error: str | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> None:
    """Write a compact, user-readable terminal summary for one run.

    This is deliberately a terminal snapshot, not a heartbeat or live state
    file.  ``activity_log.md`` remains the event timeline; this JSON only
    makes the final run result easy to inspect and link.
    """
    target = Path(path)
    output = Path(output_dir).resolve()
    git = _git_provenance(None if repo_root is None else Path(repo_root).resolve())
    default_artifact_paths: dict[str, str | Path] = {
        "config": output / "config.json",
        "manifest": output / "run_manifest.json",
        "metrics": output / "metrics.jsonl",
        "train_log": output / "train.log",
        "best_checkpoint": output / "checkpoints" / "best.pt",
        "latest_checkpoint": output / "checkpoints" / "latest.pt",
    }
    if artifact_paths:
        default_artifact_paths.update(artifact_paths)
    payload: dict[str, Any] = {
        "summary_schema": "ref2dex.run_summary.v1",
        "task": str(task),
        "run_id": str(run_name),
        "run_name": str(run_name),
        "output_dir": str(output),
        "mode": str(mode),
        "run_status": str(run_status),
        "conclusion": str(conclusion),
        "modification_version": modification_version,
        "started_at": started_at,
        "finished_at": finished_at,
        "base_commit": git.get("commit"),
        "worktree_dirty": git.get("dirty"),
        "global_step": None if global_step is None else int(global_step),
        "epoch": None if epoch is None else int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "metrics": to_jsonable(dict(metrics or {})),
        "artifacts": {
            name: str(Path(path_value).resolve()) if Path(path_value).exists() else None
            for name, path_value in default_artifact_paths.items()
        },
    }
    if error:
        payload["error"] = str(error)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
