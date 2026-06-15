from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_audit.common.mesh_io import candidate_data_roots, load_cache, read_csv, save_cache


DATASET_NAME = "OakInk2"


def build_manifest(repo_root: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache_path in sorted((repo_root / "outputs/dataset_audit/cache").glob("*.npz")):
        rows.append(_cache_row(repo_root, cache_path))
    data_roots = candidate_data_roots(repo_root, ["OAKINK2_DIR", "OAKINK2_DATASET_PREFIX"], ["OakInk-v2-hub", "data"])
    existing_roots = [path for path in data_roots if path.exists()]
    for root in existing_roots:
        for rel, status, message in (
            ("object_affordance/object_affordance.json", "affordance_metadata_available", "Affordance metadata exists; it is not per-frame geometry contact."),
            ("program/primitive_task", "primitive_task_available", "Primitive task data exists; distinguish primitive/affordance from contact."),
            ("anno_preview", "preview_annotation_available", "Preview annotations may be usable to instantiate sequence cache with the preview tool."),
        ):
            path = root / rel
            if path.exists():
                rows.append(
                    {
                        "dataset_name": DATASET_NAME,
                        "item": rel,
                        "subject": "",
                        "sequence": "",
                        "object_name": "",
                        "data_available": True,
                        "cache_path": "",
                        "source_path": _rel(repo_root, path),
                        "status": status,
                        "message": message,
                    }
                )
    if not rows:
        rows.append(
            {
                "dataset_name": DATASET_NAME,
                "item": "",
                "subject": "",
                "sequence": "",
                "object_name": "",
                "data_available": False,
                "cache_path": "",
                "source_path": "",
                "status": "data_missing",
                "message": "No OakInk-v2-hub or data directory found. Tooling will not download HuggingFace tarballs.",
            }
        )
    if limit is not None:
        rows = rows[:limit]
    summary = {
        "dataset_name": DATASET_NAME,
        "num_rows": len(rows),
        "candidate_data_roots": [str(path) for path in data_roots],
        "existing_data_roots": [str(path) for path in existing_roots],
        "contact_available": "affordance_or_primitive_metadata_only_if_present",
        "contact_source": "Affordance/primitive labels are semantic/task annotations, not per-frame geometric contact.",
    }
    return rows, summary


def prepare_sequence_cache(repo_root: Path, *, item: str, manifest_path: Path, out_cache: Path | None) -> dict[str, Any]:
    rows = read_csv(manifest_path)
    selected = next((row for row in rows if row.get("item") == item), None)
    if selected is None:
        raise SystemExit(f"item not found in manifest: {item}")
    cache_path = selected.get("cache_path") or ""
    if cache_path:
        source = repo_root / cache_path
        cache = enrich_cache(repo_root, source, load_cache(source))
        out = out_cache or source
        save_cache(out, cache)
        return {"status": "ok", "message": f"Wrote unified cache: {out}", "out_cache": _rel(repo_root, out)}
    raise SystemExit(
        "OakInk2 preview/data annotations are present but this lightweight adapter does not instantiate SMPL-X/MANO meshes. "
        "Use the official preview/toolkit inside Docker, then write a unified NPZ cache under outputs/dataset_audit/cache/."
    )


def enrich_cache(repo_root: Path, cache_path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    out = dict(cache)
    out.setdefault("dataset_name", DATASET_NAME)
    out.setdefault("source_paths", [str(cache_path)])
    out.setdefault("contact_available", False)
    out.setdefault("contact_source", "Affordance/primitive labels are not direct per-frame contact; derive geometric contact by thresholding distances.")
    return out


def object_mesh_paths(repo_root: Path, limit: int | None = None) -> list[Path]:
    roots = candidate_data_roots(repo_root, ["OAKINK2_DIR", "OAKINK2_DATASET_PREFIX"], ["OakInk-v2-hub", "data"])
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*.obj")))
    return paths[:limit] if limit is not None else paths


def inspect_contact_info(repo_root: Path, max_hits: int = 80) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "has_official_contact_info": "not_identified_as_direct_frame_contact",
        "contact_available": "affordance_or_primitive_fields_if_dataset_present",
        "contact_source": "object_affordance and primitive task metadata describe part functions/tasks; they are not frame-level contact masks.",
        "is_penetration_or_collision_test": False,
        "notes": [
            "req_preview.txt may include distance/SDF-related dependencies, but the static audit does not expose a unified penetration CLI/API.",
            "Keep affordance labels separate from distance-derived contact and signed-distance penetration.",
        ],
        "static_keyword_hits": _scan_keywords(repo_root, max_hits=max_hits),
    }


def _cache_row(repo_root: Path, cache_path: Path) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "item": cache_path.stem,
        "subject": "",
        "sequence": cache_path.stem,
        "object_name": "",
        "data_available": True,
        "cache_path": _rel(repo_root, cache_path),
        "source_path": _rel(repo_root, cache_path),
        "status": "unified_cache_available",
        "message": "Existing unified audit cache found.",
    }


def _scan_keywords(repo_root: Path, max_hits: int) -> list[dict[str, str]]:
    terms = ("contact", "proximity", "touch", "affordance", "primitive", "pysdf", "distance3d", "penetration", "collision")
    roots = [repo_root / "README.md", repo_root / "src", repo_root / "script", repo_root / "req_preview.txt"]
    return _scan_roots(repo_root, roots, terms, max_hits)


def _scan_roots(repo_root: Path, roots: list[Path], terms: tuple[str, ...], max_hits: int) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for root in roots:
        paths = [root] if root.is_file() else (sorted(root.rglob("*.py")) + sorted(root.rglob("*.md")) + sorted(root.rglob("*.txt")) if root.exists() else [])
        for path in paths:
            for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if any(term in line.lower() for term in terms):
                    hits.append({"path": _rel(repo_root, path), "line": str(line_no), "text": line.strip()[:240]})
                    if len(hits) >= max_hits:
                        return hits
    return hits


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

