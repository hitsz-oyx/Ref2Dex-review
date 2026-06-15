from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_audit.common.mesh_io import candidate_data_roots, load_cache, read_csv, save_cache


DATASET_NAME = "TACO-Instructions"


def build_manifest(repo_root: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache_path in sorted((repo_root / "outputs/dataset_audit/cache").glob("*.npz")):
        rows.append(_cache_row(repo_root, cache_path))
    list_paths = sorted((repo_root / "data_lists").glob("*.txt"))
    data_roots = candidate_data_roots(repo_root, ["TACO_DATASET_ROOT", "TACO_INSTRUCTIONS_DATASET_ROOT"], ["data", "dataset"])
    existing_roots = [path for path in data_roots if path.exists()]
    for list_path in list_paths[:1]:
        rows.append(
            {
                "dataset_name": DATASET_NAME,
                "item": _rel(repo_root, list_path),
                "subject": "",
                "sequence": "",
                "object_name": "",
                "data_available": bool(existing_roots),
                "cache_path": "",
                "source_path": _rel(repo_root, list_path),
                "status": "sequence_list_available" if existing_roots else "metadata_only_data_missing",
                "message": "Sequence list exists; dataset root and MANO/object assets are required to instantiate mesh cache.",
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
                "message": "No dataset root detected. Tooling will not download data or MANO models.",
            }
        )
    if limit is not None:
        rows = rows[:limit]
    summary = {
        "dataset_name": DATASET_NAME,
        "num_rows": len(rows),
        "candidate_data_roots": [str(path) for path in data_roots],
        "existing_data_roots": [str(path) for path in existing_roots],
        "contact_available": False,
        "contact_source": "No direct contact annotation identified; derive from hand-object distance if mesh cache is generated.",
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
        "TACO-Instructions requires dataset root, object model root, and MANO files to instantiate meshes. "
        "Use official visualization/loader logic inside Docker, then save a unified NPZ cache under outputs/dataset_audit/cache/."
    )


def enrich_cache(repo_root: Path, cache_path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    out = dict(cache)
    out.setdefault("dataset_name", DATASET_NAME)
    out.setdefault("source_paths", [str(cache_path)])
    out.setdefault("contact_available", False)
    out.setdefault("contact_source", "No direct contact annotation identified; threshold geometric distances if needed.")
    return out


def object_mesh_paths(repo_root: Path, limit: int | None = None) -> list[Path]:
    roots = candidate_data_roots(repo_root, ["TACO_OBJECT_MODEL_ROOT", "TACO_DATASET_ROOT"], ["object_models", "data", "dataset"])
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*.obj")))
    return paths[:limit] if limit is not None else paths


def inspect_contact_info(repo_root: Path, max_hits: int = 80) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "has_official_contact_info": False,
        "contact_available": "not_identified",
        "contact_source": "Static scan did not identify direct contact/penetration fields in the repository.",
        "is_penetration_or_collision_test": False,
        "notes": [
            "TACO-Instructions exposes hand-object pose/mesh visualization workflows.",
            "If no official contact field is present in downloaded annotations, derive contact from a distance threshold.",
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
    terms = ("contact", "proximity", "touch", "affordance", "penetration", "collision")
    roots = [repo_root / "README.md", repo_root / "dataset_utils", repo_root / "data_lists"]
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

