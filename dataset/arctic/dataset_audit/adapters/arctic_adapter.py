from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dataset_audit.common.mesh_io import load_cache, load_obj_mesh, read_csv, save_cache, write_json


DATASET_NAME = "ARCTIC"


def build_manifest(repo_root: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meshcat_caches = sorted((repo_root / "outputs/meshcat_cache").glob("*_world_verts.npz"))
    raw_root = repo_root / "data/arctic_data/data/raw_seqs"
    raw_sequences = sorted(raw_root.glob("*/*.mano.npy")) if raw_root.exists() else []
    seen: set[str] = set()
    for cache_path in meshcat_caches:
        subject, sequence = _parse_cache_name(cache_path)
        item = f"{subject}/{sequence}"
        seen.add(item)
        rows.append(
            {
                "dataset_name": DATASET_NAME,
                "item": item,
                "subject": subject,
                "sequence": sequence,
                "object_name": _infer_object_name(repo_root, sequence) or "",
                "data_available": True,
                "cache_path": _rel(repo_root, cache_path),
                "source_path": _rel(repo_root, cache_path),
                "status": "cache_available",
                "message": "Existing ARCTIC MeshCat world-verts cache can be enriched into unified cache.",
            }
        )
    for raw_path in raw_sequences:
        subject = raw_path.parent.name
        sequence = raw_path.name.replace(".mano.npy", "")
        item = f"{subject}/{sequence}"
        if item in seen:
            continue
        rows.append(
            {
                "dataset_name": DATASET_NAME,
                "item": item,
                "subject": subject,
                "sequence": sequence,
                "object_name": _infer_object_name(repo_root, sequence) or "",
                "data_available": True,
                "cache_path": "",
                "source_path": _rel(repo_root, raw_path),
                "status": "raw_sequence_available",
                "message": "Run prepare_sequence_cache after official/local vertex processing has produced mesh vertices.",
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
                "message": "No outputs/meshcat_cache/*_world_verts.npz or data/arctic_data/data/raw_seqs/*/*.mano.npy found.",
            }
        )
    if limit is not None:
        rows = rows[:limit]
    summary = {
        "dataset_name": DATASET_NAME,
        "num_rows": len(rows),
        "num_cache_available": sum(1 for row in rows if row.get("status") == "cache_available"),
        "num_raw_sequence_available": sum(1 for row in rows if row.get("status") == "raw_sequence_available"),
        "num_data_missing": sum(1 for row in rows if row.get("status") == "data_missing"),
        "contact_available": False,
        "contact_source": "InterField distances can derive contact with a distance threshold; no direct manual contact labels.",
    }
    return rows, summary


def prepare_sequence_cache(
    repo_root: Path,
    *,
    item: str,
    manifest_path: Path,
    out_cache: Path | None,
) -> dict[str, Any]:
    rows = read_csv(manifest_path)
    selected = next((row for row in rows if row.get("item") == item), None)
    if selected is None:
        raise SystemExit(f"item not found in manifest: {item}")
    cache_text = selected.get("cache_path") or ""
    if not cache_text:
        raise SystemExit(
            "This ARCTIC item has no existing mesh cache. Generate the local ARCTIC world-verts cache first, "
            "then rerun prepare_sequence_cache."
        )
    source_cache = repo_root / cache_text
    cache = enrich_cache(repo_root, source_cache, load_cache(source_cache))
    subject = selected.get("subject") or "unknown_subject"
    sequence = selected.get("sequence") or "unknown_sequence"
    if out_cache is None:
        out_cache = repo_root / "outputs/dataset_audit/cache" / f"arctic_{subject}_{sequence}_mesh_sequence.npz"
    save_cache(out_cache, cache)
    result = {
        "status": "ok",
        "dataset_name": DATASET_NAME,
        "item": item,
        "source_cache": _rel(repo_root, source_cache),
        "out_cache": _rel(repo_root, out_cache),
        "summary_path": repo_root / "outputs/dataset_audit/cache" / f"arctic_{subject}_{sequence}_metadata.json",
        "message": f"Wrote unified cache: {out_cache}",
    }
    write_json(result["summary_path"], {key: value for key, value in result.items() if key != "summary_path"})
    return result


def enrich_cache(repo_root: Path, cache_path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    out = dict(cache)
    subject, sequence = _parse_cache_name(cache_path)
    object_name = str(out.get("object_name") or _infer_object_name(repo_root, sequence) or "")
    out.setdefault("dataset_name", DATASET_NAME)
    out.setdefault("subject", subject)
    out.setdefault("sequence", sequence)
    out.setdefault("object_name", object_name)
    out.setdefault("source_paths", [str(cache_path)])
    out["contact_available"] = False
    out["contact_source"] = "InterField/contact is geometry-distance-derived when processed fields are available; not force contact or penetration."
    if "faces_object" not in out and object_name and "verts_object" in out:
        try:
            _, faces = _find_object_template_faces(repo_root, object_name, int(out["verts_object"].shape[1]))
            out["faces_object"] = faces
        except Exception as exc:
            out["object_faces_error"] = f"{type(exc).__name__}: {exc}"
    return out


def object_mesh_paths(repo_root: Path, limit: int | None = None) -> list[Path]:
    root = repo_root / "data/arctic_data/data/meta/object_vtemplates"
    paths = sorted(root.glob("*/*.obj")) if root.exists() else []
    return paths[:limit] if limit is not None else paths


def inspect_contact_info(repo_root: Path, max_hits: int = 80) -> dict[str, Any]:
    hits = _scan_keywords(repo_root, max_hits=max_hits)
    return {
        "dataset_name": DATASET_NAME,
        "has_official_contact_info": False,
        "contact_available": "distance_derived_if_interfield_outputs_are_generated",
        "contact_source": [
            "dist.ro: right hand vertices to object distance",
            "dist.lo: left hand vertices to object distance",
            "dist.or: object vertices to right hand distance",
            "dist.ol: object vertices to left hand distance",
            "contact is commonly derived as dist < contact_bnd, e.g. 3 mm in ARCTIC code paths",
        ],
        "is_penetration_or_collision_test": False,
        "notes": [
            "ARCTIC InterField distances are nearest-distance/proximity fields, not signed inside/outside SDF.",
            "They are not force contact labels and should not be treated as penetration depth.",
        ],
        "static_keyword_hits": hits,
    }


def _parse_cache_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    stem = stem.removesuffix("_world_verts")
    match = re.match(r"^(s\d+)_(.+)$", stem)
    if match:
        return match.group(1), match.group(2)
    return "unknown_subject", stem


def _infer_object_name(repo_root: Path, sequence: str) -> str | None:
    root = repo_root / "data/arctic_data/data/meta/object_vtemplates"
    if root.exists():
        candidates = sorted([path.name for path in root.iterdir() if path.is_dir()], key=len, reverse=True)
        for name in candidates:
            if sequence == name or sequence.startswith(f"{name}_"):
                return name
    for token in ("_use_", "_grab_"):
        if token in sequence:
            return sequence.split(token)[0]
    return sequence.split("_")[0] if sequence else None


def _find_object_template_faces(repo_root: Path, object_name: str, vertex_count: int) -> tuple[Path, Any]:
    root = repo_root / "data/arctic_data/data/meta/object_vtemplates" / object_name
    if not root.exists():
        raise FileNotFoundError(f"object template directory missing: {root}")
    candidates = sorted(root.glob("*.obj"))
    fallback: tuple[Path, Any] | None = None
    for path in candidates:
        vertices, faces = load_obj_mesh(path)
        if fallback is None:
            fallback = (path, faces)
        if len(vertices) == vertex_count:
            return path, faces
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"no OBJ templates found for object: {object_name}")


def _scan_keywords(repo_root: Path, max_hits: int) -> list[dict[str, str]]:
    terms = ("contact", "interfield", "dist2contact", "contact_bnd", "penetration", "collision")
    roots = [repo_root / "README.md", repo_root / "README_ARCTIC.md", repo_root / "docs", repo_root / "src", repo_root / "scripts_method"]
    hits: list[dict[str, str]] = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py")) + sorted(root.rglob("*.md")) if root.exists() else []
        for path in paths:
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    low = line.lower()
                    if any(term in low for term in terms):
                        hits.append({"path": _rel(repo_root, path), "line": str(line_no), "text": line.strip()[:240]})
                        if len(hits) >= max_hits:
                            return hits
            except OSError:
                continue
    return hits


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

