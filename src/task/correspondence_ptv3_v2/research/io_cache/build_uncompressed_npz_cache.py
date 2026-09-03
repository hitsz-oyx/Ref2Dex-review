"""Build a resumable, uncompressed NPZ sidecar cache for Stage 3 data.

The source Stage 3 tree remains the canonical dataset.  Cache files preserve
the source-relative path and array schema, so ``CorrStaticDatasetV2`` can
switch between source and cache without changing sample identities or seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


CACHE_FORMAT = "ref2dex_uncompressed_npz_v2"


def _cache_file(source: Path, source_root: Path, cache_root: Path) -> Path:
    return cache_root / source.relative_to(source_root)


def _valid_existing(source: Path, cache: Path, coordinate_frame: str | None = None) -> bool:
    if not cache.is_file() or cache.stat().st_size <= 0:
        return False
    try:
        with np.load(source, allow_pickle=False) as src, np.load(cache, allow_pickle=False) as dst:
            if set(src.files) != set(dst.files):
                return False
            for key in ("raw_frame_id", "obj_points", "hand_points"):
                if key not in src.files or src[key].shape != dst[key].shape or src[key].dtype != dst[key].dtype:
                    return False
            if coordinate_frame is not None:
                if str(np.asarray(dst["coordinate_frame"]).item()) != str(coordinate_frame):
                    return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def _convert_one(args: tuple[str, str, str, bool, str | None]) -> dict[str, Any]:
    source_raw, source_root_raw, cache_root_raw, verify_existing, coordinate_frame = args
    source = Path(source_raw)
    source_root = Path(source_root_raw)
    cache_root = Path(cache_root_raw)
    cache = _cache_file(source, source_root, cache_root)
    if cache.exists() and (not verify_existing or _valid_existing(source, cache, coordinate_frame)):
        return {"status": "existing", "source": str(source), "cache": str(cache), "bytes": cache.stat().st_size}

    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    if coordinate_frame == "object":
        source_frame = str(np.asarray(payload.get("coordinate_frame", "hand_root")).item())
        if source_frame == "hand_root":
            for key in ("hand_root_pose", "obj_root_pose_world"):
                if key not in payload:
                    raise KeyError(f"{source}: missing {key} for object-frame cache")
            obj_pose = np.asarray(payload["obj_root_pose_world"], dtype=np.float64)
            hand_pose = np.asarray(payload["hand_root_pose"], dtype=np.float64)
            inv_obj = np.linalg.inv(obj_pose)
            rotation = np.einsum("tij,tjk->tik", inv_obj[:, :3, :3], hand_pose[:, :3, :3])
            translation = np.einsum(
                "tij,tj->ti",
                inv_obj[:, :3, :3],
                hand_pose[:, :3, 3] - obj_pose[:, :3, 3],
            )
            for key in ("obj_points", "hand_points"):
                points = np.asarray(payload[key], dtype=np.float64)
                payload[key] = (
                    np.einsum("tni,tji->tnj", points, rotation).astype(np.float32)
                    + translation[:, None, :].astype(np.float32)
                )
            for key in ("obj_normals", "hand_normals"):
                normals = np.asarray(payload[key], dtype=np.float64)
                transformed = np.einsum("tni,tji->tnj", normals, rotation)
                transformed /= np.clip(np.linalg.norm(transformed, axis=-1, keepdims=True), 1e-8, None)
                payload[key] = transformed.astype(np.float32)
            payload["coordinate_frame"] = np.asarray("object")
        elif source_frame != "object":
            raise ValueError(f"{source}: unsupported coordinate_frame={source_frame!r}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temporary, cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "written", "source": str(source), "cache": str(cache), "bytes": cache.stat().st_size}


def build_cache(
    *,
    source_root: Path,
    cache_root: Path,
    workers: int,
    limit_files: int,
    verify_existing: bool,
    coordinate_frame: str | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    if source_root == cache_root or source_root in cache_root.parents:
        raise ValueError("cache_root must be outside source_root so cache files are never rediscovered as source data")
    sources = sorted(source_root.rglob("*.npz"))
    if limit_files > 0:
        sources = sources[:limit_files]
    if not sources:
        raise FileNotFoundError(f"No .npz files found under {source_root}")
    cache_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    jobs = [(str(path), str(source_root), str(cache_root), verify_existing, coordinate_frame) for path in sources]
    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            results.append(_convert_one(job))
            if index % 25 == 0 or index == len(jobs):
                print(f"[cache] {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_convert_one, job): job[0] for job in jobs}
            for index, future in enumerate(as_completed(future_map), 1):
                results.append(future.result())
                if index % 25 == 0 or index == len(jobs):
                    print(f"[cache] {index}/{len(jobs)}", flush=True)

    summary = {
        "format": CACHE_FORMAT,
        "source_root": str(source_root),
        "cache_root": str(cache_root),
        "files": len(results),
        "written": sum(item["status"] == "written" for item in results),
        "existing": sum(item["status"] == "existing" for item in results),
        "bytes": sum(int(item["bytes"]) for item in results),
        "elapsed_s": time.perf_counter() - started,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "coordinate_frame": coordinate_frame,
    }
    manifest = cache_root / "cache_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--coordinate-frame", choices=("hand_root", "object"), default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_cache(
        source_root=Path(args.source_root),
        cache_root=Path(args.cache_root),
        workers=max(1, int(args.workers)),
        limit_files=max(0, int(args.limit_files)),
        verify_existing=bool(args.verify_existing),
        coordinate_frame=args.coordinate_frame,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
