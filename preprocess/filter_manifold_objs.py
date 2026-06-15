#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _inspect_non_triangle_faces_obj(obj_path: Path) -> Dict[str, Any]:
    raw_face_count = 0
    non_triangle_face_count = 0

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("f "):
                raw_face_count += 1
                nverts = len(stripped.split()[1:])
                if nverts != 3:
                    non_triangle_face_count += 1

    return {
        "raw_face_count": raw_face_count,
        "has_non_triangle_faces": non_triangle_face_count > 0,
        "non_triangle_face_count": non_triangle_face_count,
    }


def _load_obj_triangles_o3d(obj_path: Path):
    import open3d as o3d
    from pytorch3d.io import load_obj

    verts, faces, _ = load_obj(str(obj_path), load_textures=False)
    verts_np = verts.detach().cpu().numpy().astype(np.float64, copy=False)
    faces_np = faces.verts_idx.detach().cpu().numpy().astype(np.int32, copy=False)

    if verts_np.size == 0 or faces_np.size == 0:
        raise ValueError("mesh empty or failed to parse triangles")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    return mesh


def _load_and_cleanup_obj_mesh(obj_path: Path):
    mesh = _load_obj_triangles_o3d(obj_path)

    if mesh.is_empty() or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("mesh empty after loading")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    return mesh


def _write_obj_from_o3d_mesh(mesh, dst_path: Path) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if vertices.size == 0 or faces.size == 0:
        raise ValueError("cleaned mesh has no valid vertices/faces")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def _export_cleaned_obj(src_path: Path, dst_path: Path) -> None:
    mesh = _load_and_cleanup_obj_mesh(src_path)
    _write_obj_from_o3d_mesh(mesh, dst_path)


def _manifold_check_like_check_script(obj_path: Path) -> Dict[str, Any]:
    mesh = _load_and_cleanup_obj_mesh(obj_path)

    is_watertight = bool(mesh.is_watertight())
    is_edge_manifold = bool(mesh.is_edge_manifold())
    is_vertex_manifold = bool(mesh.is_vertex_manifold())
    is_orientable = bool(mesh.is_orientable())
    is_self_intersecting = bool(mesh.is_self_intersecting())

    issues = []
    if not is_watertight:
        issues.append("not_watertight")
    if not is_edge_manifold:
        issues.append("non_edge_manifold")
    if not is_vertex_manifold:
        issues.append("non_vertex_manifold")
    if not is_orientable:
        issues.append("not_orientable")
    if is_self_intersecting:
        issues.append("self_intersecting")

    return {
        "is_manifold": len(issues) == 0,
        "is_watertight": is_watertight,
        "is_edge_manifold": is_edge_manifold,
        "is_vertex_manifold": is_vertex_manifold,
        "is_orientable": is_orientable,
        "is_self_intersecting": is_self_intersecting,
        "issues": issues,
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.triangles)),
    }


def _iter_obj_files(root: Path):
    return root.rglob("*.obj")


def _resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def _evaluate_single_obj(obj_path_str: str, input_root_str: str) -> Dict[str, Any]:
    obj_path = Path(obj_path_str)
    input_root = Path(input_root_str)
    rel_path = obj_path.relative_to(input_root)

    try:
        face_info = _inspect_non_triangle_faces_obj(obj_path)
        check = _manifold_check_like_check_script(obj_path)
        ok = bool(check["is_manifold"])
        reason = "ok" if ok else (check["issues"][0] if check["issues"] else "unknown")
        row = [
            str(rel_path),
            ok,
            reason,
            check["num_vertices"],
            check["num_faces"],
            check["is_watertight"],
            check["is_edge_manifold"],
            check["is_vertex_manifold"],
            check["is_orientable"],
            check["is_self_intersecting"],
            "|".join(check["issues"]),
            face_info["raw_face_count"],
            face_info["has_non_triangle_faces"],
            face_info["non_triangle_face_count"],
        ]
        return {
            "src": str(obj_path),
            "rel": str(rel_path),
            "ok": ok,
            "row": row,
            "exception": False,
        }
    except Exception as exc:
        return {
            "src": str(obj_path),
            "rel": str(rel_path),
            "ok": False,
            "row": [
                str(rel_path),
                False,
                "exception",
                0,
                0,
                False,
                False,
                False,
                False,
                False,
                "",
                0,
                False,
                0,
            ],
            "exception": True,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def filter_manifold_objs(
    input_root: Path,
    manifold_output_root: Path,
    non_manifold_output_root: Path,
    report_csv: Path,
    overwrite: bool,
    limit: Optional[int],
    workers: int,
) -> None:
    obj_files = sorted(_iter_obj_files(input_root))
    if limit is not None:
        obj_files = obj_files[:limit]

    total = len(obj_files)
    passed = 0
    failed = 0
    skipped = 0

    manifold_output_root.mkdir(parents=True, exist_ok=True)
    non_manifold_output_root.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_workers = _resolve_workers(workers)

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "relative_path",
                "is_manifold",
                "reason",
                "num_vertices",
                "num_faces",
                "is_watertight",
                "is_edge_manifold",
                "is_vertex_manifold",
                "is_orientable",
                "is_self_intersecting",
                "issues",
                "raw_face_count",
                "has_non_triangle_faces",
                "non_triangle_face_count",
            ]
        )

        if resolved_workers == 1:
            results_iter = (
                _evaluate_single_obj(str(obj_path), str(input_root)) for obj_path in obj_files
            )
            completed = 0
            for result in results_iter:
                completed += 1
                rel_path = Path(result["rel"])
                src_path = Path(result["src"])
                writer.writerow(result["row"])

                if result.get("exception", False):
                    print(f"[ERROR] {src_path}")
                    print(result.get("traceback", ""))

                if result["ok"]:
                    dst_ok = manifold_output_root / rel_path
                    if dst_ok.exists() and not overwrite:
                        skipped += 1
                    else:
                        try:
                            _export_cleaned_obj(src_path, dst_ok)
                        except Exception:
                            dst_ok.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_path, dst_ok)
                        passed += 1
                else:
                    dst_bad = non_manifold_output_root / rel_path
                    if dst_bad.exists() and not overwrite:
                        skipped += 1
                    else:
                        try:
                            _export_cleaned_obj(src_path, dst_bad)
                        except Exception:
                            dst_bad.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_path, dst_bad)
                    failed += 1

                if completed % 50 == 0 or completed == total:
                    print(f"[{completed}/{total}] passed={passed} failed={failed} skipped={skipped}")
        else:
            completed = 0
            with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                futures = [
                    executor.submit(_evaluate_single_obj, str(obj_path), str(input_root))
                    for obj_path in obj_files
                ]

                for fut in as_completed(futures):
                    result = fut.result()
                    completed += 1
                    rel_path = Path(result["rel"])
                    src_path = Path(result["src"])
                    writer.writerow(result["row"])

                    if result.get("exception", False):
                        print(f"[ERROR] {src_path}")
                        print(result.get("traceback", ""))

                    if result["ok"]:
                        dst_ok = manifold_output_root / rel_path
                        if dst_ok.exists() and not overwrite:
                            skipped += 1
                        else:
                            try:
                                _export_cleaned_obj(src_path, dst_ok)
                            except Exception:
                                dst_ok.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src_path, dst_ok)
                            passed += 1
                    else:
                        dst_bad = non_manifold_output_root / rel_path
                        if dst_bad.exists() and not overwrite:
                            skipped += 1
                        else:
                            try:
                                _export_cleaned_obj(src_path, dst_bad)
                            except Exception:
                                dst_bad.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src_path, dst_bad)
                        failed += 1

                    if completed % 50 == 0 or completed == total:
                        print(f"[{completed}/{total}] passed={passed} failed={failed} skipped={skipped}")

    print("\nFinished filtering manifold OBJ files.")
    print(f"Input root:  {input_root}")
    print(f"Manifold output root:     {manifold_output_root}")
    print(f"Non-manifold output root: {non_manifold_output_root}")
    print(f"Report CSV:  {report_csv}")
    print(f"Total:       {total}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print(f"Skipped:     {skipped}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter manifold OBJ files from extracted RoboCasa visual OBJ outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("get_assets/0_merged_visual_objs"),
        help="Folder containing extracted visual OBJ files",
    )
    parser.add_argument(
        "--manifold-output-root",
        type=Path,
        default=Path("get_assets/1_manifold_visual_objs"),
        help="Folder to store manifold OBJ files",
    )
    parser.add_argument(
        "--non-manifold-output-root",
        type=Path,
        default=Path("get_assets/1_non_manifold_visual_objs"),
        help="Folder to store non-manifold OBJ files",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("get_assets/1_manifold_visual_objs/manifold_report.csv"),
        help="CSV file path for manifold check report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied OBJ files in output folder",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process first N OBJ files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes (1 for single process, <=0 for auto)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_root = args.input_root.resolve()
    manifold_output_root = args.manifold_output_root.resolve()
    non_manifold_output_root = args.non_manifold_output_root.resolve()
    report_csv = args.report_csv.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    filter_manifold_objs(
        input_root=input_root,
        manifold_output_root=manifold_output_root,
        non_manifold_output_root=non_manifold_output_root,
        report_csv=report_csv,
        overwrite=args.overwrite,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
