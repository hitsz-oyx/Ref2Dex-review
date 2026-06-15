"""批量检测网格资产的流形性。

特性：
- 递归扫描指定目录下的 obj/stl/ply 文件
- 参考数据生成流程，对 OBJ/PLY 自动三角化后再检测
- 使用 Open3D 检查 watertight、edge manifold、vertex manifold 等属性
- 输出详细 JSON 报告和文本摘要
- 支持多进程并行检测

示例：
    python dataset/check_manifold.py --input-dir assets/
    python dataset/check_manifold.py --input-dir assets1/ assets2/
    python dataset/check_manifold.py --input-dir assets/ --output-dir reports/
    python dataset/check_manifold.py --input-dir assets/ --workers 4
    python dataset/check_manifold.py --input-dir assets/ --quiet
    python dataset/check_manifold.py --input-dir assets/ --no-recursive
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm


SUPPORTED_EXTS = (".obj", ".stl", ".ply")


def _log(message: str, quiet: bool = False, force: bool = False) -> None:
    """统一控制控制台输出。"""
    if force or not quiet:
        print(message)


def _normalize_input_dirs(input_dirs: Sequence[str]) -> List[str]:
    """展开并规范化输入目录。"""
    normalized: List[str] = []
    for input_dir in input_dirs:
        if input_dir is None:
            continue
        normalized.append(os.path.abspath(os.path.expanduser(input_dir)))
    return normalized


def _build_result_template(file_path: str) -> Dict[str, Any]:
    """构造单个文件的默认结果结构。"""
    return {
        "path": file_path,
        "name": os.path.basename(file_path),
        "is_manifold": False,
        "is_watertight": False,
        "is_edge_manifold": False,
        "is_vertex_manifold": False,
        "is_orientable": False,
        "is_self_intersecting": False,
        "num_vertices": 0,
        "num_faces": 0,
        "duration_sec": 0.0,
        "timings_sec": {
            "face_inspect": 0.0,
            "load_mesh": 0.0,
            "mesh_cleanup": 0.0,
            "manifold_eval": 0.0,
            "total": 0.0,
        },
        "raw_face_count": 0,
        "has_non_triangle_faces": False,
        "non_triangle_face_count": 0,
        "face_check_error": None,
        "issues": [],
        "error": None,
    }


def _resolve_workers(workers: int) -> int:
    """规范化进程数。0 或负数表示自动。"""
    if workers > 0:
        return workers
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def _load_mesh_triangles(file_path: str) -> Any:
    """参考 `generate_sdfae_data.py` 的网格加载方式，返回三角网格。

    - OBJ / PLY: 使用 `pytorch3d.io` 读取，自动得到三角面
    - STL: 使用 Open3D 读取三角面
    """
    import open3d as o3d
    from pytorch3d.io import load_obj, load_ply

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".obj":
        verts, faces, _ = load_obj(file_path, load_textures=False)
        verts_np = verts.detach().cpu().numpy().astype(np.float64, copy=False)
        faces_np = faces.verts_idx.detach().cpu().numpy().astype(np.int32, copy=False)
    elif ext == ".ply":
        verts, faces = load_ply(file_path)
        verts_np = verts.detach().cpu().numpy().astype(np.float64, copy=False)
        faces_np = faces.detach().cpu().numpy().astype(np.int32, copy=False)
    elif ext == ".stl":
        mesh = o3d.io.read_triangle_mesh(file_path)
        verts_np = np.asarray(mesh.vertices, dtype=np.float64)
        faces_np = np.asarray(mesh.triangles, dtype=np.int32)
    else:
        raise ValueError(f"暂不支持的网格格式: {ext}")

    if verts_np.size == 0 or faces_np.size == 0:
        raise ValueError("网格为空，或未能解析出有效三角面")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    return mesh


def _inspect_non_triangle_faces(file_path: str) -> Dict[str, Any]:
    """检查原始文件中是否存在非三角面。"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".stl":
        return {
            "raw_face_count": 0,
            "has_non_triangle_faces": False,
            "non_triangle_face_count": 0,
            "face_check_error": None,
        }

    if ext == ".obj":
        raw_face_count = 0
        non_triangle_face_count = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
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
            "face_check_error": None,
        }

    if ext == ".ply":
        # 仅实现 ASCII PLY 的精确统计；binary PLY 标记为未知。
        with open(file_path, "rb") as fb:
            header_bytes = b""
            while True:
                line = fb.readline()
                if not line:
                    raise ValueError("PLY header 不完整")
                header_bytes += line
                if line.strip() == b"end_header":
                    break
            body_bytes = fb.read()

        header_text = header_bytes.decode("utf-8", errors="ignore").splitlines()
        format_type = None
        face_count = 0
        for line in header_text:
            tokens = line.strip().split()
            if len(tokens) >= 2 and tokens[0] == "format":
                format_type = tokens[1]
            if len(tokens) >= 3 and tokens[0] == "element" and tokens[1] == "face":
                try:
                    face_count = int(tokens[2])
                except ValueError:
                    face_count = 0

        if format_type is None:
            raise ValueError("PLY header 缺少 format 字段")

        if format_type != "ascii":
            return {
                "raw_face_count": face_count,
                "has_non_triangle_faces": False,
                "non_triangle_face_count": 0,
                "face_check_error": f"未解析 {format_type} PLY 的原始面类型",
            }

        body_text = body_bytes.decode("utf-8", errors="ignore").splitlines()
        non_triangle_face_count = 0
        faces_seen = 0

        # ASCII PLY 中 face 行通常以“顶点数 + 索引列表”开头
        for line in body_text:
            if faces_seen >= face_count:
                break
            stripped = line.strip()
            if not stripped:
                continue
            first_token = stripped.split()[0]
            if not first_token.lstrip("+-").isdigit():
                continue
            nverts = int(first_token)
            faces_seen += 1
            if nverts != 3:
                non_triangle_face_count += 1

        return {
            "raw_face_count": faces_seen,
            "has_non_triangle_faces": non_triangle_face_count > 0,
            "non_triangle_face_count": non_triangle_face_count,
            "face_check_error": None,
        }

    return {
        "raw_face_count": 0,
        "has_non_triangle_faces": False,
        "non_triangle_face_count": 0,
        "face_check_error": f"未实现 {ext} 的原始面类型检查",
    }


def _collect_mesh_files(input_dirs: Sequence[str], recursive: bool = True) -> List[str]:
    """收集所有网格文件。

    Args:
        input_dirs: 输入目录列表
        recursive: 是否递归搜索子目录

    Returns:
        网格文件路径列表
    """
    files: List[str] = []
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"[WARN] 目录不存在：{input_dir}")
            continue

        if recursive:
            for root, _, filenames in os.walk(input_dir):
                for filename in filenames:
                    if filename.lower().endswith(SUPPORTED_EXTS):
                        files.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(input_dir):
                file_path = os.path.join(input_dir, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(SUPPORTED_EXTS):
                    files.append(file_path)

    # 去重并排序
    return sorted(list(dict.fromkeys(files)))


def _check_single_mesh(file_path: str) -> Dict[str, Any]:
    """检测单个网格的流形性。"""
    result = _build_result_template(file_path)
    start_time = time.perf_counter()

    try:
        import open3d as o3d

        t0 = time.perf_counter()
        face_info = _inspect_non_triangle_faces(file_path)
        result["timings_sec"]["face_inspect"] = float(time.perf_counter() - t0)
        result.update(face_info)

        t0 = time.perf_counter()
        mesh = _load_mesh_triangles(file_path)
        result["timings_sec"]["load_mesh"] = float(time.perf_counter() - t0)
        if mesh.is_empty() or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError("网格为空，或未能解析出有效三角面")

        t0 = time.perf_counter()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        result["timings_sec"]["mesh_cleanup"] = float(time.perf_counter() - t0)

        result["num_vertices"] = int(len(mesh.vertices))
        result["num_faces"] = int(len(mesh.triangles))

        t0 = time.perf_counter()
        result["is_watertight"] = bool(mesh.is_watertight())
        result["is_edge_manifold"] = bool(mesh.is_edge_manifold())
        result["is_vertex_manifold"] = bool(mesh.is_vertex_manifold())
        result["is_orientable"] = bool(mesh.is_orientable())
        result["is_self_intersecting"] = bool(mesh.is_self_intersecting())
        result["timings_sec"]["manifold_eval"] = float(time.perf_counter() - t0)

        issues: List[str] = []
        if not result["is_watertight"]:
            issues.append("not_watertight")
        if not result["is_edge_manifold"]:
            issues.append("non_edge_manifold")
        if not result["is_vertex_manifold"]:
            issues.append("non_vertex_manifold")
        if not result["is_orientable"]:
            issues.append("not_orientable")
        if result["is_self_intersecting"]:
            issues.append("self_intersecting")

        result["issues"] = issues
        result["is_manifold"] = len(issues) == 0

    except Exception as e:
        result["error"] = str(e)
    finally:
        total_sec = float(time.perf_counter() - start_time)
        result["duration_sec"] = total_sec
        result["timings_sec"]["total"] = total_sec

    return result


def _iter_results(mesh_files: Sequence[str], workers: int) -> Iterable[Dict[str, Any]]:
    """按配置串行或并行地产生检测结果。"""
    if workers <= 1 or len(mesh_files) <= 1:
        for file_path in mesh_files:
            yield _check_single_mesh(file_path)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(_check_single_mesh, file_path): file_path
            for file_path in mesh_files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                yield future.result()
            except Exception as exc:
                result = _build_result_template(file_path)
                result["error"] = f"worker 执行失败: {exc}"
                yield result


def _build_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """从结果列表中汇总统计信息。"""
    total = len(results)
    error_count = sum(1 for r in results if r["error"] is not None)
    checked_count = total - error_count
    manifold_count = sum(1 for r in results if r["error"] is None and r["is_manifold"])
    non_manifold_count = checked_count - manifold_count

    timing_total_face_inspect = sum(float(r.get("timings_sec", {}).get("face_inspect", 0.0)) for r in results)
    timing_total_load_mesh = sum(float(r.get("timings_sec", {}).get("load_mesh", 0.0)) for r in results)
    timing_total_mesh_cleanup = sum(float(r.get("timings_sec", {}).get("mesh_cleanup", 0.0)) for r in results)
    timing_total_manifold_eval = sum(float(r.get("timings_sec", {}).get("manifold_eval", 0.0)) for r in results)

    return {
        "total": total,
        "checked": checked_count,
        "manifold": manifold_count,
        "non_manifold": non_manifold_count,
        "not_watertight": sum(1 for r in results if r["error"] is None and not r["is_watertight"]),
        "non_edge_manifold": sum(1 for r in results if r["error"] is None and not r["is_edge_manifold"]),
        "non_vertex_manifold": sum(1 for r in results if r["error"] is None and not r["is_vertex_manifold"]),
        "not_orientable": sum(1 for r in results if r["error"] is None and not r["is_orientable"]),
        "self_intersecting": sum(1 for r in results if r["error"] is None and r["is_self_intersecting"]),
        "meshes_with_non_triangle_faces": sum(1 for r in results if r["has_non_triangle_faces"]),
        "non_triangle_faces_total": sum(int(r["non_triangle_face_count"]) for r in results),
        "face_check_errors": sum(1 for r in results if r["face_check_error"] is not None),
        "total_duration_sec": float(sum(float(r.get("duration_sec", 0.0)) for r in results)),
        "avg_duration_sec": float(
            sum(float(r.get("duration_sec", 0.0)) for r in results) / total if total > 0 else 0.0
        ),
        "timing_total_face_inspect_sec": float(timing_total_face_inspect),
        "timing_total_load_mesh_sec": float(timing_total_load_mesh),
        "timing_total_mesh_cleanup_sec": float(timing_total_mesh_cleanup),
        "timing_total_manifold_eval_sec": float(timing_total_manifold_eval),
        "timing_avg_face_inspect_sec": float(timing_total_face_inspect / total if total > 0 else 0.0),
        "timing_avg_load_mesh_sec": float(timing_total_load_mesh / total if total > 0 else 0.0),
        "timing_avg_mesh_cleanup_sec": float(timing_total_mesh_cleanup / total if total > 0 else 0.0),
        "timing_avg_manifold_eval_sec": float(timing_total_manifold_eval / total if total > 0 else 0.0),
        "errors": error_count,
    }


def check_manifolds(
    input_dirs: List[str],
    output_dir: Optional[str] = None,
    recursive: bool = True,
    save_json: bool = True,
    workers: int = 4,
    verbose: bool = True
) -> Dict:
    """批量检测网格流形性。"""
    input_dirs = _normalize_input_dirs(input_dirs)
    workers = _resolve_workers(workers)
    quiet = not verbose

    _log("=" * 60, quiet)
    _log("网格流形性批量检测工具", quiet)
    _log("=" * 60, quiet)
    _log(f"输入目录：{', '.join(input_dirs)}", quiet)
    _log(f"递归搜索：{'是' if recursive else '否'}", quiet)
    _log(f"并行进程：{workers}", quiet)
    _log(f"保存 JSON：{'是' if save_json else '否'}", quiet)
    _log("=" * 60, quiet)

    # 收集文件
    _log("\n正在扫描网格文件...", quiet)
    mesh_files = _collect_mesh_files(input_dirs, recursive)

    if not mesh_files:
        _log("[ERROR] 未找到任何网格文件!", quiet, force=True)
        return {}

    _log(f"共找到 {len(mesh_files)} 个网格文件\n", quiet)
    _log("开始检测流形性...\n", quiet)

    results: List[Dict[str, Any]] = []
    manifold_so_far = 0
    error_so_far = 0
    non_manifold_so_far = 0

    with tqdm(total=len(mesh_files), desc="检测进度", disable=quiet) as pbar:
        for result in _iter_results(mesh_files, workers):
            results.append(result)
            if result["error"] is not None:
                error_so_far += 1
            elif result["is_manifold"]:
                manifold_so_far += 1
            else:
                non_manifold_so_far += 1

            pbar.update(1)
            if not quiet:
                pbar.set_postfix({
                    "流形": manifold_so_far,
                    "非流形": non_manifold_so_far,
                    "错误": error_so_far,
                })

    results.sort(key=lambda item: item["path"])
    summary = _build_summary(results)
    non_manifold_results = [r for r in results if r["error"] is None and not r["is_manifold"]]
    non_triangle_face_results = [r for r in results if r["has_non_triangle_faces"]]
    error_results = [r for r in results if r["error"] is not None]

    # 打印汇总
    total = summary["total"]
    checked = summary["checked"]
    checked_ratio = (checked / total * 100.0) if total else 0.0
    manifold_ratio = (summary["manifold"] / checked * 100.0) if checked else 0.0
    non_manifold_ratio = (summary["non_manifold"] / checked * 100.0) if checked else 0.0

    print("\n" + "=" * 60)
    print("检测结果汇总")
    print("=" * 60)
    print(f"总网格数：{total}")
    print(f"成功检测：{checked} ({checked_ratio:.1f}%)")
    print(f"✓ 流形网格：{summary['manifold']} ({manifold_ratio:.1f}%)")
    print(f"✗ 非流形网格：{summary['non_manifold']} ({non_manifold_ratio:.1f}%)")
    print(f"⚠ 加载/解析错误：{summary['errors']}")
    print("-" * 60)
    print(f"不封闭（not watertight）：{summary['not_watertight']}")
    print(f"非流形边：{summary['non_edge_manifold']}")
    print(f"非流形顶点：{summary['non_vertex_manifold']}")
    print(f"不可定向：{summary['not_orientable']}")
    print(f"自相交：{summary['self_intersecting']}")
    print(f"含非三角面网格：{summary['meshes_with_non_triangle_faces']}")
    print(f"非三角面总数：{summary['non_triangle_faces_total']}")
    print(f"非三角面检查异常：{summary['face_check_errors']}")
    print(f"总耗时（逐文件累计）：{summary['total_duration_sec']:.3f}s")
    print(f"平均每文件耗时：{summary['avg_duration_sec']:.3f}s")
    print(
        "阶段累计耗时(s)："
        f"face_inspect={summary['timing_total_face_inspect_sec']:.3f}, "
        f"load_mesh={summary['timing_total_load_mesh_sec']:.3f}, "
        f"mesh_cleanup={summary['timing_total_mesh_cleanup_sec']:.3f}, "
        f"manifold_eval={summary['timing_total_manifold_eval_sec']:.3f}"
    )
    print("=" * 60)

    if non_triangle_face_results:
        _log("\n包含非三角面的资产列表:", quiet)
        _log("-" * 60, quiet)
        for i, r in enumerate(non_triangle_face_results, 1):
            _log(
                f"{i}. {r['path']} | 非三角面：{r['non_triangle_face_count']:,} | 原始面：{r['raw_face_count']:,}",
                quiet,
            )

    # 详细列出非流形网格
    if non_manifold_results:
        _log("\n非流形网格列表:", quiet)
        _log("-" * 60, quiet)
        for i, r in enumerate(non_manifold_results, 1):
            status_w = "✓" if r["is_watertight"] else "✗"
            status_e = "✓" if r["is_edge_manifold"] else "✗"
            status_v = "✓" if r["is_vertex_manifold"] else "✗"
            status_o = "✓" if r["is_orientable"] else "✗"
            status_s = "✓" if not r["is_self_intersecting"] else "✗"
            _log(f"{i}. {r['path']}", quiet)
            _log(f"   顶点：{r['num_vertices']:,} | 面：{r['num_faces']:,} | 耗时：{r['duration_sec']:.3f}s", quiet)
            _log(
                f"   封闭性：{status_w} | 边流形：{status_e} | 顶点流形：{status_v} | 可定向：{status_o} | 无自交：{status_s}",
                quiet,
            )
            _log(
                f"   原始面：{r['raw_face_count']:,} | 非三角面：{r['non_triangle_face_count']:,}",
                quiet,
            )
            _log(
                "   阶段耗时(s)："
                f"face={r['timings_sec']['face_inspect']:.3f}, "
                f"load={r['timings_sec']['load_mesh']:.3f}, "
                f"clean={r['timings_sec']['mesh_cleanup']:.3f}, "
                f"eval={r['timings_sec']['manifold_eval']:.3f}",
                quiet,
            )
            if r["face_check_error"]:
                _log(f"   面检查异常：{r['face_check_error']}", quiet)
            if r["issues"]:
                _log(f"   问题：{', '.join(r['issues'])}", quiet)

    if error_results:
        _log("\n错误网格列表:", quiet)
        _log("-" * 60, quiet)
        for i, r in enumerate(error_results, 1):
            _log(f"{i}. {r['name']}", quiet)
            _log(f"   错误：{r['error']} | 耗时：{r['duration_sec']:.3f}s", quiet)
            _log(
                "   阶段耗时(s)："
                f"face={r['timings_sec']['face_inspect']:.3f}, "
                f"load={r['timings_sec']['load_mesh']:.3f}, "
                f"clean={r['timings_sec']['mesh_cleanup']:.3f}, "
                f"eval={r['timings_sec']['manifold_eval']:.3f}",
                quiet,
            )

    # 保存 JSON 报告
    if save_json and output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 详细报告
        report = {
            "timestamp": timestamp,
            "summary": summary,
            "results": results,
            "non_manifold_list": non_manifold_results,
            "error_list": error_results,
        }

        json_path = os.path.join(output_dir, f"manifold_check_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ JSON 报告已保存：{json_path}")

        # 简洁版文本报告
        txt_path = os.path.join(output_dir, f"manifold_summary_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"网格流形性检测报告\n")
            f.write(f"时间：{timestamp}\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"总网格数：{total}\n")
            f.write(f"成功检测：{summary['checked']}\n")
            f.write(f"流形网格：{summary['manifold']}\n")
            f.write(f"非流形网格：{summary['non_manifold']}\n")
            f.write(f"加载/解析错误：{summary['errors']}\n\n")
            f.write("问题分类统计:\n")
            f.write("-" * 60 + "\n")
            f.write(f"不封闭：{summary['not_watertight']}\n")
            f.write(f"非流形边：{summary['non_edge_manifold']}\n")
            f.write(f"非流形顶点：{summary['non_vertex_manifold']}\n")
            f.write(f"不可定向：{summary['not_orientable']}\n")
            f.write(f"自相交：{summary['self_intersecting']}\n\n")
            f.write(f"含非三角面网格：{summary['meshes_with_non_triangle_faces']}\n")
            f.write(f"非三角面总数：{summary['non_triangle_faces_total']}\n")
            f.write(f"非三角面检查异常：{summary['face_check_errors']}\n\n")
            f.write(f"总耗时（逐文件累计）：{summary['total_duration_sec']:.3f}s\n")
            f.write(f"平均每文件耗时：{summary['avg_duration_sec']:.3f}s\n\n")
            f.write("阶段累计耗时(s):\n")
            f.write(f"  face_inspect={summary['timing_total_face_inspect_sec']:.3f}\n")
            f.write(f"  load_mesh={summary['timing_total_load_mesh_sec']:.3f}\n")
            f.write(f"  mesh_cleanup={summary['timing_total_mesh_cleanup_sec']:.3f}\n")
            f.write(f"  manifold_eval={summary['timing_total_manifold_eval_sec']:.3f}\n\n")

            if non_triangle_face_results:
                f.write("包含非三角面的资产列表:\n")
                f.write("-" * 60 + "\n")
                for r in non_triangle_face_results:
                    f.write(
                        f"{r['path']} | 非三角面：{r['non_triangle_face_count']:,} | 原始面：{r['raw_face_count']:,}\n"
                    )
                f.write("\n")

            f.write("非流形网格详情:\n")
            f.write("-" * 60 + "\n")
            for r in non_manifold_results:
                f.write(f"\n{r['path']}\n")
                f.write(f"  顶点：{r['num_vertices']:,} | 面：{r['num_faces']:,} | 耗时：{r['duration_sec']:.3f}s\n")
                f.write(
                    "  阶段耗时(s)："
                    f"face={r['timings_sec']['face_inspect']:.3f}, "
                    f"load={r['timings_sec']['load_mesh']:.3f}, "
                    f"clean={r['timings_sec']['mesh_cleanup']:.3f}, "
                    f"eval={r['timings_sec']['manifold_eval']:.3f}\n"
                )
                f.write(f"  原始面：{r['raw_face_count']:,} | 非三角面：{r['non_triangle_face_count']:,}\n")
                f.write(f"  封闭：{'✓' if r['is_watertight'] else '✗'} | ")
                f.write(f"边流形：{'✓' if r['is_edge_manifold'] else '✗'} | ")
                f.write(f"顶点流形：{'✓' if r['is_vertex_manifold'] else '✗'} | ")
                f.write(f"可定向：{'✓' if r['is_orientable'] else '✗'} | ")
                f.write(f"无自交：{'✓' if not r['is_self_intersecting'] else '✗'}\n")
                if r["face_check_error"]:
                    f.write(f"  面检查异常：{r['face_check_error']}\n")
                if r["issues"]:
                    f.write(f"  问题：{', '.join(r['issues'])}\n")

            if error_results:
                f.write("\n错误网格详情:\n")
                f.write("-" * 60 + "\n")
                for r in error_results:
                    f.write(f"\n{r['path']}\n")
                    f.write(f"  错误：{r['error']} | 耗时：{r['duration_sec']:.3f}s\n")
                    f.write(
                        "  阶段耗时(s)："
                        f"face={r['timings_sec']['face_inspect']:.3f}, "
                        f"load={r['timings_sec']['load_mesh']:.3f}, "
                        f"clean={r['timings_sec']['mesh_cleanup']:.3f}, "
                        f"eval={r['timings_sec']['manifold_eval']:.3f}\n"
                    )

        print(f"✓ 文本摘要已保存：{txt_path}")

    # 返回统计信息
    return summary


def main():
    parser = argparse.ArgumentParser(description="批量检测网格资产的流形性")
    
    parser.add_argument(
        "--input-dir",
        action="extend",
        nargs="+",
        required=True,
        help="输入目录，可写为 --input-dir dir1 dir2"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/manifold_check",
        help="报告输出目录（默认：reports/manifold_check）"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归搜索子目录"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行工作进程数，0 表示自动（默认：0）"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="不保存 JSON 报告"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式（仅显示汇总信息）"
    )
    
    args = parser.parse_args()
    input_dirs = args.input_dir
    
    # 执行检测
    stats = check_manifolds(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        save_json=not args.no_json,
        workers=args.workers,
        verbose=not args.quiet,
    )
    
    # 退出码：如果有非流形或解析错误则返回 1
    if stats and (stats.get("non_manifold", 0) > 0 or stats.get("errors", 0) > 0):
        print("\n[WARNING] 检测到非流形网格或解析错误！")
        sys.exit(1)
    else:
        print("\n[SUCCESS] 所有网格均为流形！")
        sys.exit(0)


if __name__ == "__main__":
    main()