#!/usr/bin/env python3
"""
Use Blender to repair non-manifold OBJ assets in batch (advanced strategy).

Advanced pipeline:
1) Basic mesh cleanup (print3d / merge by distance / fill holes / triangulate / normals)
2) Check non-manifold edges via bmesh
3) If still non-manifold, fallback to voxel remesh + decimate + cleanup

Run example:
  blender --background --python get_fixtures_assets/repair_non_manifold_with_blender_advanced.py
"""

import os
import shutil
import sys
import traceback
from contextlib import contextmanager

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

# ===== 在这里设置你的文件夹路径 =====
INPUT_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/1_non_manifold_visual_objs"
OUTPUT_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/2_non_manifold_objs_repaired_advanced"
FAILED_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/2_non_manifold_objs_failed_advanced"
FAILED_RESULT_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/2_non_manifold_failed_results_advanced"

# 可选设置
OVERWRITE = False
LIMIT = None
PRESERVE_REGISTRY_LEVEL = True

# 高级修复参数
ALWAYS_USE_COMPLEX_REPAIR = False
VOXEL_SIZE = 0.001
DECIMATE_RATIO = 0.1
MAX_HOLE_SIDES = 0
VERBOSE_PER_FILE = True
SUPPRESS_BLENDER_OP_LOGS = True
NON_PLANAR_EPSILON = 1e-5
RETRY_WITH_HALF_VOXEL_ON_FAIL = True
# =================================

REGISTRY_NAMES = {"aigen_objs", "objaverse", "lightwheel"}


COLOR_OUTPUT = True
COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"


def _paint(text, color):
    if not COLOR_OUTPUT:
        return str(text)
    return f"{color}{text}{COLOR_RESET}"


def _bool_tag(value: bool) -> str:
    return _paint(str(value), COLOR_GREEN if value else COLOR_YELLOW)


def _nm_tag(value: int) -> str:
    if value == 0:
        return _paint(str(value), COLOR_GREEN)
    if value <= 20:
        return _paint(str(value), COLOR_YELLOW)
    return _paint(str(value), COLOR_RED)


@contextmanager
def suppress_blender_output(enabled=True):
    if not enabled:
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def reset_scene():
    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_print3d_addon():
    for module in ("mesh_print3d", "object_print3d_utils"):
        try:
            bpy.ops.preferences.addon_enable(module=module)
            return True
        except Exception:
            continue
    return False


def import_obj(filepath):
    try:
        with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
            bpy.ops.wm.obj_import(filepath=filepath)
        return
    except Exception:
        pass

    try:
        with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
            bpy.ops.import_scene.obj(filepath=filepath)
        return
    except Exception as exc:
        raise RuntimeError(f"Failed to import OBJ: {filepath}") from exc


def export_obj(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh object available for export")

    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    try:
        with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
            bpy.ops.wm.obj_export(
                filepath=filepath,
                export_selected_objects=True,
                export_materials=False,
                export_uv=False,
                export_normals=True,
                export_triangulated_mesh=True,
            )
    except Exception:
        with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
            bpy.ops.export_scene.obj(
                filepath=filepath,
                use_selection=True,
                use_materials=False,
                use_normals=True,
                use_triangles=True,
            )

    mtl_path = os.path.splitext(filepath)[0] + ".mtl"
    if os.path.exists(mtl_path):
        os.remove(mtl_path)


def get_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def count_non_manifold_edges(obj):
    if obj.type != "MESH":
        return 0
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    count = sum(1 for edge in bm.edges if not edge.is_manifold)
    bm.free()
    return count


def count_scene_non_manifold_edges():
    total = 0
    for obj in get_mesh_objects():
        total += count_non_manifold_edges(obj)
    return total


def _count_non_planar_faces(bm, eps=NON_PLANAR_EPSILON):
    count = 0
    for face in bm.faces:
        if len(face.verts) <= 3:
            continue
        if face.normal.length < 1e-12:
            count += 1
            continue
        n = face.normal.normalized()
        p0 = face.verts[0].co
        non_planar = False
        for vert in face.verts[1:]:
            dist = abs((vert.co - p0).dot(n))
            if dist > eps:
                non_planar = True
                break
        if non_planar:
            count += 1
    return count


def _count_self_intersections(bm):
    if len(bm.faces) < 2:
        return 0

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    bvh = BVHTree.FromBMesh(bm, epsilon=1e-9)
    overlaps = bvh.overlap(bvh)
    if not overlaps:
        return 0

    intersect_pairs = set()
    for i, j in overlaps:
        if i >= j:
            continue
        face_i = bm.faces[i]
        face_j = bm.faces[j]

        # Ignore adjacent faces sharing vertices/edges; keep only likely real self-intersections
        verts_i = {v.index for v in face_i.verts}
        verts_j = {v.index for v in face_j.verts}
        if verts_i & verts_j:
            continue

        intersect_pairs.add((i, j))

    return len(intersect_pairs)


def _count_disconnected_components(bm):
    bm.verts.ensure_lookup_table()
    if len(bm.verts) == 0:
        return 0

    visited = set()
    components = 0
    for vert in bm.verts:
        if vert.index in visited:
            continue

        components += 1
        stack = [vert]
        visited.add(vert.index)
        while stack:
            cur = stack.pop()
            for edge in cur.link_edges:
                other = edge.other_vert(cur)
                if other.index not in visited:
                    visited.add(other.index)
                    stack.append(other)
    return components


def collect_mesh_quality_metrics(obj):
    if obj.type != "MESH":
        return {
            "non_manifold_edges": 0,
            "error_continuous_edges": 0,
            "self_intersections": 0,
            "non_planar_faces": 0,
            "disconnected_components": 0,
        }

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    non_manifold_edges = sum(1 for edge in bm.edges if not edge.is_manifold)
    error_continuous_edges = sum(1 for edge in bm.edges if edge.is_boundary or edge.is_wire)
    self_intersections = _count_self_intersections(bm)
    non_planar_faces = _count_non_planar_faces(bm)
    disconnected_components = _count_disconnected_components(bm)

    bm.free()
    return {
        "non_manifold_edges": non_manifold_edges,
        "error_continuous_edges": error_continuous_edges,
        "self_intersections": self_intersections,
        "non_planar_faces": non_planar_faces,
        "disconnected_components": disconnected_components,
    }


def collect_scene_quality_metrics():
    total = {
        "non_manifold_edges": 0,
        "error_continuous_edges": 0,
        "self_intersections": 0,
        "non_planar_faces": 0,
        "disconnected_components": 0,
        "mesh_object_count": 0,
    }
    mesh_objects = get_mesh_objects()
    total["mesh_object_count"] = len(mesh_objects)
    for obj in mesh_objects:
        m = collect_mesh_quality_metrics(obj)
        for key in total:
            if key in m:
                total[key] += m[key]
    return total


def is_quality_metrics_ok(metrics, initial_metrics=None):
    geometry_ok = (
        metrics["non_manifold_edges"] == 0
        and metrics["error_continuous_edges"] == 0
        and metrics["self_intersections"] == 0
        and metrics["non_planar_faces"] == 0
    )
    if not geometry_ok:
        return False

    if initial_metrics is None:
        return metrics["disconnected_components"] <= 1

    initial_components = initial_metrics.get("disconnected_components", 0)
    current_components = metrics.get("disconnected_components", 0)
    initial_mesh_objects = initial_metrics.get("mesh_object_count", 0)
    current_mesh_objects = metrics.get("mesh_object_count", 0)

    max_allowed_components = initial_components if initial_components > 0 else 1
    max_allowed_mesh_objects = initial_mesh_objects if initial_mesh_objects > 0 else 1

    return (
        current_components <= max_allowed_components
        and current_mesh_objects <= max_allowed_mesh_objects
    )


def make_object_active(obj):
    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def basic_cleanup_on_object(obj, use_print3d, max_hole_sides):
    make_object_active(obj)

    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")

    if use_print3d:
        try:
            with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
                bpy.ops.mesh.print3d_make_manifold()
        except Exception:
            pass

    try:
        with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
            bpy.ops.mesh.remove_doubles()
    except Exception:
        try:
            with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
                bpy.ops.mesh.merge_by_distance()
        except Exception:
            pass

    try:
        with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
            bpy.ops.mesh.fill_holes(sides=max_hole_sides)
    except Exception:
        pass

    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")


def _effective_voxel_size(obj, requested_voxel_size):
    if obj.type != "MESH":
        return requested_voxel_size

    dims = obj.dimensions
    longest_edge = max(float(dims.x), float(dims.y), float(dims.z))
    adaptive_voxel = longest_edge / 500.0 if longest_edge > 0 else 0.0
    return max(float(requested_voxel_size), adaptive_voxel)


def complex_remesh_decimate_on_object(obj, voxel_size, decimate_ratio):
    make_object_active(obj)

    remesh_mod = obj.modifiers.new(name="Temporary_Remesh", type="REMESH")
    remesh_mod.mode = "VOXEL"
    remesh_mod.voxel_size = _effective_voxel_size(obj, voxel_size)
    remesh_mod.adaptivity = 0
    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    decimate_mod = obj.modifiers.new(name="Temporary_Decimate", type="DECIMATE")
    decimate_mod.decimate_type = "COLLAPSE"
    decimate_mod.ratio = decimate_ratio
    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    with suppress_blender_output(SUPPRESS_BLENDER_OP_LOGS):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")


def repair_scene_meshes(
    use_print3d,
    always_use_complex,
    voxel_size,
    decimate_ratio,
    max_hole_sides,
):
    meshes = get_mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh object after import")

    initial_metrics = collect_scene_quality_metrics()

    # Stage 1: simple cleanup
    for obj in meshes:
        basic_cleanup_on_object(obj, use_print3d=use_print3d, max_hole_sides=max_hole_sides)

    after_basic_metrics = collect_scene_quality_metrics()
    basic_ok = is_quality_metrics_ok(after_basic_metrics, initial_metrics=initial_metrics)
    need_advanced_1 = always_use_complex or (not basic_ok)
    used_complex = False
    used_half_voxel_pass = False

    stage_after_adv1_metrics = after_basic_metrics

    # Stage 2: advanced pass #1
    if need_advanced_1:
        used_complex = True
        for obj in meshes:
            complex_remesh_decimate_on_object(
                obj,
                voxel_size=voxel_size,
                decimate_ratio=decimate_ratio,
            )

        # Check immediately after advanced pass #1
        stage_after_adv1_metrics = collect_scene_quality_metrics()

        # Stage 3: run simple cleanup only when quality still fails
        if not is_quality_metrics_ok(stage_after_adv1_metrics, initial_metrics=initial_metrics):
            for obj in meshes:
                basic_cleanup_on_object(
                    obj,
                    use_print3d=use_print3d,
                    max_hole_sides=max_hole_sides,
                )

            stage_after_adv1_metrics = collect_scene_quality_metrics()

    # Stage 4: if quality still fails, do half-voxel advanced pass
    if (
        RETRY_WITH_HALF_VOXEL_ON_FAIL
        and not is_quality_metrics_ok(stage_after_adv1_metrics, initial_metrics=initial_metrics)
    ):
        used_complex = True
        used_half_voxel_pass = True
        half_voxel_size = max(voxel_size * 0.5, 1e-7)

        for obj in meshes:
            complex_remesh_decimate_on_object(
                obj,
                voxel_size=half_voxel_size,
                decimate_ratio=decimate_ratio,
            )

        # Check immediately after advanced pass #2
        stage_after_adv2_metrics = collect_scene_quality_metrics()

        # Stage 5: run simple cleanup only when quality still fails
        if not is_quality_metrics_ok(stage_after_adv2_metrics, initial_metrics=initial_metrics):
            for obj in meshes:
                basic_cleanup_on_object(
                    obj,
                    use_print3d=use_print3d,
                    max_hole_sides=max_hole_sides,
                )

    final_metrics = collect_scene_quality_metrics()
    final_ok = is_quality_metrics_ok(final_metrics, initial_metrics=initial_metrics)
    return {
        "mesh_count": len(meshes),
        "non_manifold_initial": initial_metrics["non_manifold_edges"],
        "non_manifold_after_basic": after_basic_metrics["non_manifold_edges"],
        "non_manifold_after": final_metrics["non_manifold_edges"],
        "error_continuous_initial": initial_metrics["error_continuous_edges"],
        "error_continuous_after_basic": after_basic_metrics["error_continuous_edges"],
        "error_continuous_after": final_metrics["error_continuous_edges"],
        "self_intersections_initial": initial_metrics["self_intersections"],
        "self_intersections_after_basic": after_basic_metrics["self_intersections"],
        "self_intersections_after": final_metrics["self_intersections"],
        "non_planar_initial": initial_metrics["non_planar_faces"],
        "non_planar_after_basic": after_basic_metrics["non_planar_faces"],
        "non_planar_after": final_metrics["non_planar_faces"],
        "components_initial": initial_metrics["disconnected_components"],
        "components_after_basic": after_basic_metrics["disconnected_components"],
        "components_after": final_metrics["disconnected_components"],
        "mesh_object_count_initial": initial_metrics["mesh_object_count"],
        "mesh_object_count_after": final_metrics["mesh_object_count"],
        "basic_ok": basic_ok,
        "final_ok": final_ok,
        "used_complex": used_complex,
        "used_half_voxel_pass": used_half_voxel_pass,
    }


def move_to_failed(src_obj, failed_root, input_root):
    rel = get_output_relative_path(src_obj, input_root)
    dst = os.path.join(failed_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.abspath(src_obj) != os.path.abspath(dst):
        shutil.copy2(src_obj, dst)


def move_repaired_to_failed(repaired_obj, original_obj, failed_root, input_root):
    rel = get_output_relative_path(original_obj, input_root)
    dst = os.path.join(failed_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    src = repaired_obj if os.path.exists(repaired_obj) else original_obj
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            os.remove(dst)
        if os.path.abspath(src) == os.path.abspath(repaired_obj) and os.path.exists(repaired_obj):
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)


def save_failed_result_from_exported(repaired_obj, original_obj, failed_result_root, input_root):
    rel = get_output_relative_path(original_obj, input_root)
    dst = os.path.join(failed_result_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(repaired_obj):
        shutil.copy2(repaired_obj, dst)
        return dst
    return None


def save_failed_result_snapshot(rel, failed_result_root):
    dst = os.path.join(failed_result_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    try:
        mesh_objects = get_mesh_objects()
        if not mesh_objects:
            return None
        export_obj(dst)
        return dst
    except Exception:
        return None


def get_output_relative_path(src_obj, input_root):
    rel = os.path.relpath(src_obj, input_root)
    rel_parts = rel.split(os.sep)

    if not PRESERVE_REGISTRY_LEVEL:
        return rel

    if rel_parts and rel_parts[0] in REGISTRY_NAMES:
        return rel

    input_basename = os.path.basename(os.path.normpath(input_root))
    if input_basename in REGISTRY_NAMES:
        return os.path.join(input_basename, rel)

    return rel


def run_batch(
    input_root,
    output_root,
    failed_root,
    failed_result_root,
    overwrite,
    limit,
):
    obj_files = []
    for root, _, files in os.walk(input_root):
        for name in files:
            if name.lower().endswith(".obj"):
                obj_files.append(os.path.join(root, name))
    obj_files = sorted(obj_files)

    if limit is not None:
        obj_files = obj_files[:limit]

    total = len(obj_files)
    repaired = 0
    failed = 0
    skipped = 0
    used_complex_count = 0
    residual_issue_count = 0
    quality_failed_count = 0
    failed_result_saved_count = 0

    use_print3d = ensure_print3d_addon()
    print(
        f"{_paint('Print3D Make Manifold available:', COLOR_CYAN)} "
        f"{_bool_tag(use_print3d)}"
    )
    print(
        f"{_paint('Advanced config:', COLOR_CYAN)} "
        f"always_complex={ALWAYS_USE_COMPLEX_REPAIR}, "
        f"voxel_size={VOXEL_SIZE}, decimate_ratio={DECIMATE_RATIO}, "
        f"max_hole_sides={MAX_HOLE_SIDES}, "
        f"suppress_blender_ops={SUPPRESS_BLENDER_OP_LOGS}, "
        f"non_planar_eps={NON_PLANAR_EPSILON}, "
        f"retry_half_voxel={RETRY_WITH_HALF_VOXEL_ON_FAIL}"
    )

    for i, obj_path in enumerate(obj_files, start=1):
        rel = get_output_relative_path(obj_path, input_root)
        out_path = os.path.join(output_root, rel)
        failed_path = os.path.join(failed_root, rel)

        if (os.path.exists(out_path) or os.path.exists(failed_path)) and not overwrite:
            skipped += 1
            if VERBOSE_PER_FILE:
                reason = []
                if os.path.exists(out_path):
                    reason.append("output")
                if os.path.exists(failed_path):
                    reason.append("failed")
                print(
                    f"{_paint('[SKIP]', COLOR_DIM)} {obj_path} | "
                    f"already processed in: {','.join(reason)}"
                )
            if i % 100 == 0 or i == total:
                print(
                    f"{_paint(f'[{i}/{total}]', COLOR_DIM)} "
                    f"repaired={_paint(repaired, COLOR_GREEN)} "
                    f"failed={_paint(failed, COLOR_RED if failed > 0 else COLOR_GREEN)} "
                    f"skipped={_paint(skipped, COLOR_YELLOW if skipped > 0 else COLOR_DIM)} "
                    f"complex={_paint(used_complex_count, COLOR_BLUE if used_complex_count > 0 else COLOR_DIM)} "
                    f"quality_failed={_paint(quality_failed_count, COLOR_YELLOW if quality_failed_count > 0 else COLOR_DIM)} "
                    f"failed_result_saved={_paint(failed_result_saved_count, COLOR_BLUE if failed_result_saved_count > 0 else COLOR_DIM)} "
                    f"residual_issues={_nm_tag(residual_issue_count)}"
                )
            continue

        try:
            reset_scene()
            import_obj(obj_path)

            stats = repair_scene_meshes(
                use_print3d=use_print3d,
                always_use_complex=ALWAYS_USE_COMPLEX_REPAIR,
                voxel_size=VOXEL_SIZE,
                decimate_ratio=DECIMATE_RATIO,
                max_hole_sides=MAX_HOLE_SIDES,
            )

            if stats["used_complex"]:
                used_complex_count += 1
            if not stats["final_ok"]:
                residual_issue_count += 1

            export_obj(out_path)

            if not stats["final_ok"]:
                quality_failed_count += 1
                saved_dst = save_failed_result_from_exported(
                    repaired_obj=out_path,
                    original_obj=obj_path,
                    failed_result_root=failed_result_root,
                    input_root=input_root,
                )
                if saved_dst is not None:
                    failed_result_saved_count += 1
                move_repaired_to_failed(
                    repaired_obj=out_path,
                    original_obj=obj_path,
                    failed_root=failed_root,
                    input_root=input_root,
                )
                print(
                    f"{_paint('[WARN]', COLOR_YELLOW)} final_ok=False -> moved to failed: "
                    f"{os.path.join(failed_root, get_output_relative_path(obj_path, input_root))}"
                )
                if saved_dst is not None:
                    print(
                        f"{_paint('[WARN]', COLOR_YELLOW)} failed result saved: {saved_dst}"
                    )
            else:
                repaired += 1

            if VERBOSE_PER_FILE:
                status_prefix = _paint("[OK]", COLOR_GREEN)
                print(
                    f"{status_prefix} {_paint('[DETAIL]', COLOR_BLUE)} {obj_path} | "
                    f"meshes={_paint(stats['mesh_count'], COLOR_CYAN)} "
                    f"nm_initial={_nm_tag(stats['non_manifold_initial'])} "
                    f"nm_after_basic={_nm_tag(stats['non_manifold_after_basic'])} "
                    f"err_edge_after_basic={_nm_tag(stats['error_continuous_after_basic'])} "
                    f"intersect_after_basic={_nm_tag(stats['self_intersections_after_basic'])} "
                    f"nonplanar_after_basic={_nm_tag(stats['non_planar_after_basic'])} "
                    f"components_after_basic={_nm_tag(stats['components_after_basic'])} "
                    f"basic_ok={_bool_tag(stats['basic_ok'])} "
                    f"complex={_bool_tag(stats['used_complex'])} "
                    f"retried_half={_bool_tag(stats['used_half_voxel_pass'])} "
                    f"nm_final={_nm_tag(stats['non_manifold_after'])} "
                    f"err_edge_final={_nm_tag(stats['error_continuous_after'])} "
                    f"intersect_final={_nm_tag(stats['self_intersections_after'])} "
                    f"nonplanar_final={_nm_tag(stats['non_planar_after'])} "
                    f"components_final={_nm_tag(stats['components_after'])} "
                    f"final_ok={_bool_tag(stats['final_ok'])}"
                )
            else:
                print(
                    f"{_paint('[OK]', COLOR_GREEN)} {obj_path} | "
                    f"complex={_bool_tag(stats['used_complex'])} "
                    f"retried_half={_bool_tag(stats['used_half_voxel_pass'])} "
                    f"final_ok={_bool_tag(stats['final_ok'])}"
                )
        except Exception:
            failed += 1
            print(f"{_paint('[ERROR]', COLOR_RED)} Failed: {obj_path}")
            print(_paint(traceback.format_exc(), COLOR_RED))
            saved_snapshot = save_failed_result_snapshot(
                rel=get_output_relative_path(obj_path, input_root),
                failed_result_root=failed_result_root,
            )
            if saved_snapshot is not None:
                failed_result_saved_count += 1
                print(
                    f"{_paint('[WARN]', COLOR_YELLOW)} failed snapshot saved: {saved_snapshot}"
                )
            move_to_failed(obj_path, failed_root, input_root)

        if i % 100 == 0 or i == total:
            print(
                f"{_paint(f'[{i}/{total}]', COLOR_DIM)} "
                f"repaired={_paint(repaired, COLOR_GREEN)} "
                f"failed={_paint(failed, COLOR_RED if failed > 0 else COLOR_GREEN)} "
                f"skipped={_paint(skipped, COLOR_YELLOW if skipped > 0 else COLOR_DIM)} "
                f"complex={_paint(used_complex_count, COLOR_BLUE if used_complex_count > 0 else COLOR_DIM)} "
                f"quality_failed={_paint(quality_failed_count, COLOR_YELLOW if quality_failed_count > 0 else COLOR_DIM)} "
                f"failed_result_saved={_paint(failed_result_saved_count, COLOR_BLUE if failed_result_saved_count > 0 else COLOR_DIM)} "
                f"residual_issues={_nm_tag(residual_issue_count)}"
            )

    print(f"\n{_paint('All done.', COLOR_BOLD)}")
    print(f"{_paint('Input root:  ', COLOR_CYAN)} {input_root}")
    print(f"{_paint('Output root: ', COLOR_CYAN)} {output_root}")
    print(f"{_paint('Failed root: ', COLOR_CYAN)} {failed_root}")
    print(f"{_paint('Failed result:', COLOR_CYAN)} {failed_result_root}")
    print(f"{_paint('Total:       ', COLOR_CYAN)} {_paint(total, COLOR_BOLD)}")
    print(f"{_paint('Repaired:    ', COLOR_GREEN)} {_paint(repaired, COLOR_GREEN)}")
    print(f"{_paint('Failed:      ', COLOR_RED)} {_paint(failed, COLOR_RED if failed > 0 else COLOR_GREEN)}")
    print(f"{_paint('Skipped:     ', COLOR_YELLOW)} {_paint(skipped, COLOR_YELLOW if skipped > 0 else COLOR_DIM)}")
    print(f"{_paint('Used complex:', COLOR_BLUE)} {_paint(used_complex_count, COLOR_BLUE if used_complex_count > 0 else COLOR_DIM)}")
    print(f"{_paint('Quality fail:', COLOR_YELLOW)} {_paint(quality_failed_count, COLOR_YELLOW if quality_failed_count > 0 else COLOR_DIM)}")
    print(f"{_paint('Saved failed result:', COLOR_BLUE)} {_paint(failed_result_saved_count, COLOR_BLUE if failed_result_saved_count > 0 else COLOR_DIM)}")
    print(f"{_paint('Residual issues: ', COLOR_CYAN)} {_nm_tag(residual_issue_count)}")


def main():
    input_root = os.path.abspath(INPUT_ROOT)
    output_root = os.path.abspath(OUTPUT_ROOT)
    failed_root = os.path.abspath(FAILED_ROOT)
    failed_result_root = os.path.abspath(FAILED_RESULT_ROOT)

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(failed_root, exist_ok=True)
    os.makedirs(failed_result_root, exist_ok=True)

    run_batch(
        input_root=input_root,
        output_root=output_root,
        failed_root=failed_root,
        failed_result_root=failed_result_root,
        overwrite=OVERWRITE,
        limit=LIMIT,
    )


if __name__ == "__main__":
    main()
