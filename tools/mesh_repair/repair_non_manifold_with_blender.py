#!/usr/bin/env python3
"""
Use Blender to repair non-manifold OBJ assets in batch.

Run example:
1) Edit INPUT_ROOT / OUTPUT_ROOT / FAILED_ROOT below.
2) Run:
     blender --background --python get_assets/repair_non_manifold_with_blender.py
"""

import os
import traceback
import shutil

import bpy

# ===== 在这里设置你的文件夹路径 =====
INPUT_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/1_non_manifold_visual_objs"
OUTPUT_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/2_non_manifold_objs_repaired"
FAILED_ROOT = "/home/yang/Projects/Dataset/robocasa/get_assets/2_non_manifold_objs_failed"

# 可选设置
OVERWRITE = False
LIMIT = None  # 例如 100；None 表示处理全部
PRESERVE_REGISTRY_LEVEL = True
# =================================

REGISTRY_NAMES = {"aigen_objs", "objaverse", "lightwheel"}

def reset_scene():
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
    filepath_str = filepath

    try:
        bpy.ops.wm.obj_import(filepath=filepath_str)
        return
    except Exception:
        pass

    try:
        bpy.ops.import_scene.obj(filepath=filepath_str)
        return
    except Exception as exc:
        raise RuntimeError(f"Failed to import OBJ: {filepath_str}") from exc


def export_obj(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    filepath_str = filepath

    try:
        bpy.ops.wm.obj_export(
            filepath=filepath_str,
            export_selected_objects=True,
            export_materials=False,
            export_uv=False,
            export_normals=True,
            export_triangulated_mesh=True,
        )
    except Exception:
        bpy.ops.export_scene.obj(
            filepath=filepath_str,
            use_selection=True,
            use_materials=False,
            use_normals=True,
            use_triangles=True,
        )

    mtl_path = os.path.splitext(filepath)[0] + ".mtl"
    if os.path.exists(mtl_path):
        os.remove(mtl_path)


def fix_active_object_geometry(use_print3d: bool):
    obj = bpy.context.active_object
    if obj is None:
        raise RuntimeError("No active object after import")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    if use_print3d:
        try:
            bpy.ops.mesh.print3d_make_manifold()
        except Exception:
            pass

    try:
        bpy.ops.mesh.remove_doubles()
    except Exception:
        try:
            bpy.ops.mesh.merge_by_distance()
        except Exception:
            pass

    try:
        bpy.ops.mesh.fill_holes(sides=0)
    except Exception:
        pass

    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.mesh.normals_make_consistent(inside=False)

    bpy.ops.object.mode_set(mode="OBJECT")


def move_to_failed(src_obj, failed_root, input_root):
    rel = get_output_relative_path(src_obj, input_root)
    dst = os.path.join(failed_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.abspath(src_obj) != os.path.abspath(dst):
        shutil.copy2(src_obj, dst)


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
    overwrite: bool,
    limit: int,
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

    use_print3d = ensure_print3d_addon()
    print(f"Print3D Make Manifold available: {use_print3d}")

    for i, obj_path in enumerate(obj_files, start=1):
        rel = get_output_relative_path(obj_path, input_root)
        out_path = os.path.join(output_root, rel)

        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            if i % 100 == 0 or i == total:
                print(f"[{i}/{total}] repaired={repaired} failed={failed} skipped={skipped}")
            continue

        try:
            reset_scene()
            import_obj(obj_path)

            selected = bpy.context.selected_objects
            if not selected:
                raise RuntimeError("No object selected after import")

            obj = selected[0]
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            fix_active_object_geometry(use_print3d=use_print3d)
            export_obj(out_path)
            repaired += 1
        except Exception:
            failed += 1
            print(f"[ERROR] Failed: {obj_path}")
            print(traceback.format_exc())
            move_to_failed(obj_path, failed_root, input_root)

        if i % 100 == 0 or i == total:
            print(f"[{i}/{total}] repaired={repaired} failed={failed} skipped={skipped}")

    print("\nAll done.")
    print(f"Input root:   {input_root}")
    print(f"Output root:  {output_root}")
    print(f"Failed root:  {failed_root}")
    print(f"Total:        {total}")
    print(f"Repaired:     {repaired}")
    print(f"Failed:       {failed}")
    print(f"Skipped:      {skipped}")


def main():
    input_root = os.path.abspath(INPUT_ROOT)
    output_root = os.path.abspath(OUTPUT_ROOT)
    failed_root = os.path.abspath(FAILED_ROOT)

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(failed_root, exist_ok=True)

    run_batch(
        input_root=input_root,
        output_root=output_root,
        failed_root=failed_root,
        overwrite=OVERWRITE,
        limit=LIMIT,
    )


if __name__ == "__main__":
    main()
