"""Link GRAB meshes and create minimal Dexplore URDFs for missing objects."""

import argparse
from pathlib import Path


URDF = '''<?xml version="1.0" ?>
<robot name="{name}.urdf">
  <link name="baseLink">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry>
      <mesh filename="objects/{name}/{name}.obj" scale="1 1 1"/>
    </geometry><material name="mat"><color rgba="0.7 0.8 0.9 1"/></material></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry>
      <mesh filename="objects/{name}/{name}.obj" scale="1 1 1"/>
    </geometry></collision>
  </link>
</robot>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-objects", required=True)
    parser.add_argument("--asset-root", required=True)
    args = parser.parse_args()
    raw_root = Path(args.raw_objects)
    asset_root = Path(args.asset_root)
    objects_root = asset_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)
    linked = generated = 0
    for source_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        name = source_dir.name
        dest_dir = objects_root / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.obj"
        source = source_dir / "mesh.obj"
        if not source.exists():
            raise FileNotFoundError(source)
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(source)
            linked += 1
        urdf = asset_root / f"{name}.urdf"
        if not urdf.exists():
            urdf.write_text(URDF.format(name=name))
            generated += 1
    print(f"linked {linked} meshes; generated {generated} URDFs under {asset_root}")


if __name__ == "__main__":
    main()
