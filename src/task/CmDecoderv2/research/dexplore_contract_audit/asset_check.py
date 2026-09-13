"""Load the real Isaac Gym asset to check native DOF names, without a rollout."""
from isaacgym import gymapi

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.up_axis = gymapi.UP_AXIS_Z
    params.physx.use_gpu = False
    params.use_gpu_pipeline = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create the asset-check simulator")
    try:
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        asset = gym.load_asset(sim, str(args.urdf.resolve().parent), args.urdf.name, options)
        actual = gym.get_asset_dof_names(asset)
        xml = ET.parse(args.urdf).getroot()
        urdf_names = [e.get("name") for e in xml.findall("joint") if e.get("type") != "fixed"]
        native_to_urdf = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9]
        expected = [urdf_names[i] for i in native_to_urdf]
        if actual != expected:
            raise AssertionError(f"Isaac Gym DOF order differs: {actual} != {expected}")
        args.output.write_text(json.dumps({"passed": True, "urdf": str(args.urdf.resolve()),
                                          "native_dof_names": actual, "expected_dof_names": expected,
                                          "physics_steps": 0, "policy_rollout": False}, indent=2) + "\n")
        print(args.output.read_text())
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
