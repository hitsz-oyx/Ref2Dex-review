from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools/data/build_dexplore_v120_motion_input.py"
SPEC = importlib.util.spec_from_file_location("dexplore_v120_motion_input", TOOL)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def _write_inputs(root: Path, *, native_frames: int = 9) -> tuple[Path, Path]:
    legacy = root / "legacy"
    raw = root / "raw"
    (legacy / "s1_airplane_lift").mkdir(parents=True)
    (raw / "s1").mkdir(parents=True)
    (raw / "objects/airplane").mkdir(parents=True)
    (raw / "tools/subject_meshes/male").mkdir(parents=True)
    (raw / "objects/airplane/mesh.obj").write_bytes(b"mesh-bytes")
    (raw / "tools/subject_meshes/male/s1.ply").write_bytes(b"subject-mesh-bytes")
    body = {"params": {"transl": np.zeros((3, 3), dtype=np.float32)},
            "vtemp": "tools/subject_meshes/male/s1.ply"}
    np.savez(legacy / "s1_airplane_lift/motion.npz", n_frames=np.array(3), body=body,
             contact={"body": np.zeros((3, 2), dtype=np.int8), "object": np.zeros((3, 2), dtype=np.int8)})
    np.savez(legacy / "s1_airplane_lift/object.npz", angles=np.zeros((3, 3)), trans=np.zeros((3, 3)), name="airplane")
    contact = {"body": np.arange(native_frames * 2, dtype=np.int8).reshape(native_frames, 2),
               "object": np.arange(native_frames * 3, dtype=np.int8).reshape(native_frames, 3),
               "threshold": 0.01}
    np.savez(raw / "s1/airplane_lift.npz", contact=contact)
    return legacy, raw


def test_build_preserves_native_contact_and_legacy_object(tmp_path):
    legacy, raw = _write_inputs(tmp_path)
    manifest = ADAPTER.build(sequence="s1_airplane_lift", legacy_root=legacy, raw_root=raw,
                             raw_object_root=raw / "objects",
                             raw_tools_root=raw / "tools",
                             output_root=tmp_path / "output")
    with np.load(tmp_path / "output/sequences/s1_airplane_lift/motion.npz", allow_pickle=True) as result:
        contact = result["contact"].item()
        assert result["n_frames"].item() == 3
        assert np.array_equal(contact["body"], np.arange(18, dtype=np.int8).reshape(9, 2))
        assert np.array_equal(contact["object"], np.arange(27, dtype=np.int8).reshape(9, 3))
    assert manifest["classification"] == "reconstructed_baseline"
    assert (tmp_path / "output/sequences/s1_airplane_lift/object.npz").read_bytes() == (
        legacy / "s1_airplane_lift/object.npz").read_bytes()
    assert (tmp_path / "output/objects/airplane/airplane.obj").read_bytes() == b"mesh-bytes"
    assert (tmp_path / "output/tools/subject_meshes/male/s1.ply").read_bytes() == b"subject-mesh-bytes"


def test_build_rejects_downsampled_contact(tmp_path):
    legacy, raw = _write_inputs(tmp_path, native_frames=3)
    with pytest.raises(ValueError, match="native contact"):
        ADAPTER.build(sequence="s1_airplane_lift", legacy_root=legacy, raw_root=raw,
                      raw_object_root=raw / "objects",
                      raw_tools_root=raw / "tools",
                      output_root=tmp_path / "output")
