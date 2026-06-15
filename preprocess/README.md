# preprocess Tools

This directory contains mesh preprocessing utilities used by Ref2Dex for manifold checking, filtering, and repair. The current scripts mainly target OBJ assets; some checking utilities also support STL/PLY.

Recommended processing order:

1. Run `check_manifold.py` on the original mesh assets to get an overall quality report.
2. Run `filter_manifold_objs.py` to split OBJ files into manifold and non-manifold outputs.
3. Repair non-manifold OBJ files with the Blender repair scripts.
4. Run `check_manifold.py` again on the repaired outputs.

## Environment Dependencies

### Regular Python Environment

`check_manifold.py` and `filter_manifold_objs.py` require a regular Python environment. Python 3.10 or 3.11 is recommended:

```bash
conda create -n ref2dex-preprocess python=3.10 -y
conda activate ref2dex-preprocess

python -m pip install --upgrade pip
python -m pip install numpy tqdm open3d
```

If PyTorch3D needs to be installed from source, prepare the basic build tools first:

```bash
# Build tools inside the conda environment.
conda install -c conda-forge cmake ninja -y

# Linux also needs a system compiler, for example on Ubuntu:
# sudo apt-get update && sudo apt-get install -y build-essential

# macOS needs Xcode Command Line Tools first:
# xcode-select --install
```

The regular checking/filtering scripts use `pytorch3d.io.load_obj/load_ply` to read and triangulate OBJ/PLY files. `filter_manifold_objs.py` only handles OBJ, while `check_manifold.py` handles OBJ/PLY/STL. Therefore, PyTorch and PyTorch3D are also required. PyTorch/PyTorch3D compatibility depends on Python, CUDA, and platform versions, so install a matching PyTorch build first, then install PyTorch3D:

```bash
# CPU example. If CUDA is available, replace this with the matching PyTorch install command.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Common source installation. If this fails, follow the official PyTorch3D instructions
# for your Python/Torch/CUDA combination.
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Quick dependency check:

```bash
python -c "import numpy, open3d, torch; from pytorch3d.io import load_obj, load_ply; print('python deps ok')"
```

### Blender Environment

`repair_non_manifold_with_blender.py` and `repair_non_manifold_with_blender_advanced.py` must be run with Blender, not regular `python`, because they depend on Blender built-in modules such as `bpy`, `bmesh`, and `mathutils`.

On Linux, this usually works:

```bash
blender --background --python preprocess/repair_non_manifold_with_blender.py
```

On macOS, if the `blender` command is not available, use the full executable path:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python preprocess/repair_non_manifold_with_blender.py
```

If Blender is not installed yet, install it with the system package manager on Linux, or use the official installer / Homebrew Cask on macOS:

```bash
brew install --cask blender
```

The Blender repair scripts only use Blender bundled modules, so `bpy` does not need to be installed in the conda environment. The scripts try to enable the Print3D add-on; if the current Blender installation does not provide the Print3D Make Manifold operator, the scripts skip that step and continue with the remaining repair operations.

## File Overview

### `check_manifold.py`

Batch-check mesh assets for manifoldness and basic geometry quality.

Supported formats:

- `.obj`
- `.stl`
- `.ply`

Main features:

- Recursively scan one or more input directories.
- Read and triangulate OBJ/PLY with PyTorch3D; read STL with Open3D.
- Check Open3D metrics: `is_watertight`, `is_edge_manifold`, `is_vertex_manifold`, `is_orientable`, and `is_self_intersecting`.
- Count original non-triangle faces in OBJ and ASCII PLY files.
- Write a detailed JSON report and a text summary.
- Support multi-process parallel checking.

Common command:

```bash
python preprocess/check_manifold.py \
  --input-dir get_assets/0_merged_visual_objs \
  --output-dir reports/manifold_check \
  --workers 8
```

Multiple input directories:

```bash
python preprocess/check_manifold.py \
  --input-dir get_assets/1_manifold_visual_objs get_assets/2_non_manifold_objs_repaired_advanced \
  --output-dir reports/manifold_check_after_repair \
  --workers 8
```

Common arguments:

- `--input-dir`: Required. Accepts one or more input directories.
- `--output-dir`: Report output directory. Default: `reports/manifold_check`.
- `--no-recursive`: Only scan the first level of each input directory.
- `--workers`: Number of worker processes. `0` means auto. Default: `0`.
- `--no-json`: Do not save JSON reports; only print the summary.
- `--quiet`: Quiet mode; only show the summary.

Exit codes:

- Returns `0` when all meshes are manifold and no parsing errors occur.
- Returns `1` when non-manifold meshes or parsing errors are found. If the command is only used to generate a report and should not interrupt a shell pipeline, append `|| true`.

Outputs:

- `manifold_check_YYYYmmdd_HHMMSS.json`: Full check report.
- `manifold_summary_YYYYmmdd_HHMMSS.txt`: Text summary.

### `filter_manifold_objs.py`

Recursively reads OBJ files from an input directory, splits them into manifold and non-manifold output directories, and writes a CSV report.

Main features:

- Only processes `.obj` files.
- Uses loading, cleanup, and Open3D checks similar to `check_manifold.py`.
- Writes passing OBJ files to `manifold_output_root`.
- Writes failing or parsing-error OBJ files to `non_manifold_output_root`.
- Writes `manifold_report.csv` with each OBJ file's manifold state, failure reason, and geometry statistics.
- Re-exports a simplified OBJ containing only vertices and triangular faces; if re-export fails, falls back to copying the original file.

Common command:

```bash
python preprocess/filter_manifold_objs.py \
  --input-root get_assets/0_merged_visual_objs \
  --manifold-output-root get_assets/1_manifold_visual_objs \
  --non-manifold-output-root get_assets/1_non_manifold_visual_objs \
  --report-csv get_assets/1_manifold_visual_objs/manifold_report.csv \
  --workers 10
```

Common arguments:

- `--input-root`: Input OBJ root directory. Default: `get_assets/0_merged_visual_objs`.
- `--manifold-output-root`: Output directory for manifold OBJ files. Default: `get_assets/1_manifold_visual_objs`.
- `--non-manifold-output-root`: Output directory for non-manifold OBJ files. Default: `get_assets/1_non_manifold_visual_objs`.
- `--report-csv`: CSV report path. Default: `get_assets/1_manifold_visual_objs/manifold_report.csv`.
- `--overwrite`: Overwrite existing output OBJ files.
- `--limit`: Only process the first N OBJ files. Useful for smoke tests.
- `--workers`: Number of worker processes. `1` means single process, `<=0` means auto. Default: `10`.

### `repair_non_manifold_with_blender.py`

Batch-repairs non-manifold OBJ files with a basic Blender-based pipeline.

Repair pipeline:

1. Import OBJ.
2. Try Print3D Make Manifold.
3. Merge duplicate vertices.
4. Fill holes.
5. Convert quads/polygons to triangles.
6. Recalculate normals consistently.
7. Export OBJ and remove the extra `.mtl` file.

Usage:

1. Edit the configuration at the top of the script:

```python
INPUT_ROOT = "/path/to/1_non_manifold_visual_objs"
OUTPUT_ROOT = "/path/to/2_non_manifold_objs_repaired"
FAILED_ROOT = "/path/to/2_non_manifold_objs_failed"

OVERWRITE = False
LIMIT = None
PRESERVE_REGISTRY_LEVEL = True
```

2. Run it with Blender in background mode:

```bash
blender --background --python preprocess/repair_non_manifold_with_blender.py
```

Use cases:

- The non-manifold issues are relatively simple.
- The original shape should be preserved as much as possible.
- A quick repair pass is needed before validating with `check_manifold.py`.

### `repair_non_manifold_with_blender_advanced.py`

Batch-repairs non-manifold OBJ files with a more aggressive Blender-based strategy. It includes basic cleanup, quality metric checks, voxel remeshing, decimation, and a half-voxel retry pass.

Advanced repair pipeline:

1. Import OBJ and record initial quality metrics.
2. Run basic cleanup on each mesh object.
3. Use `bmesh` to check non-manifold edges, boundary/wire edges, self-intersections, non-planar faces, and connected component count.
4. If basic cleanup still fails, run voxel remesh and decimate.
5. If it still fails and `RETRY_WITH_HALF_VOXEL_ON_FAIL` is enabled, retry with half the voxel size.
6. Results with `final_ok=True` are written to `OUTPUT_ROOT`.
7. Results with `final_ok=False` or exceptions are written to `FAILED_ROOT`; when possible, the repaired-but-still-invalid snapshot is also saved under `FAILED_RESULT_ROOT`.

Usage:

1. Edit the paths at the top of the script:

```python
INPUT_ROOT = "/path/to/1_non_manifold_visual_objs"
OUTPUT_ROOT = "/path/to/2_non_manifold_objs_repaired_advanced"
FAILED_ROOT = "/path/to/2_non_manifold_objs_failed_advanced"
FAILED_RESULT_ROOT = "/path/to/2_non_manifold_failed_results_advanced"
```

2. Adjust advanced parameters for the asset scale:

```python
ALWAYS_USE_COMPLEX_REPAIR = False
VOXEL_SIZE = 0.001
DECIMATE_RATIO = 0.1
MAX_HOLE_SIDES = 0
RETRY_WITH_HALF_VOXEL_ON_FAIL = True
```

3. Run it with Blender in background mode:

```bash
blender --background --python preprocess/repair_non_manifold_with_blender_advanced.py
```

Key parameters:

- `ALWAYS_USE_COMPLEX_REPAIR`: Whether every file should always go through voxel remesh + decimate. By default, this is only used when basic cleanup fails.
- `VOXEL_SIZE`: Voxel remesh size. Smaller values preserve more detail but are slower and may generate more faces.
- `DECIMATE_RATIO`: Decimation ratio. Smaller values reduce face count more aggressively but lose more detail.
- `MAX_HOLE_SIDES`: Maximum side count for Blender hole filling. `0` means fill as many holes as possible.
- `VERBOSE_PER_FILE`: Whether to print detailed metrics for every file.
- `SUPPRESS_BLENDER_OP_LOGS`: Whether to hide verbose Blender operator logs.
- `NON_PLANAR_EPSILON`: Tolerance for non-planar face detection.
- `RETRY_WITH_HALF_VOXEL_ON_FAIL`: Whether to retry with half the voxel size after failure.

Use cases:

- Basic repair still leaves non-manifold edges, boundary edges, self-intersections, or non-planar faces.
- Voxel remesh topology and shape changes are acceptable.
- Failed `final_ok=False` results need to be preserved for manual inspection.

## Recommended Workflow Example

```bash
# 1. Check the original merged OBJ files.
python preprocess/check_manifold.py \
  --input-dir get_assets/0_merged_visual_objs \
  --output-dir reports/manifold_check_raw \
  --workers 8 || true

# 2. Split manifold and non-manifold OBJ files.
python preprocess/filter_manifold_objs.py \
  --input-root get_assets/0_merged_visual_objs \
  --manifold-output-root get_assets/1_manifold_visual_objs \
  --non-manifold-output-root get_assets/1_non_manifold_visual_objs \
  --report-csv get_assets/1_manifold_visual_objs/manifold_report.csv \
  --workers 10

# 3. Edit the paths at the top of repair_non_manifold_with_blender_advanced.py,
# then repair with Blender.
blender --background --python preprocess/repair_non_manifold_with_blender_advanced.py

# 4. Check the repaired outputs.
python preprocess/check_manifold.py \
  --input-dir get_assets/2_non_manifold_objs_repaired_advanced \
  --output-dir reports/manifold_check_repaired \
  --workers 8 || true
```

## Notes

- Run the regular Python scripts from the repository root, or pass absolute paths, so default relative paths do not point to the wrong location.
- The Blender repair scripts currently do not expose command-line arguments. Paths and repair parameters must be edited at the top of each script.
- `filter_manifold_objs.py` and `check_manifold.py` rely on PyTorch3D for OBJ/PLY reading. If only Open3D is installed, OBJ/PLY loading will still fail.
- `check_manifold.py` does not precisely count original non-triangle faces for binary PLY files; it records a face-check warning in the report instead.
- Blender voxel remesh changes topology and detail. For assets that must preserve the original geometry closely, try the basic repair script first, then use the advanced repair script only if needed.
- Always rerun `check_manifold.py` after repair. Do not rely only on Blender script logs to decide whether assets are usable.
- `PRESERVE_REGISTRY_LEVEL=True` tries to preserve registry directory levels such as `aigen_objs`, `objaverse`, and `lightwheel`, making outputs easier to map back to their original asset source.
