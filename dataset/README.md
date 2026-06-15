# Ref2Dex Dataset Workspace

This folder groups the local dataset source folders used for Ref2Dex-style hand/body-object interaction auditing:

```text
dataset/
  arctic/
  OakInk/
  OakInk2/
  TACO-Instructions/
  GRAB/
```

Each subdirectory is vendored into Ref2Dex as a normal source folder. The upstream `.git` metadata has been removed, so GitHub will show the files directly under this repository instead of showing submodule links.

## 1. Folder Checkout and Upload Model

Clone this repository normally. No submodule command is required:

```bash
git clone <ref2dex-repo-url>
cd Ref2Dex
```

The `dataset/` tree should be committed from the Ref2Dex parent repository like ordinary files. Do not restore nested `.git` directories unless you intentionally want to reintroduce submodules or separate upstream checkouts.

## 2. Environment Setup

### macOS / Docker path

Do not install dependencies or run dataset processing directly on the macOS host. Use each repository's Docker/Miniconda environment:

```bash
cd /Users/key_z/Documents/github/dex/Ref2Dex/dataset/<repo>
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate <repo-dev-env>
```

Environment names:

```text
arctic              arctic-dev
OakInk              oakink-dev
OakInk2             oakink2-dev
TACO-Instructions   taco-instructions-dev
GRAB                grab-dev
```

### Ubuntu native path

On Ubuntu 22.04/24.04, a native conda environment is acceptable for audit scripts and lightweight dataset inspection:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

cd dataset/<repo>
conda create -n <repo-dev-env> python=<python-version> -y
conda activate <repo-dev-env>
python -m pip install --upgrade pip
python -m pip install numpy scipy trimesh meshcat rtree
```

Dataset-specific environment configuration:

| Dataset | Env | Python | Official/toolkit dependencies | Data root variables used by audit tools |
| --- | --- | --- | --- | --- |
| ARCTIC | `arctic-dev` | `3.9` | `python -m pip install -r requirements.txt` | local `data/arctic_data/` and `outputs/meshcat_cache/` |
| OakInk | `oakink-dev` | `3.9` | `python -m pip install -r requirements.txt`; adjust PyTorch/PyTorch3D for the Ubuntu CUDA stack | `OAKINK_DIR` or local `data/`, `OakInk/`, `oakink/` |
| OakInk2 | `oakink2-dev` | `3.10` | `python -m pip install -e .`; use `req_preview.txt` only for the full preview stack | `OAKINK2_DIR`, `OAKINK2_DATASET_PREFIX`, or local `OakInk-v2-hub/`, `data/` |
| TACO-Instructions | `taco-instructions-dev` | `3.9` | `python -m pip install -r requirements.txt` | `TACO_DATASET_ROOT`, `TACO_INSTRUCTIONS_DATASET_ROOT`, `TACO_OBJECT_MODEL_ROOT` |
| GRAB | `grab-dev` | `3.9` | `python -m pip install -r requirements.txt` | `GRAB_DATASET_PATH`, `GRAB_PATH`, `GRAB_OBJECT_ROOT` |

OakInk2's full preview stack references third-party source folders under `thirdparty/`. In this no-submodule layout, those folders must be copied or vendored as ordinary files before installing `req_preview.txt`. The lightweight audit smoke test does not require them.

After installing the lightweight audit dependencies, every dataset repo should pass the no-data smoke test:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

The tools never download datasets, MANO/SMPL-X models, or model weights automatically. Download licenses and model assets must be handled manually according to each official dataset.

## 3. Data Download and Asset Placement

Use the original dataset documentation for licensing and download links. In this workspace, the original upstream README content is kept in repository-specific backup files when present, such as `README_ARCTIC.md`, `README_OAKINK.md`, `README_OAKINK2.md`, `README_TACO.md`, and `README_GRAB.md`.

Recommended local data roots:

| Dataset | Expected local inputs |
| --- | --- |
| ARCTIC | `data/arctic_data/`, `data/body_models/`, object templates under `data/arctic_data/data/meta/object_vtemplates/` |
| OakInk | `OAKINK_DIR` or local `data/` with OakInk image/shape data |
| OakInk2 | `OakInk-v2-hub/` or `data/`, plus MANO/SMPL-X assets for preview/toolkit reconstruction |
| TACO-Instructions | dataset root, object model root, and MANO files under the official visualization path |
| GRAB | `GRAB_DATASET_PATH` / `GRAB_PATH`, SMPL-X/MANO models, and object assets |

Generated files and local data are ignored by `.gitignore`: `outputs/`, `.codex/`, downloaded archives, model files, dataset roots, and Python/macOS caches should not be uploaded to GitHub.

## 4. Manifest and Cache Workflow

Inside one dataset repository:

```bash
python dataset_audit/scripts/dataset_manifest.py \
  --out outputs/dataset_audit/manifest.csv \
  --limit 5
```

For a real sequence:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py --item <sample_item>
```

For a dependency-light smoke test:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
```

The shared cache format is an `.npz` mesh sequence with fields such as:

```text
verts_right / verts_left / verts_hand / verts_body
verts_object
faces_right / faces_left / faces_hand / faces_body
faces_object
object_name
subject
sequence
dataset_name
frame_indices
source_paths
contact_available
contact_source
```

Adapters are responsible for translating each dataset's original files into this shared format.

## 5. MeshCat Visualization

Run MeshCat from the dataset container and map container port `7000` to Mac port `7011`:

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name <repo>-meshcat \
  -p 7011:7000 \
  dev bash
```

Then run:

```bash
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/cache/<sample>.npz \
  --fps 3 \
  --stride 20 \
  --loop
```

For a non-browser validation:

```bash
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

## 6. Penetration and Negative-Distance Statistics

The audit convention is:

```text
signed_distance_mm < 0  query point is inside the object/proxy
signed_distance_mm = 0  contact boundary
signed_distance_mm > 0  outside
negative_distance_mm = min(signed_distance_mm, 0)
penetration_depth_mm = max(0, -signed_distance_mm)
```

Run point-SDF/negative-distance audit:

```bash
python dataset_audit/scripts/check_point_sdf_penetration.py \
  --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/penetration.json \
  --out-csv outputs/dataset_audit/penetration.csv \
  --surface-mode vertices \
  --stride 20
```

Run multi-sequence aggregation:

```bash
python dataset_audit/scripts/sample_penetration_statistics.py \
  --manifest outputs/dataset_audit/manifest.csv \
  --sample-size 5 \
  --frames-per-seq 100 \
  --out-json outputs/dataset_audit/sample_penetration_statistics.json \
  --out-csv outputs/dataset_audit/sample_penetration_statistics.csv
```

Important fields:

| Field | Meaning |
| --- | --- |
| `frame_min_signed_distance_mm` | Most negative signed distance in a frame |
| `frame_max_negative_depth_mm` | Maximum non-negative penetration depth in a frame |
| `mean_frame_max_negative_depth_mm` | Per-sequence mean of frame maximum depths |
| `var_frame_max_negative_depth_mm` | Per-sequence variance of frame maximum depths |
| `max_frame_max_negative_depth_mm` | Per-sequence maximum depth |
| `total_inside_ratio` | Query points inside object/proxy divided by all query points |
| `raw_negative_frame_ratio` | Fraction of frames with at least one inside point |
| `minor_penetration_frame_ratio` | Fraction of frames above minor depth/ratio thresholds |
| `significant_penetration_frame_ratio` | Fraction of frames above significant depth/ratio/count thresholds |

## 7. Surface Collision

```bash
python dataset_audit/scripts/check_surface_collision.py \
  --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json \
  --out-csv outputs/dataset_audit/surface_collision.csv \
  --stride 50
```

Surface collision is not penetration depth. It reports surface-overlap/intersection cues and triangle ratios. It can count boundary touch, coplanar touch, or visual-mesh fitting artifacts, so review suspicious frames in MeshCat.

## 8. Mesh Topology

```bash
python dataset_audit/scripts/audit_object_mesh_topology.py \
  --limit 20 \
  --out-csv outputs/dataset_audit/object_mesh_topology.csv \
  --out-json outputs/dataset_audit/object_mesh_topology_summary.json
```

This reports watertightness, boundary edges, non-manifold edges, connected components, Euler number, volume, area, and bounding boxes. Non-watertight visual meshes are marked as heuristic for point-SDF signs.

## 9. Contact Information

```bash
python dataset_audit/scripts/inspect_contact_info.py \
  --out-json outputs/dataset_audit/contact_info_summary.json
```

Dataset-specific summary:

| Dataset | Contact information |
| --- | --- |
| ARCTIC | InterField nearest-distance fields can derive contact by threshold; not force contact and not penetration |
| OakInk | No direct contact annotation identified in the static audit; derive from hand-object distance if needed |
| OakInk2 | Affordance/primitive metadata is semantic/task information, not per-frame geometric contact |
| TACO-Instructions | No direct contact/penetration field identified in the static audit; derive from distance if needed |
| GRAB | Official body-object binary contact map labels object vertices by body/hand part contact; not penetration depth |

## 10. Implementation Logic

The audit layer is intentionally separate from official dataset code:

1. `adapters/*_adapter.py` discovers local data, reads existing caches, and records dataset-specific contact notes.
2. `common/mesh_io.py` loads/saves the shared `.npz` cache and synthetic toy cache.
3. `common/topology.py` computes mesh topology metrics without requiring full dataset processing.
4. `common/point_sdf.py` queries hand/body surface points against object meshes. If `trimesh` is available it uses `trimesh.proximity.signed_distance` and converts the sign to the shared convention; otherwise it falls back to a lightweight triangle-distance plus ray-cast heuristic.
5. `common/surface_collision.py` provides a lightweight surface-overlap proxy for smoke tests and review routing.
6. `scripts/*.py` are stable command-line entry points for manifest generation, cache preparation, visualization, penetration statistics, topology audit, and contact inspection.

Use point-SDF statistics, topology reliability, surface collision, and MeshCat review together. No single metric should be treated as ground-truth contact or collision.
