# Dataset Audit Override

## 1. Purpose

This directory adds local, shareable data-quality audit tools for ARCTIC. It is not official ARCTIC source code. The tools live under `dataset_audit/` and write generated outputs only under `outputs/dataset_audit/`.

## 2. Docker / Miniconda usage

Do not install dependencies or run dataset scripts directly on the macOS host. Enter the repository Docker/Miniconda environment first:

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate arctic-dev
```

For MeshCat, expose container port `7000` on Mac port `7011`:

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name arctic-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu native usage

On Ubuntu 22.04/24.04, run the audit tools in a native conda environment from the ARCTIC repository root:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n arctic-dev python=3.9 -y
conda activate arctic-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install meshcat rtree
```

Run a no-data smoke test before using real ARCTIC assets:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

For real data, place ARCTIC under `data/arctic_data/`, body models under `data/body_models/`, and object templates under `data/arctic_data/data/meta/object_vtemplates/`. If the Ubuntu machine is remote, use `--headless-check` for validation or forward the MeshCat port before opening the browser viewer.

## 4. Dataset availability

The audit tools do not download ARCTIC data, MANO/SMPL-X files, or object assets. They can reuse existing local caches such as `outputs/meshcat_cache/*_world_verts.npz` and can enrich them with object faces from `data/arctic_data/data/meta/object_vtemplates/` when available.

## 5. Visualization

```bash
python dataset_audit/scripts/dataset_manifest.py --out outputs/dataset_audit/manifest.csv
python dataset_audit/scripts/prepare_sequence_cache.py --item s01/capsulemachine_use_01
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/cache/arctic_s01_capsulemachine_use_01_mesh_sequence.npz --fps 3 --stride 20 --loop
```

For a non-browser cache check:

```bash
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz --headless-check
```

## 6. Penetration / negative-distance statistics

```bash
python dataset_audit/scripts/check_point_sdf_penetration.py \
  --cache outputs/dataset_audit/cache/arctic_s01_capsulemachine_use_01_mesh_sequence.npz \
  --out-json outputs/dataset_audit/arctic_capsulemachine_penetration.json \
  --out-csv outputs/dataset_audit/arctic_capsulemachine_penetration.csv \
  --surface-mode vertices --stride 50

python dataset_audit/scripts/sample_penetration_statistics.py \
  --manifest outputs/dataset_audit/manifest.csv \
  --sample-size 5 --frames-per-seq 100 \
  --out-json outputs/dataset_audit/sample_penetration_statistics.json \
  --out-csv outputs/dataset_audit/sample_penetration_statistics.csv
```

`frame_min_signed_distance_mm < 0` means a hand/body query point is inside the object or collision proxy. `frame_max_negative_depth_mm` is the corresponding non-negative penetration depth. Sequence fields include `mean_frame_max_negative_depth_mm`, `var_frame_max_negative_depth_mm`, `max_frame_max_negative_depth_mm`, `total_inside_ratio`, `raw_negative_frame_ratio`, `minor_penetration_frame_ratio`, and `significant_penetration_frame_ratio`.

## 7. Surface collision

```bash
python dataset_audit/scripts/check_surface_collision.py \
  --cache outputs/dataset_audit/cache/arctic_s01_capsulemachine_use_01_mesh_sequence.npz \
  --out-json outputs/dataset_audit/arctic_capsulemachine_surface_collision.json \
  --out-csv outputs/dataset_audit/arctic_capsulemachine_surface_collision.csv \
  --stride 100
```

Surface collision checks triangle-center inside/object-proxy overlap and reports collision ratios. It does not report stable penetration depth.

## 8. Contact information

ARCTIC does not provide direct manual force/contact labels. Its InterField path provides nearest-distance fields such as `dist.ro`, `dist.lo`, `dist.or`, and `dist.ol`; contact is typically derived with a threshold such as `dist < contact_bnd`. This is geometry-distance-derived contact, not force contact and not a penetration/collision test.

## 9. Limitations

Point-SDF reliability depends on watertight object meshes or a collision proxy. Non-watertight visual meshes are marked `heuristic`. Surface collision can flag boundary touch, coplanar touch, or mesh fitting artifacts, and it does not output penetration depth. Review suspicious frames in MeshCat before treating statistics as dataset labels.
