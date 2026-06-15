# Dataset Audit Override

## 1. Purpose

This directory adds local, shareable data-quality audit tools for TACO-Instructions. It is not official TACO-Instructions source code. The tools live under `dataset_audit/` and write generated outputs only under `outputs/dataset_audit/`.

## 2. Docker / Miniconda usage

Do not install dependencies or run dataset scripts directly on the macOS host. Enter the repository Docker/Miniconda environment first:

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate taco-instructions-dev
```

For MeshCat, expose container port `7000` on Mac port `7011`:

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name taco-instructions-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu native usage

On Ubuntu 22.04/24.04, run the audit tools in a native conda environment from the TACO-Instructions repository root:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n taco-instructions-dev python=3.9 -y
conda activate taco-instructions-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install meshcat rtree
```

Run a no-data smoke test before using real TACO-Instructions assets:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

Set `TACO_DATASET_ROOT` or `TACO_INSTRUCTIONS_DATASET_ROOT` to the downloaded dataset root, and set `TACO_OBJECT_MODEL_ROOT` when object models live outside that root. If the Ubuntu machine is remote, use `--headless-check` or forward the MeshCat port before opening the browser viewer.

## 4. Dataset availability

The audit tools do not download the TACO-Instructions dataset, object models, or MANO files. The manifest checks data-list files and local/env dataset roots, and reports missing assets explicitly.

The official project page and README provide two data entry points: the pre-release version on OneDrive, and the full **Whole Dataset Version 1** on Dropbox, with an additional Dropbox backup link. The pre-release version also has a BaiduNetDisk backup. See the [TACO project site](https://taco2024.github.io/) and `README_TACO.md`.

Download guidance:

1. For a quick data-structure check, the pre-release version is enough. It contains 244 high-quality motion sequences, 206 high-resolution object models, hand-object pose/mesh annotations, egocentric RGB-D videos, and 8-view allocentric RGB videos.
2. For a full audit or experiment reproduction, download Whole Dataset Version 1. It contains 2317 motion sequences, 206 object models, hand-object pose/mesh annotations, egocentric RGB-D videos, 12-view allocentric RGB videos, camera parameters, automatic hand-object 2D segmentations, and marker-removed allocentric RGB videos.
3. If using the BaiduNetDisk pre-release backup, some video archives are split. Merge them first:

```bash
cat Allocentric_RGB_Videos_split.* > Allocentric_RGB_Videos.zip
cat Egocentric_Depth_Videos_split.* > Egocentric_Depth_Videos.zip
```

After extraction, the dataset root should contain the directories described by the official README:

```text
TACO_Dataset_V1/
├── Allocentric_RGB_Videos/
├── Egocentric_Depth_Videos/
├── Egocentric_RGB_Videos/
├── Hand_Poses/
├── Object_Poses/
├── Object_Models/
└── Marker_Removed_Allocentric_RGB_Videos/
```

Expose the downloaded dataset to the audit tools:

```bash
export TACO_DATASET_ROOT=/path/to/TACO_Dataset_V1
export TACO_INSTRUCTIONS_DATASET_ROOT=$TACO_DATASET_ROOT
export TACO_OBJECT_MODEL_ROOT=$TACO_DATASET_ROOT/Object_Models
```

The official visualization code also needs `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` from the [MANO site](https://mano.is.tue.mpg.de/), placed at:

```text
dataset_utils/manopth/mano/models/
├── MANO_LEFT.pkl
└── MANO_RIGHT.pkl
```

The lists under `data_lists/` only describe available sequences and splits. They are not a substitute for the downloaded dataset, object models, or MANO files.

## 5. Visualization

```bash
python dataset_audit/scripts/dataset_manifest.py --out outputs/dataset_audit/manifest.csv
python dataset_audit/scripts/prepare_sequence_cache.py --item <sample_item>
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/cache/<sample>.npz --fps 3 --stride 20 --loop
```

For a no-data smoke test:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py --toy --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz --headless-check
```

## 6. Penetration / negative-distance statistics

```bash
python dataset_audit/scripts/check_point_sdf_penetration.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/penetration.json --out-csv outputs/dataset_audit/penetration.csv

python dataset_audit/scripts/sample_penetration_statistics.py \
  --manifest outputs/dataset_audit/manifest.csv --sample-size 5 --frames-per-seq 100 \
  --out-json outputs/dataset_audit/sample_penetration_statistics.json \
  --out-csv outputs/dataset_audit/sample_penetration_statistics.csv
```

`frame_min_signed_distance_mm` is the most negative signed distance for a checked frame. `frame_max_negative_depth_mm` is the maximum depth after converting negative signed distance to a non-negative penetration value. Sequence and aggregate summaries report means, variances, maxima, and frame ratios.

## 7. Surface collision

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

Surface collision reports intersection cues and triangle ratios, not stable penetration depth.

## 8. Contact information

The static audit does not identify official contact or penetration fields. TACO-Instructions provides hand-object pose/mesh annotation and visualization workflows. If downloaded annotations do not contain a direct contact field, contact should be derived from hand-object geometric distance thresholds.

## 9. Limitations

Visual meshes are not guaranteed collision meshes. Point-SDF on non-watertight objects is heuristic. Surface collision can count touch or fitting artifacts. MANO/object coordinate alignment must be validated with real TACO-Instructions sequences and MeshCat review.
