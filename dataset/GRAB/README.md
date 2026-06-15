# Dataset Audit Override

## 1. Purpose

This directory adds local, shareable data-quality audit tools for GRAB. It is not official GRAB source code. The tools live under `dataset_audit/` and write generated outputs only under `outputs/dataset_audit/`.

## 2. Docker / Miniconda usage

Do not install dependencies or run dataset scripts directly on the macOS host. Enter the repository Docker/Miniconda environment first:

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate grab-dev
```

For MeshCat, expose container port `7000` on Mac port `7011`:

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name grab-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu native usage

On Ubuntu 22.04/24.04, run the audit tools in a native conda environment from the GRAB repository root:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n grab-dev python=3.9 -y
conda activate grab-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install meshcat rtree
```

Run a no-data smoke test before using real GRAB assets:

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

Set `GRAB_DATASET_PATH` or `GRAB_PATH` to the downloaded GRAB motion data root. Set `GRAB_OBJECT_ROOT` when object assets live outside the data root. If the Ubuntu machine is remote, use `--headless-check` or forward the MeshCat port before opening the browser viewer.

## 4. Dataset availability

The audit tools do not download GRAB data, SMPL-X, MANO, or object assets. The manifest checks `GRAB_DATASET_PATH`, `GRAB_PATH`, and local data directories. Missing data is reported explicitly.

Follow the official flow: register and log in on the [GRAB site](https://grab.is.tue.mpg.de/), then accept the license. The Download section is only available after login. The site also requires following its instructions to get `object_meshes.zip`. The preserved upstream README in this workspace is `README_GRAB.md`.

Download flow:

1. Download all GRAB dataset ZIP files from the official Download section. Do not unzip them manually first.
2. Make sure `object_meshes.zip` has also been obtained as instructed by the official site; otherwise mesh visualization and audit cache generation cannot fully reconstruct objects.
3. Put all GRAB ZIP files in one directory, for example `/storage/downloads/grab_zips/`.
4. Use the official unzip script:

```bash
python grab/unzip_grab.py \
  --grab-path /storage/downloads/grab_zips \
  --extract-path /storage/data/GRAB
```

The extracted data should look like:

```text
GRAB/
├── grab/
│   ├── s1/
│   ├── ...
│   └── s10/
├── tools/
│   ├── object_meshes/
│   ├── object_settings/
│   ├── subject_meshes/
│   ├── subject_settings/
│   └── smplx_correspondence/
└── mocap/                 # optional
```

Set the paths used by the audit tools:

```bash
export GRAB_DATASET_PATH=/storage/data/GRAB
export GRAB_PATH=$GRAB_DATASET_PATH
export GRAB_OBJECT_ROOT=$GRAB_DATASET_PATH/tools/object_meshes
```

Official preprocessing, vertex extraction, and visualization also require SMPL-X and MANO models from the [SMPL-X site](https://smpl-x.is.tue.mpg.de/). Pass the model directory through `--model-path`, for example:

```bash
export SMPLX_MODEL_FOLDER=/storage/models/smplx
python grab/grab_preprocessing.py \
  --grab-path "$GRAB_DATASET_PATH" \
  --model-path "$SMPLX_MODEL_FOLDER" \
  --out-path outputs/grab_preprocessed
```

The GRAB license has explicit restrictions on subject data and allowed use. Do not commit downloaded data, object meshes, SMPL-X/MANO models, or generated caches to Git.

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

The unified convention is `signed_distance_mm < 0` for body/hand query points inside the object or proxy. `frame_max_negative_depth_mm` is the maximum non-negative depth in a frame. Aggregate summaries average per-sequence mean/variance/max depth and frame ratios.

## 7. Surface collision

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

Surface collision is separate from point-SDF penetration depth. It can flag surface overlap/touch but does not output stable depth.

## 8. Contact information

GRAB has official body-object binary contact maps. For each motion frame, object vertices are labeled as no contact or contact with a body/hand part. This is a proximity/contact label, not a signed-distance penetration or triangle-collision test. It should be recorded separately from geometric negative-distance statistics.

## 9. Limitations

GRAB contact maps and geometric penetration checks answer different questions. Visual meshes are not guaranteed collision meshes. Point-SDF requires watertight meshes or proxies for reliable signs. Surface collision may flag touch and fitting artifacts. Validate SMPL-X/MANO/object alignment and review suspicious frames in MeshCat.
