# Dataset Audit Override

## 1. Purpose

This directory adds local, shareable data-quality audit tools for OakInk2. It is not official OakInk2 source code. The tools live under `dataset_audit/` and write generated outputs only under `outputs/dataset_audit/`.

## 2. Docker / Miniconda usage

Do not install dependencies or run dataset scripts directly on the macOS host. Enter the repository Docker/Miniconda environment first:

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate oakink2-dev
```

For MeshCat, expose container port `7000` on Mac port `7011`:

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name oakink2-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu native usage

On Ubuntu 22.04/24.04, run the audit tools in a native conda environment from the OakInk2 repository root:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n oakink2-dev python=3.10 -y
conda activate oakink2-dev
python -m pip install --upgrade pip
python -m pip install numpy scipy trimesh meshcat rtree
python -m pip install -e .
```

Use `python -m pip install -r req_preview.txt` only when you need the full official preview stack. That file pins CUDA/PyTorch wheels and references third-party source folders under `thirdparty/`; in this vendored no-submodule layout, copy those folders in as ordinary files before installing it.

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

Set `OAKINK2_DIR` or `OAKINK2_DATASET_PREFIX` to the downloaded OakInk2 root, or place the hub under local `OakInk-v2-hub/` or `data/`. If the Ubuntu machine is remote, use `--headless-check` or forward the MeshCat port before opening the browser viewer.

## 4. Dataset availability

The audit tools do not download OakInk2 tarballs, MANO, or SMPL-X assets. The manifest checks `OakInk-v2-hub`, `data`, and relevant environment variables. Missing data is reported rather than downloaded.

The official OakInk2 data is hosted on [Hugging Face OakInk-v2](https://huggingface.co/datasets/kelvin34501/OakInk-v2), and the [OakInk2 project site](https://oakink.net/v2/) Download section points to the same page. The official README notes that some annotation files were re-uploaded after 2024-12, so keep the Hugging Face snapshot at the latest commit when using the dataset; since 2025-04, `object_preview.tar` is also provided again for backward compatibility. See `README_OAKINK2.md` and `script/download.py` for the upstream instructions kept in this workspace.

The recommended path is to download the snapshot from the OakInk2 repository root. Make sure there is enough space for both the Hugging Face cache and local directory:

```bash
python -m pip install -U huggingface_hub
python script/download.py
```

The equivalent manual command is:

```bash
huggingface-cli download kelvin34501/OakInk-v2 \
  --repo-type dataset \
  --local-dir ./OakInk-v2-hub \
  --cache-dir ./hub
```

At minimum, you need the `data/` tarball and matching `anno_preview/` annotation for at least one sequence, plus `object_raw.tar`, `object_repair.tar`, `object_affordance.tar`, and `program.tar`; `object_preview.tar` is only for old-path compatibility. After download, extract or arrange the files under a single dataset prefix:

```text
data/
├── data/
│   └── scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS/
├── anno_preview/
│   └── scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.pkl
├── object_preview/        # deprecated, old-path compatibility only
├── object_raw/
├── object_repair/
├── object_affordance/
└── program/
```

If using `OakInk-v2-hub/` directly as the data root, set:

```bash
export OAKINK2_DIR=$PWD/OakInk-v2-hub
export OAKINK2_DATASET_PREFIX=$PWD/OakInk-v2-hub
```

If you extract the tarballs into local `data/`, environment variables are optional because the manifest checks local `data/` automatically. The official preview tool also needs SMPL-X v1.1 from the [SMPL-X site](https://smpl-x.is.tue.mpg.de/download.php), placed under `asset/smplx_v1_1/`. The optional segment preview needs MANO v1.2 from the [MANO site](https://mano.is.tue.mpg.de/), placed under `asset/mano_v1_2/`. Do not commit these model files or download caches to Git.

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

The signed-distance convention is unified across datasets: negative means the query hand/body point is inside the object or proxy, and `frame_max_negative_depth_mm` is the maximum non-negative depth for that frame.

## 7. Surface collision

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

Surface collision is an overlap/intersection cue. It is not a penetration-depth metric.

## 8. Contact information

The static audit does not identify a direct built-in penetration CLI/API. OakInk2 has affordance and primitive task metadata when data is present. Affordance describes object-part function or task semantics; it is not the same as per-frame geometric contact. If no direct contact field is present, derive contact by thresholding hand-object distance.

## 9. Limitations

Point-SDF signs are reliable only with watertight meshes or proxies. Non-watertight visual meshes are heuristic. Surface collision may count boundary touch or mesh fitting noise. OakInk2 preview/toolkit reconstruction must be validated per sequence before aggregate statistics are trusted.
