# Dataset Audit Override

## 1. Purpose

This directory adds local, shareable data-quality audit tools for OakInk. It is not official OakInk source code. The tools live under `dataset_audit/` and write generated outputs only under `outputs/dataset_audit/`.

## 2. Docker / Miniconda usage

Do not install dependencies or run dataset scripts directly on the macOS host. Enter the repository Docker/Miniconda environment first:

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate oakink-dev
```

For MeshCat, expose container port `7000` on Mac port `7011`:

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name oakink-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu native usage

On Ubuntu 22.04/24.04, run the audit tools in a native conda environment from the OakInk repository root:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n oakink-dev python=3.9 -y
conda activate oakink-dev
python -m pip install --upgrade pip
python -m pip install numpy scipy trimesh meshcat rtree
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PyTorch, PyTorch3D, or Open3D wheels do not match the Ubuntu CUDA stack, install the matching wheels first and then rerun the remaining requirements. For audit-only smoke tests, the lightweight `numpy scipy trimesh meshcat rtree` install is enough.

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

Set `OAKINK_DIR` to the downloaded OakInk data root, or place the data under local `data/`, `OakInk/`, or `oakink/`. If the Ubuntu machine is remote, use `--headless-check` or forward the MeshCat port before opening the browser viewer.

## 4. Dataset availability

The audit tools do not download OakInk data or MANO/object assets. The manifest checks `OAKINK_DIR` plus local `data/`, `OakInk/`, and `oakink/` paths. If data is missing, manifest generation should still complete with `data_available=false`.

According to the official docs, OakInk has three parts: **OakBase**, **OakInk-Image**, and **OakInk-Shape**. The current project site links downloads to [Hugging Face OakInk-v1](https://huggingface.co/datasets/oakink/OakInk-v1); users in China can also use the Baidu mirror referenced by the project site. `anno_v2.1.zip` requires the Google Form linked from the official site. See `README_OAKINK.md`, `docs/datasets.md`, and the [OakInk project site](https://oakink.net/) for the upstream documentation. Some older local docs use the spelling `anno_v2_1.zip`; when arranging files, prefer the `anno_v2.1.zip` name used by the project site and `docs/checksum.json`.

```bash
export OAKINK_DIR=/storage/data/OakInk
mkdir -p "$OAKINK_DIR/zipped"

python -m pip install -U huggingface_hub
huggingface-cli download oakink/OakInk-v1 \
  --repo-type dataset \
  --local-dir "$OAKINK_DIR/zipped"
```

After download, arrange the archives in the official layout. If the Hugging Face snapshot or mirror creates a different directory structure, move files manually to match this layout:

```text
$OAKINK_DIR/zipped/
├── OakBase.zip
├── image/
│   ├── anno_v2.1.zip        # obtained through Google Form
│   ├── obj.zip
│   └── stream_zipped/
│       ├── oakink_image_v2.z01
│       ├── ...
│       └── oakink_image_v2.zip
└── shape/
    ├── metaV2.zip
    ├── OakInkObjectsV2.zip
    ├── oakink_shape_v2.zip
    └── OakInkVirtualObjectsV2.zip
```

The official unzip script requires 7zip:

```bash
sudo apt-get install -y p7zip-full
python scripts/verify_checksum.py
python scripts/unzip_all.py
```

After extraction, `$OAKINK_DIR` should contain `OakBase/`, `image/anno/`, `image/obj/`, `image/stream_release_v2/`, `shape/metaV2/`, `shape/OakInkObjectsV2/`, `shape/oakink_shape_v2/`, and `shape/OakInkVirtualObjectsV2/`. To let this workspace auto-detect the data, you can also place the extracted OakInk root under local `data/`, `OakInk/`, or `oakink/`.

If you need to reconstruct MANO hands with the official toolkit, download `mano_v1_2.zip` from the [MANO site](https://mano.is.tue.mpg.de/), unzip it, and place it under `assets/` as described in the official `docs/install.md`. Do not commit these model files to Git.

## 5. Visualization

```bash
python dataset_audit/scripts/dataset_manifest.py --out outputs/dataset_audit/manifest.csv
python dataset_audit/scripts/prepare_sequence_cache.py --item <sample_item>
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/cache/<sample>.npz --fps 3 --stride 20 --loop
```

If real OakInk data is not available, create and inspect a toy cache:

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

`frame_min_signed_distance_mm < 0` means a hand query point is inside the object or proxy. `frame_max_negative_depth_mm` is the maximum non-negative penetration depth for a frame. Sequence summaries include mean, variance, max depth, inside ratio, and minor/significant penetration frame ratios.

## 7. Surface collision

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

Surface collision is separate from point-SDF depth. It reports overlap/collision ratios and does not produce stable penetration depth.

## 8. Contact information

The static audit does not identify a direct OakInk per-frame contact annotation or built-in penetration checker. OakInk provides data loading/splitting/visualization and can reconstruct MANO hand and object geometry when data and assets are present. If no official contact field exists in the local data, derive contact with a hand-object distance threshold and record it as derived contact.

## 9. Limitations

Point-SDF requires watertight object meshes or proxies for reliable inside/outside signs. Non-watertight visual meshes are heuristic. Surface collision may flag boundary touch or mesh fitting artifacts. Coordinate systems and MANO/object reconstruction must be validated with real OakInk sequences before using aggregate statistics.
