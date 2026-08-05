#!/usr/bin/env bash
# scripts/setup_grab_data.sh
#
# One-shot helper that re-creates the symlinks `process/GRAB/raw.py` and
# `process/common/stage2.py` need to find GRAB canonical object meshes and
# per-subject hand templates when the raw dataset lives outside the repo
# (for example, under `data/raw_data/GRAB`).
#
# Without these symlinks, GRAB stage 2 will fail with
# `FileNotFoundError: GRAB object mesh not found: <obj>` and
# `WARNING: missing subject v_template ... Falling back to default MANO mean shape.`
#
# Idempotent: re-running will replace stale symlinks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_ROOT="${REF2DEX_RAW_GRAB_ROOT:-${REPO_ROOT}/data/raw_data/GRAB}"

if [[ ! -d "${RAW_ROOT}" ]]; then
    echo "[setup_grab_data] raw GRAB root not found at ${RAW_ROOT}" >&2
    echo "[setup_grab_data] set REF2DEX_RAW_GRAB_ROOT to the right path" >&2
    exit 1
fi

# 1. /assets/grab/objects/ -> ${RAW_ROOT}/objects/<name>
ASSETS_GRAB_OBJECTS="${REPO_ROOT}/assets/grab/objects"
mkdir -p "${ASSETS_GRAB_OBJECTS}"
for obj_dir in "${RAW_ROOT}/objects/"*; do
    [[ -d "${obj_dir}" ]] || continue
    name="$(basename "${obj_dir}")"
    link="${ASSETS_GRAB_OBJECTS}/${name}"
    if [[ -L "${link}" && "$(readlink "${link}")" == "${obj_dir}" ]]; then
        continue
    fi
    ln -sfn "${obj_dir}" "${link}"
done

# 2. /dataset/GRAB/tools/{object_meshes,object_settings,subject_meshes,subject_settings}
TOOLS_DST="${REPO_ROOT}/dataset/GRAB/tools"
for sub in object_meshes object_settings subject_meshes subject_settings; do
    src="${RAW_ROOT}/tools/${sub}"
    dst="${TOOLS_DST}/${sub}"
    if [[ ! -d "${src}" ]]; then
        echo "[setup_grab_data] WARNING: missing ${src}, skipping link" >&2
        continue
    fi
    if [[ -L "${dst}" && "$(readlink "${dst}")" == "${src}" ]]; then
        continue
    fi
    # Remove any pre-existing real directory of the same name.
    if [[ -e "${dst}" && ! -L "${dst}" ]]; then
        echo "[setup_grab_data] WARNING: ${dst} exists and is not a symlink, leaving it alone" >&2
        continue
    fi
    rm -f "${dst}"
    ln -sfn "${src}" "${dst}"
done

echo "[setup_grab_data] done; sources under ${RAW_ROOT}"
