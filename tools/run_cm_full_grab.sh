#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home2/wyy/miniconda3/envs/graspenv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home2/wyy/miniconda3/envs/graspenv/bin/torchrun}"
STAGE4_ROOT="$ROOT/data/processed_data/stage4/data"
DATA_ROOT="$STAGE4_ROOT/grab"
SPLIT_ROOT="$ROOT/data/processed_data/stage4/splits/full_grab_v1"
LOG_ROOT="$ROOT/outputs/Cm/preprocess_full_grab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

run_preprocess() {
  local gpu="$1"
  shift
  for subject in "$@"; do
    "$PYTHON_BIN" -m process.GRAB.stage4_cm \
      --grab-root "$ROOT/dataset/GRAB/data" \
      --output-root "$DATA_ROOT" \
      --seq "$subject" \
      --device "cuda:${gpu}" \
      --side both \
      --frame-batch-size 4 \
      --nn-batch-size 8
  done
}

(run_preprocess 4 s1 s2 s3 s4 s5) >"$LOG_ROOT/gpu4.log" 2>&1 &
PID_4=$!
(run_preprocess 5 s6 s7 s8 s9) >"$LOG_ROOT/gpu5.log" 2>&1 &
PID_5=$!
wait "$PID_4"
wait "$PID_5"

"$PYTHON_BIN" -m process.GRAB.build_cm_split \
  --data-root "$DATA_ROOT" --output-root "$SPLIT_ROOT" --path-prefix grab --seed 42

cd "$ROOT"
CUDA_VISIBLE_DEVICES=4,5 "$TORCHRUN_BIN" --standalone --nproc_per_node=2 \
  -m src.task.Cm.src.train \
  --config src/task/Cm/configs/archive/sequence_full_grab_16slot.yaml \
  --distributed \
  --set 'train.description=Full GRAB (1335 raw sequences): 16 slots, no gate, physical time conditioning, two-GPU DDP.'
