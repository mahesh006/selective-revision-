#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

python3 evaluate.py \
  --input_files "$DATASET_DIR/*.json" \
  --output_dir "$OUTPUT_DIR" \
  --inference_mode score \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --device_map "$DEVICE_MAP" \
  --resume \
  "$@"
