#!/usr/bin/env bash
# Stage 1: Thermal-only surface reconstruction
# Usage: bash stage1.sh DATASET_ROOT [DEPTHS_DIR]
# Example: bash stage1.sh ~/SmokeSeer/data/real/red_sub images_depth

set -euo pipefail

DATASET_ROOT=${1:?"Usage: bash stage1.sh DATASET_ROOT [DEPTHS_DIR]"}
DEPTHS_DIR=${2:-images_depth}

echo "========================================"
echo "Stage 1: Thermal-only reconstruction"
echo "Dataset : $DATASET_ROOT"
echo "Depths  : $DEPTHS_DIR"
echo "========================================"

python -W ignore train_stage1_thermal.py \
    -s "$DATASET_ROOT" \
    --eval \
    --use_thermal \
    --depths "$DEPTHS_DIR"

echo "Stage 1 complete. Output saved to ./output/$(basename $DATASET_ROOT)/"