#!/usr/bin/env bash
# Stage 2: RGB+Thermal finetuning with smoke decomposition
# Usage: bash stage2.sh DATASET_ROOT STAGE1_MODEL_DIR [DEPTHS_DIR]
# Example: bash stage2.sh ~/SmokeSeer/data/real/red_sub ./output/red_sub/20260414-12-51-38 images_depth
#
# If STAGE1_MODEL_DIR is not given, the most recent Stage 1 output is used automatically.

set -euo pipefail

DATASET_ROOT=${1:?"Usage: bash stage2.sh DATASET_ROOT [STAGE1_MODEL_DIR] [DEPTHS_DIR]"}
DEPTHS_DIR=${3:-images_depth}

# Auto-detect latest Stage 1 model if not provided
if [ -z "${2:-}" ]; then
    DATASET_NAME=$(basename "$DATASET_ROOT")
    STAGE1_MODEL_DIR=$(ls -dt ./output/"$DATASET_NAME"/*/ 2>/dev/null | grep -v ft_thermal | head -1)
    if [ -z "$STAGE1_MODEL_DIR" ]; then
        echo "Error: No Stage 1 model found under ./output/$DATASET_NAME/. Run stage1.sh first."
        exit 1
    fi
    echo "Auto-detected Stage 1 model: $STAGE1_MODEL_DIR"
else
    STAGE1_MODEL_DIR=$2
fi

echo "========================================"
echo "Stage 2: RGB+Thermal smoke decomposition"
echo "Dataset     : $DATASET_ROOT"
echo "Stage1 model: $STAGE1_MODEL_DIR"
echo "Depths      : $DEPTHS_DIR"
echo "========================================"

python -W ignore train_finetune_thermal.py \
    --eval \
    -s "$DATASET_ROOT" \
    -m "$STAGE1_MODEL_DIR" \
    --use_thermal \
    --depths "$DEPTHS_DIR"

echo "Stage 2 complete. Output saved to ${STAGE1_MODEL_DIR}/"