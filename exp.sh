#!/usr/bin/env bash
# Gaussian count reduction experiments (exp2 ~ exp4)
# Run after exp1 (grad_threshold=0.0003) is complete.
#
# Experiment design:
#   baseline : grad=0.0001, prune=0.005, pct=0.01  -> ~268만
#   exp1     : grad=0.0003, prune=0.005, pct=0.01  -> ~100만  (already running)
#   exp2     : grad=0.0005, prune=0.005, pct=0.01  -> ~60만
#   exp3     : grad=0.0003, prune=0.010, pct=0.01  -> ~80만
#   exp4     : grad=0.0003, prune=0.005, pct=0.02  -> ~90만

set -euo pipefail

DATASET=~/SmokeSeer/data/real/red_sub
DEPTHS=images_depth

run_experiment() {
    local name=$1
    local grad=$2
    local prune=$3
    local pct=$4

    echo ""
    echo "========================================"
    echo "Running $name"
    echo "  densify_grad_threshold_surface : $grad"
    echo "  prune_opacity_surface_threshold: $prune"
    echo "  percent_dense                  : $pct"
    echo "========================================"

    # Stage 1
    python -W ignore train_stage1_thermal.py \
        -s "$DATASET" \
        --eval \
        --use_thermal \
        --depths "$DEPTHS" \
        --densify_grad_threshold_surface "$grad" \
        --prune_opacity_surface_threshold "$prune" \
        --percent_dense "$pct"

    # Get latest Stage 1 output
    STAGE1=$(ls -dt ~/SmokeSeer/output/red_sub/*/ | grep -v ft_thermal | head -1)
    echo "Stage 1 output: $STAGE1"

    # Stage 2
    python -W ignore train_finetune_thermal.py \
        --eval \
        -s "$DATASET" \
        -m "$STAGE1" \
        --use_thermal \
        --depths "$DEPTHS" \
        --densify_grad_threshold_surface "$grad" \
        --prune_opacity_surface_threshold "$prune" \
        --percent_dense "$pct"

    # Get Stage 2 output (same folder as Stage 1)
    STAGE2=$(ls -dt ~/SmokeSeer/output/red_sub/*/ | grep -v ft_thermal | head -1)

    # Metrics
    echo "Computing metrics for $name..."
    python metrics.py -m "$STAGE2"

    # Log Gaussian count
    python -W ignore -c "
import torch, glob, os
ckpts = glob.glob('${STAGE2}ft_chkpnt_surface_thermal30000.pth')
if ckpts:
    ckpt = torch.load(ckpts[0], map_location='cpu')
    print(f'[$name] Surface Gaussians: {ckpt[0][1].shape[0]:,}')
ckpts2 = glob.glob('${STAGE2}ft_chkpnt_smoke_thermal30000.pth')
if ckpts2:
    ckpt2 = torch.load(ckpts2[0], map_location='cpu')
    print(f'[$name] Smoke Gaussians  : {ckpt2[0][1].shape[0]:,}')
" 2>/dev/null

    echo "$name done. Results in $STAGE2"
}

# # baseline
# run_experiment "baseline" "0.0001" "0.005" "0.01"

# # exp1
# run_experiment "exp1" "0.0003" "0.005" "0.01"

# # exp2: higher grad threshold
# run_experiment "exp2" "0.0005" "0.005" "0.01"

# # exp3: higher prune threshold
# run_experiment "exp3" "0.0003" "0.010" "0.01"

# # exp4: higher percent_dense
# run_experiment "exp4" "0.0003" "0.005" "0.02"
run_experiment "exp5" "0.0007" "0.005" "0.01"
run_experiment "exp6" "0.001"  "0.005" "0.01"
run_experiment "exp7" "0.002"  "0.005" "0.01"

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Compare metrics.json in each output folder."
echo "========================================"