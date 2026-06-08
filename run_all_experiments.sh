#!/usr/bin/env bash
# =============================================================================
# SmokeSeer 경량화 실험 전체 실행 스크립트
# Usage: bash run_all_experiments.sh DATASET_ROOT [DEPTHS_DIR]
# Example: bash run_all_experiments.sh ~/SmokeSeer/data/real/red_full images_depth
#
# 실행 순서:
#   1. baseline        (main 브랜치)
#   2. mini_splatting  (mini-splatting 브랜치)
#   3. light_gaussian  (light-gaussian 브랜치, prune 60%)
#   4. compact_3dgs    (compact-3dgs 브랜치)
#
# 각 실험 결과는 ./output/{DATASET_NAME}/{EXP_NAME}/ 에 저장됩니다.
# metrics.json (PSNR/SSIM/LPIPS/FPS)도 자동으로 계산됩니다.
# =============================================================================
set -euo pipefail

DATASET_ROOT=${1:?"Usage: bash run_all_experiments.sh DATASET_ROOT [DEPTHS_DIR]"}
DEPTHS_DIR=${2:-images_depth}
DATASET_NAME=$(basename "$DATASET_ROOT")
OUTPUT_BASE="./output/$DATASET_NAME"
LOG_DIR="./logs/$DATASET_NAME"

mkdir -p "$LOG_DIR"

# 색상 출력
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"; }
err() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }

# 현재 브랜치 저장
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
log "현재 브랜치: $ORIGINAL_BRANCH"

# 종료 시 원래 브랜치로 복귀
cleanup() {
    log "원래 브랜치($ORIGINAL_BRANCH)로 복귀..."
    git checkout "$ORIGINAL_BRANCH" 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================================
# 공통 함수: Stage 1 + Stage 2 실행 후 metrics 측정
# =============================================================================
run_experiment() {
    local EXP_NAME=$1
    local BRANCH=$2
    local OUT_DIR="$OUTPUT_BASE/$EXP_NAME"

    echo ""
    echo "============================================================"
    log "실험 시작: $EXP_NAME (브랜치: $BRANCH)"
    echo "============================================================"

    # 이미 완료된 실험 스킵
    if [ -f "$OUT_DIR/metrics.json" ]; then
        warn "$EXP_NAME 이미 완료됨. 스킵합니다."
        return 0
    fi

    # 브랜치 전환
    log "브랜치 전환: $BRANCH"
    git stash 2>/dev/null || true
    git checkout "$BRANCH"
    git stash pop 2>/dev/null || true

    # Stage 1
    log "[$EXP_NAME] Stage 1 시작..."
    python -W ignore train_stage1_thermal.py \
        -s "$DATASET_ROOT" \
        --eval \
        --use_thermal \
        --depths "$DEPTHS_DIR" \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_stage1.log"

    # Stage 1 output 감지
    STAGE1_OUT=$(ls -dt "$OUTPUT_BASE"/*/  2>/dev/null | grep -v "ft_thermal\|$EXP_NAME\|baseline\|mini_splatting\|light_gaussian\|compact_3dgs" | head -1)
    if [ -z "$STAGE1_OUT" ]; then
        err "Stage 1 output을 찾을 수 없습니다."
        return 1
    fi
    log "Stage 1 output: $STAGE1_OUT"

    # Stage 2
    log "[$EXP_NAME] Stage 2 시작..."
    python -W ignore train_finetune_thermal.py \
        --eval \
        -s "$DATASET_ROOT" \
        -m "$STAGE1_OUT" \
        --use_thermal \
        --depths "$DEPTHS_DIR" \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_stage2.log"

    # 결과 폴더 이름 변경
    LATEST=$(ls -dt "$OUTPUT_BASE"/*/ 2>/dev/null | grep -v "ft_thermal\|baseline\|mini_splatting\|light_gaussian\|compact_3dgs" | head -1)
    if [ -n "$LATEST" ] && [ "$LATEST" != "$OUT_DIR/" ]; then
        mv "$LATEST" "$OUT_DIR"
        log "결과 폴더: $OUT_DIR"
    fi

    # Metrics 측정
    log "[$EXP_NAME] Metrics 측정..."
    python -W ignore metrics.py -m "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_metrics.log"

    log "$EXP_NAME 완료!"
}

# =============================================================================
# LightGaussian 전용 함수 (pruning + finetune)
# =============================================================================
run_light_gaussian() {
    local EXP_NAME="light_gaussian"
    local BRANCH="light-gaussian"
    local PRUNE_PERCENT=${1:-0.6}
    local OUT_DIR="$OUTPUT_BASE/$EXP_NAME"

    echo ""
    echo "============================================================"
    log "실험 시작: $EXP_NAME (브랜치: $BRANCH, prune: ${PRUNE_PERCENT})"
    echo "============================================================"

    if [ -f "$OUT_DIR/metrics.json" ]; then
        warn "$EXP_NAME 이미 완료됨. 스킵합니다."
        return 0
    fi

    # baseline 결과 필요
    local BASELINE_DIR="$OUTPUT_BASE/baseline"
    if [ ! -d "$BASELINE_DIR" ]; then
        err "baseline 결과가 없습니다. baseline 실험을 먼저 실행하세요."
        return 1
    fi

    log "브랜치 전환: $BRANCH"
    git stash 2>/dev/null || true
    git checkout "$BRANCH"
    git stash pop 2>/dev/null || true

    # Pruning
    log "[$EXP_NAME] Pruning 시작 (prune_percent=$PRUNE_PERCENT)..."
    python -W ignore prune_smokeseer.py \
        -m "$BASELINE_DIR" \
        --prune_percent "$PRUNE_PERCENT" \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_prune.log"

    PRUNE_INT=$(python3 -c "print(int(${PRUNE_PERCENT}*100))")
    SURFACE_CKPT="$BASELINE_DIR/ft_chkpnt_surface_thermal30000_pruned${PRUNE_INT}.pth"
    SMOKE_CKPT="$BASELINE_DIR/ft_chkpnt_smoke_thermal30000_pruned${PRUNE_INT}.pth"

    # Finetune
    log "[$EXP_NAME] Finetuning 시작..."
    python -W ignore train_finetune_thermal.py \
        --eval \
        -s "$DATASET_ROOT" \
        -m "$BASELINE_DIR" \
        --use_thermal \
        --depths "$DEPTHS_DIR" \
        --start_checkpoint "$SURFACE_CKPT" \
        --start_checkpoint_smoke "$SMOKE_CKPT" \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_finetune.log"

    # 결과 이동
    LATEST=$(ls -dt "$OUTPUT_BASE"/*/ 2>/dev/null | grep -v "ft_thermal\|baseline\|mini_splatting\|light_gaussian\|compact_3dgs" | head -1)
    if [ -n "$LATEST" ]; then
        mv "$LATEST" "$OUT_DIR"
        log "결과 폴더: $OUT_DIR"
    fi

    # Metrics
    log "[$EXP_NAME] Metrics 측정..."
    python -W ignore metrics.py -m "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_metrics.log"

    log "$EXP_NAME 완료!"
}

# =============================================================================
# 최종 결과 출력
# =============================================================================
print_summary() {
    echo ""
    echo "============================================================"
    log "전체 실험 결과 요약"
    echo "============================================================"
    python3 -c "
import json, os, torch
exps = ['baseline', 'mini_splatting', 'light_gaussian', 'compact_3dgs']
print(f'{'실험':<20} {'Gaussians':>12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FPS':>8}')
print('-'*68)
for exp in exps:
    mpath = os.path.join('$OUTPUT_BASE', exp, 'metrics.json')
    if not os.path.exists(mpath):
        print(f'{exp:<20} {'N/A':>12}')
        continue
    with open(mpath) as f:
        m = json.load(f)
    avg = m.get('desmoked_vs_gt_desmoked', {}).get('avg', {})
    spd = m.get('render_speed', {})
    # Gaussian count
    ckpt_path = os.path.join('$OUTPUT_BASE', exp, 'ft_chkpnt_surface_thermal30000.pth')
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        n_surface = ckpt[0][1].shape[0]
    except:
        n_surface = -1
    psnr  = avg.get('PSNR', 0)
    ssim  = avg.get('SSIM', 0)
    lpips = avg.get('LPIPS', 0)
    fps   = spd.get('fps', 0)
    print(f'{exp:<20} {n_surface:>12,} {psnr:>8.4f} {ssim:>8.4f} {lpips:>8.4f} {fps:>8.1f}')
" 2>/dev/null
}

# =============================================================================
# 실험 실행
# =============================================================================
echo ""
log "SmokeSeer 경량화 전체 실험 시작"
log "데이터셋: $DATASET_ROOT"
log "출력 경로: $OUTPUT_BASE"
echo ""

run_experiment  "baseline"      "main"
run_experiment  "mini_splatting" "mini-splatting"
run_light_gaussian 0.6
run_experiment  "compact_3dgs"  "compact-3dgs"

print_summary

log "모든 실험 완료!"