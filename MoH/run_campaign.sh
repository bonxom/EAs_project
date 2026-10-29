#!/usr/bin/env bash
# run_campaign.sh
# Unified runner for MoH meta-training and zero-shot deployment campaigns.
#
# Usage:
#   bash run_campaign.sh [--stage train|deploy|all] [--seed SEED] [--modes "mode1 mode2 ..."]
#
# Examples:
#   bash run_campaign.sh --stage all --seed 0
#   bash run_campaign.sh --stage train --seed 42 --modes "outer_hifo dual_hifo_independent"
#   bash run_campaign.sh --stage deploy --seed 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Detect uv binary
if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    echo "ERROR: uv binary not found in PATH or ~/.local/bin/uv" >&2
    exit 1
fi

# Defaults
STAGE="all"
SEED=0
MODES=("full_moh" "outer_hifo" "dual_hifo_independent" "inner_hifo")
TARGET_SIZES="[500,1000]"
LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://172.25.16.1:20128/v1}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-sk-af7e9199bf70201d-4u8mp0-739ae71f}"
export UV_LINK_MODE="copy"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --modes)
            read -r -a MODES <<< "$2"
            shift 2
            ;;
        --target-sizes)
            TARGET_SIZES="$2"
            shift 2
            ;;
        --base-url)
            LLM_BASE_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--stage train|deploy|all] [--seed SEED] [--modes \"mode1 mode2...\"] [--target-sizes \"[500,1000]\"]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

TRAIN_BASE="$PROJECT_DIR/outputs/full_budget_seed${SEED}"
DEPLOY_BASE="$PROJECT_DIR/outputs/deployment/full_budget_seed${SEED}"
CAMPAIGN_LOG="$TRAIN_BASE/campaign_master.log"

mkdir -p "$TRAIN_BASE" "$DEPLOY_BASE"

echo "=========================================================="
echo "MoH Campaign Runner"
echo "Stage:        $STAGE"
echo "Seed:         $SEED"
echo "Modes:        ${MODES[*]}"
echo "Target Sizes: $TARGET_SIZES"
echo "LLM Base URL: $LLM_BASE_URL"
echo "Timestamp:    $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=========================================================="

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# STAGE 1: META-TRAINING
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
    echo ">>> Starting Meta-Training Phase..."
    SUMMARY_MD="$TRAIN_BASE/summary.md"
    if [ ! -f "$SUMMARY_MD" ]; then
        cat << 'EOF' > "$SUMMARY_MD"
# Full-Budget Meta-Training Report

| Mode | Status | Meta-Utility | TSP-20 | TSP-50 | TSP-100 | TSP-200 | Duration (s) |
|---|---|---:|---:|---:|---:|---:|---:|
EOF
    fi

    for MODE in "${MODES[@]}"; do
        TAG="fb_${MODE}_seed${SEED}"
        RUN_DIR="$TRAIN_BASE/$TAG"
        RUN_LOG="$TRAIN_BASE/${TAG}.log"

        echo "--------------------------------------------------------"
        echo ">>> [TRAIN] Mode: $MODE (Tag: $TAG) at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

        if [ -f "$RUN_DIR/checkpoint/metadata.json" ]; then
            echo "Checkpoint already exists for $TAG, skipping training."
            continue
        fi

        if [ -d "$RUN_DIR" ]; then
            echo "Cleaning incomplete prior run directory $RUN_DIR..."
            rm -rf "$RUN_DIR"
        fi

        START_SEC=$(date +%s)

        "$UV_BIN" run python main.py \
            --config-name=dualhifo \
            dual_hifo.mode="$MODE" \
            seed="$SEED" \
            n_iterations=10 \
            pop_size=10 \
            max_eval_calls=1000 \
            timeout=90 \
            optimizer_timeout=1800 \
            isolate_base_solutions=true \
            llm_client@heu=local \
            llm_client@meta=local \
            heu.base_url="$LLM_BASE_URL" \
            meta.base_url="$LLM_BASE_URL" \
            provider_budget.max_requests=5000 \
            provider_budget.max_attempts=7500 \
            provider_budget.max_retries_per_call=5 \
            hydra.run.dir="$RUN_DIR" > "$RUN_LOG" 2>&1 || true

        DURATION=$(( $(date +%s) - START_SEC ))

        if [ -f "$RUN_DIR/checkpoint/metadata.json" ]; then
            META_UTIL=$(grep '"training_meta_utility":' "$RUN_DIR/checkpoint/metadata.json" | head -n 1 | awk '{print $2}' | tr -d ',')
            echo ">>> [TRAIN] $MODE finished successfully in ${DURATION}s (Meta-Utility: $META_UTIL)"
            echo "| $MODE | completed | $META_UTIL | - | - | - | - | $DURATION |" >> "$SUMMARY_MD"
        else
            echo ">>> [TRAIN] $MODE failed or incomplete after ${DURATION}s. Check $RUN_LOG"
            echo "| $MODE | failed | N/A | - | - | - | - | $DURATION |" >> "$SUMMARY_MD"
        fi
    done
fi

# ---------------------------------------------------------------------------
# STAGE 2: ZERO-SHOT DEPLOYMENT
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "deploy" || "$STAGE" == "all" ]]; then
    echo ">>> Starting Zero-Shot Deployment Phase..."
    DEPLOY_LOG="$DEPLOY_BASE/deployment_campaign.log"

    for MODE in "${MODES[@]}"; do
        TAG="deploy_fb_${MODE}_seed${SEED}"
        CKPT_DIR="$TRAIN_BASE/fb_${MODE}_seed${SEED}/checkpoint"
        RUN_DIR="$DEPLOY_BASE/$TAG"
        RUN_LOG="$DEPLOY_BASE/${TAG}.log"

        echo "--------------------------------------------------------"
        echo ">>> [DEPLOY] Mode: $MODE (Tag: $TAG) at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

        if [ ! -f "$CKPT_DIR/metadata.json" ]; then
            echo "ERROR: Checkpoint not found at $CKPT_DIR, skipping mode $MODE."
            continue
        fi

        if [ -f "$RUN_DIR/report.json" ] && [ -s "$RUN_DIR/report.json" ]; then
            echo "Report already exists for $TAG, skipping deployment."
            continue
        fi

        if [ -d "$RUN_DIR" ]; then
            echo "Cleaning incomplete prior deployment directory $RUN_DIR..."
            rm -rf "$RUN_DIR"
        fi

        START_SEC=$(date +%s)

        "$UV_BIN" run python deploy.py \
            checkpoint="$CKPT_DIR" \
            "target_sizes=$TARGET_SIZES" \
            inference_iterations=10 \
            val_n_instances=8 \
            test_n_instances=100 \
            workers=8 \
            seed="$SEED" \
            heu.base_url="$LLM_BASE_URL" \
            hydra.run.dir="$RUN_DIR" > "$RUN_LOG" 2>&1 || true

        DURATION=$(( $(date +%s) - START_SEC ))

        if [ -f "$RUN_DIR/report.json" ]; then
            echo ">>> [DEPLOY] $MODE completed in ${DURATION}s."
            "$UV_BIN" run python tools/export_deployment_table.py "$RUN_DIR" > "$DEPLOY_BASE/${TAG}_table.tsv" 2>/dev/null || true
        else
            echo ">>> [DEPLOY] $MODE failed. Check $RUN_LOG"
        fi
    done

    # Consolidate comparison across all completed modes
    if compgen -G "$DEPLOY_BASE/deploy_fb_*" > /dev/null; then
        "$UV_BIN" run python tools/export_deployment_table.py "$DEPLOY_BASE"/deploy_fb_* > "$DEPLOY_BASE/consolidated_summary.tsv" 2>/dev/null || true
        echo "Consolidated summary updated at $DEPLOY_BASE/consolidated_summary.tsv"
    fi
fi

echo "=========================================================="
echo "Campaign finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=========================================================="
