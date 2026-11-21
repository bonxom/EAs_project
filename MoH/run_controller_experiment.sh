#!/usr/bin/env bash
# run_controller_experiment.sh
# Equal-budget 3-arm comparison of Dual-HiFo controllers:
#   A) dual_hifo_independent    (two independent local navigators)
#   B) dual_hifo_global_rule    (cross-level rule-based global controller)
#   C) dual_hifo_global_bandit  (cross-level LinUCB contextual bandit)
#
# All arms receive the exact same provider_budget.max_requests and high
# n_iterations so the shared LLM-call budget terminates each run rather than
# iteration count. Existing completed runs (with checkpoint/metadata.json) are
# skipped and nothing is deleted.
#
# The deployment stage follows the moh-deployment checkpoint-driven flow: each
# trained arm is deployed zero-shot on unseen TSP-500 and TSP-1000 targets
# (10 inference iterations, 8 validation instances, 100 independent test
# instances, 8 workers). The validation winner per target is frozen and scored
# on test once; test scores never feed generation, ranking, or selection.
#
# Usage:
#   bash run_controller_experiment.sh [--stage all|train|deploy]
#       [--seeds "0 1 2"] [--max-requests 1500] [--n-iterations 40]
#       [--out-dir outputs/controller_exp] [--deploy-dir <out-dir>/deployment]
#       [--target-sizes "[500,1000]"]
#       [--modes "dual_hifo_independent dual_hifo_global_rule dual_hifo_global_bandit"]

set -euo pipefail

usage() {
    sed -n '19,24p' "$0"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    echo "ERROR: uv binary not found in PATH or ~/.local/bin/uv" >&2
    exit 1
fi

SEEDS=(0 1 2)
MODES=("dual_hifo_independent" "dual_hifo_global_rule" "dual_hifo_global_bandit")
STAGE="all"
MAX_REQUESTS=1500
MAX_ATTEMPTS=""
N_ITERATIONS=40
POP_SIZE=10
MAX_EVAL_CALLS=1000
TIMEOUT=90
OPTIMIZER_TIMEOUT=1800
LLM_PROFILE="local"
LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://172.25.16.1:20128/v1}"
OUT_DIR="$PROJECT_DIR/outputs/controller_exp"
DEPLOY_DIR=""
TARGET_SIZES="[500,1000]"

# Deployment constants mirror cfg/deployment.yaml and the moh-deployment run D
# benchmark: 10 inference iterations, 8 validation and 100 test instances.
INFERENCE_ITERATIONS=10
VAL_N_INSTANCES=8
TEST_N_INSTANCES=100
DEPLOY_WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --seeds)
            read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --modes)
            read -r -a MODES <<< "$2"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="$2"
            shift 2
            ;;
        --max-attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --n-iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --deploy-dir)
            DEPLOY_DIR="$2"
            shift 2
            ;;
        --target-sizes)
            TARGET_SIZES="$2"
            shift 2
            ;;
        --llm-profile)
            LLM_PROFILE="$2"
            shift 2
            ;;
        --base-url)
            LLM_BASE_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

case "$STAGE" in
    all|train|deploy) ;;
    *)
        echo "ERROR: --stage must be one of: all, train, deploy (got '$STAGE')" >&2
        exit 1
        ;;
esac

if [ -z "$TARGET_SIZES" ]; then
    echo "ERROR: --target-sizes cannot be empty" >&2
    exit 1
fi

# Fail before Hydra creates a run directory that the never-delete retry policy
# would then treat as an incomplete run.
if [ "$LLM_PROFILE" = "local" ]; then
    if [ -z "${LOCAL_LLM_API_KEY:-}" ]; then
        echo "ERROR: LOCAL_LLM_API_KEY must be set for --llm-profile=local." >&2
        exit 1
    fi
elif [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY must be set for --llm-profile=$LLM_PROFILE." >&2
    exit 1
fi

if [ -z "$DEPLOY_DIR" ]; then
    DEPLOY_DIR="$OUT_DIR/deployment"
fi

if [ -z "$MAX_ATTEMPTS" ]; then
    MAX_ATTEMPTS=$(( MAX_REQUESTS * 2 ))
fi

mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Dual-HiFo Global Controller 3-Arm Experiment"
echo "Stage:        $STAGE"
echo "Modes:        ${MODES[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Max requests: $MAX_REQUESTS (max attempts: $MAX_ATTEMPTS)"
echo "Iterations:   $N_ITERATIONS"
echo "Output dir:   $OUT_DIR"
echo "Deploy dir:   $DEPLOY_DIR"
echo "Target sizes: $TARGET_SIZES"
echo "=========================================================="

# ---------------------------------------------------------------------------
# STAGE 1: EQUAL-BUDGET META-TRAINING
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
    RUN_DIRS=()

    for SEED in "${SEEDS[@]}"; do
        for MODE in "${MODES[@]}"; do
            RUN_DIR="$OUT_DIR/${MODE}_seed${SEED}"
            RUN_LOG="$OUT_DIR/${MODE}_seed${SEED}.log"
            RUN_DIRS+=("$RUN_DIR")

            if [ -f "$RUN_DIR/checkpoint/metadata.json" ]; then
                echo ">>> [SKIP] $MODE seed=$SEED already completed ($RUN_DIR)"
                continue
            fi
            if [ -e "$RUN_DIR" ]; then
                echo ">>> [SKIP] $RUN_DIR exists without checkpoint; move or inspect it before re-running." >&2
                continue
            fi

            echo ">>> [RUN] $MODE seed=$SEED -> $RUN_DIR"
            if [ "$LLM_PROFILE" = "local" ]; then
                EXTRA_ARGS=(
                    "llm_client@heu=local"
                    "llm_client@meta=local"
                    "heu.base_url=$LLM_BASE_URL"
                    "meta.base_url=$LLM_BASE_URL"
                )
            else
                EXTRA_ARGS=(
                    "llm_client@heu=openai"
                    "llm_client@meta=openai"
                )
            fi

            (
                cd "$PROJECT_DIR"
                "$UV_BIN" run python main.py \
                    --config-name=dualhifo \
                    dual_hifo.mode="$MODE" \
                    seed="$SEED" \
                    n_iterations="$N_ITERATIONS" \
                    pop_size="$POP_SIZE" \
                    max_eval_calls="$MAX_EVAL_CALLS" \
                    timeout="$TIMEOUT" \
                    optimizer_timeout="$OPTIMIZER_TIMEOUT" \
                    isolate_base_solutions=true \
                    provider_budget.max_requests="$MAX_REQUESTS" \
                    provider_budget.max_attempts="$MAX_ATTEMPTS" \
                    provider_budget.max_retries_per_call=5 \
                    "${EXTRA_ARGS[@]}" \
                    hydra.run.dir="$RUN_DIR" > "$RUN_LOG" 2>&1 || true
            )
        done
    done

    echo ">>> Aggregating training report..."
    (
        cd "$PROJECT_DIR"
        "$UV_BIN" run python -m tools.controller_report \
            "${RUN_DIRS[@]}" \
            --md-out "$OUT_DIR/controller_report.md" \
            --json-out "$OUT_DIR/controller_report.json" || true
    )
    echo "Training report written to $OUT_DIR/controller_report.md"
fi

# ---------------------------------------------------------------------------
# STAGE 2: ZERO-SHOT DEPLOYMENT TEST ON TSP-500 / TSP-1000
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "deploy" || "$STAGE" == "all" ]]; then
    echo ">>> Starting zero-shot TSP-500/TSP-1000 deployment test..."
    mkdir -p "$DEPLOY_DIR"
    COMPLETED_DEPLOY=()

    for SEED in "${SEEDS[@]}"; do
        for MODE in "${MODES[@]}"; do
            CKPT_DIR="$OUT_DIR/${MODE}_seed${SEED}/checkpoint"
            RUN_DIR="$DEPLOY_DIR/${MODE}_seed${SEED}"
            RUN_LOG="$DEPLOY_DIR/${MODE}_seed${SEED}.log"

            if [ ! -f "$CKPT_DIR/metadata.json" ]; then
                echo ">>> [DEPLOY][SKIP] no checkpoint at $CKPT_DIR; train $MODE seed=$SEED first." >&2
                continue
            fi
            if [ -s "$RUN_DIR/report.json" ]; then
                echo ">>> [DEPLOY][SKIP] $MODE seed=$SEED already complete ($RUN_DIR)"
                COMPLETED_DEPLOY+=("$RUN_DIR")
                continue
            fi
            if [ -e "$RUN_DIR" ]; then
                echo ">>> [DEPLOY][SKIP] $RUN_DIR exists without report.json; move or inspect it before re-running." >&2
                continue
            fi

            echo ">>> [DEPLOY][RUN] $MODE seed=$SEED -> $RUN_DIR"
            if [ "$LLM_PROFILE" = "local" ]; then
                EXTRA_DEPLOY_ARGS=(
                    "llm_client@heu=local"
                    "heu.base_url=$LLM_BASE_URL"
                )
            else
                EXTRA_DEPLOY_ARGS=("llm_client@heu=openai")
            fi

            (
                cd "$PROJECT_DIR"
                "$UV_BIN" run python deploy.py \
                    checkpoint="$CKPT_DIR" \
                    "target_sizes=$TARGET_SIZES" \
                    inference_iterations="$INFERENCE_ITERATIONS" \
                    val_n_instances="$VAL_N_INSTANCES" \
                    test_n_instances="$TEST_N_INSTANCES" \
                    workers="$DEPLOY_WORKERS" \
                    evaluate_test=true \
                    seed="$SEED" \
                    "${EXTRA_DEPLOY_ARGS[@]}" \
                    hydra.run.dir="$RUN_DIR" > "$RUN_LOG" 2>&1 || true
            )

            if [ -s "$RUN_DIR/report.json" ]; then
                COMPLETED_DEPLOY+=("$RUN_DIR")
                echo ">>> [DEPLOY] $MODE seed=$SEED completed."
            else
                echo ">>> [DEPLOY] $MODE seed=$SEED failed or incomplete. Check $RUN_LOG" >&2
            fi
        done
    done

    if [ "${#COMPLETED_DEPLOY[@]}" -gt 0 ]; then
        (
            cd "$PROJECT_DIR"
            "$UV_BIN" run python tools/export_deployment_table.py \
                "${COMPLETED_DEPLOY[@]}" > "$DEPLOY_DIR/deployment_tables.tsv" 2>/dev/null || true
            "$UV_BIN" run python -m tools.controller_deployment_report \
                "${COMPLETED_DEPLOY[@]}" \
                --md-out "$DEPLOY_DIR/deployment_report.md" \
                --json-out "$DEPLOY_DIR/deployment_report.json" || true
        )
        echo "Deployment report written to $DEPLOY_DIR/deployment_report.md"
    else
        echo ">>> No completed deployment runs to report."
    fi
fi

echo "=========================================================="
echo "Controller experiment finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=========================================================="
