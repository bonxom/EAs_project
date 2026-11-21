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
# Usage:
#   bash run_controller_experiment.sh [--seeds "0 1 2"] [--max-requests 1500]
#       [--n-iterations 40] [--out-dir outputs/controller_exp]
#       [--modes "dual_hifo_independent dual_hifo_global_rule dual_hifo_global_bandit"]

set -euo pipefail

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

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --llm-profile)
            LLM_PROFILE="$2"
            shift 2
            ;;
        --base-url)
            LLM_BASE_URL="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$MAX_ATTEMPTS" ]; then
    MAX_ATTEMPTS=$(( MAX_REQUESTS * 2 ))
fi

mkdir -p "$OUT_DIR"
RUN_DIRS=()

echo "=========================================================="
echo "Dual-HiFo Global Controller 3-Arm Experiment"
echo "Modes:        ${MODES[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Max requests: $MAX_REQUESTS (max attempts: $MAX_ATTEMPTS)"
echo "Iterations:   $N_ITERATIONS"
echo "Output dir:   $OUT_DIR"
echo "=========================================================="

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
        EXTRA_ARGS=()
        if [ "$LLM_PROFILE" = "local" ]; then
            EXTRA_ARGS+=(
                "llm_client@heu=local"
                "llm_client@meta=local"
                "heu.base_url=$LLM_BASE_URL"
                "meta.base_url=$LLM_BASE_URL"
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

echo ">>> Aggregating report..."
(
    cd "$PROJECT_DIR"
    "$UV_BIN" run python -m tools.controller_report \
        "${RUN_DIRS[@]}" \
        --md-out "$OUT_DIR/controller_report.md" \
        --json-out "$OUT_DIR/controller_report.json" || true
)
echo "Report written to $OUT_DIR/controller_report.md"
