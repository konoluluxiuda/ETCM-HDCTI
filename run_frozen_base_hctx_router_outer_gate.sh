#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

MANIFEST="configs/frozen_base_hctx_router_outer_evaluation.json"
DATASETS=(tcmsuite tcmsp symmap etcm_mention10)

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Frozen-base Hctx-P outer four-dataset Gate:\n'
    printf '  manifest: %s\n' "$MANIFEST"
    for dataset in "${DATASETS[@]}"; do
        printf '  dataset:  %s\n' "$dataset"
    done
    exit 0
fi

if [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/frozen_base_hctx_router_outer_$RUN_TIMESTAMP}"
mkdir -p "$RUN_DIR"

export HDCTI_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

failed=0
reports=()
for dataset in "${DATASETS[@]}"; do
    output_dir="$RUN_DIR/$dataset"
    log_path="$RUN_DIR/${dataset}.log"
    printf '\nStarting frozen outer evaluation: %s\n' "$dataset"
    python tools/evaluate_frozen_base_hctx_router_outer.py \
        --manifest "$MANIFEST" \
        --dataset "$dataset" \
        --output-dir "$output_dir" 2>&1 | tee "$log_path"
    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 || ! -f "$output_dir/report.json" ]]; then
        printf 'Outer evaluation failed: %s\n' "$dataset" >&2
        failed=$((failed + 1))
        continue
    fi
    reports+=("$output_dir/report.json")
done

if [[ "$failed" -gt 0 ]]; then
    printf '%d outer evaluation(s) failed.\n' "$failed" >&2
    exit 1
fi

summary_command=(
    python tools/summarize_frozen_base_hctx_router_outer.py
    --output-dir "$RUN_DIR"
    --require-all-pass
)
for report in "${reports[@]}"; do
    summary_command+=(--report "$report")
done
"${summary_command[@]}"
status=$?
printf '\nBatch results: %s\n' "$RUN_DIR/summary.md"
exit "$status"
