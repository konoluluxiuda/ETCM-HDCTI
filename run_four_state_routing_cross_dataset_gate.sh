#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCMSP|tcmsp|configs/HDCTI_tcmsp_four_state_no_context_unit_pilot.conf|configs/HDCTI_tcmsp_four_state_isolated_routing_unit_pilot.conf"
    "SymMap2.0|symmap|configs/HDCTI_symmap_four_state_no_context_unit_pilot.conf|configs/HDCTI_symmap_four_state_isolated_routing_unit_pilot.conf"
    "ETCM2.0-mention10|etcm_mention10|configs/HDCTI_etcm_mention10_four_state_no_context_unit_pilot.conf|configs/HDCTI_etcm_mention10_four_state_isolated_routing_unit_pilot.conf"
)
TCMSUITE_REPORT="$REPOSITORY_ROOT/results/four_state_routing_gate/tcmsuite_isolated_routing/report.json"

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Frozen four-state routing Gate jobs:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset slug baseline_config candidate_config <<< "$job"
        printf '  %-20s baseline: %s\n' "$dataset" "$baseline_config"
        printf '  %-20s V2:       %s\n' '' "$candidate_config"
    done
    exit 0
fi

if [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/four_state_routing_gate_$RUN_TIMESTAMP}"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"
mkdir -p "$RUN_DIR"

if [[ ! -f "$TCMSUITE_REPORT" ]]; then
    printf 'Missing frozen TCM-Suite report: %s\n' "$TCMSUITE_REPORT" >&2
    exit 1
fi

if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
    {
        printf 'batch_started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'repository=%s\n' "$REPOSITORY_ROOT"
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'python=%s\n' "$(command -v python || printf unknown)"
        python -c 'import platform; print("python_version=" + platform.python_version())' 2>/dev/null || true
        python -c 'import tensorflow as tf; print("tensorflow_version=" + tf.__version__)' 2>/dev/null || true
        printf '\ngit_status:\n'
        git status --short 2>/dev/null || true
    } > "$ENVIRONMENT_FILE"
fi

extract_checkpoint() {
    local log_path="$1"
    sed -n 's/^.*模型权重保存成功: //p' "$log_path" | tail -n 1
}

report_matches_config() {
    local report_path="$1"
    local config_path="$2"
    local report_hash
    local config_hash
    report_hash="$(
        python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["config"]["sha256"])' \
            "$report_path"
    )" || return 1
    config_hash="$(sha256sum "$config_path" | awk '{print $1}')"
    [[ "$report_hash" == "$config_hash" ]]
}

run_and_evaluate() {
    local dataset="$1"
    local variant="$2"
    local slug="$3"
    local config_path="$4"
    local baseline_report="${5:-}"
    local output_dir="$RUN_DIR/${slug}_${variant}"
    local report_path="$output_dir/report.json"
    local train_log="$RUN_DIR/${slug}_${variant}_train.log"
    local evaluation_log="$RUN_DIR/${slug}_${variant}_evaluation.log"

    if [[ ! -f "$config_path" ]]; then
        printf 'Missing config: %s\n' "$config_path" >&2
        return 1
    fi
    if [[ -f "$report_path" ]]; then
        if report_matches_config "$report_path" "$config_path"; then
            printf 'Reusing completed report: %s\n' "$report_path"
            return 0
        fi
        printf 'Refusing stale report with a different config hash: %s\n' \
            "$report_path" >&2
        return 1
    fi

    printf '\nStarting %s %s\nConfig: %s\n' \
        "$dataset" "$variant" "$config_path"
    ./run_hdcti.sh "$config_path" 2>&1 | tee "$train_log"
    local train_exit=${PIPESTATUS[0]}
    if [[ "$train_exit" -ne 0 ]]; then
        printf '%s %s training failed with exit code %d.\n' \
            "$dataset" "$variant" "$train_exit" >&2
        return "$train_exit"
    fi

    local checkpoint
    checkpoint="$(extract_checkpoint "$train_log")"
    if [[ -z "$checkpoint" ]]; then
        printf 'Could not extract checkpoint from: %s\n' "$train_log" >&2
        return 1
    fi

    local command=(
        python tools/evaluate_four_state_checkpoint.py
        --config "$config_path"
        --checkpoint "$checkpoint"
        --output-dir "$output_dir"
    )
    if [[ -n "$baseline_report" ]]; then
        command+=(--baseline-report "$baseline_report")
    fi
    "${command[@]}" 2>&1 | tee "$evaluation_log"
    local evaluation_exit=${PIPESTATUS[0]}
    if [[ "$evaluation_exit" -ne 0 || ! -f "$report_path" ]]; then
        printf '%s %s evaluation failed.\n' "$dataset" "$variant" >&2
        return 1
    fi
}

failed_jobs=0
candidate_reports=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r dataset slug baseline_config candidate_config <<< "$job"
    baseline_report="$RUN_DIR/${slug}_baseline/report.json"
    candidate_report="$RUN_DIR/${slug}_v2/report.json"

    if ! run_and_evaluate \
        "$dataset" baseline "$slug" "$baseline_config"; then
        failed_jobs=$((failed_jobs + 1))
        continue
    fi
    if ! run_and_evaluate \
        "$dataset" v2 "$slug" "$candidate_config" "$baseline_report"; then
        failed_jobs=$((failed_jobs + 1))
        continue
    fi
    candidate_reports+=("$dataset=$candidate_report")
done

if [[ "$failed_jobs" -gt 0 ]]; then
    printf '%d dataset pair(s) failed; summary was not finalized.\n' \
        "$failed_jobs" >&2
    exit 1
fi

summary_command=(
    python tools/summarize_four_state_routing_gate.py
    --output-dir "$RUN_DIR"
    --require-all-pass
    --report "TCM-Suite=$TCMSUITE_REPORT"
)
for report in "${candidate_reports[@]}"; do
    summary_command+=(--report "$report")
done

"${summary_command[@]}"
summary_exit=$?
printf '\nBatch results: %s\n' "$RUN_DIR/summary.md"
if [[ "$summary_exit" -ne 0 ]]; then
    printf 'At least one frozen dataset Gate failed.\n' >&2
    exit "$summary_exit"
fi
printf 'All four frozen dataset Gates passed.\n'
