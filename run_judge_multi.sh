#!/bin/bash
# Run 4 judges in parallel on all desta2 prediction files
# Usage: bash run_judge_multi.sh [pred_pattern]
LOG=~/run_judge_multi.log
echo "START $(date)" | tee "$LOG"

if [ -f /livingrooms/tincan/sp/FirstSemester/hw4/work/.env ]; then
    export $(grep -v '^#' /livingrooms/tincan/sp/FirstSemester/hw4/work/.env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set" | tee -a "$LOG"; exit 1
fi
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set" | tee -a "$LOG"; exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate af3

WORKDIR=/livingrooms/tincan/sp/FirstSemester/hw4/work
PRED_PATTERN="${1:-$WORKDIR/preds/desta2_*.json}"
JUDGES=("gpt-4o" "gpt-4o-mini" "gemini-2.5-flash-lite" "gemini-3.1-flash-lite")

PIDS=()
for JUDGE in "${JUDGES[@]}"; do
    OUT_DIR="$WORKDIR/judge/$JUDGE"
    mkdir -p "$OUT_DIR"
    (
        for PRED in $PRED_PATTERN; do
            NAME=$(basename "$PRED")
            echo "=== [$JUDGE] Judging $NAME ===" | tee -a "$LOG"
            python "$WORKDIR/llm_judge_multi.py" \
                -i "$PRED" \
                -o "$OUT_DIR" \
                --judge "$JUDGE" \
                --workers 20 \
                2>&1 | tee -a "$LOG"
        done
        echo "[$JUDGE] ALL DONE" | tee -a "$LOG"
    ) &
    PIDS+=($!)
    echo "Launched judge '$JUDGE' (pid $!)" | tee -a "$LOG"
done

echo "Waiting for all 4 judges to finish..." | tee -a "$LOG"
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo "" | tee -a "$LOG"
echo "=== Accuracy Summary ===" | tee -a "$LOG"
for JUDGE in "${JUDGES[@]}"; do
    echo "--- Judge: $JUDGE ---" | tee -a "$LOG"
    for JUDGE_FILE in "$WORKDIR/judge/$JUDGE"/*_judgements.json; do
        NAME=$(basename "$JUDGE_FILE" _judgements.json)
        printf "  %-40s " "$NAME" | tee -a "$LOG"
        python "$WORKDIR/SAKURA/evaluation/calculate_acc.py" \
            -i "$JUDGE_FILE" 2>&1 | grep "Accuracy:" | tee -a "$LOG"
    done
done

echo "DONE $(date)" | tee -a "$LOG"
