#!/bin/bash
# Run GPT-4o judge on af3 / afthink pred files, save results to judge/
set -a && source /livingrooms/tincan/sp/FirstSemester/hw4/work/.env && set +a

LOG=~/run_judge.log
echo "START $(date)" | tee "$LOG"
echo "node: $(hostname)" | tee -a "$LOG"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set" | tee -a "$LOG"
    exit 1
fi

PYTHON=~/miniconda3/envs/af3/bin/python
WORKDIR=/livingrooms/tincan/sp/FirstSemester/hw4/work
mkdir -p "$WORKDIR/judge_mini"

for PRED in "$WORKDIR"/preds/af3_*.json "$WORKDIR"/preds/afthink_*.json; do
    [ -f "$PRED" ] || continue
    NAME=$(basename "$PRED")
    echo "=== Judging $NAME ===" | tee -a "$LOG"
    $PYTHON "$WORKDIR/SAKURA/evaluation/llm_judge.py" \
        -i "$PRED" \
        -o "$WORKDIR/judge_mini" \
        -w 8 \
        2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Accuracy Summary (gpt-4o-mini) ===" | tee -a "$LOG"
for JUDGE in "$WORKDIR"/judge_mini/*_judgements.json; do
    [ -f "$JUDGE" ] || continue
    NAME=$(basename "$JUDGE" _judgements.json)
    ACC=$($PYTHON "$WORKDIR/SAKURA/evaluation/calculate_acc.py" \
        -i "$JUDGE" 2>&1 | grep "Accuracy:")
    printf "%-40s %s\n" "$NAME" "$ACC" | tee -a "$LOG"
done

echo "DONE $(date)" | tee -a "$LOG"
