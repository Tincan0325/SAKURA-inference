#!/bin/bash
LOG=~/run_desta2.log
echo "START $(date)" > "$LOG"
echo "node: $(hostname)" >> "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh >> "$LOG" 2>&1

# Load API keys from .env
if [ -f /livingrooms/tincan/sp/FirstSemester/hw4/work/.env ]; then
    export $(grep -v '^#' /livingrooms/tincan/sp/FirstSemester/hw4/work/.env | xargs)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set" >> "$LOG"
    exit 1
fi

conda run -n af3 python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" >> "$LOG" 2>&1

WORKDIR=/livingrooms/tincan/sp/FirstSemester/hw4/work
echo "Running DeSTA2 inference from $WORKDIR" >> "$LOG"

cd "$WORKDIR" && conda run -n af3 \
    python run_desta2.py \
        --sakura-dir SAKURA \
        --pred-dir preds \
        --hf-token "$HF_TOKEN" \
    2>&1 | tee -a "$LOG"

echo "DONE $(date)" >> "$LOG"
