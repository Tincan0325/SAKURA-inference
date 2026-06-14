#!/bin/bash
LOG=~/run_af3.log
echo "START $(date)" > "$LOG"
echo "node: $(hostname)" >> "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh >> "$LOG" 2>&1

conda run -n af3 python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.cuda.is_available())" >> "$LOG" 2>&1

WORKDIR=/livingrooms/tincan/sp/FirstSemester/hw4/work
echo "Running AF3 inference from $WORKDIR" >> "$LOG"

cd "$WORKDIR" && TMPDIR=~/pip_tmp conda run -n af3 python run_af3.py \
    --sakura-dir SAKURA \
    --pred-dir preds \
    2>&1 | tee -a "$LOG"

echo "DONE $(date)" >> "$LOG"
