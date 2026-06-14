#!/bin/bash
LOG=~/setup_kimi.log
echo "START $(date)" > "$LOG"
echo "node: $(hostname)" >> "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh >> "$LOG" 2>&1

# 若 kimi env 已存在則移除重建（確保乾淨）
conda env remove -n kimi -y >> "$LOG" 2>&1 || true

echo "[1/4] Creating conda env kimi (python=3.10)..." >> "$LOG"
conda create -n kimi python=3.10 -y >> "$LOG" 2>&1
echo "conda create exit: $?" >> "$LOG"

echo "[2/4] Installing requirements (skip flash_attn/deepspeed, need GPU to build)..." >> "$LOG"
# 過濾掉需要在 GPU 節點編譯的套件
grep -vE "^(flash_attn|deepspeed)" ~/Kimi-Audio/requirements.txt > /tmp/kimi_req_filtered.txt
echo "Filtered requirements:" >> "$LOG"
cat /tmp/kimi_req_filtered.txt >> "$LOG"

conda run -n kimi pip install -r /tmp/kimi_req_filtered.txt >> "$LOG" 2>&1
echo "pip requirements exit: $?" >> "$LOG"

echo "[3/4] Installing kimia_infer (editable install from local repo)..." >> "$LOG"
conda run -n kimi pip install -e ~/Kimi-Audio/ --no-deps >> "$LOG" 2>&1
echo "kimia_infer install exit: $?" >> "$LOG"

echo "[4/4] Verifying..." >> "$LOG"
conda run -n kimi python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
from kimia_infer.api.kimia import KimiAudio
print('kimia_infer: OK')
" >> "$LOG" 2>&1
echo "verify exit: $?" >> "$LOG"

echo "DONE $(date)" >> "$LOG"
