#!/bin/bash
LOG=~/setup_af3.log
echo "START $(date)" > "$LOG"
echo "node: $(hostname)" >> "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh >> "$LOG" 2>&1

echo "[1/4] Creating conda env af3 (python=3.10)..." >> "$LOG"
conda create -n af3 python=3.10 -y >> "$LOG" 2>&1
echo "conda create exit: $?" >> "$LOG"

echo "[2/4] Installing PyTorch (cu121)..." >> "$LOG"
conda run -n af3 pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 >> "$LOG" 2>&1
echo "torch install exit: $?" >> "$LOG"

echo "[3/4] Installing transformers>=4.57 + extras..." >> "$LOG"
conda run -n af3 pip install "transformers>=4.57" accelerate soundfile librosa >> "$LOG" 2>&1
echo "transformers install exit: $?" >> "$LOG"

echo "[4/4] Verifying..." >> "$LOG"
conda run -n af3 python -c "
import torch, transformers
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
" >> "$LOG" 2>&1

echo "DONE $(date)" >> "$LOG"
