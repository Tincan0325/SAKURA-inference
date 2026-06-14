#!/bin/bash
LOG=~/setup_flash_attn.log
echo "START $(date)" > "$LOG"
echo "node: $(hostname)" >> "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh >> "$LOG" 2>&1

conda run -n kimi python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" >> "$LOG" 2>&1

# TMPDIR 設在 home，避免 cross-device link 錯誤 (/livingrooms != /home)
mkdir -p ~/pip_tmp

echo "Installing flash_attn via pre-built wheel (cu12 + torch2.6 + cp310)..." >> "$LOG"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

cd ~ && TMPDIR=~/pip_tmp conda run -n kimi pip install "$WHEEL_URL" >> "$LOG" 2>&1
STATUS=$?
echo "wheel install exit: $STATUS" >> "$LOG"

if [ $STATUS -ne 0 ]; then
    echo "Wheel failed, building from source with TMPDIR fix..." >> "$LOG"
    cd ~ && TMPDIR=~/pip_tmp conda run -n kimi pip install flash_attn==2.7.4.post1 >> "$LOG" 2>&1
    echo "source build exit: $?" >> "$LOG"
fi

echo "Verifying flash_attn..." >> "$LOG"
conda run -n kimi python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)" >> "$LOG" 2>&1

echo "Verifying kimia_infer..." >> "$LOG"
conda run -n kimi python -c "
from kimia_infer.api.kimia import KimiAudio
print('kimia_infer: OK')
" >> "$LOG" 2>&1
echo "verify exit: $?" >> "$LOG"

echo "DONE $(date)" >> "$LOG"
