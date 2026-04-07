#!/bin/bash
#
# One-shot environment setup for Memory Affordance mask generation on GPU server.
#
# Usage:
#   bash setup_server.sh
#
# Prerequisites:
#   - GPU server with CUDA 12.1+, ≥24GB VRAM
#   - Conda installed (Anaconda/Miniconda)
#   - ~50GB free disk for models + intermediate results
#
# Steps:
#   1. Create conda env `mem-aff` with Python 3.11
#   2. Install PyTorch 2.5.1 + CUDA 12.1
#   3. Install Flash Attention 2.7.4 (pre-built wheel)
#   4. Install Rex-Omni and SAM2
#   5. Download model weights from HuggingFace
#   6. Verify GPU is visible and models load

set -euo pipefail

ENV_NAME="${ENV_NAME:-mem-aff}"
PYTHON_VERSION="3.11"
PROJECT_DIR="${PROJECT_DIR:-$HOME/KRA_src}"

echo "=========================================="
echo "Memory Affordance — Server Setup"
echo "=========================================="
echo "Env name:     $ENV_NAME"
echo "Project dir:  $PROJECT_DIR"
echo "Hostname:     $(hostname)"
echo

# 1. Verify GPU
echo "[1/6] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "  ERROR: nvidia-smi not found. Is CUDA installed?"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

# 2. Create conda env
echo "[2/6] Creating conda env $ENV_NAME (python $PYTHON_VERSION)..."
if conda env list | grep -q "^$ENV_NAME "; then
  echo "  Env $ENV_NAME already exists. Reusing."
else
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate (works in scripts via this trick)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "  Activated: $(which python)"
echo

# 3. Install PyTorch + CUDA 12.1
echo "[3/6] Installing PyTorch 2.5.1 + CUDA 12.1..."
if python -c "import torch; assert torch.__version__.startswith('2.5')" 2>/dev/null; then
  echo "  PyTorch 2.5 already installed."
else
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print(f'  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo

# 4. Install Flash Attention (pre-built wheel for cu121 + torch 2.5 + py311)
echo "[4/6] Installing Flash Attention 2.7.4..."
if python -c "import flash_attn" 2>/dev/null; then
  echo "  Flash Attention already installed."
else
  FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
  pip install "$FLASH_WHEEL" || {
    echo "  Pre-built wheel failed; trying source build (slow, ~30min)..."
    pip install flash-attn==2.7.4 --no-build-isolation
  }
fi
echo

# 5. Install Rex-Omni + SAM2 + project deps
echo "[5/6] Installing Rex-Omni, SAM2, and project deps..."
pip install -q opencv-python-headless pillow tqdm numpy
pip install -q openai modelscope requests
pip install -q git+https://github.com/IDEA-Research/Rex-Omni.git --no-deps
pip install -q git+https://github.com/facebookresearch/sam2.git
# Rex-Omni transitive deps
pip install -q transformers accelerate sentencepiece
echo

# 6. Download model weights from HuggingFace
echo "[6/6] Downloading model weights..."
pip install -q huggingface_hub
huggingface-cli download facebook/sam2.1-hiera-large 2>&1 | tail -3
huggingface-cli download IDEA-Research/Rex-Omni 2>&1 | tail -3
echo

# Verification
echo "=========================================="
echo "Verification"
echo "=========================================="
python <<'PY'
import torch
print(f"  torch: {torch.__version__}")
print(f"  cuda: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f}GB)")

try:
    import flash_attn
    print(f"  flash_attn: {flash_attn.__version__}")
except Exception as e:
    print(f"  flash_attn: FAILED ({e})")

try:
    from rex_omni import RexOmniWrapper
    print(f"  rex_omni: OK")
except Exception as e:
    print(f"  rex_omni: FAILED ({e})")

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"  sam2: OK")
except Exception as e:
    print(f"  sam2: FAILED ({e})")
PY

echo
echo "=========================================="
echo "Setup complete."
echo
echo "Activate env: conda activate $ENV_NAME"
echo "Project:      $PROJECT_DIR"
echo
echo "Next:"
echo "  1. cd $PROJECT_DIR"
echo "  2. python benchmark/scripts/generate_masks.py \\"
echo "       --qa_check_dir results/episode_qa \\"
echo "       --keyframes_root data/keyframes/ego4d_batch \\"
echo "       --output_dir results/episode_masks"
echo "=========================================="
