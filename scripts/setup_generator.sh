#!/usr/bin/env bash
# Create/verify the isolated generator environment and download the pinned PixArt
# files into the inherited Hugging Face cache. Run from anywhere: bash scripts/setup_generator.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
GENERATOR_VENV=external/generator/.venv
REQUIREMENTS=configs/requirements-generator.txt
MODEL_ID=PixArt-alpha/PixArt-XL-2-512x512
MODEL_REVISION=50f702106901db6d0f8b67eb88e814c56ded2692
LOG_DIR=outputs/generator-integration/logs

mkdir -p "$LOG_DIR"
if [ ! -x "$GENERATOR_VENV/bin/python" ]; then
    uv venv --python 3.11 --seed "$GENERATOR_VENV"
fi
uv pip install --python "$GENERATOR_VENV/bin/python" -r "$REQUIREMENTS" \
    2>&1 | tee -a "$LOG_DIR/install.log"

# Needed weights only: fp32 transformer/VAE, fp16 T5 text encoder, tokenizer, configs.
MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" "$GENERATOR_VENV/bin/python" -I - <<'PY' \
    2>&1 | tee -a "$LOG_DIR/download.log"
import os, time
from huggingface_hub import snapshot_download
start = time.time()
path = snapshot_download(
    os.environ['MODEL_ID'], revision=os.environ['MODEL_REVISION'],
    allow_patterns=[
        'model_index.json', 'scheduler/*', 'tokenizer/*', 'transformer/*', 'vae/config.json',
        'vae/diffusion_pytorch_model.safetensors', 'text_encoder/config.json',
        'text_encoder/model.fp16-*.safetensors', 'text_encoder/model.safetensors.index.fp16.json'])
print(f'snapshot={path} download_seconds={time.time() - start:.1f}')
PY

# Verify the downloaded files against the Hub LFS sha256 values recorded in the repository.
SNAPSHOT=$(MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" "$GENERATOR_VENV/bin/python" -I -c \
    'import os; from huggingface_hub import snapshot_download as s; print(s(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], local_files_only=True))')
(cd "$SNAPSHOT" && sha256sum -c "$OLDPWD/configs/generator-weights.sha256") \
    2>&1 | tee -a "$LOG_DIR/download.log"
