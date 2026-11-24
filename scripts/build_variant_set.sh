#!/usr/bin/env bash
# Build the appearance-control "one image per ID" sets as G-style manifests under
# outputs/generated_variants/<name>/, consumed by: MODE=G GENERATED_MANIFEST=<manifest> bash scripts/train_da3.sh
#   T_reinhard         Reinhard colour transfer to a seeded C1-train frame
#   A_strong           strong photometric augmentation (no target data)
#   F_fda_b01/b05      Fourier Domain Adaptation to the same C1-train frame as T_reinhard (beta 0.01 / 0.05)
# Builds the 2,000 frozen downstream IDs on the CPU in a few minutes per set.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
VARIANTS=${VARIANTS:-"T_reinhard A_strong F_fda_b01 F_fda_b05"}
WORKERS=${WORKERS:-12}
PYTHON=external/da3-venv/bin/python
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for name in $VARIANTS; do
    "$PYTHON" -m generative_sim2real.variants.build build --name "$name" --workers "$WORKERS" --overwrite
done
