#!/usr/bin/env bash
# Source-only DA3 head training for one RGB mode. Refuses to run until the frozen
# downstream ID list named by CONFIG exists (scripts/prepare_data.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
CONFIG=${CONFIG:-configs/da3.yaml}   # CONFIG=configs/da3_bblora.yaml for the backbone-LoRA variant
# Only MODE (R|A|G|M) and, for G/M, GENERATED_MANIFEST may differ between comparison runs;
# every other setting is locked in CONFIG and validated by the trainer.
MODE=${MODE:-R}   # override per launch: MODE=A bash scripts/train_da3.sh
GENERATED_MANIFEST=${GENERATED_MANIFEST:-}   # required for MODE=G|M, forbidden for R|A
SEED=${SEED:-0}   # training seed (0 is the primary run; 1 and 2 give variance estimates)
VARIANT=; grep -q 'trainable: head+backbone_lora' "$CONFIG" && VARIANT=_bblora   # run name differs for the bbLoRA config
RUN_NAME=${RUN_NAME:-da3_${MODE}${VARIANT}_seed${SEED}}
OUTPUT=outputs/${RUNS_DIR:-runs_c2}/$RUN_NAME
GPU=${GPU:-0}   # physical GPU index; check nvidia-smi first
DEVICE=cuda:0
PYTHON=external/da3-venv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" -m generative_sim2real.depth.train --config "$CONFIG" --output "$OUTPUT" --device "$DEVICE" \
    --mode "$MODE" --seed "$SEED" ${GENERATED_MANIFEST:+--generated-manifest "$GENERATED_MANIFEST"}
