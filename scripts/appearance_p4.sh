#!/usr/bin/env bash
# P4 appearance checks: source vs target LoRA grids for source validation and C1 RGB validation
# and the pre-declared pixel-statistics checks. Needs both trained LoRAs and a controller.
#   CONTROLLER_RUN=controller_pilot CONTROLLER_STEP=2000 GPU=2 bash scripts/appearance_p4.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
CONTROLLER_RUN=${CONTROLLER_RUN:-controller_pilot}
CONTROLLER_STEP=${CONTROLLER_STEP:-2000}
LORA_STEP=${LORA_STEP:-1000}
GPU=${GPU:-2}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}
CONFIG=configs/p4.yaml
GENERATOR_PYTHON=external/generator/.venv/bin/python
OUTPUT_ROOT=outputs/p4-appearance
STEP_DIR=step_$(printf %06d "$LORA_STEP")
SOURCE_LORA=$OUTPUT_ROOT/runs/source_lora/checkpoints/$STEP_DIR/lora_source.safetensors
TARGET_LORA=$OUTPUT_ROOT/runs/target_lora/checkpoints/$STEP_DIR/lora_target.safetensors
CONTROLLER=$OUTPUT_ROOT/runs/$CONTROLLER_RUN/checkpoints/step_$(printf %06d "$CONTROLLER_STEP")/controller.safetensors
OUT=$OUTPUT_ROOT/appearance/lora${LORA_STEP}_${CONTROLLER_RUN}${CONTROLLER_STEP}
LOG=$OUTPUT_ROOT/logs/appearance_lora${LORA_STEP}.log

used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
if [ -z "$used" ] || [ "$used" -gt "$MAX_BUSY_MIB" ]; then
    echo "GPU $GPU is busy or missing (${used:-?} MiB used); not starting." >&2
    exit 1
fi
mkdir -p "$OUT" "$OUTPUT_ROOT/logs"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
"$GENERATOR_PYTHON" -m generative_sim2real.generation.p4_eval appearance \
    --source-lora "$SOURCE_LORA" --target-lora "$TARGET_LORA" --controller "$CONTROLLER" \
    --out "$OUT" --config "$CONFIG" 2>&1 | tee -a "$LOG"
