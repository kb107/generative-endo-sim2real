#!/usr/bin/env bash
# P4 repair: target/source LoRA appearance review (W1 and diversity against all four real reference sets,
# nearest-neighbour memorisation check). Declared in configs/p4.yaml `appearance_review`.
#   GPU=2 bash scripts/appearance_review_p4.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
GPU=${GPU:-2}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}
LORA_STEP=${LORA_STEP:-1000}
TARGET_RUN=${TARGET_RUN:-target_lora}        # v2: TARGET_RUN=target_lora_v2 TARGET_STEP=2000
TARGET_STEP=${TARGET_STEP:-$LORA_STEP}
CONFIG=configs/p4.yaml
GENERATOR_PYTHON=external/generator/.venv/bin/python
OUTPUT_ROOT=outputs/p4-appearance
STEP_DIR=step_$(printf %06d "$LORA_STEP")
SOURCE_LORA=$OUTPUT_ROOT/runs/source_lora/checkpoints/$STEP_DIR/lora_source.safetensors
TARGET_LORA=$OUTPUT_ROOT/runs/$TARGET_RUN/checkpoints/step_$(printf %06d "$TARGET_STEP")/lora_target.safetensors
if [ "$TARGET_RUN" = target_lora ]; then
    OUT=$OUTPUT_ROOT/appearance_review/lora${LORA_STEP}
    LOG=$OUTPUT_ROOT/logs/appearance_review.log
else
    OUT=$OUTPUT_ROOT/appearance_review/${TARGET_RUN}_${TARGET_STEP}
    LOG=$OUTPUT_ROOT/logs/appearance_review_${TARGET_RUN}.log
fi

used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
if [ -z "$used" ] || [ "$used" -gt "$MAX_BUSY_MIB" ]; then
    echo "GPU $GPU is busy or missing (${used:-?} MiB used); not starting." >&2
    exit 1
fi
mkdir -p "$OUT" "$OUTPUT_ROOT/logs"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
"$GENERATOR_PYTHON" -m generative_sim2real.generation.p4_eval reference \
    --source-lora "$SOURCE_LORA" --target-lora "$TARGET_LORA" --out "$OUT" --config "$CONFIG" \
    2>&1 | tee -a "$LOG"
