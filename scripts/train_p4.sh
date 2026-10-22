#!/usr/bin/env bash
# P4 trainer for one run of configs/p4.yaml. Run from anywhere; durable launch example:
#   RUN=source_lora GPU=2 setsid nohup bash scripts/train_p4.sh > /dev/null 2>&1 &
# RUN: source_lora | target_lora | controller_overfit | controller_pilot
# RESUME=1 continues an interrupted run from its last.pt. SMOKE_STEPS=N runs a labeled smoke
# (outputs/p4-appearance/smoke/RUN); STOP_AFTER=K interrupts a smoke at update K for a resume test.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
RUN=${RUN:-source_lora}
GPU=${GPU:-2}                         # physical GPU index; check nvidia-smi first
RESUME=${RESUME:-0}
SMOKE_STEPS=${SMOKE_STEPS:-}
STOP_AFTER=${STOP_AFTER:-}
OUTPUT=${OUTPUT:-}                    # optional run-directory override (smoke comparisons)
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}    # refuse to start if the GPU already holds more than this
CONFIG=${CONFIG:-configs/p4.yaml}
GENERATOR_PYTHON=external/generator/.venv/bin/python
OUTPUT_ROOT=outputs/p4-appearance
LOG_DIR=$OUTPUT_ROOT/logs

used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
if [ -z "$used" ] || [ "$used" -gt "$MAX_BUSY_MIB" ]; then
    echo "GPU $GPU is busy or missing (${used:-?} MiB used); not starting." >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
suffix=${SMOKE_STEPS:+_smoke}
args=(--run "$RUN" --config "$CONFIG")
[ "$RESUME" = 1 ] && args+=(--resume)
[ -n "$SMOKE_STEPS" ] && args+=(--smoke-steps "$SMOKE_STEPS")
[ -n "$STOP_AFTER" ] && args+=(--stop-after-step "$STOP_AFTER")
[ -n "$OUTPUT" ] && args+=(--output "$OUTPUT")
"$GENERATOR_PYTHON" -m generative_sim2real.generation.train "${args[@]}" \
    2>&1 | tee -a "$LOG_DIR/${RUN}${suffix}.log"
