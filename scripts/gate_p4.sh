#!/usr/bin/env bash
# P4 source-controller gate for one controller checkpoint: render (generator venv, GPU), proxy
# scoring (DA3 venv, same GPU, separate step), then apply the pre-declared rules (configs/p4.yaml).
#   RUN=controller_overfit STEP=500 SET=overfit GPU=2 bash scripts/gate_p4.sh
#   RUN=controller_pilot   STEP=2000 SET=validation GPU=2 bash scripts/gate_p4.sh
# RUBRIC=<json> supplies the reviewer's per-condition visual ratings after the grids were inspected
# (re-run with STAGES=report to recompute gate.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
RUN=${RUN:-controller_overfit}
STEP=${STEP:-500}
SET=${SET:-overfit}                   # overfit (8 training pairs) | validation (8 source-validation conditions)
GPU=${GPU:-2}
RUNS=${RUNS:-runs}                     # runs | smoke (labeled smoke checkpoints)
LORA_STEP=${LORA_STEP:-1000}
STAGES=${STAGES:-"render proxy report"}
RUBRIC=${RUBRIC:-}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}
CONFIG=${CONFIG:-configs/p4.yaml}
GENERATOR_PYTHON=external/generator/.venv/bin/python
DA3_PYTHON=external/da3-venv/bin/python
OUTPUT_ROOT=outputs/p4-appearance
SOURCE_LORA=$OUTPUT_ROOT/$RUNS/source_lora/checkpoints/step_$(printf %06d "$LORA_STEP")/lora_source.safetensors
CONTROLLER=$OUTPUT_ROOT/$RUNS/$RUN/checkpoints/step_$(printf %06d "$STEP")/controller.safetensors
OUT=$OUTPUT_ROOT/gate_$RUNS/${RUN}_step${STEP}_${SET}${TAG:+_$TAG}
LOG=$OUTPUT_ROOT/logs/gate_${RUN}_step${STEP}_${SET}${TAG:+_$TAG}.log

mkdir -p "$OUT" "$OUTPUT_ROOT/logs"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
case " $STAGES " in *" render "*|*" proxy "*)
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
    if [ -z "$used" ] || [ "$used" -gt "$MAX_BUSY_MIB" ]; then
        echo "GPU $GPU is busy or missing (${used:-?} MiB used); not starting." >&2
        exit 1
    fi;;
esac
{
    case " $STAGES " in *" render "*)
        "$GENERATOR_PYTHON" -m generative_sim2real.generation.p4_eval render --set "$SET" \
            --controller "$CONTROLLER" --source-lora "$SOURCE_LORA" --step "$STEP" --out "$OUT" \
            --runs "$RUNS" --config "$CONFIG";;
    esac
    case " $STAGES " in *" proxy "*)
        "$DA3_PYTHON" -m generative_sim2real.generation.p4_proxy --render-dir "$OUT";;
    esac
    case " $STAGES " in *" report "*)
        "$GENERATOR_PYTHON" -m generative_sim2real.generation.p4_report --render-dir "$OUT" \
            --config "$CONFIG" ${RUBRIC:+--rubric "$RUBRIC"};;
    esac
} 2>&1 | tee -a "$LOG"
