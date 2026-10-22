#!/usr/bin/env bash
# P5 reduced swap matrix (configs/p5.yaml): render 72 images (generator venv, GPU), score them with the frozen
# DA3 proxy (DA3 venv, same GPU, separate step), then apply the declared within-adapter rules.
#   GPU=2 bash scripts/p5_matrix.sh            # STAGES="render proxy report" (default)
#   STAGES=report bash scripts/p5_matrix.sh    # recompute matrix.json from existing proxy.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
GPU=${GPU:-2}
STAGES=${STAGES:-"render proxy report"}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}
CONFIG=${CONFIG:-configs/p5.yaml}          # v2: CONFIG=configs/p5_v2.yaml OUT=outputs/p5-diagnostics/matrix_reduced_v2
GENERATOR_PYTHON=external/generator/.venv/bin/python
DA3_PYTHON=external/da3-venv/bin/python
OUT=${OUT:-outputs/p5-diagnostics/matrix_reduced}
LOG=${OUT}.log

mkdir -p "$OUT"
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
        "$GENERATOR_PYTHON" -m generative_sim2real.generation.p5_matrix render --config "$CONFIG";;
    esac
    case " $STAGES " in *" proxy "*)
        "$DA3_PYTHON" -m generative_sim2real.generation.p4_proxy --render-dir "$OUT";;
    esac
    case " $STAGES " in *" report "*)
        "$GENERATOR_PYTHON" -m generative_sim2real.generation.p5_matrix report --config "$CONFIG";;
    esac
} 2>&1 | tee -a "$LOG"
