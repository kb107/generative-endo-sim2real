#!/usr/bin/env bash
# P5b bulk generation: one image per frozen downstream ID with a frozen recipe (src/.../generation/bulk.py).
# Atomic, idempotent shards: rerun after any interruption to resume. Stages: generate (GPU) | manifest | verify.
#   GPU=3 bash scripts/generate_downstream.sh                       # production recipe, all 2,000 IDs
#   RECIPE=outputs/p5-diagnostics/bulk/smoke/recipe_smoke.json OUT_ROOT=outputs/p5-diagnostics/bulk/smoke \
#       LIMIT=16 GPU=3 bash scripts/generate_downstream.sh          # labelled smoke (first 16 IDs)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Batch size comes from the recipe (production: 1, hashed and fingerprinted); this script cannot override it.
# Resuming after a code/library/GPU change needs ALLOW_ENV_CHANGE=1 (recorded in run_history.jsonl).

# Named settings
GPU=${GPU:-3}
RECIPE=${RECIPE:-outputs/p5-diagnostics/recipe_v2_production.json}
IDS=${IDS:-outputs/manifests/milestone_v1/downstream_ids.json}
SOURCE=${SOURCE:-outputs/manifests/milestone_v1/source_train.jsonl}
OUT_ROOT=${OUT_ROOT:-outputs}          # images land in $OUT_ROOT/generated/<recipe_hash>/
LIMIT=${LIMIT:-}                       # first N IDs; smoke recipes only
ALLOW_ENV_CHANGE=${ALLOW_ENV_CHANGE:-}
STAGES=${STAGES:-"generate manifest verify"}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}
GENERATOR_PYTHON=external/generator/.venv/bin/python
DA3_PYTHON=external/da3-venv/bin/python
LOG_DIR=${LOG_DIR:-outputs/p5-diagnostics/bulk}

mkdir -p "$LOG_DIR"
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID
args=(--recipe "$RECIPE" --ids "$IDS" --source "$SOURCE" --out-root "$OUT_ROOT")
[ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
run_args=("${args[@]}")
[ -n "$ALLOW_ENV_CHANGE" ] && run_args+=(--allow-environment-change)
case " $STAGES " in *" generate "*)
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
    if [ -z "$used" ] || [ "$used" -gt "$MAX_BUSY_MIB" ]; then
        echo "GPU $GPU is busy or missing (${used:-?} MiB used); not starting." >&2
        exit 1
    fi;;
esac
{
    case " $STAGES " in *" generate "*)
        CUDA_VISIBLE_DEVICES=$GPU "$GENERATOR_PYTHON" -m generative_sim2real.generation.bulk run \
            "${run_args[@]}" --no-manifest;;
    esac
    # Manifest build + rgb_modes G/M validation needs no GPU or generator libraries: use the DA3 venv.
    case " $STAGES " in *" manifest "*)
        CUDA_VISIBLE_DEVICES= "$DA3_PYTHON" -m generative_sim2real.generation.bulk manifest "${args[@]}";;
    esac
    case " $STAGES " in *" verify "*)
        CUDA_VISIBLE_DEVICES= "$DA3_PYTHON" -m generative_sim2real.generation.bulk verify "${args[@]}";;
    esac
} 2>&1 | tee -a "$LOG_DIR/generate_$(date +%Y%m%d).log"
