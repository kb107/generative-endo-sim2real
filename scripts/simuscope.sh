#!/usr/bin/env bash
# SimuScope-style baseline (diffusers re-implementation; see docs/baseline-simuscope.md).
# Stages: select (C1 frame choice) | lora (kohya style LoRA, GPU) | smoke (16 images, GPU) |
#         generate (all frozen downstream IDs, GPU) | manifest | verify | score (generation metrics)
#   GPU=2 bash scripts/simuscope.sh                       # everything
#   GPU=2 STAGES="smoke" bash scripts/simuscope.sh
# Rerun after any interruption: generation resumes from atomic shards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
GPU=${GPU:-2}
STAGES=${STAGES:-"select lora smoke generate manifest verify score"}
CONFIG=${CONFIG:-configs/simuscope.yaml}
SMOKE_LIMIT=${SMOKE_LIMIT:-16}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1000}
OUT_DIR=${OUT_DIR:-outputs/baselines_simuscope}
GENERATED_MANIFEST=${GENERATED_MANIFEST:-outputs/generated_baselines/simuscope/generated_manifest.jsonl}
GM_DIR=${GM_DIR:-outputs/sweeps/gm}
PYTHON=external/generator/.venv/bin/python

mkdir -p "$OUT_DIR/logs" "$GM_DIR"
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID
module=generative_sim2real.baselines.simuscope
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
need_gpu() {
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
    if [ -z "$used" ] || [ "$used" -gt "$MAX_BUSY_MIB" ]; then
        echo "GPU $GPU is busy or missing (${used:-?} MiB used); not starting." >&2
        exit 1
    fi
}
{
    has select && "$PYTHON" -m $module select --config "$CONFIG"
    if has lora; then need_gpu; CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m $module train-lora --config "$CONFIG"; fi
    if has smoke; then
        need_gpu
        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m $module run --config "$CONFIG" --smoke --limit "$SMOKE_LIMIT"
    fi
    if has generate; then need_gpu; CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m $module run --config "$CONFIG" --no-manifest; fi
    has manifest && CUDA_VISIBLE_DEVICES= "$PYTHON" -m $module manifest --config "$CONFIG"
    has verify && CUDA_VISIBLE_DEVICES= "$PYTHON" -m $module verify --config "$CONFIG"
    if has score; then
        for ref in target_appearance_validation source_validation; do
            CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m generative_sim2real.generation_metrics.compare \
                --generated "$GENERATED_MANIFEST" --reference "outputs/manifests/milestone_v1/$ref.jsonl" \
                --out "$GM_DIR/simuscope_vs_${ref}.json"
        done
    fi
} 2>&1 | tee -a "$OUT_DIR/logs/simuscope_$(date +%Y%m%d).log"
