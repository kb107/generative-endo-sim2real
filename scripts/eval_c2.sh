#!/usr/bin/env bash
# Locked C2 evaluation of every method in the frozen registry, in one campaign.
# Refuses unless the registry is frozen and every hashed input still matches. DRY_RUN=1 validates only:
# it builds the real adapter (parses pose files, lists files) but loads no model and decodes no C2 pixel.
# Ledger, seals and results live in the fixed directory outputs/c2_campaign (not configurable).
# A second campaign under a different registry is a new disclosed experiment and needs
# DISCLOSE_NEW_EXPERIMENT (see docs/evaluation.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
REGISTRY=${REGISTRY:-outputs/c2_campaign/registry.json}
EXPECTED_REGISTRY_SHA256=${EXPECTED_REGISTRY_SHA256-1ceaa08c5bd314c118a93c3e542dd3d9d1bb6d00811a01807bffdf7ff6b81ccc}   # campaign 1, frozen 2026-10-19T21:47:55Z
DISCLOSE_NEW_EXPERIMENT=${DISCLOSE_NEW_EXPERIMENT:-}             # non-empty reason only when the ledger shows another registry
DRY_RUN=${DRY_RUN:-1}                # DRY_RUN=0 bash scripts/eval_c2.sh runs the campaign
GPU=${GPU:-1}                        # physical GPU index; check nvidia-smi first
DEVICE=cuda:0
NUM_WORKERS=0
PYTHON=external/da3-venv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p outputs/c2_campaign
if [ "$DRY_RUN" = 1 ]; then
    "$PYTHON" -m generative_sim2real.evaluation.c2 run --registry "$REGISTRY" --dry-run \
        ${EXPECTED_REGISTRY_SHA256:+--expected-registry-sha256 "$EXPECTED_REGISTRY_SHA256"}
else
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    "$PYTHON" -m generative_sim2real.evaluation.c2 run --registry "$REGISTRY" \
        --device "$DEVICE" --num-workers "$NUM_WORKERS" \
        ${EXPECTED_REGISTRY_SHA256:+--expected-registry-sha256 "$EXPECTED_REGISTRY_SHA256"} \
        ${DISCLOSE_NEW_EXPERIMENT:+--disclose-new-experiment "$DISCLOSE_NEW_EXPERIMENT"} \
        2>&1 | tee -a outputs/c2_campaign/eval_c2.log
fi
