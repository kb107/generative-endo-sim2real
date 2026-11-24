#!/usr/bin/env bash
# FD/KID at matched n (the same <=300 subset300 IDs) of every image arm vs C1 appearance validation (RGB only).
# Builds symlink sets (foundation/fd_sets.py), then scores each with generation_metrics.compare (DINOv2-base CLS).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
GPU=${GPU:-0}                       # physical GPU index; check nvidia-smi first
SETS=outputs/foundation/fd300/sets
OUT=outputs/foundation/fd300/gm
REFERENCE=outputs/manifests/milestone_v1/target_appearance_validation.jsonl
GEN_PYTHON=external/generator/.venv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
mkdir -p "$OUT"
"$GEN_PYTHON" -m generative_sim2real.foundation.fd_sets --out "$SETS"
for dir in "$SETS"/*/; do
    arm=$(basename "$dir")
    "$GEN_PYTHON" -m generative_sim2real.generation_metrics.compare --generated "$dir" --reference "$REFERENCE" \
        --out "$OUT/$arm.json"
done
"$GEN_PYTHON" -m generative_sim2real.generation_metrics.table "$OUT"/*.json | tee "$OUT/table.md"
