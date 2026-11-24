#!/usr/bin/env bash
# Geometry-fidelity proxy (registration + frozen DA3-Metric depth structure) in the DA3 venv.
#   GPU=2 SET=subset300 ARMS="gemini=outputs/foundation/gemini/gemini-nano-banana-2.1/subset300/{index:03d}.png" \
#       bash scripts/foundation_geometry.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
GPU=${GPU:-2}
SET=${SET:-fixed8}                  # fixed8 | matrix32 | subset300
ARMS=${ARMS:-}                      # space-separated NAME=TEMPLATE extra arms ({cond} or {index:03d})
OUT=${OUT:-outputs/foundation/geometry/$SET.json}
DA3_PYTHON=external/da3-venv/bin/python

extra=()
for arm in $ARMS; do extra+=(--arm "$arm"); done
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src "$DA3_PYTHON" -m generative_sim2real.foundation.geometry \
    --set "$SET" --out "$OUT" "${extra[@]}"
