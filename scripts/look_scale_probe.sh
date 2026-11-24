#!/usr/bin/env bash
# Look -> scale probe on the 32 source-validation conditions of the lc3000 matrix (source labels only).
# Outputs: outputs/analysis/look_scale_probe/{results.json,table.md}. Inference only; recheck nvidia-smi for a free GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
GPU=${GPU:-3}               # physical GPU index
MATRIX=outputs/p5-diagnostics/matrix_lc3000_seed17
OUT=${OUT:-outputs/analysis/look_scale_probe}
# Run directories; best.pt and last.pt of each are probed, plus the official pretrained head (Z).
HEADS="outputs/runs_c2/da3_R_seed0 outputs/runs_c2/da3_G_seed0 outputs/runs_c2_v4/da3_G_seed0 outputs/runs_c2_simuscope/da3_G_seed0"
PYTHON=external/da3-venv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
"$PYTHON" -m generative_sim2real.analysis.look_scale --matrix "$MATRIX" --out "$OUT" --device cuda:0 --heads $HEADS
