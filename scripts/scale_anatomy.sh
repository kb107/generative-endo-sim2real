#!/usr/bin/env bash
# Label-free scale anatomy of every head registered in the final C2 campaign (source labels only; no C2 results read).
# Stages (space separated, in order): probe kappa cx_first cx_more cx_rest assemble. Inference only.
# Outputs: outputs/analysis/look_scale_probe_c5/, outputs/analysis/scale_anatomy/{kappa,cx,predictions.*}.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
GPU=${GPU:-1}               # physical GPU index (1 only: 0 runs the C2 campaign, 2-3 are reserved for training)
STAGES=${STAGES:-"probe kappa cx_first cx_more cx_rest assemble"}
REGISTRY=outputs/c2_campaign/registry_c5.json
MATRIX=outputs/p5-diagnostics/matrix_lc3000_seed17
PROBE_OUT=outputs/analysis/look_scale_probe_c5
OUT=outputs/analysis/scale_anatomy
# c_X cost is 2000 forwards per (set, head); stages are ordered by what the campaign 6A arms need first.
CX_FIRST_SETS="G raw"; CX_FIRST_HEADS="R R_s1"
CX_MORE_HEADS="R.last R_s1.last G"        # same sets, remaining heads
CX_REST_SETS="Gp S T A_strong"; CX_REST_HEADS="R R_s1"
PYTHON=external/da3-venv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
mkdir -p "$OUT"
for stage in $STAGES; do
  echo "== $stage $(date -Is)"
  case $stage in
    probe) "$PYTHON" -m generative_sim2real.analysis.look_scale --matrix "$MATRIX" --out "$PROBE_OUT" --device cuda:0 \
             --registry "$REGISTRY" --no-gain-sweeps ;;
    kappa) "$PYTHON" -m generative_sim2real.analysis.scale_anatomy kappa --out "$OUT" --registry "$REGISTRY" ;;
    cx_first) "$PYTHON" -m generative_sim2real.analysis.scale_anatomy cx --out "$OUT" --registry "$REGISTRY" --sets $CX_FIRST_SETS --heads $CX_FIRST_HEADS ;;
    cx_more) "$PYTHON" -m generative_sim2real.analysis.scale_anatomy cx --out "$OUT" --registry "$REGISTRY" --sets $CX_FIRST_SETS --heads $CX_MORE_HEADS ;;
    cx_rest) "$PYTHON" -m generative_sim2real.analysis.scale_anatomy cx --out "$OUT" --registry "$REGISTRY" --sets $CX_REST_SETS --heads $CX_REST_HEADS ;;
    assemble) "$PYTHON" -m generative_sim2real.analysis.scale_anatomy assemble --out "$OUT" --registry "$REGISTRY" \
                --probe "$PROBE_OUT/results.json" ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
