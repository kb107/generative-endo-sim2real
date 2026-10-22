#!/usr/bin/env bash
# Regenerate every result figure/table that does not need real-phantom (C2) results, from saved evidence in
# outputs/ (git-ignored). CPU only: no model, no GPU, no dataset access. Writes docs/results/.
# Uses the DA3 venv (it pins matplotlib, pillow, numpy, torch). Safe to re-run: G/M cells in the DA3 table
# are 'pending' until their validation summaries exist, then a re-run fills them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
PYTHON=external/da3-venv/bin/python
OUTPUTS=outputs
RESULTS=docs/results
C2_STATUS=$'not run here\nfrozen registry required\n(excluded from these assets)'

export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=src MPLBACKEND=Agg
mkdir -p "$RESULTS" "$OUTPUTS/p7-results"
"$PYTHON" -m generative_sim2real.reporting.build --outputs "$OUTPUTS" --out "$RESULTS" --c2-status "$C2_STATUS" \
    2>&1 | tee "$OUTPUTS/p7-results/make_figures.log"
