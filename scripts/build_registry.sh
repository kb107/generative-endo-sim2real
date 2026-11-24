#!/usr/bin/env bash
# Build, freeze or verify the C2 evaluation registry (no model inference, no target pixels).
#   ACTION=build   hash all inputs into a DRAFT (frozen: false); official digests come from the pinned weights file
#   ACTION=freeze  mark the draft frozen exactly once (re-verifies every input first)
#   ACTION=verify  re-check against the files on disk, recomputing official digests and re-hashing C2 pose files
# Register the final method set in SPEC before freezing; a frozen registry is never overwritten.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
SPEC=${SPEC:-configs/registry.yaml}
ACTION=${ACTION:-build}          # override per launch: ACTION=freeze bash scripts/build_registry.sh
EXPECTED_REGISTRY_SHA256=        # optional: hash printed by ACTION=freeze; checked by ACTION=verify
PYTHON=external/da3-venv/bin/python
export CUDA_VISIBLE_DEVICES=     # CPU only
export PYTHONPATH=src
mkdir -p outputs/p6-evaluation
"$PYTHON" -m generative_sim2real.evaluation.registry "$ACTION" --spec "$SPEC" \
    ${EXPECTED_REGISTRY_SHA256:+--expected-sha256 "$EXPECTED_REGISTRY_SHA256"} \
    2>&1 | tee -a "outputs/p6-evaluation/build_registry_${ACTION}.log"
