#!/usr/bin/env bash
# Fixed comparison inputs for the foundation-model study: 8 conditions, 2 C1 references, 300 IDs.
# Needs the DA3 venv (torch + dataset code); reads SimCol RGB and C1-train RGB only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

DA3_PYTHON=external/da3-venv/bin/python

PYTHONPATH=src "$DA3_PYTHON" -m generative_sim2real.foundation.inputs
