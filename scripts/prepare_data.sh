#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env
CONFIG=configs/data.yaml
PYTHON=.venv/bin/python
"$PYTHON" -m generative_sim2real.data.manifests --config "$CONFIG"
