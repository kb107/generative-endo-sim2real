#!/usr/bin/env bash
# Gemini image-edit client (closed-editor comparison). DRY RUN by default (no network, no key needed).
# Real, billed calls need EXECUTE=1 and GEMINI_API_KEY in the environment (never written to disk or logs).
# Needs external/foundation-venv with google-genai and pillow.
#   SET=fixed8 bash scripts/foundation_gemini.sh                     # dry run + cost estimate
#   EXECUTE=1 MAX_REQUESTS=8 SET=fixed8 bash scripts/foundation_gemini.sh
#   CONDITION=depth SET=subset300 bash scripts/foundation_gemini.sh  # depth map + 2 phantom refs (no SimCol frame)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && source .env

# Named settings
SET=${SET:-fixed8}                  # fixed8 | subset300
MODEL=${MODEL:-gemini-nano-banana-2.1}   # Nano Banana Pro: gemini-3-pro-image
IMAGE_SIZE=${IMAGE_SIZE:-1K}
MAX_REQUESTS=${MAX_REQUESTS:-8}     # hard cap on attempted requests (retries included)
EXECUTE=${EXECUTE:-}
EVERY=${EVERY:-1}                   # use every k-th item (evenly spread subset)
CONDITION=${CONDITION:-frame}          # frame | depth (subset300; needs foundation.inputs --depth300)
SUFFIX=$([ "$CONDITION" = depth ] && echo -depth || true)
OUT=${OUT:-outputs/foundation/gemini/${MODEL}${SUFFIX}/$SET}
FOUNDATION_PYTHON=external/foundation-venv/bin/python

extra=()
[ -n "$EXECUTE" ] && extra+=(--execute)
PYTHONPATH=src "$FOUNDATION_PYTHON" -m generative_sim2real.foundation.gemini_edit \
    --set "$SET" --out "$OUT" --model "$MODEL" --image-size "$IMAGE_SIZE" \
    --max-requests "$MAX_REQUESTS" --condition "$CONDITION" --every "$EVERY" "${extra[@]}"
