#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .env

DA3_DIR=external/Depth-Anything-3
DA3_REPOSITORY=https://github.com/ByteDance-Seed/Depth-Anything-3.git
DA3_REVISION=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
DA3_VENV=external/da3-venv
PYTHON_VERSION=3.11
REQUIREMENTS=configs/requirements-da3.txt
DOWNLOAD_CHECKPOINT=1
CHECKPOINT=depth-anything/DA3METRIC-LARGE
CHECKPOINT_REVISION=4010e39f3634a45bc60553321fb49fb760bd594e

: "${HF_HOME:?HF_HOME must be set by the shell or in .env}"

if [[ ! -d "$DA3_DIR/.git" ]]; then
    if [[ -e "$DA3_DIR" ]]; then
        echo "$DA3_DIR exists but is not a Git repository" >&2
        exit 1
    fi
    git clone "$DA3_REPOSITORY" "$DA3_DIR"
fi
if [[ -n "$(git -C "$DA3_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "$DA3_DIR has tracked local changes; refusing to change revisions" >&2
    exit 1
fi
if [[ "$(git -C "$DA3_DIR" rev-parse HEAD)" != "$DA3_REVISION" ]]; then
    git -C "$DA3_DIR" fetch origin "$DA3_REVISION"
    git -C "$DA3_DIR" checkout --detach "$DA3_REVISION"
fi

if [[ ! -x "$DA3_VENV/bin/python" ]]; then
    uv venv "$DA3_VENV" --python "$PYTHON_VERSION" --seed
fi
# NumPy<2 stack is fully pinned in the lock; upstream DA3 is installed without
# dependency resolution so its unpinned metadata cannot change the lock.
uv pip install --python "$DA3_VENV/bin/python" -r "$REQUIREMENTS"
uv pip install --python "$DA3_VENV/bin/python" --no-deps --editable "$DA3_DIR"
uv pip check --python "$DA3_VENV/bin/python" || echo "uv pip check reported upstream metadata conflicts (see log)"

if [[ "$DOWNLOAD_CHECKPOINT" == 1 ]]; then
    "$DA3_VENV/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$CHECKPOINT', revision='$CHECKPOINT_REVISION'))"
fi
echo "DA3 environment ready at $DA3_VENV (DA3 revision $DA3_REVISION)"
