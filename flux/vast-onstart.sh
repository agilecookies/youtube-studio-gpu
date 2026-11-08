#!/usr/bin/env bash
# Vast.ai onstart script for the FLUX reference-image service.
# The Vast bootstrap clones this repo to /workspace/gpu before calling this script.
set -euo pipefail

REPO_DIR="/workspace/gpu"
LOG_FILE="/workspace/flux-service.log"
HF_HOME="${HF_HOME:-/workspace/flux/cache/huggingface}"
FLUX_MODEL_DIR="${FLUX_MODEL_DIR:-/workspace/flux/models}"
PORT="${PORT:-8080}"
REQS_HASH_FILE="/workspace/.flux-reqs-hash"

export HF_HOME

# ── HF_TOKEN diagnostic + alias ───────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[BOOT] HF_TOKEN present: yes (length ${#HF_TOKEN})"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[BOOT] HF_TOKEN present: no"
fi

echo "[STEP 1/4] Repo cloned to ${REPO_DIR} — starting bootstrap"

# ── Install dependencies (skip if unchanged) ──────────────────────────────────

REQS="${REPO_DIR}/flux/requirements.txt"
REQS_HASH="$(sha256sum "${REQS}" | awk '{print $1}')"
CACHED_HASH="$(cat "${REQS_HASH_FILE}" 2>/dev/null || echo '')"
if [ "${REQS_HASH}" != "${CACHED_HASH}" ]; then
  echo "[STEP 2/4] Installing FLUX dependencies (uv)..."
  STEP2_START=$(date +%s)
  pip install --quiet uv
  uv pip install --system --quiet \
    -r "${REQS}" \
    -c "${REPO_DIR}/flux/constraints.txt"
  echo "${REQS_HASH}" > "${REQS_HASH_FILE}"
  echo "[STEP 2/4] pip done in $(( $(date +%s) - STEP2_START ))s"
else
  echo "[STEP 2/4] Dependencies unchanged — skipping."
fi

# ── Download model weights if missing ────────────────────────────────────────

if [ ! -d "${FLUX_MODEL_DIR}" ] || [ -z "$(ls -A "${FLUX_MODEL_DIR}" 2>/dev/null)" ]; then
  echo "[STEP 3/4] Downloading FLUX.2-klein-4B weights to ${FLUX_MODEL_DIR}..."
  python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    local_dir=os.environ["FLUX_MODEL_DIR"],
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=["*.md", "*.txt", "*.bin"],
)
print("[STEP 3/4] Model weights downloaded.")
PYEOF
else
  echo "[STEP 3/4] Model weights already present at ${FLUX_MODEL_DIR}."
fi

# ── Start service ─────────────────────────────────────────────────────────────

cd "${REPO_DIR}/flux"
echo "[STEP 4/4] Starting FLUX service on port ${PORT}..."
exec python3 app.py
