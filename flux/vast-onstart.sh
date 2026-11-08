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

# ── Install dependencies (skip if unchanged) ──────────────────────────────────

REQS="${REPO_DIR}/flux/requirements.txt"
REQS_HASH="$(sha256sum "${REQS}" | awk '{print $1}')"
CACHED_HASH="$(cat "${REQS_HASH_FILE}" 2>/dev/null || echo '')"
if [ "${REQS_HASH}" != "${CACHED_HASH}" ]; then
  echo "[onstart] Installing FLUX dependencies..."
  pip install --quiet -r "${REQS}"
  echo "${REQS_HASH}" > "${REQS_HASH_FILE}"
  echo "[onstart] Dependencies installed."
else
  echo "[onstart] Dependencies unchanged — skipping pip install."
fi

# ── Download model weights if missing ────────────────────────────────────────

if [ ! -d "${FLUX_MODEL_DIR}" ] || [ -z "$(ls -A "${FLUX_MODEL_DIR}" 2>/dev/null)" ]; then
  echo "[onstart] Downloading FLUX.2-klein-4B weights to ${FLUX_MODEL_DIR}..."
  python3 - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        "black-forest-labs/FLUX.2-klein-4B",
        local_dir=os.environ["FLUX_MODEL_DIR"],
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.md", "*.txt", "*.bin"],
    )
    print("[onstart] Model weights downloaded.")
except Exception as e:
    print(f"[onstart] WARNING: weight download failed ({e}). Service will download on first load.")
PYEOF
else
  echo "[onstart] Model weights already present at ${FLUX_MODEL_DIR}."
fi

# ── Start service ─────────────────────────────────────────────────────────────

cd "${REPO_DIR}/flux"
echo "[onstart] Starting FLUX service on port ${PORT}..."
nohup python3 app.py >> "${LOG_FILE}" 2>&1 &
echo $! > /workspace/.flux-service.pid
echo "[onstart] FLUX service started (PID $(cat /workspace/.flux-service.pid)). Logs: ${LOG_FILE}"
