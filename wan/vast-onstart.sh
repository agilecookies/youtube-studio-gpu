#!/usr/bin/env bash
# Vast.ai onstart script for Wan2.2-S2V service.
# Cloned repo is at /workspace/gpu before this script runs.

_boot() {
  set -euo pipefail

  REPO_DIR="/workspace/gpu"
  HF_HOME="${HF_HOME:-/workspace/wan/cache/huggingface}"
  OUTPUT_DIR="${WAN_OUTPUT_DIR:-/workspace/wan/outputs}"
  ASSETS_DIR="${WAN_ASSETS_DIR:-/workspace/wan/assets}"
  LOG_DIR="/workspace/wan/logs"
  PORT="${PORT:-8080}"
  MODEL_ID="${WAN_S2V_MODEL_ID:-Wan-AI/Wan2.2-S2V-14B}"
  REQS_HASH_FILE="/workspace/.wan-reqs-hash"

  export HF_HOME
  export HF_XET_HIGH_PERFORMANCE=1
  export HF_HUB_DISABLE_XET=0

  # Wan2.2-S2V-14B is Apache 2.0 — no HF_TOKEN required.
  if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  fi

  mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${ASSETS_DIR}"

  # Self-update
  if [ "${_WAN_SELF_UPDATED:-0}" = "0" ]; then
    git -C "${REPO_DIR}" fetch --depth=1 origin main 2>/dev/null \
      && git -C "${REPO_DIR}" reset --hard origin/main 2>/dev/null \
      && echo "[BOOT] updated to $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null)" \
      || echo "[BOOT] git pull skipped"
    export _WAN_SELF_UPDATED=1
    exec bash "${REPO_DIR}/wan/vast-onstart.sh"
  fi

  apt-get update -qq 2>/dev/null || true
  apt-get install -y -q gcc g++ ffmpeg 2>/dev/null || true

  # ── Install Python deps ───────────────────────────────────────────────────────
  REQS="${REPO_DIR}/wan/requirements.txt"
  CACHE_KEY="$(sha256sum "${REQS}" | awk '{print $1}')"
  CACHED_KEY="$(cat "${REQS_HASH_FILE}" 2>/dev/null || echo '')"
  if [ "${CACHE_KEY}" != "${CACHED_KEY}" ]; then
    echo "[STEP 2/4] Installing Wan dependencies..."
    pip install --quiet uv
    uv pip install --system --quiet -r "${REQS}"
    echo "${CACHE_KEY}" > "${REQS_HASH_FILE}"
    echo "[STEP 2/4] done"
  else
    echo "[STEP 2/4] Dependencies unchanged — skipping"
  fi

  # ── Download model weights ────────────────────────────────────────────────────
  MODEL_DIR="${HF_HOME}/hub/models--${MODEL_ID/\//__}"
  echo "[STEP 3/4] Checking model weights for ${MODEL_ID}..."
  python3 - <<PYEOF
import os, sys, time
from pathlib import Path

model_id = os.environ.get("WAN_S2V_MODEL_ID", "Wan-AI/Wan2.2-S2V-14B")
hf_home  = os.environ.get("HF_HOME", "/workspace/wan/cache/huggingface")
token    = os.environ.get("HF_TOKEN", "")

try:
    from huggingface_hub import snapshot_download, login
except ImportError:
    print("[STEP 3/4] huggingface_hub not installed — skipping download", flush=True)
    sys.exit(0)

try:
    if token:
        login(token=token, add_to_git_credential=False)
    print(f"[STEP 3/4] Downloading {model_id} ...", flush=True)
    t0 = time.time()
    local = snapshot_download(
        model_id,
        cache_dir=hf_home,
        token=token or None,
        ignore_patterns=["*.md", "*.txt", "*.bin"],  # prefer .safetensors
    )
    print(f"[STEP 3/4] Done in {time.time()-t0:.0f}s → {local}", flush=True)
except Exception as e:
    print(f"[BOOT] WARNING: model download failed: {e}", flush=True)
    print("[BOOT] Service will start but /generate will fail until model is accessible", flush=True)
PYEOF
  echo "[STEP 3/4] done"

  # ── Start service ─────────────────────────────────────────────────────────────
  cd "${REPO_DIR}/wan"
  echo "[STEP 4/4] Starting Wan2.2-S2V service on port ${PORT}..."
  exec python3 app.py
}

if ! _boot; then
  echo "[BOOT] FAILED — idling container to prevent restart loop"
  sleep infinity
fi
