#!/usr/bin/env bash
# Vast.ai onstart for Wan2.2-S2V service.
# Our repo (youtube-studio-gpu) is at /workspace/gpu before this script runs.

_boot() {
  set -euo pipefail

  REPO_DIR="/workspace/gpu"
  WAN2_DIR="/workspace/wan2/Wan2.2"
  MODEL_DIR="${WAN_MODEL_DIR:-/workspace/wan/models/Wan2.2-S2V-14B}"
  OUTPUT_DIR="${WAN_OUTPUT_DIR:-/workspace/wan/outputs}"
  ASSETS_DIR="${WAN_ASSETS_DIR:-/workspace/wan/assets}"
  LOG_DIR="/workspace/wan/logs"
  PORT="${PORT:-8080}"
  REQS_HASH_FILE="/workspace/.wan-reqs-hash"

  mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${ASSETS_DIR}"

  # Self-update our repo
  if [ "${_WAN_SELF_UPDATED:-0}" = "0" ]; then
    git -C "${REPO_DIR}" fetch --depth=1 origin main 2>/dev/null \
      && git -C "${REPO_DIR}" reset --hard origin/main 2>/dev/null \
      && echo "[BOOT] updated to $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null)" \
      || echo "[BOOT] git pull skipped"
    export _WAN_SELF_UPDATED=1
    exec bash "${REPO_DIR}/wan/vast-onstart.sh"
  fi

  apt-get update -qq 2>/dev/null || true
  apt-get install -y -q gcc g++ ffmpeg git 2>/dev/null || true

  # ── Clone Wan2.2 official repo ────────────────────────────────────────────────
  echo "[STEP 1/4] Wan2.2 repo..."
  if [ ! -d "${WAN2_DIR}/.git" ]; then
    mkdir -p /workspace/wan2
    git clone --depth=1 https://github.com/Wan-Video/Wan2.2.git "${WAN2_DIR}"
    echo "[STEP 1/4] Cloned Wan2.2"
  else
    git -C "${WAN2_DIR}" pull --ff-only 2>/dev/null || echo "[STEP 1/4] Wan2.2 already up to date"
  fi

  # ── Install Python deps ───────────────────────────────────────────────────────
  echo "[STEP 2/4] Dependencies..."
  REQS="${WAN2_DIR}/requirements.txt"
  OUR_REQS="${REPO_DIR}/wan/requirements.txt"
  CACHE_KEY="$(sha256sum "${REQS}" "${OUR_REQS}" 2>/dev/null | sha256sum | awk '{print $1}')"
  CACHED_KEY="$(cat "${REQS_HASH_FILE}" 2>/dev/null || echo '')"
  if [ "${CACHE_KEY}" != "${CACHED_KEY}" ]; then
    pip install --quiet uv
    # Wan2.2 requirements first (includes flash-attn etc.)
    uv pip install --system --quiet -r "${REQS}" || \
      pip install --quiet -r "${REQS}"
    # Our service requirements (fastapi, uvicorn, etc.)
    uv pip install --system --quiet -r "${OUR_REQS}"
    echo "${CACHE_KEY}" > "${REQS_HASH_FILE}"
    echo "[STEP 2/4] done"
  else
    echo "[STEP 2/4] unchanged — skipping"
  fi

  # ── Download model weights ────────────────────────────────────────────────────
  echo "[STEP 3/4] Model weights..."
  if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
    mkdir -p "${MODEL_DIR}"
    echo "[STEP 3/4] Downloading Wan-AI/Wan2.2-S2V-14B (Apache 2.0, no token needed)..."
    pip install --quiet "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir "${MODEL_DIR}"
    echo "[STEP 3/4] Download complete"
  else
    echo "[STEP 3/4] Weights already present at ${MODEL_DIR}"
  fi

  # ── Start service ─────────────────────────────────────────────────────────────
  export WAN_REPO_DIR="${WAN2_DIR}"
  export WAN_MODEL_DIR="${MODEL_DIR}"
  export WAN_OUTPUT_DIR="${OUTPUT_DIR}"
  export WAN_ASSETS_DIR="${ASSETS_DIR}"

  cd "${REPO_DIR}/wan"
  echo "[STEP 4/4] Starting Wan2.2-S2V service on port ${PORT}..."
  exec python3 app.py
}

if ! _boot; then
  echo "[BOOT] FAILED — idling container"
  sleep infinity
fi
