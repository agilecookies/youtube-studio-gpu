#!/usr/bin/env bash
# Vast.ai onstart script for the LTX-2.5 video generation service.
# The Vast bootstrap clones this repo to /workspace/gpu before calling this script.
set -euo pipefail

REPO_DIR="/workspace/gpu"
LOG_FILE="/workspace/ltx-service.log"
LOG_DIR="/workspace/ltx/logs"
MODEL_DIR="${LTX_MODEL_DIR:-/workspace/ltx/models/ltx-2.5}"
HF_HOME="${HF_HOME:-/workspace/ltx/cache/huggingface}"
OUTPUT_DIR="${LTX_OUTPUT_DIR:-/workspace/ltx/outputs}"
PORT="${PORT:-8080}"
REQS_HASH_FILE="/workspace/.ltx-reqs-hash"

export HF_HOME
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
mkdir -p "${MODEL_DIR}/diffusion_models" "${MODEL_DIR}/text_encoders" "${MODEL_DIR}/vae"
mkdir -p "${MODEL_DIR}/model_patches" "${MODEL_DIR}/latent_upscale_models"

# ── Install dependencies (skip if unchanged) ──────────────────────────────────

REQS="${REPO_DIR}/ltx/requirements.txt"
REQS_HASH="$(sha256sum "${REQS}" | awk '{print $1}')"
CACHED_HASH="$(cat "${REQS_HASH_FILE}" 2>/dev/null || echo '')"
if [ "${REQS_HASH}" != "${CACHED_HASH}" ]; then
  echo "[onstart] Installing LTX dependencies..."
  # Install ltx_pipelines from Lightricks (commit pinned in model.py)
  pip install --quiet "git+https://github.com/Lightricks/LTX-Video.git@a95ab856" 2>&1 | tail -3
  pip install --quiet -r "${REQS}"
  # tesseract binary for OCR QC
  apt-get install -y -qq tesseract-ocr 2>/dev/null || true
  echo "${REQS_HASH}" > "${REQS_HASH_FILE}"
  echo "[onstart] Dependencies installed."
else
  echo "[onstart] Dependencies unchanged — skipping pip install."
fi

# ── Download model weights if missing ────────────────────────────────────────

REQUIRED_FILES=(
  "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
  "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  "vae/ltx-2.5-video-vae-bf16.safetensors"
  "vae/ltx-2.5-audio-vae-bf16.safetensors"
  "model_patches/ltx-2.5-duration-head-bf16.safetensors"
  "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)

HF_REPO="https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
for f in "${REQUIRED_FILES[@]}"; do
  dest="${MODEL_DIR}/${f}"
  if [[ -f "$dest" ]]; then
    echo "[onstart] Already present: ${f}"
    continue
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[onstart] Downloading: ${f}"
  curl -L \
    --fail-with-body \
    --retry 3 \
    --retry-delay 10 \
    --connect-timeout 30 \
    --show-error \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -o "${dest}.tmp" \
    "${HF_REPO}/${f}" 2>&1 | tee -a "${LOG_DIR}/startup.log"
  mv "${dest}.tmp" "${dest}"
  sync
  echo "[onstart] -> ${f} ($(du -sh "${dest}" | cut -f1))"
done

# ── Start service ─────────────────────────────────────────────────────────────

cd "${REPO_DIR}/ltx"
echo "[onstart] Starting LTX service on port ${PORT}..."
nohup python3 app.py >> "${LOG_FILE}" 2>&1 &
echo $! > /workspace/.ltx-service.pid
echo "[onstart] LTX service started (PID $(cat /workspace/.ltx-service.pid)). Logs: ${LOG_FILE}"
