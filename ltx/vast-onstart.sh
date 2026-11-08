#!/usr/bin/env bash
# Vast.ai onstart script for the LTX-2.5 video generation service.
# The Vast bootstrap clones this repo to /workspace/gpu before calling this script.
set -euo pipefail

REPO_DIR="/workspace/gpu"
MODEL_DIR="${LTX_MODEL_DIR:-/workspace/ltx/models/ltx-2.5}"
HF_HOME="${HF_HOME:-/workspace/ltx/cache/huggingface}"
OUTPUT_DIR="${LTX_OUTPUT_DIR:-/workspace/ltx/outputs}"
LOG_DIR="/workspace/ltx/logs"
PORT="${PORT:-8080}"
REQS_HASH_FILE="/workspace/.ltx-reqs-hash"

export HF_HOME
# XET fast-download: HIGH_PERFORMANCE enables multi-connection XET protocol;
# DISABLE_XET=0 keeps it enabled (double-negative is the official HF flag name).
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_XET=0

# ── HF_TOKEN diagnostic ───────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[BOOT] HF_TOKEN present: yes (length ${#HF_TOKEN})"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[BOOT] HF_TOKEN present: no — LTX-2.5 is gated; model downloads will fail"
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[STEP 1/4] Repo cloned to ${REPO_DIR} — starting bootstrap"

# ── Install dependencies (skip if unchanged) ──────────────────────────────────
REQS="${REPO_DIR}/ltx/requirements.txt"
REQS_HASH="$(sha256sum "${REQS}" | awk '{print $1}')"
CACHED_HASH="$(cat "${REQS_HASH_FILE}" 2>/dev/null || echo '')"
if [ "${REQS_HASH}" != "${CACHED_HASH}" ]; then
  echo "[STEP 2/4] Installing LTX dependencies (uv)..."
  STEP2_START=$(date +%s)
  pip install --quiet uv
  uv pip install --system --quiet \
    -r "${REQS}" \
    -c "${REPO_DIR}/ltx/constraints.txt"
  apt-get install -y -qq tesseract-ocr 2>/dev/null || true
  echo "${REQS_HASH}" > "${REQS_HASH_FILE}"
  echo "[STEP 2/4] pip done in $(( $(date +%s) - STEP2_START ))s"
else
  echo "[STEP 2/4] Dependencies unchanged — skipping."
fi

# ── Print hf_xet version ──────────────────────────────────────────────────────
python3 -c "
import importlib.metadata as m
try:
    print('[BOOT] hf_xet ' + m.version('hf_xet'))
except Exception:
    print('[BOOT] hf_xet not installed')
"

# ── Download model weights ────────────────────────────────────────────────────
echo "[STEP 3/4] Checking LTX-2.5 model weights..."
STEP3_START=$(date +%s)
python3 - <<'PYEOF'
import os, sys, threading, time
from pathlib import Path

# Disable XET for bulk download — same reason as FLUX: XET cold-start is slow.
os.environ["HF_HUB_DISABLE_XET"] = "1"
try:
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("[STEP 3/4] fast backend: hf_transfer", flush=True)
except ImportError:
    print("[STEP 3/4] fast backend: standard", flush=True)

local_dir = os.environ["LTX_MODEL_DIR"]
token     = os.environ["HF_TOKEN"]
HF_REPO   = "Lightricks/LTX-2.5"

# Exact files DistilledPipeline loads (from model.py REQUIRED_FILES +
# spatial_upsampler_path). Only these are downloaded — skips everything else
# in the LTX-2.5 repo (fp16 checkpoints, GGUF, docs, configs, etc.).
REQUIRED = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]

missing = [f for f in REQUIRED if not Path(local_dir, f).exists()]
if not missing:
    print("[STEP 3/4] All model files already present — skipping download.", flush=True)
    sys.exit(0)

present = len(REQUIRED) - len(missing)
print(f"[STEP 3/4] {present}/{len(REQUIRED)} files present, need to download {len(missing)}:", flush=True)
for f in missing:
    print(f"  {f}", flush=True)

from huggingface_hub import HfApi, login, snapshot_download

login(token=token, add_to_git_credential=False)
api = HfApi()

# ── List repo files for sizes ─────────────────────────────────────────────────
print(f"[STEP 3/4] Fetching file list for {HF_REPO}...", flush=True)
info = api.model_info(HF_REPO, files_metadata=True, token=token)
all_remote = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
total_gb   = sum(all_remote.values()) / 1e9
print(f"[STEP 3/4] Repo: {len(all_remote)} files, {total_gb:.2f} GB total", flush=True)
for fname, size in sorted(all_remote.items(), key=lambda x: -x[1])[:15]:
    print(f"  {size/1e9:>6.2f} GB  {fname}", flush=True)

dl_gb = sum(all_remote.get(f, 0) for f in missing) / 1e9
print(f"[STEP 3/4] Will download: {len(missing)} files, {dl_gb:.2f} GB", flush=True)
for f in missing:
    size = all_remote.get(f, 0)
    print(f"  {size/1e9:>6.2f} GB  {f}", flush=True)


# ── Progress monitor ──────────────────────────────────────────────────────────
def _dir_bytes(d):
    total = 0
    for f in d.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _newest_file(d):
    best, best_mt = None, 0.0
    for f in d.rglob("*"):
        if f.is_file():
            try:
                mt = f.stat().st_mtime
                if mt > best_mt:
                    best_mt, best = mt, f
            except OSError:
                pass
    return best.name if best else "—"


class _Monitor(threading.Thread):
    daemon = True
    INTERVAL   = 30     # seconds between samples
    SLOW_MBS   = 20.0   # MB/s threshold
    GRACE      = 300    # seconds before slow-host check activates
    SLOW_LIMIT = 300    # seconds below threshold after grace before fail

    def __init__(self, local, expected_gb):
        super().__init__()
        self.local       = local
        self.expected_gb = expected_gb
        self._stop       = threading.Event()
        self._prev       = 0
        self._slow_since = None
        self._start_time = time.time()

    def run(self):
        while not self._stop.wait(self.INTERVAL):
            cur     = _dir_bytes(self.local)
            speed   = (cur - self._prev) / (self.INTERVAL * 1e6)
            elapsed = time.time() - self._start_time
            fname   = _newest_file(self.local)
            print(
                f"[STEP 3/4] {cur/1e9:.2f} GB / {self.expected_gb:.2f} GB"
                f"  {speed:.1f} MB/s  current: {fname}",
                flush=True,
            )
            self._prev = cur
            if elapsed < self.GRACE:
                continue
            if speed < self.SLOW_MBS:
                if self._slow_since is None:
                    self._slow_since = time.time()
                elif time.time() - self._slow_since >= self.SLOW_LIMIT:
                    print(
                        f"[STEP 3/4] FAIL — slow host: {speed:.1f} MB/s < "
                        f"{self.SLOW_MBS} MB/s for {self.SLOW_LIMIT}s — "
                        f"terminate and pick a faster offer",
                        flush=True,
                    )
                    os._exit(1)
            else:
                self._slow_since = None

    def stop(self):
        self._stop.set()


local_path = Path(local_dir)
local_path.mkdir(parents=True, exist_ok=True)
monitor = _Monitor(local_path, dl_gb)
monitor.start()
t0 = time.time()

try:
    snapshot_download(
        HF_REPO,
        local_dir=local_dir,
        token=token,
        allow_patterns=missing,
    )
finally:
    monitor.stop()
    monitor.join(timeout=5)

elapsed  = time.time() - t0
final_gb = sum(all_remote.get(f, 0) for f in REQUIRED if Path(local_dir, f).exists()) / 1e9
print(f"[STEP 3/4] download done in {elapsed:.0f}s, {final_gb:.2f} GB", flush=True)
PYEOF
echo "[STEP 3/4] done in $(( $(date +%s) - STEP3_START ))s"

# ── Start service ─────────────────────────────────────────────────────────────
cd "${REPO_DIR}/ltx"
echo "[STEP 4/4] Starting LTX service on port ${PORT}..."
exec python3 app.py
