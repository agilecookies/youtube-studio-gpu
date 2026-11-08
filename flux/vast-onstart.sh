#!/usr/bin/env bash
# Vast.ai onstart script for the FLUX reference-image service.
# The Vast bootstrap clones this repo to /workspace/gpu before calling this script.
set -euo pipefail

REPO_DIR="/workspace/gpu"
HF_HOME="${HF_HOME:-/workspace/flux/cache/huggingface}"
FLUX_MODEL_DIR="${FLUX_MODEL_DIR:-/workspace/flux/models}"
PORT="${PORT:-8080}"
REQS_HASH_FILE="/workspace/.flux-reqs-hash"

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
  echo "[BOOT] HF_TOKEN present: no"
  exit 1
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

# ── Print hf_xet version ──────────────────────────────────────────────────────
python3 -c "
import importlib.metadata as m
try:
    print('[BOOT] hf_xet ' + m.version('hf_xet'))
except Exception:
    print('[BOOT] hf_xet not installed')
"

# ── Download model weights ────────────────────────────────────────────────────
echo "[STEP 3/4] Checking FLUX.2-klein-4B weights..."
STEP3_START=$(date +%s)
python3 - <<'PYEOF'
import fnmatch, os, sys, threading, time
from pathlib import Path

local_dir = os.environ["FLUX_MODEL_DIR"]
token     = os.environ["HF_TOKEN"]
MODEL_ID  = "black-forest-labs/FLUX.2-klein-4B"

# Fast exit on restart when weights already present.
if Path(local_dir, "model_index.json").exists():
    print("[STEP 3/4] Model weights already present — skipping download.", flush=True)
    sys.exit(0)

from huggingface_hub import HfApi, login, snapshot_download

login(token=token, add_to_git_credential=False)
api = HfApi()

# ── List repo files ───────────────────────────────────────────────────────────
print(f"[STEP 3/4] Fetching file list for {MODEL_ID}...", flush=True)
info = api.model_info(MODEL_ID, files_metadata=True, token=token)
all_files = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
total_remote_gb = sum(all_files.values()) / 1e9
print(f"[STEP 3/4] Repo: {len(all_files)} files, {total_remote_gb:.2f} GB total", flush=True)
for fname, size in sorted(all_files.items(), key=lambda x: -x[1])[:15]:
    print(f"  {size/1e6:>8.1f} MB  {fname}", flush=True)

# ── Ignore patterns ───────────────────────────────────────────────────────────
# Diffusers Flux2KleinPipeline loads: model_index.json, transformer/,
# text_encoder/, text_encoder_2/, vae/, scheduler/, tokenizer/, tokenizer_2/
# Skip: GGUF quantized variants, .ckpt single-file checkpoints,
#        .bin PyTorch checkpoints (safetensors preferred), docs.
IGNORE = ["*.gguf", "*.ckpt", "*.bin", "*.md"]
# Skip fp32 variants when bf16 equivalents exist in the same repo.
if any("bf16" in f or "bfloat16" in f for f in all_files):
    IGNORE += ["*fp32*", "*float32*"]

to_download = {f: s for f, s in all_files.items()
               if not any(fnmatch.fnmatch(f, p) for p in IGNORE)}
dl_gb = sum(to_download.values()) / 1e9
print(f"[STEP 3/4] Will download: {len(to_download)} files, {dl_gb:.2f} GB", flush=True)
for fname, size in sorted(to_download.items(), key=lambda x: -x[1])[:10]:
    print(f"  {size/1e6:>8.1f} MB  {fname}", flush=True)


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
    SLOW_LIMIT = 120    # seconds below threshold before fail

    def __init__(self, local, expected_gb):
        super().__init__()
        self.local       = local
        self.expected_gb = expected_gb
        self._stop       = threading.Event()
        self._prev       = 0
        self._slow_since = None

    def run(self):
        while not self._stop.wait(self.INTERVAL):
            cur   = _dir_bytes(self.local)
            speed = (cur - self._prev) / (self.INTERVAL * 1e6)
            fname = _newest_file(self.local)
            print(
                f"[STEP 3/4] {cur/1e9:.2f} GB / {self.expected_gb:.2f} GB"
                f"  {speed:.1f} MB/s  current: {fname}",
                flush=True,
            )
            self._prev = cur
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
        MODEL_ID,
        local_dir=local_dir,
        token=token,
        ignore_patterns=IGNORE,
    )
finally:
    monitor.stop()
    monitor.join(timeout=5)

elapsed   = time.time() - t0
final_gb  = _dir_bytes(local_path) / 1e9
print(f"[STEP 3/4] download done in {elapsed:.0f}s, {final_gb:.2f} GB", flush=True)
PYEOF
echo "[STEP 3/4] done in $(( $(date +%s) - STEP3_START ))s"

# ── Start service ─────────────────────────────────────────────────────────────
cd "${REPO_DIR}/flux"
echo "[STEP 4/4] Starting FLUX service on port ${PORT}..."
exec python3 app.py
