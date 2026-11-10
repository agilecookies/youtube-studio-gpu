#!/usr/bin/env bash
# Vast.ai onstart script for the FLUX reference-image service.
# The Vast bootstrap clones this repo to /workspace/gpu before calling this script.
#
# If boot fails for any reason, prints [BOOT] FAILED and idles the container
# (sleep infinity) so the orchestrator can read the logs and pause the run cleanly.

_boot() {
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
  echo "[BOOT] FAILED: HF_TOKEN not set — add HF_TOKEN to container env"
  return 1
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
import fnmatch, json, os, sys, threading, time
from pathlib import Path

# Disable XET for bulk download — XET cold-start is 0.0-0.4 MB/s and triggers
# our slow-host kill before it can ramp up. Standard downloader or hf_transfer
# achieves 20-167 MB/s on the same hardware.
os.environ["HF_HUB_DISABLE_XET"] = "1"
try:
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("[STEP 3/4] fast backend: hf_transfer", flush=True)
except ImportError:
    print("[STEP 3/4] fast backend: standard", flush=True)

local_dir  = os.environ["FLUX_MODEL_DIR"]
token      = os.environ["HF_TOKEN"]
MODEL_ID   = "black-forest-labs/FLUX.2-klein-4B"
local_path = Path(local_dir)

# ── Shard integrity check ─────────────────────────────────────────────────────
def check_shard_integrity(base: Path) -> list:
    """
    For every *.safetensors.index.json under base, verify every shard listed in
    weight_map exists on disk with size > 0. Returns missing/empty shard paths.
    Partial downloads leave model_index.json present but shards absent; this
    catches that state so the fast-exit path does not silently skip re-download.
    """
    missing = []
    for idx_file in base.rglob("*.safetensors.index.json"):
        try:
            idx = json.loads(idx_file.read_text())
            for shard in sorted(set(idx.get("weight_map", {}).values())):
                p = idx_file.parent / shard
                if not p.exists():
                    missing.append(str(p.relative_to(base)))
                elif p.stat().st_size == 0:
                    missing.append(f"{p.relative_to(base)} (zero bytes)")
        except Exception as e:
            missing.append(f"{idx_file.relative_to(base)} (unreadable: {e})")
    return missing

# ── Build allow_patterns from model_index.json ────────────────────────────────
def build_allow_patterns(model_index_path: Path) -> list:
    """
    Parse model_index.json to discover component dirs (transformer, text_encoder,
    text_encoder_2, vae, scheduler, tokenizer, tokenizer_2, …) and return
    allow_patterns that cover the complete set of files each component needs.

    For every component folder we use 'c/*' (no extension filter) so we always
    download *.jinja, *.tiktoken, *.txt, *.json, *.safetensors, etc.  Missing
    tokenizer files (e.g. chat_template.jinja for Qwen3-based models) caused
    'tokenizer.chat_template is not set' on every /generate call.

    Root-level '*.safetensors' is kept for the FLUX.2-klein-4B transformer
    checkpoint, which lives at the repo root rather than inside transformer/.
    Files outside the components that are NOT root-level safetensors are skipped.
    """
    idx   = json.loads(model_index_path.read_text())
    comps = [k for k in idx if not k.startswith("_")]
    # Root: model index + transformer checkpoint (repo-root safetensors for FLUX.2-klein-4B)
    patterns = ["model_index.json", "*.safetensors"]
    # Every component folder: all files, no extension filtering
    for c in comps:
        patterns.append(f"{c}/*")
    return patterns

from huggingface_hub import HfApi, login, snapshot_download

login(token=token, add_to_git_credential=False)
local_path.mkdir(parents=True, exist_ok=True)
model_index_path = local_path / "model_index.json"

# ── Fast exit on restart: skip only when all shards verified present ──────────
# model_index.json existing is NOT sufficient — a partial download leaves it in
# place while component shards are still missing. check_shard_integrity() reads
# every *.safetensors.index.json and confirms each listed shard is on disk.
if model_index_path.exists():
    missing = check_shard_integrity(local_path)
    if not missing:
        print("[STEP 3/4] Model weights already present and verified — skipping download.", flush=True)
        _t_dir  = local_path / "transformer"
        _target = _t_dir / "diffusion_pytorch_model.safetensors"
        if _t_dir.exists() and not _target.exists() and not _target.is_symlink():
            _root_sf = sorted(local_path.glob("*.safetensors"), key=lambda f: f.stat().st_size, reverse=True)
            if _root_sf:
                _target.symlink_to(f"../{_root_sf[0].name}")
                print(f"[STEP 3/4] linked transformer (restart): {_root_sf[0].name}", flush=True)
        sys.exit(0)
    else:
        print(f"[STEP 3/4] model_index.json present but {len(missing)} shard(s) missing — will re-download:", flush=True)
        for f in missing[:10]:
            print(f"  missing: {f}", flush=True)

# ── Determine allow_patterns ──────────────────────────────────────────────────
# If model_index.json is absent (first boot), download it alone first so we can
# read the component list before choosing allow_patterns for the main download.
if not model_index_path.exists():
    print("[STEP 3/4] Fetching model_index.json to determine components...", flush=True)
    snapshot_download(MODEL_ID, local_dir=local_dir, token=token,
                      allow_patterns=["model_index.json"])

allow_patterns = build_allow_patterns(model_index_path)
print(f"[STEP 3/4] allow_patterns ({len(allow_patterns)}): {allow_patterns}", flush=True)

# ── List repo files for progress monitor size estimate ────────────────────────
print(f"[STEP 3/4] Fetching file list for {MODEL_ID}...", flush=True)
api  = HfApi()
info = api.model_info(MODEL_ID, files_metadata=True, token=token)
all_files = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
dl_gb = sum(s for f, s in all_files.items()
            if any(fnmatch.fnmatch(f, p) for p in allow_patterns)) / 1e9
print(f"[STEP 3/4] Repo: {len(all_files)} files; will download up to {dl_gb:.2f} GB", flush=True)
for fname, size in sorted(all_files.items(), key=lambda x: -x[1])[:10]:
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
    GRACE      = 300    # seconds before slow-host check activates (hf_transfer ramp-up)
    SLOW_LIMIT = 300    # seconds below threshold after grace before fail

    def __init__(self, local, expected_gb):
        super().__init__()
        self.local         = local
        self.expected_gb   = expected_gb
        self._done_event   = threading.Event()   # renamed: Thread._stop is an internal method
        self._prev         = 0
        self._slow_since   = None
        self._start_time   = time.time()

    def run(self):
        while not self._done_event.wait(self.INTERVAL):
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
            # Skip slow-host check during GRACE period to allow download ramp-up.
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
        self._done_event.set()


monitor = _Monitor(local_path, dl_gb)
monitor.start()
t0 = time.time()

try:
    snapshot_download(
        MODEL_ID,
        local_dir=local_dir,
        token=token,
        allow_patterns=allow_patterns,
    )
finally:
    monitor.stop()
    monitor.join(timeout=5)

elapsed   = time.time() - t0
final_gb  = _dir_bytes(local_path) / 1e9
print(f"[STEP 3/4] download done in {elapsed:.0f}s, {final_gb:.2f} GB", flush=True)

# ── Post-download shard verification (retry once if anything missing) ─────────
for attempt in range(1, 3):
    missing = check_shard_integrity(local_path)
    if not missing:
        print("[STEP 3/4] All shard files verified present.", flush=True)
        break
    if attempt == 1:
        print(f"[STEP 3/4] {len(missing)} shard(s) still missing after download — retrying:", flush=True)
        for f in missing[:10]:
            print(f"  missing: {f}", flush=True)
        snapshot_download(MODEL_ID, local_dir=local_dir, token=token,
                          allow_patterns=allow_patterns)
    else:
        for f in missing:
            print(f"[BOOT] FAILED: missing {f}", flush=True)
        sys.exit(1)

# ── Transformer symlink ───────────────────────────────────────────────────────
# FLUX.2-klein-4B stores the transformer at the repo root as flux-2-klein-4b.safetensors.
# Diffusers expects transformer/diffusion_pytorch_model.safetensors.  Symlink it now so
# app.py's from_pretrained() can find it without touching the downloaded files.
_t_dir  = local_path / "transformer"
_target = _t_dir / "diffusion_pytorch_model.safetensors"
if _t_dir.exists() and not _target.exists() and not _target.is_symlink():
    _root_sf = sorted(local_path.glob("*.safetensors"), key=lambda f: f.stat().st_size, reverse=True)
    if _root_sf:
        _target.symlink_to(f"../{_root_sf[0].name}")
        print(f"[STEP 3/4] linked transformer: {_root_sf[0].name} -> transformer/diffusion_pytorch_model.safetensors", flush=True)
PYEOF
echo "[STEP 3/4] done in $(( $(date +%s) - STEP3_START ))s"

# ── Start service ─────────────────────────────────────────────────────────────
  cd "${REPO_DIR}/flux"
  echo "[STEP 4/4] Starting FLUX service on port ${PORT}..."
  exec python3 app.py
}

if ! _boot; then
  echo "[BOOT] FAILED: boot script exited with non-zero status — see output above. Container will idle to prevent restart loop."
  sleep infinity
fi
