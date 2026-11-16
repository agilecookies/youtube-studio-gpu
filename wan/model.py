"""
Wan2.2-S2V model — calls generate.py from the official Wan2.2 GitHub repo.

Diffusers integration is not yet released (unchecked in the todo list on the
HF model card as of 2026-11). We use the official generate.py CLI directly.

Repo:   https://github.com/Wan-Video/Wan2.2
Model:  Wan-AI/Wan2.2-S2V-14B  (Apache 2.0, no HF_TOKEN needed)
VRAM:   80 GB (single A100 with --offload_model True --convert_model_dtype)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("wan.model")

WAN_REPO_DIR  = Path(os.environ.get("WAN_REPO_DIR",   "/workspace/wan2/Wan2.2"))
MODEL_DIR     = Path(os.environ.get("WAN_MODEL_DIR",  "/workspace/wan/models/Wan2.2-S2V-14B"))
OUTPUT_DIR    = Path(os.environ.get("WAN_OUTPUT_DIR", "/workspace/wan/outputs"))

_model_ready  = False
_load_error: Optional[str] = None


def is_loaded() -> bool:
    return _model_ready


def load_model() -> None:
    """Warm-check: verify the checkpoint dir exists and generate.py is importable."""
    global _model_ready, _load_error
    if _model_ready:
        return
    gen_script = WAN_REPO_DIR / "generate.py"
    if not gen_script.exists():
        _load_error = f"generate.py not found at {gen_script}"
        raise RuntimeError(_load_error)
    if not MODEL_DIR.exists():
        _load_error = f"Model weights not found at {MODEL_DIR}"
        raise RuntimeError(_load_error)
    _model_ready = True
    logger.info(f"[wan.model] Ready — repo={WAN_REPO_DIR} ckpt={MODEL_DIR}")


def generate(
    audio_path: str,
    prompt: str,
    image_path: Optional[str] = None,
    seed: int = 42,
    fps: int = 24,
    size: str = "1024*704",
) -> str:
    """Run generate.py. Returns absolute path to the output MP4.

    We attempt two save strategies and log each outcome so we know which works:
      1. --save_file <explicit_path>  (flag may not exist in older generate.py versions)
      2. samples/<timestamp>.mp4      (generate.py's default output location)
    """
    if not _model_ready:
        raise RuntimeError("Model not ready — call load_model() first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = OUTPUT_DIR / out_name

    cmd = [
        sys.executable, str(WAN_REPO_DIR / "generate.py"),
        "--task",             "s2v-14B",
        "--size",             size,
        "--ckpt_dir",         str(MODEL_DIR),
        "--offload_model",    "True",
        "--convert_model_dtype",
        "--prompt",           prompt,
        "--audio",            audio_path,
        "--save_file",        str(out_path),
        "--base_seed",        str(seed),
    ]
    if image_path:
        cmd += ["--image", image_path]

    logger.info(f"[wan.model] CMD: {' '.join(cmd)}")
    logger.info(f"[wan.model] Expected --save_file path: {out_path}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(WAN_REPO_DIR),
        capture_output=False,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - t0

    logger.info(f"[wan.model] generate.py exited {result.returncode} after {elapsed:.1f}s")

    if result.returncode != 0:
        raise RuntimeError(f"generate.py exited {result.returncode} after {elapsed:.0f}s")

    # ── Strategy 1: --save_file path ─────────────────────────────────────────
    if out_path.exists():
        sz = out_path.stat().st_size
        logger.info(f"[wan.model] STRATEGY 1 SUCCESS: --save_file worked → {out_path} ({sz} bytes)")
        return str(out_path)
    else:
        logger.warning(f"[wan.model] STRATEGY 1 MISS: {out_path} not found — --save_file flag may be unrecognised")

    # ── Strategy 2: samples/ default output dir ───────────────────────────────
    fallback_dir = WAN_REPO_DIR / "samples"
    logger.info(f"[wan.model] STRATEGY 2: checking {fallback_dir} for newest .mp4")
    if fallback_dir.exists():
        candidates = sorted(fallback_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            newest = candidates[0]
            # Only accept files modified within the last hour (this run)
            age = time.time() - newest.stat().st_mtime
            if age < 3600:
                dest = OUTPUT_DIR / out_name
                newest.rename(dest)
                sz = dest.stat().st_size
                logger.info(f"[wan.model] STRATEGY 2 SUCCESS: moved {newest.name} → {dest} ({sz} bytes, age={age:.0f}s)")
                return str(dest)
            else:
                logger.warning(f"[wan.model] STRATEGY 2 MISS: newest .mp4 in samples/ is {age:.0f}s old — skipping stale file")
        else:
            logger.warning(f"[wan.model] STRATEGY 2 MISS: {fallback_dir} exists but contains no .mp4 files")
    else:
        logger.warning(f"[wan.model] STRATEGY 2 MISS: {fallback_dir} does not exist")

    raise RuntimeError(
        f"generate.py completed (exit 0) but output not found.\n"
        f"  strategy-1 path: {out_path}\n"
        f"  strategy-2 dir:  {fallback_dir}"
    )
