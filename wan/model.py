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
    """Run generate.py. Returns absolute path to the output MP4."""
    if not _model_ready:
        raise RuntimeError("Model not ready — call load_model() first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"

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
        "--seed",             str(seed),
    ]
    if image_path:
        cmd += ["--image", image_path]

    logger.info(f"[wan.model] Running: {' '.join(cmd[:6])} ...")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(WAN_REPO_DIR),
        capture_output=False,
        text=True,
        timeout=3600,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"generate.py exited {result.returncode} after {elapsed:.0f}s")

    # generate.py may save to --save_file directly, or fall back to its default
    # output location (samples/<timestamp>.mp4). Check both.
    if out_path.exists():
        logger.info(f"[wan.model] Done in {elapsed:.1f}s → {out_path}")
        return str(out_path)

    # Fallback: find the newest .mp4 in the repo's default output dir
    fallback_dir = WAN_REPO_DIR / "samples"
    if fallback_dir.exists():
        candidates = sorted(fallback_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            # Move to our output dir for consistent serving
            dest = OUTPUT_DIR / out_path.name
            candidates[0].rename(dest)
            logger.info(f"[wan.model] Done in {elapsed:.1f}s → {dest} (from fallback)")
            return str(dest)

    raise RuntimeError(f"generate.py completed but output not found at {out_path}")
