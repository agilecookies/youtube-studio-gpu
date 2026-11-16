"""
Wan2.2-S2V FastAPI service — Vast.ai deployment.

Endpoints:
  GET  /health              — fast liveness check (no model load)
  POST /stage-audio         — upload audio file, returns pod_path
  POST /stage-image         — upload first-frame image, returns pod_path
  POST /generate            — submit job { prompt, audio_path, image_path?, seed }
  GET  /job/{id}            — poll job status
  GET  /output/{filename}   — download generated video
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import model as wan_model
from schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatusResponse,
    StageFileResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("wan.app")

app = FastAPI(title="Wan2.2-S2V service", version="1.0.0")

SERVICE_START   = time.time()
SKIP_MODEL_LOAD = os.environ.get("SKIP_MODEL_LOAD", "0") == "1"
OUTPUT_DIR      = Path(os.environ.get("WAN_OUTPUT_DIR",  "/workspace/wan/outputs"))
ASSETS_DIR      = Path(os.environ.get("WAN_ASSETS_DIR",  "/workspace/wan/assets"))
PORT            = int(os.environ.get("PORT", "8080"))

_SAFE_ID_RE     = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$')
_MAX_FILE_BYTES = 100 * 1024 * 1024   # 100 MB

_current_job: Optional[dict] = None


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if SKIP_MODEL_LOAD:
        logger.info("[wan.app] SKIP_MODEL_LOAD=1 — skipping model check")
        return
    # load_model() is a fast check (no GPU loading) — verify repo + weights exist
    try:
        wan_model.load_model()
        logger.info("[wan.app] Model check passed — ready")
    except Exception as e:
        logger.warning(f"[wan.app] Model check failed (will retry on first /generate): {e}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import torch
    return HealthResponse(
        status="ok",
        provider="wan-s2v",
        gpu_available=torch.cuda.is_available(),
        model_loaded=wan_model.is_loaded(),
    )


# ── File staging ──────────────────────────────────────────────────────────────

def _safe_stage(file: UploadFile, prefix: str, allowed_exts: set[str]) -> StageFileResponse:
    ext = Path(file.filename or "upload").suffix.lower() or ".bin"
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported extension {ext!r}. Allowed: {allowed_exts}")
    asset_id = f"{prefix}-{uuid.uuid4().hex}{ext}"
    dest = ASSETS_DIR / asset_id
    data = file.file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(400, "File too large (max 100 MB)")
    dest.write_bytes(data)
    logger.info(f"[wan.app] Staged {prefix}: {dest} ({len(data)} bytes)")
    return StageFileResponse(pod_path=str(dest), size_bytes=len(data))


@app.post("/stage-audio", response_model=StageFileResponse)
async def stage_audio(file: UploadFile = File(...)) -> StageFileResponse:
    return _safe_stage(file, "audio", {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"})


@app.post("/stage-image", response_model=StageFileResponse)
async def stage_image(file: UploadFile = File(...)) -> StageFileResponse:
    return _safe_stage(file, "image", {".jpg", ".jpeg", ".png", ".webp"})


# ── Generation ────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    global _current_job
    if _current_job and _current_job["status"] == "running":
        raise HTTPException(409, "A job is already running")

    if not wan_model.is_loaded():
        raise HTTPException(503, "Model not loaded yet")

    if not Path(req.audio_path).exists():
        raise HTTPException(400, f"Audio file not found: {req.audio_path}")
    if req.image_path and not Path(req.image_path).exists():
        raise HTTPException(400, f"Image file not found: {req.image_path}")

    job_id = uuid.uuid4().hex
    _current_job = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "duration_seconds": None,
        "output_file": None,
        "error": None,
        "req": req.model_dump(),
    }

    asyncio.get_event_loop().run_in_executor(None, _run_job, job_id, req)
    return GenerateResponse(job_id=job_id, status="queued")


def _run_job(job_id: str, req: GenerateRequest) -> None:
    global _current_job
    t0 = time.time()
    try:
        out_path = wan_model.generate(
            audio_path=req.audio_path,
            prompt=req.prompt,
            image_path=req.image_path,
            seed=req.seed,
            fps=req.fps,
        )
        elapsed = time.time() - t0
        _current_job.update({
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": elapsed,
            "output_file": Path(out_path).name,
        })
        logger.info(f"[wan.app] Job {job_id} done in {elapsed:.1f}s → {out_path}")
    except Exception as e:
        elapsed = time.time() - t0
        _current_job.update({
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": elapsed,
            "error": str(e),
        })
        logger.error(f"[wan.app] Job {job_id} failed: {e}")


# ── Poll ──────────────────────────────────────────────────────────────────────

@app.get("/job/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    if not _current_job or _current_job["job_id"] != job_id:
        raise HTTPException(404, "Job not found")
    j = _current_job
    return JobStatusResponse(
        job_id=j["job_id"],
        status=j["status"],
        started_at=j.get("started_at"),
        finished_at=j.get("finished_at"),
        duration_seconds=j.get("duration_seconds"),
        output_file=j.get("output_file"),
        error=j.get("error"),
    )


# ── Download ──────────────────────────────────────────────────────────────────

@app.get("/output/{filename}")
def download_output(filename: str) -> FileResponse:
    safe = Path(filename).name
    fp = OUTPUT_DIR / safe
    if not fp.exists():
        raise HTTPException(404, "Output not found")
    return FileResponse(str(fp), media_type="video/mp4", filename=safe)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
