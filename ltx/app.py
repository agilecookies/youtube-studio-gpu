"""
LTX-2.5 FastAPI service — Vast.ai deployment.

Endpoints:
  GET  /health    — fast, no model load
  GET  /status    — GPU info, model state, job state
  POST /generate  — joint audio+video shot generation
  GET  /job/{id}  — poll generation status
  GET  /output/{filename} — download generated video

Watchdog (optional):
  Background task checks machine runtime every 60s.
  If > LTX_MAX_RUNTIME_SECONDS, terminates the pod via RunPod GraphQL API.
  Set LTX_MAX_RUNTIME_SECONDS=0 (default) to disable.
  RUNPOD_POD_ID and RUNPOD_API_KEY are only needed if the watchdog is enabled.
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

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import model as ltx_model
from schemas import (
    AssetStageResponse,
    ConditioningCapabilities,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatusResponse,
    StatusResponse,
    GpuInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ltx.app")

app = FastAPI(title="LTX-2.5 service", version="1.0.0")

# ── Runtime state ─────────────────────────────────────────────────────────────

SERVICE_START = time.time()
# Watchdog ceiling in seconds.  0 (or unset) = DISABLED.  Never defaults to a
# non-zero value so the service does not self-terminate unexpectedly.
MAX_RUNTIME   = int(os.environ.get("LTX_MAX_RUNTIME_SECONDS", "0"))
# Set SKIP_MODEL_LOAD=1 to skip model loading at startup (import-check / dry-run mode).
SKIP_MODEL_LOAD = os.environ.get("SKIP_MODEL_LOAD", "0") == "1"
OUTPUT_DIR    = Path(os.environ.get("LTX_OUTPUT_DIR",    "/workspace/ltx/outputs"))
LOG_DIR       = Path(os.environ.get("LTX_LOG_DIR",      "/workspace/ltx/logs"))
# Staging area for conditioning images uploaded from the Video Engine.
# Isolated per-project; never overlaps with model weights or outputs.
ASSETS_DIR    = Path(os.environ.get("LTX_ASSETS_DIR",   "/workspace/video-engine/assets"))

_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$')
_ALLOWED_MIME = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
_MAX_ASSET_BYTES = 10 * 1024 * 1024  # 10 MB

_current_job: Optional[dict] = None
_last_gen_seconds: Optional[float] = None
_model_load_error: Optional[str] = None


def _pod_started_at_iso() -> Optional[str]:
    p = LOG_DIR / "machine_started_at.txt"
    try:
        ts = int(p.read_text().strip())
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _gpu_info() -> Optional[GpuInfo]:
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name   = pynvml.nvmlDeviceGetName(handle)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gb     = 1024 ** 3
        return GpuInfo(
            name=name if isinstance(name, str) else name.decode(),
            vram_total_gb=round(mem.total / gb, 1),
            vram_free_gb=round(mem.free  / gb, 1),
            vram_used_gb=round(mem.used  / gb, 1),
            utilization_percent=util.gpu,
        )
    except Exception:
        return None


# ── Watchdog ──────────────────────────────────────────────────────────────────

async def _watchdog() -> None:
    """
    Optional billing kill-switch.  Only runs when LTX_MAX_RUNTIME_SECONDS > 0.
    Uses RUNPOD_POD_ID (auto-injected by RunPod) and RUNPOD_API_KEY (pod env var).
    """
    while True:
        await asyncio.sleep(60)
        elapsed = time.time() - SERVICE_START
        if elapsed > MAX_RUNTIME:
            logger.critical(
                "WATCHDOG: pod runtime %.0fs exceeds ceiling %ds — terminating pod.",
                elapsed, MAX_RUNTIME,
            )
            _write_watchdog_log(elapsed)
            _attempt_self_terminate()
            import sys
            sys.exit(99)


def _write_watchdog_log(elapsed: float) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        import json
        with open(LOG_DIR / "watchdog.jsonl", "a") as fh:
            fh.write(json.dumps({
                "event":           "watchdog_triggered",
                "pod_id":          os.environ.get("RUNPOD_POD_ID", "unknown"),
                "elapsed_seconds": elapsed,
                "ceiling_seconds": MAX_RUNTIME,
                "ts":              time.time(),
            }) + "\n")
    except Exception:
        pass


def _attempt_self_terminate() -> None:
    """Terminate this RunPod pod via GraphQL API."""
    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not (pod_id and api_key):
        logger.error(
            "Watchdog: RUNPOD_POD_ID or RUNPOD_API_KEY not available — cannot self-terminate. "
            "Terminate manually via RunPod console."
        )
        return
    try:
        resp = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'},
            timeout=15,
        )
        logger.info("Watchdog: RunPod terminate → HTTP %d", resp.status_code)
    except Exception as exc:
        logger.error("Watchdog: RunPod terminate API failed: %s", exc)


async def _preload_model() -> None:
    """
    Load LTX-2.5 pipeline at service startup in a background thread so /health
    reports model_loaded=true (and status="ready") as soon as the pipeline is
    ready. Exits non-zero if loading fails so the orchestrator detects the failure
    immediately rather than waiting for the health-poll timeout.
    """
    global _model_load_error
    logger.info("[startup] Pre-loading LTX-2.5 pipeline in background...")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available — no GPU assigned to this container. "
                "Verify Vast.ai instance has a GPU allocated."
            )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ltx_model.load_pipeline)
        logger.info("[startup] LTX-2.5 pipeline loaded — running warm-up inference...")
        await loop.run_in_executor(None, ltx_model.warmup_inference)
        logger.info("[startup] LTX-2.5 warm-up succeeded — service is ready")
    except Exception as exc:
        import traceback as _tb
        _model_load_error = f"{exc}\n{_tb.format_exc()}"
        logger.critical("[startup] LTX-2.5 pipeline FAILED to load: %s — exiting", exc)
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(1)


@app.on_event("startup")
async def _startup() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID", "unknown")

    # Startup diagnostics — never print secrets
    logger.info("=== LTX-2.5 service startup ===")
    logger.info("Pod:            %s", pod_id)
    logger.info("Port:           %s", os.environ.get("PORT", "8080"))
    logger.info("Model dir:      %s", os.environ.get("LTX_MODEL_DIR", "/workspace/ltx/models/ltx-2.5"))
    logger.info("HF cache dir:   %s", os.environ.get("HF_HOME", "/workspace/ltx/cache/huggingface"))
    logger.info("Output dir:     %s", os.environ.get("LTX_OUTPUT_DIR", "/workspace/ltx/outputs"))
    logger.info("Log dir:        %s", os.environ.get("LTX_LOG_DIR", "/workspace/ltx/logs"))

    gpu = _gpu_info()
    if gpu:
        logger.info("GPU detected:   %s (%.1f GB VRAM)", gpu.name, gpu.vram_total_gb)
    else:
        logger.info("GPU detected:   NONE (pynvml unavailable or no GPU present)")

    try:
        import torch
        logger.info("CUDA available: %s", torch.cuda.is_available())
    except ImportError:
        logger.info("CUDA available: UNKNOWN (torch not imported yet)")

    files_ok = ltx_model.model_files_present()
    logger.info("Model files:    %s", "PRESENT" if files_ok else "MISSING — will fail on first /generate")

    if MAX_RUNTIME > 0:
        asyncio.create_task(_watchdog())
        logger.info("Watchdog:       ARMED — ceiling=%ds (%.1fmin)", MAX_RUNTIME, MAX_RUNTIME / 60)
    else:
        logger.info("Watchdog:       DISABLED (LTX_MAX_RUNTIME_SECONDS=0 or unset)")

    if SKIP_MODEL_LOAD:
        logger.info("SKIP_MODEL_LOAD=1 — skipping model load (import-check / dry-run mode)")
        logger.info("=== LTX-2.5 service starting — model load skipped ===")
        return
    asyncio.create_task(_preload_model())
    logger.info("=== LTX-2.5 service starting — model loading in background ===")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Responds immediately without loading the model.
    Used by RunPod health checks and by npm run ltx:health.
    """
    gpu_info = _gpu_info()
    loaded = ltx_model.is_loaded()
    return HealthResponse(
        status="ready" if loaded else ("error" if _model_load_error else "loading"),
        provider="ltx-2.5",
        gpu_available=gpu_info is not None,
        model_files_present=ltx_model.model_files_present(),
        model_loaded=loaded,
    )


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        provider="ltx-2.5",
        gpu=_gpu_info(),
        model_dir=str(ltx_model.MODEL_DIR),
        model_files_present=ltx_model.model_files_present(),
        model_loaded=ltx_model.is_loaded(),
        current_job=_current_job.get("job_id") if _current_job else None,
        service_uptime_seconds=round(time.time() - SERVICE_START, 1),
        last_generation_seconds=_last_gen_seconds,
        machine_started_at=_pod_started_at_iso(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Trigger LTX-2.5 generation.

    DO NOT CALL DURING INITIAL SETUP.
    Only invoke after explicit user approval and full installation verification.
    """
    global _current_job, _last_gen_seconds

    if _current_job and _current_job.get("status") == "running":
        raise HTTPException(status_code=409, detail="A generation is already in progress")

    job_id = str(uuid.uuid4())[:8]
    _current_job = {
        "job_id":      job_id,
        "status":      "running",
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "result":      None,
        "error":       None,
    }

    async def _run() -> None:
        global _current_job, _last_gen_seconds
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available — GPU is not assigned to this container. "
                    "Verify pod GPU assignment (gpu.count > 0) before retrying."
                )
            await asyncio.get_event_loop().run_in_executor(None, ltx_model.load_pipeline)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ltx_model.generate(
                    prompt=req.prompt,
                    duration_seconds=req.duration_seconds,
                    aspect_ratio=req.aspect_ratio,
                    seed=req.seed,
                    audio=req.audio,
                    width=req.width,
                    height=req.height,
                    job_id=job_id,
                    conditioning_image_path=req.conditioning_image_path,
                    negative_prompt=req.negative_prompt,
                    auto_duration=req.auto_duration,
                ),
            )
            _last_gen_seconds = result["generation_seconds"]
            _current_job.update({
                "status":      "complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result":      result,
            })
        except Exception as exc:
            logger.exception("Generation failed: %s", exc)
            _current_job.update({
                "status":      "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error":       str(exc),
            })

    asyncio.create_task(_run())
    return GenerateResponse(job_id=job_id, status="running", message="Generation started")


@app.get("/job/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    if not _current_job or _current_job.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = _current_job.get("result") or {}
    return JobStatusResponse(
        job_id=job_id,
        status=_current_job["status"],
        started_at=_current_job.get("started_at"),
        finished_at=_current_job.get("finished_at"),
        duration_seconds=result.get("generation_seconds"),
        output_file=result.get("output_filename"),
        error=_current_job.get("error"),
        ocr_passed=result.get("ocr_passed"),
        ocr_detail=result.get("ocr_detail"),
    )


@app.get("/output/{filename}")
def download_output(filename: str) -> FileResponse:
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Output file not found: {filename}")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.post("/assets", response_model=AssetStageResponse)
async def stage_asset(
    project_id: str      = Form(..., description="Narration project ID"),
    asset_id:   str      = Form(..., description="Character or location ID"),
    file:       UploadFile = File(..., description="Reference image (JPEG/PNG/WebP, max 10 MB)"),
) -> AssetStageResponse:
    """
    Upload and stage a conditioning reference image.

    The image is stored at:
        /workspace/video-engine/assets/{projectId}/{assetId}.{ext}

    The returned pod_path is what callers must pass to /generate as
    conditioning_image_path.  A local Mac filesystem path MUST NEVER be
    sent to /generate — only values returned by this endpoint.

    Security:
    - project_id and asset_id must match [a-zA-Z0-9][a-zA-Z0-9_-]{0,127}
    - File type validated against Content-Type (jpeg / png / webp only)
    - File size capped at 10 MB
    - Path is constructed from sanitised components — no path traversal possible
    """
    if not _SAFE_ID_RE.match(project_id):
        raise HTTPException(status_code=400, detail=f"Invalid project_id: {project_id!r}")
    if not _SAFE_ID_RE.match(asset_id):
        raise HTTPException(status_code=400, detail=f"Invalid asset_id: {asset_id!r}")

    mime = file.content_type or ""
    if mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {mime!r}. Allowed: {list(_ALLOWED_MIME)}"
        )

    data = await file.read()
    if len(data) > _MAX_ASSET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes (max {_MAX_ASSET_BYTES})"
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    ext        = _ALLOWED_MIME[mime]
    dest_dir   = ASSETS_DIR / project_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path  = dest_dir / f"{asset_id}{ext}"

    dest_path.write_bytes(data)
    logger.info(
        "Staged asset: project=%s asset=%s → %s (%d bytes)",
        project_id, asset_id, dest_path, len(data),
    )

    return AssetStageResponse(
        asset_id   = asset_id,
        pod_path   = str(dest_path),
        mime_type  = mime,
        size_bytes = len(data),
    )


@app.get("/capabilities", response_model=ConditioningCapabilities)
def capabilities() -> ConditioningCapabilities:
    """
    Preflight check: verify ImageConditioningInput is importable from
    ltx_pipelines.utils.args and can be constructed.  Does NOT load the model.

    API verified against LTX-2 commit a95ab856:
      ImageConditioningInput(path, frame_idx, strength, crf=None)
    from ltx_pipelines.utils.args.
    """
    import importlib
    import tempfile

    import_error: Optional[str] = None

    try:
        mod = importlib.import_module("ltx_pipelines.utils.args")
        ImageConditioningInput = getattr(mod, "ImageConditioningInput", None)
        if ImageConditioningInput is None:
            raise ImportError("ImageConditioningInput not found in ltx_pipelines.utils.args")
    except Exception as exc:
        return ConditioningCapabilities(
            conditioning=False,
            conditioning_type=None,
            conditioning_item_import=None,
            fields=[],
            error=f"ltx_pipelines.utils.args: {exc}",
        )

    # NamedTuple exposes _fields
    try:
        fields = list(getattr(ImageConditioningInput, "_fields", []))
    except Exception as exc:
        fields = []
        import_error = f"fields: {exc}"

    # Smoke-test: construct with a temp file path (no image decode — tests the type only)
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n")
            tmp_path = tmp.name
        _ = ImageConditioningInput(path=tmp_path, frame_idx=0, strength=1.0)
        Path(tmp_path).unlink(missing_ok=True)
        smoke_error: Optional[str] = None
    except Exception as exc:
        smoke_error = f"smoke-test failed: {exc}"

    return ConditioningCapabilities(
        conditioning=smoke_error is None,
        conditioning_type="first_frame" if smoke_error is None else None,
        conditioning_item_import="ltx_pipelines.utils.args" if smoke_error is None else None,
        fields=fields,
        error=smoke_error or import_error,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
