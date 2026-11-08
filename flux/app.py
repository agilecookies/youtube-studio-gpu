"""
FLUX reference-conditioned image generation service — Vast.ai deployment.

Endpoints:
  GET  /health            — runtime status and capability declaration
  POST /generate          — text-to-image (references optional)
  POST /edit              — image-to-image with one or more reference images (required)
  GET  /job/{job_id}      — poll job status
  GET  /output/{filename} — download generated PNG

Configuration (environment variables):
  FLUX_MODEL_ID          HuggingFace model ID (default: black-forest-labs/FLUX.2-klein-4B)
  FLUX_MODEL_DIR         Local model weights directory (default: /workspace/flux/models)
  FLUX_INFERENCE_STEPS   Number of denoising steps (default: 4)
  FLUX_GUIDANCE_SCALE    Guidance scale — 1.0 for FLUX.2-klein-4B distilled (default: 1.0)
  OUTPUT_DIR             Directory for generated outputs (default: /tmp/flux_outputs)
  METRICS_FILE           JSONL file for cost/telemetry records
                         (default: /workspace/flux-generation-metrics.jsonl)

GPU lifecycle: NONE.
This service NEVER starts, stops, or restarts any GPU instance.
Image generation occurs ONLY in response to POST /generate or POST /edit requests.
SERVER STARTUP DOES NOT auto-generate anything.

Reference conditioning:
  Uses Flux2KleinPipeline from diffusers, loaded via from_pretrained().
  Reference images are passed via the native image= argument — NOT via IP-Adapter.
  Reference ordering: characters first, then location, then props (deterministic).
  reference_conditioning=true when the native pipeline loads successfully.
  If reference_conditioning=false, POST /edit is rejected (no text-only fallback).

One output image per request (num_images_per_prompt=1, no collage).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("flux.app")

# ── Configuration ─────────────────────────────────────────────────────────────

FLUX_MODEL_ID   = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
FLUX_MODEL_DIR  = os.environ.get("FLUX_MODEL_DIR", "/workspace/flux/models")
INFERENCE_STEPS = int(os.environ.get("FLUX_INFERENCE_STEPS", "4"))
GUIDANCE_SCALE  = float(os.environ.get("FLUX_GUIDANCE_SCALE", "1.0"))
OUTPUT_DIR      = Path(os.environ.get("OUTPUT_DIR", "/tmp/flux_outputs"))
METRICS_FILE    = Path(os.environ.get("METRICS_FILE", "/workspace/flux-generation-metrics.jsonl"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference role ordering for deterministic image= argument construction.
# Characters first → location → props, consistent with sceneAnchorPipeline.ts ref ordering.
_ROLE_ORDER: dict[str, int] = {"character": 0, "location": 1, "prop": 2}

# ── Runtime state ─────────────────────────────────────────────────────────────

SERVICE_START = time.time()

_pipeline      = None          # Loaded pipeline after startup
_model_loaded  = False
_ref_cond_ok   = False         # True when native FLUX.2 pipeline loaded successfully
_model_load_started_at: Optional[str] = None
_model_ready_at:        Optional[str] = None
_model_load_seconds:    Optional[float] = None

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="FLUX image generation service", version="1.0.0")

# ── Schemas ───────────────────────────────────────────────────────────────────

class ReferenceInput(BaseModel):
    role: str                         # "character" | "location" | "prop"
    asset_id: str
    character_id: Optional[str] = None
    label: str
    image_base64: str                 # Base-64 encoded PNG

class GenerateRequest(BaseModel):
    prompt: str
    width: int  = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=576,  ge=64, le=2048)
    output_format: str = "png"
    seed: int   = Field(default=42)
    references: list[ReferenceInput] = Field(default_factory=list)

class GenerateResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str                       # "queued" | "running" | "complete" | "failed"
    started_at:    Optional[str] = None
    finished_at:   Optional[str] = None
    duration_seconds: Optional[float] = None
    output_file:   Optional[str] = None
    error:         Optional[str] = None
    telemetry:     Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    reference_conditioning: bool
    model_id: str
    device: str
    startup_seconds: float

# ── Model loading ─────────────────────────────────────────────────────────────

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _load_models() -> None:
    global _pipeline, _model_loaded, _ref_cond_ok
    global _model_load_started_at, _model_ready_at, _model_load_seconds

    _model_load_started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    device = _detect_device()

    logger.info("Loading FLUX pipeline: %s on %s", FLUX_MODEL_ID, device)
    try:
        # Prefer the native Flux2KleinKVPipeline class (available in diffusers ≥ 0.32).
        # Fall back to DiffusionPipeline auto-detection for environments where the
        # specific class is not yet exported.
        model_path = FLUX_MODEL_DIR if os.path.isdir(FLUX_MODEL_DIR) else FLUX_MODEL_ID
        logger.info("Loading from: %s", model_path)
        try:
            from diffusers import Flux2KleinPipeline
            pipe = Flux2KleinPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            logger.info("Using Flux2KleinPipeline (native class import)")
        except (ImportError, AttributeError):
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            logger.info("Flux2KleinPipeline not in this diffusers build; using DiffusionPipeline auto-detection")

        pipe.to(device)

        # Log the actual resolved pipeline class for verification.
        # Native reference conditioning is via the image= argument, not IP-Adapter.
        pipe_cls_name = type(pipe).__name__
        logger.info("Loaded pipeline class: %s", pipe_cls_name)

        _pipeline = pipe
        _ref_cond_ok = True
        logger.info(
            "FLUX pipeline loaded — reference_conditioning=true (native image= path). "
            "Pipeline class: %s",
            pipe_cls_name,
        )
    except Exception as exc:
        logger.error("Failed to load FLUX pipeline: %s", exc)
        _model_loaded = False
        _ref_cond_ok = False
        _model_load_seconds = time.time() - t0
        _model_ready_at = datetime.now(timezone.utc).isoformat()
        return

    _model_loaded = True
    _model_load_seconds = time.time() - t0
    _model_ready_at = datetime.now(timezone.utc).isoformat()
    logger.info("Model ready in %.1fs. reference_conditioning=%s", _model_load_seconds, _ref_cond_ok)


@app.on_event("startup")
async def startup_event() -> None:
    """Load model at startup — DOES NOT generate any images automatically."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load_models)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device = _detect_device()
    return HealthResponse(
        status="ok" if _model_loaded else "loading",
        model_loaded=_model_loaded,
        reference_conditioning=_ref_cond_ok,
        model_id=FLUX_MODEL_ID,
        device=device,
        startup_seconds=round(time.time() - SERVICE_START, 1),
    )

# ── Generation ────────────────────────────────────────────────────────────────

def _sort_references(references: list[ReferenceInput]) -> list[ReferenceInput]:
    """Sort references: characters first → location → props (deterministic)."""
    return sorted(references, key=lambda r: _ROLE_ORDER.get(r.role, 99))

def _decode_reference_images(references: list[ReferenceInput]) -> list[Image.Image]:
    images = []
    for ref in references:
        raw = base64.b64decode(ref.image_base64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        images.append(img)
    return images

def _gpu_telemetry(device: str) -> dict:
    tel: dict[str, Any] = {}
    if device == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            tel["gpu_name"] = props.name
        except Exception:
            pass
        try:
            tel["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            tel["peak_cuda_memory_reserved_bytes"]  = torch.cuda.max_memory_reserved()
        except Exception:
            pass
    return tel

def _write_metrics(record: dict) -> None:
    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write metrics: %s", exc)

def _run_generation(job_id: str, req: GenerateRequest) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    device = _detect_device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        # Sort references: characters → location → props before decoding.
        sorted_refs = _sort_references(req.references) if req.references else []
        ref_images = _decode_reference_images(sorted_refs) if sorted_refs else []

        generator = torch.Generator(device=device).manual_seed(req.seed)

        if ref_images and _ref_cond_ok:
            # Native FLUX.2-klein-4B reference conditioning via image= argument.
            # Reference images are passed in deterministic order (chars → loc → props).
            result = _pipeline(
                prompt=req.prompt,
                image=ref_images,
                width=req.width,
                height=req.height,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=INFERENCE_STEPS,
                generator=generator,
                num_images_per_prompt=1,
            )
        else:
            # Text-only path — only reached when no references are supplied.
            result = _pipeline(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=INFERENCE_STEPS,
                generator=generator,
                num_images_per_prompt=1,
            )

        image: Image.Image = result.images[0]
        filename = f"{job_id}.png"
        out_path = OUTPUT_DIR / filename
        image.save(out_path, format="PNG")

        generation_seconds = round(time.time() - t_start, 2)
        finished_at = datetime.now(timezone.utc).isoformat()

        gpu_tel = _gpu_telemetry(device)
        telemetry = {
            "model_id":                  FLUX_MODEL_ID,
            "model_load_started_at":     _model_load_started_at,
            "model_ready_at":            _model_ready_at,
            "model_load_seconds":        _model_load_seconds,
            "job_submitted_at":          job["submitted_at"],
            "generation_started_at":     job["started_at"],
            "generation_completed_at":   finished_at,
            "generation_seconds":        generation_seconds,
            "total_request_seconds":     round(time.time() - time.mktime(
                time.strptime(job["submitted_at"][:19], "%Y-%m-%dT%H:%M:%S")
            ), 2),
            "inference_steps":           INFERENCE_STEPS,
            "width":                     req.width,
            "height":                    req.height,
            "seed":                      req.seed,
            "number_of_reference_images": len(ref_images),
            "reference_conditioning_used": bool(ref_images and _ref_cond_ok),
            **gpu_tel,
        }

        _write_metrics({**telemetry, "job_id": job_id, "output_file": filename})

        job.update({
            "status":       "complete",
            "finished_at":  finished_at,
            "duration_seconds": generation_seconds,
            "output_file":  filename,
            "telemetry":    telemetry,
        })
        logger.info("Job %s complete in %.2fs → %s", job_id, generation_seconds, filename)

    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        logger.error("Job %s failed: %s", job_id, exc)
        job.update({
            "status":      "failed",
            "finished_at": finished_at,
            "duration_seconds": round(time.time() - t_start, 2),
            "error":       str(exc),
        })


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.references and not _ref_cond_ok:
        raise HTTPException(
            status_code=422,
            detail=(
                "reference_conditioning=false: this runtime does not support reference images. "
                "No text-only fallback — request rejected."
            ),
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":       job_id,
        "status":       "queued",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "started_at":   None,
        "finished_at":  None,
        "output_file":  None,
        "error":        None,
    }

    loop = asyncio.get_event_loop()
    asyncio.create_task(loop.run_in_executor(_executor, _run_generation, job_id, req))
    logger.info("Job %s queued (seed=%d, %dx%d, %d refs)", job_id, req.seed, req.width, req.height, len(req.references))
    return GenerateResponse(job_id=job_id, status="queued")


@app.post("/edit", response_model=GenerateResponse)
async def edit(req: GenerateRequest) -> GenerateResponse:
    """Multi-reference image editing — requires at least one reference image."""
    if not req.references:
        raise HTTPException(status_code=422, detail="edit requires at least one reference image in 'references'")
    return await generate(req)


@app.get("/job/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return JobStatusResponse(**{k: v for k, v in job.items() if k != "submitted_at"})


@app.get("/output/{filename}")
def download_output(filename: str) -> FileResponse:
    safe = Path(filename).name
    path = OUTPUT_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"output {filename} not found")
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")
