from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str                  # "ok"
    provider: str                # "ltx-2.5"
    gpu_available: bool
    model_files_present: bool
    model_loaded: bool


class GpuInfo(BaseModel):
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    utilization_percent: int


class StatusResponse(BaseModel):
    provider: str
    gpu: Optional[GpuInfo]
    model_dir: str
    model_files_present: bool
    model_loaded: bool
    current_job: Optional[str]       # job_id or None
    service_uptime_seconds: float
    last_generation_seconds: Optional[float]
    machine_started_at: Optional[str]  # ISO timestamp


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description=(
        "Director-style prompt: characters, dialogue, environment, "
        "shot sequence, camera movement, lighting, sound, acting direction."
    ))
    duration_seconds: Optional[int] = Field(
        default=None,
        ge=4,
        le=60,
        description=(
            "Clip length in seconds. Frame count is rounded to satisfy num_frames % 8 == 1. "
            "Omit (or set null) together with auto_duration=true to let the duration head "
            "predict the best length from the prompt (clamped to [1s, 20s])."
        ),
    )
    auto_duration: bool = Field(
        default=False,
        description=(
            "When true and duration_seconds is omitted, the LTX-2.5 duration head "
            "predicts num_frames from the prompt. Requires the duration-head checkpoint. "
            "Falls back to 20s if the checkpoint is absent."
        ),
    )
    aspect_ratio: str = Field(
        default="9:16",
        description="Target aspect ratio. Supported: '9:16', '16:9', '1:1'.",
    )
    seed: int = Field(default=42)
    audio: bool = Field(
        default=True,
        description="Enable joint synchronized audio/video generation via the LTX-2.5 audio VAE.",
    )
    # Resolution override (must be divisible by 32).
    # Defaults to 576x1024 for 9:16 (exact ratio, div by 32, manageable VRAM).
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    conditioning_image_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path to a staged reference image for first-frame conditioning. "
            "Obtain via POST /assets → pod_path. "
            "API: ltx_pipelines.utils.args.ImageConditioningInput(path, frame_idx, strength)."
        ),
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Negative guidance prompt. Forward-compatible field — the distilled checkpoint "
            "runs at CFG=1 (guidance-free) and silently ignores this value. Non-distilled "
            "checkpoints will apply it. On-screen text suppression is enforced via the "
            "VISUAL OUTPUT RULE section of the positive prompt regardless."
        ),
    )


class GenerateResponse(BaseModel):
    job_id: str
    status: str        # "queued" | "running" | "complete" | "failed"
    message: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    status: str            # "running" | "complete" | "failed"
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    output_file: Optional[str]   # filename under /data/ltx/outputs/
    error: Optional[str]
    # §10 OCR validation gate result — only set when status == "complete"
    # ocr_passed=False means the clip requires owner review before assembly.
    ocr_passed: Optional[bool] = None
    ocr_detail: Optional[str] = None


class AssetStageResponse(BaseModel):
    asset_id: str          # echoed back
    pod_path: str          # absolute path inside the RunPod container
    mime_type: str         # detected MIME type
    size_bytes: int        # bytes written to disk


class ConditioningCapabilities(BaseModel):
    conditioning: bool
    conditioning_type: Optional[str]        # "first_frame" | None
    conditioning_item_import: Optional[str] # e.g. "ltx_pipelines.conditioning"
    fields: list[str]                       # e.g. ["image", "frame_index", "strength"]
    error: Optional[str]
