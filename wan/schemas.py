from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str           # "ok"
    provider: str         # "wan-s2v"
    gpu_available: bool
    model_loaded: bool


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of what should happen in the shot.")
    # Audio path staged on the pod via POST /stage-audio
    audio_path: str = Field(..., description="Absolute path to the staged audio file on the pod.")
    # Optional first-frame image staged via POST /stage-image
    image_path: Optional[str] = Field(default=None, description="Absolute path to the staged first-frame image on the pod.")
    seed: int = Field(default=42)
    fps: int = Field(default=24)


class GenerateResponse(BaseModel):
    job_id: str
    status: str   # "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str            # "running" | "complete" | "failed"
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    output_file: Optional[str]
    error: Optional[str]


class StageFileResponse(BaseModel):
    pod_path: str
    size_bytes: int
