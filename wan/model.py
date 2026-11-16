"""
Wan2.2-S2V model loader and inference.

Model:  Wan-AI/Wan2.2-S2V-14B (16B params BF16, ~35 GB, fits A100 80GB).
        Apache 2.0 — no HF_TOKEN required.
        Uses the standard 🤗 diffusers WanS2VPipeline.

Input:  audio file (WAV/MP3) + optional first-frame image + text prompt.
Output: MP4 video synchronized to the audio. Duration = audio length.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("wan.model")

MODEL_ID   = os.environ.get("WAN_S2V_MODEL_ID", "Wan-AI/Wan2.2-S2V-14B")
OUTPUT_DIR = Path(os.environ.get("WAN_OUTPUT_DIR", "/workspace/wan/outputs"))
FPS        = int(os.environ.get("WAN_FPS", "24"))

_pipeline = None
_model_load_error: Optional[str] = None


def is_loaded() -> bool:
    return _pipeline is not None


def load_model() -> None:
    global _pipeline, _model_load_error
    if _pipeline is not None:
        return
    try:
        logger.info(f"[wan.model] Loading {MODEL_ID} via diffusers …")
        t0 = time.time()

        from diffusers import WanS2VPipeline  # type: ignore[import]

        _pipeline = WanS2VPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        _pipeline.to("cuda")

        logger.info(f"[wan.model] Model loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        _model_load_error = str(e)
        logger.error(f"[wan.model] Load failed: {e}")
        raise


def generate(
    audio_path: str,
    prompt: str,
    image_path: Optional[str] = None,
    seed: int = 42,
    fps: int = FPS,
) -> str:
    """Run inference. Returns absolute path to the output MP4."""
    if _pipeline is None:
        raise RuntimeError("Model not loaded — call load_model() first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"

    import torchaudio  # type: ignore[import]
    from PIL import Image  # type: ignore[import]

    # Load audio → mono waveform
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # Optional first frame
    first_frame = Image.open(image_path).convert("RGB") if image_path else None

    generator = torch.Generator("cuda").manual_seed(seed)

    logger.info(f"[wan.model] Generating: prompt={prompt[:60]!r} audio={audio_path} image={image_path}")
    t0 = time.time()

    output = _pipeline(
        prompt=prompt,
        image=first_frame,
        audio=waveform,
        audio_sample_rate=sample_rate,
        fps=fps,
        generator=generator,
    )

    elapsed = time.time() - t0
    logger.info(f"[wan.model] Done in {elapsed:.1f}s")

    # Export to MP4 — diffusers WanS2VPipeline returns a WanPipelineOutput
    if hasattr(output, "export_to_video"):
        output.export_to_video(str(out_path), fps=fps)
    elif hasattr(output, "frames"):
        import imageio  # type: ignore[import]
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264")
        for frame in output.frames[0]:  # frames[0] = first (only) video
            import numpy as np
            writer.append_data(np.array(frame))
        writer.close()
    else:
        raise RuntimeError(f"Unexpected pipeline output type: {type(output)}")

    return str(out_path)
