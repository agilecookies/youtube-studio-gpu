"""
LTX-2.5 lazy model loader — Vast.ai deployment.

Pipeline is NOT loaded at import time or on /health.
It is loaded on the first POST /generate request.

Model weights live at /workspace/ltx/models/ (downloaded by vast-onstart.sh on
first instance boot, ~66 GiB, 20-40 min). Subsequent instances with a persistent
disk skip the download.

Audio design (joint pass):
  Every shot = one DistilledPipeline call that produces video frames + synchronized
  audio in one pass from the same text prompt. Speech/dialogue/narration goes into
  the main prompt using the official LTX-2.5 format:

      SPOKEN AUDIO:
      Speaker (delivery, voice description): "Verbatim dialogue text."
      MUSIC: [music cue]

  There is no separate T2A pipeline, no ffmpeg audio muxing, no post-processing.
  LTX is the sole source of all audio: speech, music, ambience, SFX.

Negative prompt:
  Not supported by the distilled checkpoint (runs at CFG=1; guidance-free sampling).
  The `negative_prompt` field is accepted and passed through for forward-compatibility
  with non-distilled checkpoints. The distilled pipeline silently ignores it.
  On-screen text is suppressed via the VISUAL OUTPUT RULE in the positive prompt
  and by stripping the rule section before passing to the model (see _strip_visual_output_rule).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ltx.model")

MODEL_DIR  = Path(os.environ.get("LTX_MODEL_DIR",  "/workspace/ltx/models/ltx-2.5"))
OUTPUT_DIR = Path(os.environ.get("LTX_OUTPUT_DIR", "/workspace/ltx/outputs"))
LOG_DIR    = Path(os.environ.get("LTX_LOG_DIR",    "/workspace/ltx/logs"))
FPS        = int(os.environ.get("LTX_FPS",         "24"))

REQUIRED_FILES = [
    MODEL_DIR / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    MODEL_DIR / "text_encoders"    / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    MODEL_DIR / "vae"              / "ltx-2.5-video-vae-bf16.safetensors",
    MODEL_DIR / "vae"              / "ltx-2.5-audio-vae-bf16.safetensors",
    MODEL_DIR / "model_patches"    / "ltx-2.5-duration-head-bf16.safetensors",
    MODEL_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]

# Default negative prompt passed to non-distilled checkpoints.
# The distilled checkpoint (CFG=1) silently ignores it; kept for forward-compat.
# On-screen text is enforced via VISUAL OUTPUT RULE in the positive prompt, not here.
_DEFAULT_NEGATIVE_PROMPT = (
    "text, words, letters, caption, captions, subtitles, title card, title cards, "
    "lower thirds, typography, font, watermark, logo, signage, UI text, timestamp, "
    "numbers, writing, handwriting, labels, overlaid text"
)

_VISUAL_OUTPUT_RULE_RE = re.compile(
    r"\n*VISUAL OUTPUT RULE:.*?(?=\n\nMOTION RULE:|\Z)", re.DOTALL
)

# ── State ─────────────────────────────────────────────────────────────────────

_pipeline: Optional[object] = None
_model_loaded_at: Optional[float] = None
_load_seconds: Optional[float] = None


def model_files_present() -> bool:
    return all(f.exists() for f in REQUIRED_FILES)


def is_loaded() -> bool:
    return _pipeline is not None


def _write_timing(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "timings.jsonl", "a") as fh:
        fh.write(json.dumps({**record, "ts": time.time()}) + "\n")


def _frames_for_duration(duration_seconds: int) -> int:
    """LTX-2.5 requires: num_frames % 8 == 1  (e.g. 1, 9, 17, ..., 481)."""
    target = duration_seconds * FPS
    remainder = (target - 1) % 8
    if remainder == 0:
        return target
    return target + (8 - remainder)


def _ocr_gate(video_path: str, num_frames: int, confidence_threshold: float = 0.6) -> dict:
    """
    §10 validation gate: sample 3 frames (25/50/75%), run Tesseract OCR on each.
    Returns {"passed": bool, "detail": str}.

    FAIL-CLOSED: any dependency missing, any ffmpeg error, any unexpected exception
    → passed=False and the clip is sent to owner review. Never silently passes
    a clip that couldn't be checked.
    """
    try:
        import subprocess
        import tempfile
        import pytesseract  # type: ignore[import]
        from PIL import Image  # type: ignore[import]
    except ImportError as exc:
        detail = f"OCR dependency missing ({exc}) — clip requires owner review"
        logger.error("OCR gate FAILED — dependency not installed: %s", exc)
        return {"passed": False, "detail": detail}

    if num_frames < 3:
        detail = "clip too short to sample (< 3 frames) — clip requires owner review"
        logger.error("OCR gate FAILED — %s", detail)
        return {"passed": False, "detail": detail}

    sample_positions = [0.25, 0.50, 0.75]
    text_detections: list[str] = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for pos in sample_positions:
                frame_idx = int(num_frames * pos)
                out_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
                result = subprocess.run(
                    [
                        "ffmpeg", "-i", video_path,
                        "-vf", f"select=eq(n\\,{frame_idx})",
                        "-vframes", "1", "-y", out_path,
                    ],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0 or not os.path.exists(out_path):
                    detail = f"ffmpeg frame extract failed at pos {pos:.0%} — clip requires owner review"
                    logger.error("OCR gate FAILED — %s", detail)
                    return {"passed": False, "detail": detail}

                img = Image.open(out_path)
                img_height = img.height or 1
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                for i, text in enumerate(data["text"]):
                    conf = data["conf"][i]
                    token = text.strip()
                    if not (token and isinstance(conf, (int, float)) and conf >= confidence_threshold * 100):
                        continue

                    # Ignore 1–2 char tokens: single-letter brand icons (e.g. 'N', 'G')
                    # on character clothing must not trigger the OCR gate.
                    if len(token) <= 2:
                        continue

                    # Check if this token is in the lower-third of the frame.
                    # pytesseract's 'top' is in pixels from the top of the image.
                    # Anything with top > 70% of frame height is the caption/subtitle zone.
                    word_top = data["top"][i] if i < len(data["top"]) else 0
                    in_lower_third = word_top > img_height * 0.70

                    if in_lower_third:
                        text_detections.append(
                            f"pos {pos:.0%}: CAPTION '{token}' at y={word_top}/{img_height} (conf {conf:.0f})"
                        )
                    else:
                        text_detections.append(f"pos {pos:.0%}: '{token}' (conf {conf:.0f})")

    except subprocess.TimeoutExpired:
        detail = "OCR timed out — clip requires owner review"
        logger.error("OCR gate FAILED — timeout for %s", video_path)
        return {"passed": False, "detail": detail}
    except Exception as exc:  # pylint: disable=broad-except
        detail = f"OCR error ({exc}) — clip requires owner review"
        logger.error("OCR gate FAILED — unexpected error for %s: %s", video_path, exc)
        return {"passed": False, "detail": detail}

    if text_detections:
        detail = "TEXT DETECTED — requires owner review: " + "; ".join(text_detections)
        logger.warning("OCR gate FAILED for %s: %s", video_path, detail)
        return {"passed": False, "detail": detail}

    return {"passed": True, "detail": f"clean ({len(sample_positions)} frames sampled)"}


def _resolution_for_aspect(aspect_ratio: str, width: Optional[int], height: Optional[int]) -> tuple[int, int]:
    """Return (width, height) divisible by 32. Default: 576×1024 for 9:16."""
    if width and height:
        return (width // 32) * 32, (height // 32) * 32
    defaults = {
        "9:16":  (576,  1024),
        "16:9":  (1024,  576),
        "1:1":   (768,   768),
    }
    return defaults.get(aspect_ratio, (576, 1024))


def _make_model_paths():
    """Construct ModelPaths pointing at the installed weights on /workspace."""
    from ltx_pipelines.utils.model_paths import ModelPaths  # type: ignore[import]
    return ModelPaths.from_split(
        transformer_path   = str(MODEL_DIR / "diffusion_models"       / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        text_encoder_path  = str(MODEL_DIR / "text_encoders"          / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        video_vae_path     = str(MODEL_DIR / "vae"                    / "ltx-2.5-video-vae-bf16.safetensors"),
        audio_vae_path     = str(MODEL_DIR / "vae"                    / "ltx-2.5-audio-vae-bf16.safetensors"),
        duration_head_path = str(MODEL_DIR / "model_patches"          / "ltx-2.5-duration-head-bf16.safetensors"),
    )


def load_pipeline() -> None:
    """
    Load the LTX-2.5 DistilledPipeline into GPU memory.
    Called lazily on the first /generate request — never on /health.
    """
    global _pipeline, _model_loaded_at, _load_seconds

    if _pipeline is not None:
        return

    if not model_files_present():
        raise RuntimeError(
            "Model files are missing. Ensure vast-onstart.sh has completed the "
            "download step and /workspace/ltx/models/ exists on the instance."
        )

    logger.info("Loading LTX-2.5 distilled pipeline…")
    t0 = time.time()
    _write_timing({"event": "model_load_started"})

    try:
        from ltx_pipelines.distilled import DistilledPipeline          # type: ignore[import]
        from ltx_pipelines.utils.types import OffloadMode              # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Could not import ltx_pipelines. "
            "Ensure uv sync --extra natten was run inside /opt/ltx2."
        ) from exc

    model_paths = _make_model_paths()
    # CPU offload: transformer and prompt-encoder weights stay in CPU RAM,
    # streamed layer-by-layer to a small GPU buffer. Keeps VRAM use under ~5 GB
    # during weight streaming, making the 22B BF16 model fit on A100 80GB.
    logger.info("Pipeline offload_mode=cpu (transformer + prompt encoder streamed from CPU RAM)")
    _pipeline = DistilledPipeline(
        model_paths            = model_paths,
        spatial_upsampler_path = str(MODEL_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        loras                  = [],
        offload_mode           = OffloadMode.CPU,
    )

    _load_seconds = time.time() - t0
    _model_loaded_at = time.time()
    _write_timing({"event": "model_load_finished", "load_seconds": _load_seconds})
    logger.info("Pipeline loaded in %.1f seconds (offload_mode=cpu)", _load_seconds)


def warmup_inference() -> None:
    """
    Generate one tiny (256×256, 1s) clip to verify the pipeline works end-to-end.
    Called once after load_pipeline() during service startup.
    /health only reports model_loaded=true after this succeeds.
    Raises on any failure so the service exits non-zero and the orchestrator
    detects the error at boot rather than on the first real /generate request.
    """
    if _pipeline is None:
        raise RuntimeError("Pipeline not loaded — call load_pipeline() first")
    logger.info("[warmup] Running warm-up inference: 256×256, 1s, seed=0 ...")
    t0 = time.time()
    result = generate(
        prompt="A solid blue square.",
        duration_seconds=1,
        aspect_ratio="1:1",
        seed=0,
        audio=False,
        width=256,
        height=256,
        job_id="warmup",
    )
    elapsed = time.time() - t0
    logger.info("[warmup] Warm-up succeeded in %.1fs → %s", elapsed, result.get("output_filename", "?"))


def _strip_visual_output_rule(prompt: str) -> str:
    """Remove the enumerated 'VISUAL OUTPUT RULE' anti-text block from the positive
    prompt. Those tokens (captions/title cards/typography) cue diffusion models to
    render garbled on-screen text; the ban is enforced via the negative prompt
    instead. All other sections (incl. the trailing MOTION RULE) are preserved."""
    return _VISUAL_OUTPUT_RULE_RE.sub("", prompt).strip()


def generate(
    prompt: str,
    duration_seconds: Optional[int],
    aspect_ratio: str,
    seed: int,
    audio: bool,
    width: Optional[int] = None,
    height: Optional[int] = None,
    job_id: str = "",
    conditioning_image_path: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    auto_duration: bool = False,
) -> dict:
    """
    Run LTX-2.5 joint audio+video inference.

    Speech, music, ambience, and SFX are all generated in this single call
    from the text prompt. The SPOKEN AUDIO section of the prompt drives audio
    generation. There is no T2A post-processing, no separate audio pipeline,
    and no ffmpeg audio muxing.

    negative_prompt: passed through for forward-compatibility. The distilled
    checkpoint runs at CFG=1 and silently ignores it. On-screen text is
    suppressed by _strip_visual_output_rule (removes the VISUAL OUTPUT RULE
    section from the positive prompt before diffusion).

    DO NOT CALL DURING SETUP — only after explicit user approval.
    """
    if _pipeline is None:
        raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

    # Resolve frame count.
    if auto_duration and duration_seconds is None:
        try:
            from ltx_pipelines.distilled import DEFAULT_AUTO_DURATION  # type: ignore[import]
            num_frames = DEFAULT_AUTO_DURATION
            logger.info("auto_duration=True — duration head will predict num_frames from prompt")
        except ImportError:
            duration_seconds = 20
            num_frames = _frames_for_duration(duration_seconds)
            logger.warning("DEFAULT_AUTO_DURATION not available — falling back to 20s")
    else:
        num_frames = _frames_for_duration(duration_seconds or 20)

    w, h = _resolution_for_aspect(aspect_ratio, width, height)
    neg = negative_prompt or _DEFAULT_NEGATIVE_PROMPT

    logger.info(
        "Starting generation: %ds → %d frames at %dx%d (aspect %s) seed=%d audio=%s",
        duration_seconds or 0, num_frames, w, h, aspect_ratio, seed, audio,
    )
    _write_timing({
        "event":            "generation_started",
        "job_id":           job_id,
        "prompt_chars":     len(prompt),
        "duration_seconds": duration_seconds,
        "num_frames":       num_frames,
        "width":            w,
        "height":           h,
        "seed":             seed,
        "audio":            audio,
    })

    t0 = time.time()
    output_filename = f"{job_id or 'output'}.mp4"
    output_path = str(OUTPUT_DIR / output_filename)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch                                                    # type: ignore[import]
    from ltx_pipelines.utils.media_io import encode_video          # type: ignore[import]
    from ltx_core.model.video_vae import get_video_chunks_number  # type: ignore[import]

    # Build conditioning items if an image path was supplied.
    images_list = []
    if conditioning_image_path:
        if not Path(conditioning_image_path).exists():
            raise ValueError(f"Conditioning image not found: {conditioning_image_path}")
        try:
            from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                f"Image conditioning requested but ImageConditioningInput import failed "
                f"(expected ltx_pipelines.utils.args — LTX-2 commit a95ab856). Error: {exc}"
            ) from exc
        images_list = [ImageConditioningInput(path=conditioning_image_path, frame_idx=0, strength=1.0)]
        logger.info("Image conditioning: %s (first-frame, strength=1.0)", conditioning_image_path)

    # On-screen text suppression:
    # (1) Strip the VISUAL OUTPUT RULE block from the positive prompt — listing
    #     caption/title-card tokens causes the model to render garbled on-screen text.
    # (2) Pass the negative_prompt for non-distilled checkpoints. Distilled ignores it
    #     (CFG=1, guidance-free sampling).
    clean_prompt = _strip_visual_output_rule(prompt)

    def _run_pipeline(use_negative: bool):
        kwargs = dict(
            prompt=clean_prompt, seed=seed, height=h, width=w,
            num_frames=num_frames, frame_rate=float(FPS), images=images_list,
        )
        if use_negative:
            kwargs["negative_prompt"] = neg
        return _pipeline(**kwargs)  # type: ignore[operator]

    # Video pass runs under torch.inference_mode() — matches the official LTX
    # distilled CLI and is measurably more memory-efficient than no_grad.
    with torch.inference_mode():
        try:
            result = _run_pipeline(use_negative=True)
        except TypeError as exc:
            # Older DistilledPipeline signatures may not accept negative_prompt.
            if "negative_prompt" in str(exc):
                logger.warning("DistilledPipeline has no negative_prompt; using stripped positive prompt only")
                result = _run_pipeline(use_negative=False)
            else:
                raise

    # Joint audio from DistilledPipeline — generated in the same pass as the video.
    # This is the only audio path. There is no T2A post-processing.
    final_audio = result.audio if audio else None

    # encode_video drives the VAE decode of result.video under inference_mode so
    # the decode stays graph-free and mode-consistent with the forward pass above.
    with torch.inference_mode():
        encode_video(
            video=result.video,
            fps=FPS,
            audio=final_audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        )

    elapsed = time.time() - t0
    _write_timing({
        "event":              "generation_finished",
        "job_id":             job_id,
        "generation_seconds": elapsed,
        "output_path":        output_path,
    })
    logger.info("Generation done in %.1f seconds → %s", elapsed, output_path)

    # §10 validation gate — OCR check: sample 3 frames, reject if text detected.
    # Fail-open when pytesseract or ffmpeg is unavailable.
    ocr_result = _ocr_gate(output_path, num_frames)

    return {
        "output_path":        output_path,
        "output_filename":    output_filename,
        "generation_seconds": elapsed,
        "num_frames":         num_frames,
        "width":              w,
        "height":             h,
        "fps":                FPS,
        "ocr_passed":         ocr_result["passed"],
        "ocr_detail":         ocr_result["detail"],
    }
