"""
Smoke tests for the LTX-2.5 FastAPI service.

Designed to run without GPU, CUDA, or model weights.
pynvml / torch / ltx_pipelines are stubbed before any service import.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional
from unittest.mock import MagicMock, patch

# ── Stub heavy GPU/ML modules before service code is imported ─────────────────

_pynvml = MagicMock()
_pynvml.nvmlInit.side_effect = RuntimeError("no GPU in test env")
sys.modules["pynvml"] = _pynvml

_torch = MagicMock()
_torch.cuda.is_available.return_value = False
sys.modules["torch"] = _torch

_ltx_pip = MagicMock()
sys.modules["ltx_pipelines"] = _ltx_pip
sys.modules["ltx_pipelines.distilled"] = _ltx_pip
sys.modules["ltx_pipelines.utils"] = _ltx_pip
sys.modules["ltx_pipelines.utils.types"] = _ltx_pip
sys.modules["ltx_pipelines.utils.model_paths"] = _ltx_pip

# ── Stub ltx_pipelines.utils.args with a real NamedTuple ─────────────────────
# The capabilities endpoint and generate() import ImageConditioningInput from here.
# Providing a real NamedTuple (not a bare MagicMock) lets _fields introspection
# and construction work correctly in tests.

class _FakeImageConditioningInput(NamedTuple):
    path: str
    frame_idx: int
    strength: float
    crf: Optional[int] = None

_ltx_args = MagicMock()
_ltx_args.ImageConditioningInput = _FakeImageConditioningInput
sys.modules["ltx_pipelines.utils.args"] = _ltx_args

# ── Path setup: allow importing services/ltx/*.py directly ───────────────────

_SVC_DIR = str(Path(__file__).parent.parent.resolve())
if _SVC_DIR not in sys.path:
    sys.path.insert(0, _SVC_DIR)

# Remove LTX_MAX_RUNTIME_SECONDS from env so we test the real default
os.environ.pop("LTX_MAX_RUNTIME_SECONDS", None)

import app as svc_app   # noqa: E402
import model as ltx_model  # noqa: E402



# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_startup_capture_tasks(target_max_runtime: int) -> list[str]:
    """Run _startup() with a patched MAX_RUNTIME, collect create_task names."""
    created: list[str] = []

    def fake_task(coro):
        # Synchronous — side_effect must not be a coroutine function
        name = getattr(coro, "__name__", repr(coro))
        created.append(name)
        coro.close()  # silence "coroutine was never awaited"
        return MagicMock()

    orig = svc_app.MAX_RUNTIME
    svc_app.MAX_RUNTIME = target_max_runtime
    try:
        with patch.object(svc_app.asyncio, "create_task", side_effect=fake_task):
            await svc_app._startup()
    finally:
        svc_app.MAX_RUNTIME = orig
    return created


# ── Watchdog tests ────────────────────────────────────────────────────────────

class TestWatchdog:
    def test_default_max_runtime_is_zero(self) -> None:
        """app.py must default to MAX_RUNTIME=0 (watchdog DISABLED) when env is unset."""
        assert svc_app.MAX_RUNTIME == 0, (
            f"MAX_RUNTIME={svc_app.MAX_RUNTIME!r}; expected 0. "
            "The env var LTX_MAX_RUNTIME_SECONDS was not set during import."
        )

    def test_explicit_nonzero_value_parses(self) -> None:
        """LTX_MAX_RUNTIME_SECONDS=7200 must parse to 7200."""
        with patch.dict(os.environ, {"LTX_MAX_RUNTIME_SECONDS": "7200"}):
            val = int(os.environ.get("LTX_MAX_RUNTIME_SECONDS", "0"))
        assert val == 7200

    def test_watchdog_not_scheduled_when_zero(self) -> None:
        """_startup() must NOT schedule _watchdog when MAX_RUNTIME == 0."""
        created = asyncio.get_event_loop().run_until_complete(
            _run_startup_capture_tasks(target_max_runtime=0)
        )
        watchdog_tasks = [t for t in created if "watchdog" in t.lower()]
        assert watchdog_tasks == [], (
            f"Watchdog was scheduled despite MAX_RUNTIME=0: {watchdog_tasks}"
        )

    def test_watchdog_scheduled_when_armed(self) -> None:
        """_startup() MUST schedule _watchdog when MAX_RUNTIME > 0."""
        created = asyncio.get_event_loop().run_until_complete(
            _run_startup_capture_tasks(target_max_runtime=3600)
        )
        watchdog_tasks = [t for t in created if "watchdog" in t.lower()]
        assert watchdog_tasks, (
            f"Expected watchdog task for MAX_RUNTIME=3600, got tasks: {created}"
        )


# ── Workspace path tests ──────────────────────────────────────────────────────

class TestWorkspacePaths:
    def test_model_dir_under_workspace(self) -> None:
        assert str(ltx_model.MODEL_DIR).startswith("/workspace"), (
            f"MODEL_DIR={ltx_model.MODEL_DIR}"
        )

    def test_output_dir_under_workspace(self) -> None:
        assert str(ltx_model.OUTPUT_DIR).startswith("/workspace"), (
            f"OUTPUT_DIR={ltx_model.OUTPUT_DIR}"
        )

    def test_log_dir_under_workspace(self) -> None:
        assert str(ltx_model.LOG_DIR).startswith("/workspace"), (
            f"LOG_DIR={ltx_model.LOG_DIR}"
        )


# ── Health / route tests ──────────────────────────────────────────────────────

class TestRoutes:
    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(svc_app.app, raise_server_exceptions=True)

    def test_health_returns_200_no_gpu(self) -> None:
        """GET /health must return 200 without any GPU present."""
        with patch.object(svc_app, "_gpu_info", return_value=None), \
             patch.object(ltx_model, "model_files_present", return_value=False), \
             patch.object(ltx_model, "is_loaded", return_value=False):
            resp = self._client().get("/health")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["status"] == "ok"
        assert body["provider"] == "ltx-2.5"
        assert body["gpu_available"] is False
        assert "model_files_present" in body
        assert "model_loaded" in body

    def test_status_returns_200_no_gpu(self) -> None:
        """GET /status must return 200 without any GPU."""
        with patch.object(svc_app, "_gpu_info", return_value=None), \
             patch.object(ltx_model, "model_files_present", return_value=False), \
             patch.object(ltx_model, "is_loaded", return_value=False):
            resp = self._client().get("/status")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_generate_route_exists(self) -> None:
        """POST /generate must exist (returns something other than 404)."""
        with patch.object(svc_app, "_gpu_info", return_value=None), \
             patch.object(ltx_model, "model_files_present", return_value=True), \
             patch.object(ltx_model, "load_pipeline", side_effect=RuntimeError("no GPU")), \
             patch.object(ltx_model, "is_loaded", return_value=False):
            client = TestClient(svc_app.app, raise_server_exceptions=False)  # type: ignore[import]
            resp = client.post("/generate", json={"prompt": "test prompt"})

        assert resp.status_code != 404, "POST /generate route is missing"

    def test_job_route_returns_404_for_unknown(self) -> None:
        """GET /job/{id} must return 404 for an unknown job ID."""
        with patch.object(svc_app, "_gpu_info", return_value=None), \
             patch.object(ltx_model, "model_files_present", return_value=False), \
             patch.object(ltx_model, "is_loaded", return_value=False):
            resp = self._client().get("/job/nonexistent-job-id")

        assert resp.status_code == 404


# ── CPU offload tests ─────────────────────────────────────────────────────────

class TestCpuOffload:
    """Verify that load_pipeline() constructs DistilledPipeline with OffloadMode.CPU."""

    def test_pipeline_constructed_with_cpu_offload(self) -> None:
        """DistilledPipeline must be called with offload_mode whose .value == 'cpu'."""
        from enum import Enum
        from unittest.mock import MagicMock, patch

        class FakeOffloadMode(Enum):
            NONE = "none"
            CPU  = "cpu"
            DISK = "disk"

        fake_distilled_cls   = MagicMock(return_value=MagicMock())
        fake_model_paths_cls = MagicMock()
        fake_model_paths_cls.from_split.return_value = MagicMock()

        # Patch sys.modules so the imports inside load_pipeline() resolve to our fakes.
        patched_modules = {
            "ltx_pipelines.distilled":         MagicMock(DistilledPipeline=fake_distilled_cls),
            "ltx_pipelines.utils.model_paths": MagicMock(ModelPaths=fake_model_paths_cls),
            "ltx_pipelines.utils.types":        MagicMock(OffloadMode=FakeOffloadMode),
        }

        # Reset pipeline state before and after.
        original_pipeline = ltx_model._pipeline
        ltx_model._pipeline = None
        try:
            with patch.dict("sys.modules", patched_modules), \
                 patch.object(ltx_model, "model_files_present", return_value=True), \
                 patch.object(ltx_model, "_write_timing"):
                ltx_model.load_pipeline()
        finally:
            ltx_model._pipeline = original_pipeline

        assert fake_distilled_cls.called, "DistilledPipeline was never instantiated"
        _, kwargs = fake_distilled_cls.call_args
        assert "offload_mode" in kwargs, (
            f"offload_mode not passed to DistilledPipeline. Got kwargs: {list(kwargs)}"
        )
        offload_val = kwargs["offload_mode"]
        assert offload_val == FakeOffloadMode.CPU, (
            f"Expected OffloadMode.CPU (value='cpu'), got {offload_val!r}"
        )



# ── Inference-mode tests ──────────────────────────────────────────────────────

class TestInferenceMode:
    """Verify that generate() runs the video pipeline inside torch.inference_mode().

    inference_mode is more memory-efficient than no_grad (it skips version-counter
    bookkeeping and allows in-place buffer reuse), which the ~22B VAE decode needs to
    fit on an 80 GB A100 — a no_grad video pass OOM'd. The autograd "inference
    tensors cannot be saved for backward" issue is avoided architecturally: T2A
    narration runs in a SEPARATE request (/generate-narration), never in the same
    process as this inference_mode video pass.
    """

    def test_pipeline_called_inside_inference_mode(self) -> None:
        """_pipeline() must be invoked within a torch.inference_mode() context."""
        import contextlib
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        in_context: list[bool] = []
        pipeline_saw_context: list[bool] = []

        @contextlib.contextmanager
        def fake_inference_mode():
            in_context.append(True)
            try:
                yield
            finally:
                in_context.pop()

        def capturing_pipeline(**kwargs):
            pipeline_saw_context.append(bool(in_context))
            result = MagicMock()
            result.video = iter([])
            result.audio = None
            result.num_frames = 9
            result.tiling_config = None
            return result

        fake_torch = MagicMock()
        fake_torch.inference_mode = fake_inference_mode
        # no_grad must NOT be what wraps the pipeline — make it a no-op context that
        # does not register in_context, so a regression to no_grad would leave
        # pipeline_saw_context[0] False.
        fake_torch.no_grad = contextlib.nullcontext

        fake_media_io = MagicMock()
        fake_vae = MagicMock()
        fake_vae.get_video_chunks_number.return_value = 1

        original_pipeline   = ltx_model._pipeline
        original_output_dir = ltx_model.OUTPUT_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            ltx_model._pipeline   = capturing_pipeline
            ltx_model.OUTPUT_DIR  = Path(tmpdir)
            try:
                with patch.dict("sys.modules", {
                    "torch":                        fake_torch,
                    "ltx_pipelines.utils.media_io": fake_media_io,
                    "ltx_core":                     MagicMock(),
                    "ltx_core.model":               MagicMock(),
                    "ltx_core.model.video_vae":     fake_vae,
                }), patch.object(ltx_model, "_write_timing"):
                    ltx_model.generate(
                        prompt="test prompt",
                        duration_seconds=4,
                        aspect_ratio="9:16",
                        seed=42,
                        audio=False,
                        job_id="test-inference-mode",
                    )
            finally:
                ltx_model._pipeline  = original_pipeline
                ltx_model.OUTPUT_DIR = original_output_dir

        assert pipeline_saw_context, "Pipeline was never called — check generate() logic"
        assert pipeline_saw_context[0], (
            "Pipeline was NOT called inside torch.inference_mode(). "
            "Wrap _pipeline() with `with torch.inference_mode():` in model.py:generate()."
        )


from fastapi.testclient import TestClient  # noqa: E402 (used in tests above)


# ── Capabilities tests ────────────────────────────────────────────────────────

class TestCapabilities:
    """Verify /capabilities returns conditioning:true when the import is available."""

    def _client(self):
        return TestClient(svc_app.app, raise_server_exceptions=True)

    def test_capabilities_returns_conditioning_true(self) -> None:
        """GET /capabilities must return conditioning:true when ltx_pipelines.utils.args is present."""
        with patch.object(svc_app, "_gpu_info", return_value=None):
            resp = self._client().get("/capabilities")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["conditioning"] is True, (
            f"Expected conditioning:true, got: {body}"
        )

    def test_capabilities_import_path(self) -> None:
        """conditioning_item_import must reference ltx_pipelines.utils.args."""
        with patch.object(svc_app, "_gpu_info", return_value=None):
            resp = self._client().get("/capabilities")
        body = resp.json()
        assert body.get("conditioning_item_import") == "ltx_pipelines.utils.args", (
            f"Wrong import path: {body.get('conditioning_item_import')!r}"
        )

    def test_capabilities_fields_include_path_and_frame_idx(self) -> None:
        """NamedTuple fields must include path and frame_idx."""
        with patch.object(svc_app, "_gpu_info", return_value=None):
            resp = self._client().get("/capabilities")
        body = resp.json()
        fields = body.get("fields", [])
        assert "path" in fields,      f"'path' not in fields: {fields}"
        assert "frame_idx" in fields, f"'frame_idx' not in fields: {fields}"
        assert "strength" in fields,  f"'strength' not in fields: {fields}"

    def test_capabilities_no_error(self) -> None:
        """error field must be null when conditioning is true."""
        with patch.object(svc_app, "_gpu_info", return_value=None):
            resp = self._client().get("/capabilities")
        body = resp.json()
        assert body.get("error") is None, f"Unexpected error: {body.get('error')}"


# ── Assets tests ──────────────────────────────────────────────────────────────

class TestAssets:
    """Verify POST /assets accepts a multipart image upload."""

    def _client(self):
        return TestClient(svc_app.app, raise_server_exceptions=True)

    def test_assets_route_accepts_png_upload(self) -> None:
        """POST /assets must accept a valid PNG upload and return pod_path."""
        import tempfile
        from io import BytesIO

        # Minimal valid PNG (8-byte signature + IHDR + IDAT + IEND)
        _TINY_PNG = (
            b"\x89PNG\r\n\x1a\n"               # signature
            b"\x00\x00\x00\rIHDR"              # IHDR length+type
            b"\x00\x00\x00\x01\x00\x00\x00\x01"  # 1×1
            b"\x08\x02\x00\x00\x00"            # 8-bit RGB
            b"\x90wS\xde"                       # IHDR CRC
            b"\x00\x00\x00\x0cIDAT"            # IDAT length+type
            b"\x08\xd7c\xf8\xcf\xc0\x00\x00\x00\x02\x00\x01"  # compressed row
            b"\xe2!\xbc3"                        # IDAT CRC
            b"\x00\x00\x00\x00IEND"            # IEND
            b"\xaeB`\x82"                       # IEND CRC
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(svc_app, "ASSETS_DIR", Path(tmpdir)):
                resp = self._client().post(
                    "/assets",
                    data={"project_id": "test-proj-001", "asset_id": "char-main"},
                    files={"file": ("ref.png", BytesIO(_TINY_PNG), "image/png")},
                )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["asset_id"] == "char-main"
        assert "pod_path" in body
        assert body["mime_type"] == "image/png"
        assert body["size_bytes"] > 0

    def test_assets_rejects_invalid_mime(self) -> None:
        """POST /assets must reject non-image content types."""
        from io import BytesIO
        resp = self._client().post(
            "/assets",
            data={"project_id": "test-proj", "asset_id": "test-asset"},
            files={"file": ("file.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 415, f"Expected 415, got {resp.status_code}"

    def test_assets_rejects_empty_file(self) -> None:
        """POST /assets must reject zero-byte uploads."""
        from io import BytesIO
        resp = self._client().post(
            "/assets",
            data={"project_id": "test-proj", "asset_id": "test-asset"},
            files={"file": ("empty.png", BytesIO(b""), "image/png")},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"


# ── Conditioning pipeline acceptance test ─────────────────────────────────────

class TestConditioningPipelineAcceptance:
    """Verify generate() constructs ImageConditioningInput correctly (no GPU needed)."""

    def test_generate_passes_image_conditioning_input_to_pipeline(self) -> None:
        """generate() must pass ImageConditioningInput(path, frame_idx=0, strength=1.0)."""
        import contextlib
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        received_images: list = []

        @contextlib.contextmanager
        def fake_inference_mode():
            yield

        def capturing_pipeline(**kwargs):
            received_images.extend(kwargs.get("images", []))
            result = MagicMock()
            result.video = iter([])
            result.audio = None
            result.num_frames = 9
            result.tiling_config = None
            return result

        fake_torch = MagicMock()
        fake_torch.inference_mode = fake_inference_mode
        fake_media_io = MagicMock()
        fake_vae = MagicMock()
        fake_vae.get_video_chunks_number.return_value = 1

        original_pipeline = ltx_model._pipeline
        original_output_dir = ltx_model.OUTPUT_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a real (non-empty) temp file to satisfy the existence check
            img_path = Path(tmpdir) / "ref.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00test")

            ltx_model._pipeline = capturing_pipeline
            ltx_model.OUTPUT_DIR = Path(tmpdir)
            try:
                with patch.dict("sys.modules", {
                    "torch":                        fake_torch,
                    "ltx_pipelines.utils.media_io": fake_media_io,
                    "ltx_core":                     MagicMock(),
                    "ltx_core.model":               MagicMock(),
                    "ltx_core.model.video_vae":     fake_vae,
                }), patch.object(ltx_model, "_write_timing"):
                    ltx_model.generate(
                        prompt="test conditioning",
                        duration_seconds=4,
                        aspect_ratio="9:16",
                        seed=42,
                        audio=False,
                        job_id="test-conditioning",
                        conditioning_image_path=str(img_path),
                    )
            finally:
                ltx_model._pipeline = original_pipeline
                ltx_model.OUTPUT_DIR = original_output_dir

        assert len(received_images) == 1, (
            f"Expected 1 conditioning item passed to pipeline, got {len(received_images)}"
        )
        item = received_images[0]
        assert isinstance(item, _FakeImageConditioningInput), (
            f"Expected ImageConditioningInput, got {type(item)}"
        )
        assert item.path == str(img_path),  f"Wrong path: {item.path!r}"
        assert item.frame_idx == 0,         f"Wrong frame_idx: {item.frame_idx}"
        assert item.strength == 1.0,        f"Wrong strength: {item.strength}"


# ── Narration endpoint tests ──────────────────────────────────────────────────

class TestNarrationEndpoint:
    """Tests for POST /generate-narration (audio-only, no video regeneration)."""

    def _client(self, raise_exceptions: bool = True):
        return TestClient(svc_app.app, raise_server_exceptions=raise_exceptions)

    @staticmethod
    def _minimal_mp4() -> bytes:
        # ftyp box — enough bytes for a non-empty upload
        return b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42" + b"\x00" * 64

    def test_generate_narration_route_exists(self) -> None:
        """POST /generate-narration must exist (not 404)."""
        from io import BytesIO
        import tempfile

        original = svc_app._current_job
        svc_app._current_job = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(svc_app, "OUTPUT_DIR", Path(tmpdir)):
                    client = self._client(raise_exceptions=False)
                    resp = client.post(
                        "/generate-narration",
                        data={"audio_prompt": "Test narration.", "num_frames": "145", "seed": "42"},
                        files={"video_file": ("test.mp4", BytesIO(self._minimal_mp4()), "video/mp4")},
                    )
        finally:
            svc_app._current_job = original

        assert resp.status_code != 404, "POST /generate-narration route is missing"

    def test_generate_narration_returns_409_when_job_running(self) -> None:
        """POST /generate-narration must return 409 when a job is already running."""
        from io import BytesIO

        original = svc_app._current_job
        svc_app._current_job = {"job_id": "existing-job", "status": "running"}
        try:
            resp = self._client().post(
                "/generate-narration",
                data={"audio_prompt": "Test narration.", "num_frames": "145", "seed": "42"},
                files={"video_file": ("test.mp4", BytesIO(self._minimal_mp4()), "video/mp4")},
            )
        finally:
            svc_app._current_job = original

        assert resp.status_code == 409, f"Expected 409 for busy service, got {resp.status_code}"

    def test_generate_narration_rejects_empty_video(self) -> None:
        """POST /generate-narration must return 400 for a zero-byte video upload."""
        from io import BytesIO

        original = svc_app._current_job
        svc_app._current_job = None
        try:
            resp = self._client().post(
                "/generate-narration",
                data={"audio_prompt": "Test narration.", "num_frames": "145", "seed": "42"},
                files={"video_file": ("empty.mp4", BytesIO(b""), "video/mp4")},
            )
        finally:
            svc_app._current_job = original

        assert resp.status_code == 400, (
            f"Expected 400 for empty video, got {resp.status_code}: {resp.text}"
        )

    def test_generate_narration_accepts_valid_upload_returns_job_id(self) -> None:
        """POST /generate-narration returns 200 with job_id for a valid multipart upload."""
        from io import BytesIO
        import tempfile

        original = svc_app._current_job
        svc_app._current_job = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(svc_app, "OUTPUT_DIR", Path(tmpdir)):
                    client = self._client(raise_exceptions=False)
                    resp = client.post(
                        "/generate-narration",
                        data={"audio_prompt": "Test narration.", "num_frames": "145", "seed": "42"},
                        files={"video_file": ("test.mp4", BytesIO(self._minimal_mp4()), "video/mp4")},
                    )
        finally:
            svc_app._current_job = original

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "job_id" in body, f"No job_id in response: {body}"
        assert body.get("status") == "running", f"Expected status=running: {body}"
