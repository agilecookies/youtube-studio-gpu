# youtube-studio-gpu

GPU service code for the YouTube Studio autopilot pipeline.

Two services, each deployed independently as Vast.ai instances:

- **`flux/`** — FLUX.2-klein-4B reference image generation (24+ GB VRAM)
- **`ltx/`** — LTX-2.5 shot generation with joint audio+video (48+ GB VRAM)

Each service is a FastAPI app started by its `vast-onstart.sh`.

## Deployment

Vast.ai bootstrap clones this public repo to `/workspace/gpu` and runs the onstart script.
No secrets are stored here. Required env vars are injected at Vast instance creation:

| Var | Required by | Purpose |
|-----|------------|---------|
| `HF_TOKEN` | both | HuggingFace model download |
| `GPU_REPO_URL` | bootstrap | URL of this repo (default: `https://github.com/agilecookies/youtube-studio-gpu.git`) |
| `GPU_REPO_BRANCH` | bootstrap | branch to clone (default: `main`) |

## No secrets, keys, story data or production scripts in this repo.
