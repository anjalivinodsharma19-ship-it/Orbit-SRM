# OrbitSRM backend

FastAPI backend for **OrbitSRM**, a satellite imagery super-resolution and analysis
platform. Metadata is persisted as JSON files and rasters are kept separately on disk.

Honest status: this local development API has **no authentication, no ownership checks
and no durable database** yet, and it should not be exposed publicly as-is. Background
processing runs in-process, so jobs do not survive a restart. See
[Still to implement](#still-to-implement) for the outstanding work.

## Run

Python 3.12+ recommended (3.14.6 is what the test suite runs on here); the optional
OpenSR stack requires 3.12+.

PyTorch is optional for the baseline API. It is needed only for `trained_srcnn`
(`pip install torch`) and for the optional OpenSR wrapper, which has its own
dependency file: `pip install -r requirements-opensr.txt`.

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

For reproducible installs (CI or a deployment) use the tested lock file instead:
`pip install -r requirements-lock.txt`.

Swagger is at `http://127.0.0.1:8000/docs`. Run tests with `pytest` from `backend`.

## Configuration

`APP_NAME` (default `OrbitSRM`) and `APP_VERSION` label the API metadata, logs and health responses. `DATA_DIR` controls raster and JSON storage (defaults to `backend/data`); `MAX_UPLOAD_MB` defaults to 200; `MAX_RASTER_WIDTH`, `MAX_RASTER_HEIGHT` and `MAX_RASTER_PIXELS` bound accepted imagery; `MAX_BASELINE_OUTPUT_PIXELS` bounds baseline output size; `METRIC_CHUNK_SIZE` bounds metric reads; `CORS_ORIGINS` is a comma-separated origin list; `MODEL_CHECKPOINT` optionally points at a compatible `state_dict` checkpoint; `DEVICE` may be `auto`, `cpu`, or `cuda`. `OPENSR_ENABLED`, `OPENSR_CHECKPOINT`, `OPENSR_DEVICE`, `OPENSR_SAMPLING_STEPS`, `OPENSR_MAX_SAMPLING_STEPS`, `OPENSR_WINDOW`, `OPENSR_MAX_WINDOW_SIZE`, `OPENSR_OVERLAP`, `OPENSR_BATCH_SIZE`, `OPENSR_MAX_BATCH_SIZE`, `OPENSR_DN_SCALE`, `OPENSR_MAX_TILES`, and `OPENSR_TORCH_THREADS` configure the optional OpenSR method described below. Persistence is JSON metadata plus raster files under `DATA_DIR`: **no database is configured**, which `/api/ready` reports explicitly as `"database": "not_configured"`.

## Health and readiness

`GET /api/health` is a **liveness** probe: it returns 200 with `{"status": "ok", ...}` plus the application name/version, the rasterio dependency and the OpenSR configuration report. It does not prove the service can actually do work.

`GET /api/ready` is a **readiness** probe: it returns 200 only when the hard checks pass (`storage_writable`, `rasterio`, `scikit_image`), otherwise 503 with a `failures` map. It also reports capability labels honestly:

```json
{
  "status": "ready",
  "application": {"name": "OrbitSRM", "version": "1.1.0"},
  "checks": {
    "storage_writable": true,
    "rasterio": true,
    "scikit_image": true,
    "storage": "local_filesystem",
    "database": "not_configured",
    "background_executor": "in_process",
    "opensr": {"enabled": false, "configured": false, "available": false, "reason": "..."}
  },
  "generated_at": "2026-01-01T00:00:00+00:00"
}
```

OpenSR is optional, so a disabled or unconfigured model never makes the service unready. Neither probe returns credentials, environment values or filesystem paths.

## Still to implement

From the Stage 1 backend audit - not started, because they need decisions or credentials that were not provided:

* durable persistence (storage engine not chosen) and migration of the existing JSON metadata
* users, authentication, ownership checks and rate limiting (there is no auth today)
* imagery listing/deletion and orphan cleanup
* durable job execution, restart recovery and cancellation (currently in-process)
* JSON-safe encoding for infinite/undefined metric values
* deployment assets (container, pinned installs, CI) and a hosting decision for the GPU model

## Optional OpenSR (LDSR-S2)

`app/ml/opensr.py` wraps the ESA OpenSR LDSR-S2 latent-diffusion model and `app/opensr_pipeline.py` runs it over large GeoTIFFs in tiles. OpenSR is selectable as method `opensr_ldsrs2` through the existing job workflow (never the default), appears in `/api/models`, and is reported in `/api/health`. The interpolation baseline, `trained_srcnn`, and the existing route/response shapes are otherwise unchanged.

```text
POST /api/processing/start
{ "image_id": "...", "method": "opensr_ldsrs2", "scale_factor": 4,
  "parameters": { "sampling_steps": 100, "overlap": 12, "batch_size": 1 } }
```

`scale_factor` must be `4`, and `parameters` is optional and accepts only those three keys. A request is rejected with 422 while OpenSR is disabled, when the checkpoint or dependencies are missing, for rasters that are not exactly 4-band, or for rasters without a CRS.

Install the optional stack (Python 3.12+, plus a CUDA build of torch for GPU inference):

```powershell
pip install -r requirements.txt
pip install -r requirements-opensr.txt
```

Configure it in `.env`. It stays inert unless `OPENSR_ENABLED` is set, and `OPENSR_CHECKPOINT` must be a local file because checkpoints are never downloaded automatically.

```text
OPENSR_ENABLED=true
OPENSR_CHECKPOINT=C:\path\to\opensr-ldsrs2_v1_0_0.ckpt
OPENSR_DEVICE=auto        # auto = CUDA only, never CPU; use cpu to opt in explicitly
OPENSR_SAMPLING_STEPS=100 # DDIM sampling steps, model default
OPENSR_WINDOW=128         # LR window in pixels; output is 4x per axis
OPENSR_OVERLAP=12         # LR pixels shared between neighbouring tiles (0 disables blending)
OPENSR_BATCH_SIZE=1       # tiles per model call; 1 keeps per-tile spectral matching behaviour
OPENSR_DN_SCALE=10000     # Sentinel-2 DN scale for input normalisation and uint16 output
OPENSR_MAX_TILES=4096     # guard against runaway jobs; raise it for very large rasters
OPENSR_TORCH_THREADS=0    # 0 keeps the torch default
```

Contract and behaviour:

- Input is exactly four channels ordered **B04, B03, B02, B08**, a square window of at least 128 pixels, Sentinel-2 L2A reflectance in `[0, 1]`. Use `opensr.to_reflectance(dn, dn_scale=10000.0)` for integer DN.
- Output is 4x the window size on both axes, reflectance in `[0, 1]`. Use `opensr.to_uint16_reflectance(...)` to scale back to Sentinel-2 DN before writing a uint16 GeoTIFF.
- `torch` and `opensr-model` are imported lazily and the model is built on first use and cached; nothing is imported or loaded at application startup.
- `auto` never falls back to CPU. CPU inference is impractically slow (measured in this project: ~79 s fixed cost plus ~6.7 s per DDIM step per 128x128 window, ~2 GB resident for the weights) and must be requested explicitly with `OPENSR_DEVICE=cpu`.
- `opensr.status()` returns a non-raising availability report (device, checkpoint, dependency and device-probe results, and the reason it is unavailable); `opensr.model_card()` returns a descriptor shaped like the `/api/models` entries.
- Failures raise `OpenSRUnavailable`, `OpenSRInputError`, or `OpenSRInferenceError`, all subclasses of `OpenSRError`, including CUDA/host memory exhaustion, so existing job error handling keeps working.
- Large rasters are processed in tiles of `OPENSR_WINDOW` LR pixels read one at a time, so host memory is bounded by a tile rather than the raster. Tile offsets are clamped inwards so the model never receives a short window; rasters smaller than one tile are read whole and edge-padded to exactly the window size, then cropped back on write.
- Neighbouring tiles share `OPENSR_OVERLAP` LR pixels and are feathered together (a linear ramp across the whole shared band, now applied vertically and then horizontally). The blend uses only the previous tile's right band and the previous tile row's bottom band, so no full-raster accumulator and no read-back of a partially written file is needed (`rasterio` exposes no flush on a writer).
- Output keeps the source CRS, nodata and band descriptions, is written as uint16 DN at `OPENSR_DN_SCALE`, is 4x the size per axis, and has the affine transform scaled to a quarter of the source pixel size. `OPENSR_*` provenance tags are written into the GeoTIFF.
- Failures (missing CUDA, unusable checkpoint, invalid raster, out-of-memory, tile-limit exceeded) mark the job `failed` with the reason and remove the temporary output, preview and any partial output file.
- Not included: vendor `opensr-utils` tiling (not installed), calibrated uncertainty, and progress reporting inside a running job. Blending is an approximation and seams may remain. Output is model-inferred detail and is not scientifically validated here.
- Batched execution (`OPENSR_BATCH_SIZE > 1`) is wired and shape-verified, and the model's per-image spectral correction was measured to be equivalent batched vs per-tile; a full batched forward pass could not be completed on this CPU-only host, so the default stays `1`.

Testing without a GPU: run `pytest` from `backend`. `tests/test_opensr.py` covers the disabled path, a missing checkpoint, missing dependencies, the no-CPU-fallback device rule, band order and DN scaling, and the guarantee that importing the module neither imports torch nor loads a model. `tests/test_opensr_pipeline.py` covers tiling geometry, edge tiles, blending, output scaling/georeferencing, batching wiring, cleanup on failure, and the API method-selection/validation/health/model-registry contracts with inference mocked. `tests/test_gpu_validation_script.py` covers the Phase 3 harness helpers and its `--dry-run` path. None of them needs a CUDA device, torch, or the checkpoint.

Running on a GPU (Phase 3): `scripts/gpu_validation.py` is a standalone harness that exercises the wrapper, the tiled pipeline and the API on CUDA with real weights, checking shapes, georeferencing, cleanup and resource use, and reporting wall times plus peak GPU memory. It uses a low DDIM step count for the smoke test, never changes production defaults, never downloads checkpoints and cleans up after itself. `scripts/GPU_VALIDATION.md` has the Google Colab walkthrough, the exact API request, expected output and troubleshooting. It validates plumbing only - not image quality.

## API

- `GET /api/health`
- `POST /api/imagery/upload` (multipart field `file`); `GET /api/imagery/{image_id}` and `/preview`
- `POST /api/processing/start` with `{ "image_id": "...", "method": "baseline", "scale_factor": 2 }`; `method` is `baseline`, `trained_srcnn`, or `opensr_ldsrs2` (see the OpenSR section for its parameters and 4x-only scale)
- `GET /api/jobs`, `/api/jobs/{job_id}`, `/result`, `/preview`, `/download`; `DELETE /api/jobs/{job_id}`
- `GET /api/models`, `/api/models/{model_id}`
- `POST /api/validation/compare?output_job_id=...&reference_image_id=...`

GeoTIFF is required for geospatial processing. PNG/JPEG are preview-only. The baseline uses cubic interpolation and does not recover newly observed detail or establish sub-4m ground resolution. No trained checkpoint is included. `trained_srcnn` is available only when compatible weights are configured, and those weights must have been trained for the raster channels and radiometry in use. The basic CNN loader does not imply scientific suitability. Validation computes PSNR, SSIM, and RMSE only for exact CRS/transform/dimension/band matches; supply a true aligned high-resolution reference. No calibrated uncertainty is produced.

Outputs preserve CRS and scale the affine transform to match output pixel dimensions. The baseline and `trained_srcnn` use bounded 512-pixel input windows; OpenSR uses bounded `OPENSR_WINDOW` tiles with feathered overlap. Deploy behind authenticated access and persistent storage for production; background execution is in-process and is lost if the server restarts.
