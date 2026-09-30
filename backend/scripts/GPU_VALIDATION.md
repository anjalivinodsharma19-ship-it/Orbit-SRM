# OpenSR GPU validation (Phase 3)

`scripts/gpu_validation.py` runs the completed OpenSR integration on a CUDA machine
(Google Colab T4 is enough) with real weights, and writes a machine-readable report.

**What a pass means:** model loading, tiled inference, blending, uint16 GeoTIFF
writing, preview generation, temp-file cleanup, API wiring and georeferencing all
worked on CUDA, with recorded timings and peak GPU memory.

**What a pass does not mean:** nothing about image quality or scientific validity.
The harness uses a low DDIM step count (default 2) on purpose, so the output is a
smoke-test artifact, not a usable product. It also never changes production defaults
(`OPENSR_SAMPLING_STEPS` stays 100) and never modifies application code.

## 1. Prerequisites

| Item | Requirement |
|---|---|
| GPU | any CUDA device; Colab "T4 GPU" runtime is sufficient |
| torch | a **CUDA** build (`torch.version.cuda` must not be `None`) |
| checkpoint | `opensr-ldsrs2_v1_0_0.ckpt` (~1.05 GiB) as a local file; the harness never downloads it |
| project | the `backend/` folder available in the Colab session |
| input | a 4-band (B04, B03, B02, B08) Sentinel-2 GeoTIFF, or `--fetch-stac` to build one |
| Python | 3.12+ (an `opensr-model` requirement) |

## 2. Google Colab setup (cell by cell)

### Cell 1 - confirm the GPU

```python
!nvidia-smi
import torch
print(torch.__version__, "cuda build:", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU, then rerun"
```

### Cell 2 - get the project into Colab

Mount Drive if the project lives there:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/SIH_2_App/backend
```

Or upload a zip of `backend/` instead:

```python
from google.colab import files
import zipfile
uploaded = files.upload()                      # pick backend.zip
zipfile.ZipFile(next(iter(uploaded))).extractall("/content")
%cd /content/backend
```

### Cell 3 - install dependencies (base + optional OpenSR stack)

```python
!pip install -q -r requirements.txt
!pip install -q -r requirements-opensr.txt
import torch, importlib.util
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("opensr-model installed:", importlib.util.find_spec("opensr_model") is not None)
assert torch.version.cuda is not None, (
    "torch is a CPU-only build; reinstall the CUDA wheel, e.g. "
    "!pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121"
)
```

Colab already ships a CUDA torch that satisfies `torch>=2.2.0`, so pip normally leaves
it alone. The assertion above catches the case where it did not.

### Cell 4 - fetch the checkpoint explicitly (once, ~1.05 GiB)

The harness refuses to download weights implicitly, so fetch them yourself:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id="simon-donike/RS-SR-LTDF", filename="opensr-ldsrs2_v1_0_0.ckpt")
print(ckpt)
```

### Cell 5 - plan first (no inference, no model load)

```python
!python scripts/gpu_validation.py --dry-run --raster /path/to/your_s2_rgbn.tif --checkpoint "$ckpt"
```

This prints the resolved window/overlap, the tile count, the exact API payload and the
equivalent curl commands, plus which preflight checks pass. Use it to confirm the
checkpoint, the packaged config and the raster contract before spending GPU time.

### Cell 6 - the real run (one tile, then the full pipeline, then the API)

```python
!python scripts/gpu_validation.py --fetch-stac --checkpoint "$ckpt" \
    --sampling-steps 2 --modes tile,pipeline,api --json gpu_validation_report.json
```

* `--fetch-stac` builds a small real Sentinel-2 L2A chip (B04, B03, B02, B08) from
  public COGs; pass `--raster /path/to/chip.tif` for a controlled chip, or `--synthetic`
  for a mechanics-only run with clearly fake data.
* `--sampling-steps 2` is the smoke-test setting. Raise it (for example
  `--sampling-steps 20`) only if you have the time budget; production stays at 100.
* `--modes tile,pipeline,api` runs the single-tile test first and aborts before later
  phases if it fails, so the multi-tile run only happens after a successful one-tile run.

### Cell 7 - read the report

```python
import json
report = json.load(open("gpu_validation_report.json"))
print(report["environment"].get("gpu_name"), report["environment"].get("gpu_total_memory_mb"), "MB")
for phase in report["phases"]:
    print(phase["name"], phase["status"], phase.get("seconds"),
          phase.get("gpu_peak_allocated_mb"), phase.get("tiles"))
```

## 3. What the run does

| Phase | Action | Key checks |
|---|---|---|
| `preflight` | config, checkpoint, packaged YAML, raster contract, tile plan | four bands, CRS present, `validate_source`, tile count within `OPENSR_MAX_TILES` |
| `tile` | one 128 px LR window -> one model call | output `(4, 512, 512)`, float32, finite, range `[0, 1]`, not all zero; informational correlation vs a nearest-neighbour upscale |
| `pipeline` | `opensr_pipeline.process` over a multi-tile chip | tile count > 1, provenance device/steps/bands, 4x dimensions, quarter pixel size, CRS and nodata preserved, band descriptions, `OPENSR_MODEL` tag, DN range, preview written, temp file cleaned, outputs removed afterwards |
| `api` | `TestClient`: upload -> `/api/processing/start` -> `/result`, `/preview`, `/download` | 202 accepted, job completed, downloaded GeoTIFF size/CRS/count/dtype, result hides internal paths, job deleted afterwards |

The equivalent API request (also printed by the harness):

```json
{"image_id": "<uploaded id>", "method": "opensr_ldsrs2", "scale_factor": 4,
 "parameters": {"sampling_steps": 2, "overlap": 12, "batch_size": 1}}
```

Equivalent curl against a live server:

```bash
curl -s -F file=@chip.tif http://127.0.0.1:8000/api/imagery/upload
curl -s -X POST http://127.0.0.1:8000/api/processing/start \
  -H 'Content-Type: application/json' \
  -d '{"image_id":"<id>","method":"opensr_ldsrs2","scale_factor":4,"parameters":{"sampling_steps":2}}'
curl -s http://127.0.0.1:8000/api/jobs/<job_id>/result
curl -s -o sr.tif http://127.0.0.1:8000/api/jobs/<job_id>/download
```

Direct pipeline call (no HTTP), as used by the `pipeline` phase:

```python
from app import opensr_pipeline
job = {"id": "gpuvalid1", "method": "opensr_ldsrs2", "scale_factor": 4,
       "parameters": {"sampling_steps": 2, "overlap": 12}}
result = opensr_pipeline.process("chip_244.tif", job)   # writes into DATA_DIR
print(result["provenance"])                             # device, tiles, steps, checkpoint
```

## 4. Expected output

```
[  PASSED] preflight: environment, checkpoint, raster contract and tile plan verified
[  PASSED] tile: one 128px tile at 2 DDIM steps (12.3s)
[  PASSED] pipeline: 4 tiles through opensr_pipeline.process (48.0s)
[  PASSED] api: upload -> processing/start -> download through the API (52.1s)
--- summary ---
preflight    passed
tile         passed        12.3s gpu_peak_alloc=3410.5MB reserved=4192.0MB
pipeline     passed        48.0s tiles=4 gpu_peak_alloc=3520.9MB reserved=4300.0MB
api          passed        52.1s tiles=4 gpu_peak_alloc=3520.9MB reserved=4300.0MB
GPU VALIDATION PASSED - functional smoke test only
```

Timings and memory are hardware dependent. On a CPU run (only with `--allow-cpu`) the
phase details end with `on CPU (NOT GPU VALIDATION)` and the verdict reads
`CPU SMOKE TEST COMPLETED - THIS IS NOT A GPU VALIDATION`.

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| exit code 2, `NOT RUN - no CUDA device` | no CUDA in the runtime | Runtime > Change runtime type > T4 GPU, verify `torch.version.cuda`; `--allow-cpu` runs a labelled CPU smoke test instead |
| `torch is a CPU-only build` assertion | pip replaced Colab's torch | `!pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121` |
| `FAIL opensr_model_installed` | optional stack missing | `pip install -r requirements-opensr.txt` |
| `FAIL checkpoint_configured` | `--checkpoint` not passed | point it at the downloaded `.ckpt`; the harness never downloads weights |
| `FAIL config_yaml_found` | package installed without its data files | reinstall `opensr-model` |
| `raster check failed: ... 4 bands` | wrong band count | supply a 4-band B04/B03/B02/B08 GeoTIFF |
| `raster check failed: ... 8-bit` | uint8 raster | provide uint16 DN Sentinel-2 data (10000 = 1.0) |
| `--fetch-stac` fails (timeout, 403, no scenes) | STAC/network not reachable from the runtime | pass `--raster your_chip.tif`, or `--synthetic` for a mechanics-only run |
| `CUDA out of memory` during `pipeline` | VRAM too small for the batch/window | `--batch-size 1`, `--multi-size 128`, fewer `--sampling-steps` |
| `ABORT: phase 'tile' failed` | one-tile test failed, later phases skipped by design | fix the tile failure first; its failing check names the problem |
| inference silently on CPU | `OPENSR_DEVICE=cpu` already set in the environment | unset it; `auto` never falls back to CPU |
| `can't open file 'scripts/gpu_validation.py'` | wrong working directory | `cd backend` first (the script adds `backend/` to `sys.path`) |

## 6. Cleanup and scope

The harness writes its chips into `--work-dir` (default: the system temp directory) and
its pipeline output into `DATA_DIR` (`backend/data/{outputs,previews,temporary}`).
Pipeline outputs and previews are removed after the pipeline phase unless
`--keep-outputs` is given, and the API phase deletes its job through
`DELETE /api/jobs/{id}`. Nothing is deployed, no configuration file is modified, and
production defaults (`OPENSR_SAMPLING_STEPS=100`, `OPENSR_DEVICE=auto`) are untouched.

## 7. Limits of this harness

* It validates plumbing, georeferencing and resource use - **not** image quality. Two
  DDIM steps cannot produce a meaningful product.
* `--fetch-stac` reads a window from the scene origin rather than the requested bbox:
  it is there to supply real pixels, not a spatially exact chip. Its network path was
  written from the public STAC/COG interface and was **not executed** on the
  development machine (which has no CUDA and no network run was performed).
* `--batch-size > 1` is passed through to the verified batched wrapper, but batching
  was never validated end to end on the development host; test it separately if you
  want to raise it.
* Uncertainty maps, quality metrics against reference imagery, and COG output are out
  of scope for this phase.
