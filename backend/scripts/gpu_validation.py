#!/usr/bin/env python
"""GPU validation harness for the OpenSR (LDSR-S2) integration.

Phase 3: run this on a CUDA machine or Google Colab to exercise the Phase 1 wrapper
(`app/ml/opensr.py`) and the Phase 2 tiled pipeline (`app/opensr_pipeline.py`) with
real model weights.

What this script does
---------------------
* warm-up checks: dependencies, checkpoint, packaged config, raster contract, tile plan
* one-tile test (default 128x128 LR window) at a LOW DDIM step count, for speed
* full pipeline test through `opensr_pipeline.process` (model load, tiled inference,
  uint16 GeoTIFF writing, preview generation, temp-file cleanup)
* an optional API test through the real FastAPI routes with `TestClient`
* structured report with per-phase wall time and peak CUDA memory

What this script does NOT do
----------------------------
* it does not change production defaults (the model default stays 100 DDIM steps;
  the low step count is passed per job/``parameters`` only)
* it does not modify application code, download checkpoints, or deploy anything
* it does not assess image quality. A smoke test validates plumbing, shapes,
  georeferencing and resource use only - nothing about scientific validity.

Exit codes: 0 = all requested phases passed, 2 = not run (no CUDA and no --allow-cpu),
1 = a phase failed.

Typical use (see scripts/GPU_VALIDATION.md for the Colab walkthrough):

    python scripts/gpu_validation.py --fetch-stac --sampling-steps 2
    python scripts/gpu_validation.py --raster /path/to/s2_rgbn.tif \\
        --checkpoint /path/to/opensr-ldsrs2_v1_0_0.ckpt --sampling-steps 2
    python scripts/gpu_validation.py --dry-run --synthetic   # plan only, no inference
"""
import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DISCLAIMER = ("Smoke test only: shapes, georeferencing and resource use are checked; "
              "no scientific image-quality assessment is made.")
DEFAULT_TILE_SIZE = 128
DEFAULT_SAMPLING_STEPS = 2
STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"


@dataclass
class Phase:
    """Result of one validation phase."""

    name: str
    status: str = "skipped"          # passed | failed | skipped | not-run
    detail: str = ""
    seconds: float = None
    tiles: int = None
    checks: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    gpu_peak_allocated_mb: float = None
    gpu_peak_reserved_mb: float = None

    def ok(self):
        return self.status == "passed"

    def as_dict(self):
        return {key: value for key, value in self.__dict__.items() if value is not None}


@dataclass
class Report:
    """Whole-run report, printable as text or JSON."""

    environment: dict = field(default_factory=dict)
    phases: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def add(self, phase):
        self.phases.append(phase)
        print(f"[{phase.status.upper():>8}] {phase.name}: {phase.detail}"
              + (f" ({phase.seconds:.1f}s)" if phase.seconds else ""), flush=True)
        for check, result in phase.checks.items():
            print(f"            {'ok  ' if result else 'FAIL'} {check}")
        return phase

    def as_dict(self):
        return {"environment": self.environment, "phases": [p.as_dict() for p in self.phases],
                "notes": self.notes, "disclaimer": DISCLAIMER}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenSR GPU validation harness (Phase 3). " + DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("input raster (choose one)")
    source.add_argument("--raster", type=Path, help="existing 4-band GeoTIFF (B04, B03, B02, B08)")
    source.add_argument("--fetch-stac", action="store_true",
                        help=f"build a chip from public Sentinel-2 L2A COGs via STAC ({STAC_SEARCH_URL})")
    source.add_argument("--synthetic", action="store_true",
                        help="build a SYNTHETIC 4-band raster (mechanics only, not real imagery)")
    source.add_argument("--bbox", default="11.55,48.13,11.58,48.15",
                        help="STAC search bbox as west,south,east,north")
    source.add_argument("--datetime", dest="datetime_range", default="2024-06-01/2024-08-31",
                        help="STAC datetime range for --fetch-stac")
    source.add_argument("--source-size", type=int, default=512,
                        help="side length in pixels of the chip built by --fetch-stac/--synthetic")

    run = parser.add_argument_group("run configuration")
    run.add_argument("--checkpoint", type=Path, help="local opensr-ldsrs2_v1_0_0.ckpt (never downloaded)")
    run.add_argument("--device", default="auto", help="OPENSR_DEVICE value (auto = CUDA only, never CPU)")
    run.add_argument("--window", type=int, default=0, help="OPENSR_WINDOW override (default: keep config)")
    run.add_argument("--overlap", type=int, default=-1, help="per-job tile overlap (default: keep config)")
    run.add_argument("--batch-size", type=int, default=0, help="per-job batch size (default: keep config)")
    run.add_argument("--sampling-steps", type=int, default=DEFAULT_SAMPLING_STEPS,
                     help=f"DDIM steps for this smoke test only (production default stays 100); "
                          f"default {DEFAULT_SAMPLING_STEPS}")
    run.add_argument("--modes", default="tile,pipeline,api",
                     help="comma list of tile,pipeline,api (preflight always runs first)")
    run.add_argument("--work-dir", type=Path, default=None,
                     help="where chips and reports are written (default: system temp)")
    run.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                     help="LR window for the one-tile test (must be >= 128)")
    run.add_argument("--multi-size", type=int, default=0,
                     help="LR side length for the multi-tile test (0 = auto: window + stride)")
    run.add_argument("--allow-cpu", action="store_true",
                     help="allow a CPU smoke test when CUDA is absent (NOT a GPU validation)")
    run.add_argument("--dry-run", action="store_true",
                     help="preflight and print the plan/API request, then stop without inference")
    run.add_argument("--keep-outputs", action="store_true", help="keep pipeline outputs and preview")
    run.add_argument("--json", type=Path, default=None, help="write the report as JSON to this path")
    return parser.parse_args(argv)


def apply_environment(args):
    """Configure OpenSR through the documented OPENSR_* variables, without touching files.

    Values the operator already exported win, so this only supplies what is missing.
    Sampling steps, overlap and batch size are deliberately not set here: they travel
    per job through the documented ``parameters`` dict (see run_pipeline).
    """
    applied = {}
    defaults = {"OPENSR_ENABLED": "true", "OPENSR_DEVICE": args.device}
    if args.checkpoint:
        defaults["OPENSR_CHECKPOINT"] = str(args.checkpoint.resolve())
    if args.window:
        defaults["OPENSR_WINDOW"] = str(args.window)
    for name, value in defaults.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def synthetic_sentinel2(path, size, dn_scale=10000.0):
    """Build a deliberately SYNTHETIC 10 m 4-band raster (B04, B03, B02, B08).

    Only for validating plumbing when no real chip is available. The run notes and the
    report say so explicitly: results from synthetic input say nothing about how the
    model behaves on real Sentinel-2 imagery.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    columns = np.linspace(0.02, 0.55, size, dtype=np.float32)
    rows = np.linspace(0.03, 0.35, size, dtype=np.float32)
    base = np.clip(rows[:, None] + columns[None, :], 0.0, 1.0)
    texture = (np.sin(np.arange(size) / 7.0)[None, :] * np.cos(np.arange(size) / 11.0)[:, None]) * 0.05
    data = np.zeros((4, size, size), dtype="uint16")
    for band, gain in enumerate((1.0, 0.9, 0.75, 1.25)):  # R, G, B, NIR ordering
        data[band] = np.rint(np.clip((base + texture) * gain, 0.0, 1.0) * dn_scale).astype("uint16")
    profile = dict(driver="GTiff", width=size, height=size, count=4, dtype="uint16",
                   crs="EPSG:32643", transform=from_origin(500000, 5330000, 10, 10),
                   nodata=0, compress="deflate")
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)
        ds.descriptions = ("B04", "B03", "B02", "B08")
    return path


def crop_raster(source, dest, size, offset=(0, 0)):
    """Crop a square window from a raster, preserving CRS, transform, dtype and nodata."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(source) as src:
        width = min(size, src.width - offset[0])
        height = min(size, src.height - offset[1])
        window = Window(offset[0], offset[1], width, height)
        profile = src.profile.copy()
        profile.update(width=width, height=height, transform=src.window_transform(window), compress="deflate")
        if not profile.get("tiled"):
            # A copied strip size is meaningless for the crop and upsets GDAL.
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        data = src.read(window=window)
        descriptions = tuple(src.descriptions or ())
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data)
        if len(descriptions) == 4 and all(descriptions):
            dst.descriptions = descriptions
    return dest, (width, height)


def fetch_sentinel2_chip(dest, bbox, datetime_range, size):
    """Build a real 4-band chip (B04, B03, B02, B08) from public Sentinel-2 L2A COGs.

    Queries the openly accessible Earth Search STAC API and reads a window from the
    public ``sentinel-cogs`` assets. NOTE: this network path was not executed from the
    development machine (no network run was performed there), and it reads the window
    from the scene origin rather than the requested bbox - it only needs to yield real
    Sentinel-2 pixels for a smoke test. Use ``--raster`` for a controlled chip.
    """
    import numpy as np
    import rasterio
    import requests
    from rasterio.windows import Window

    west, south, east, north = (float(value) for value in str(bbox).split(","))
    payload = {"collections": [STAC_COLLECTION], "bbox": [west, south, east, north],
               "datetime": datetime_range, "limit": 5,
               "query": {"eo:cloud_cover": {"lt": 30}}}
    response = requests.post(STAC_SEARCH_URL, json=payload, timeout=60)
    if response.status_code >= 400:  # some deployments reject the query extension
        payload.pop("query", None)
        response = requests.post(STAC_SEARCH_URL, json=payload, timeout=60)
    response.raise_for_status()
    features = response.json().get("features") or []
    if not features:
        raise RuntimeError("STAC search returned no scenes; widen --bbox or --datetime")
    assets = features[0]["assets"]
    bands = []
    profile = None
    for band in ("B04", "B03", "B02", "B08"):
        if band not in assets:
            raise RuntimeError(f"scene {features[0].get('id')} has no {band} asset")
        with rasterio.open(assets[band]["href"]) as ds:
            window = Window(0, 0, min(size, ds.width), min(size, ds.height))
            bands.append(ds.read(1, window=window).astype("uint16"))
            if profile is None:
                profile = ds.profile.copy()
                profile.update(width=int(window.width), height=int(window.height), count=4,
                               transform=ds.window_transform(window), nodata=0, compress="deflate")
    data = np.stack(bands)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ("B04", "B03", "B02", "B08")
    return dest, features[0].get("id", "unknown"), (int(data.shape[-1]), int(data.shape[-2]))


def safe_correlation(first, second):
    """Informational Pearson correlation, or None when a side has no variation."""
    import numpy as np

    if float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return None
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def verify_output(path, source, dn_scale, opensr, scale=4):
    """Compare a written GeoTIFF with its source contract; returns (checks, values)."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as out, rasterio.open(source) as src:
        checks = {
            "dimensions_4x": (out.width, out.height) == (src.width * scale, src.height * scale),
            "band_count_4": out.count == 4,
            "dtype_uint16": out.dtypes[0] == "uint16",
            "crs_preserved": out.crs == src.crs,
            "nodata_preserved": out.nodata == src.nodata,
            "pixel_size_quarter": (abs(out.transform.a - src.transform.a / scale) < 1e-9
                                   and abs(out.transform.e - src.transform.e / scale) < 1e-9),
            "origin_preserved": (abs(out.transform.c - src.transform.c) < 1e-9
                                 and abs(out.transform.f - src.transform.f) < 1e-9),
            "band_descriptions": tuple(out.descriptions) == tuple(opensr.SENTINEL2_BAND_ORDER),
            "model_tag": out.tags().get("OPENSR_MODEL") == opensr.MODEL_ID,
        }
        data = out.read()
    checks["dn_range"] = bool(int(data.min()) >= 0 and int(data.max()) <= dn_scale)
    checks["not_empty"] = bool(int(data.max()) > 0)
    checks["has_variation"] = bool(int(data.max()) != int(data.min()))
    values = {"min_dn": int(data.min()), "max_dn": int(data.max()),
              "mean_dn": round(float(data.mean()), 2), "shape": list(data.shape)}
    return checks, values


def api_request_example(image_id, args):
    """The exact API payload this harness sends, for manual reproduction."""
    parameters = {"sampling_steps": args.sampling_steps}
    if args.overlap >= 0:
        parameters["overlap"] = args.overlap
    if args.batch_size:
        parameters["batch_size"] = args.batch_size
    return {"image_id": image_id, "method": "opensr_ldsrs2", "scale_factor": 4, "parameters": parameters}


def curl_example(image_id, args, base_url="http://127.0.0.1:8000"):
    """Equivalent curl commands for the same run against a live server."""
    body = json.dumps(api_request_example(image_id, args))
    return [
        f"curl -s -F file=@chip.tif {base_url}/api/imagery/upload",
        f"curl -s -X POST {base_url}/api/processing/start -H 'Content-Type: application/json' -d '{body}'",
        f"curl -s {base_url}/api/jobs/<job_id>/result",
        f"curl -s -o sr.tif {base_url}/api/jobs/<job_id>/download",
    ]


def run_preflight(args, settings, opensr, opensr_pipeline, report, source, origin, dn_scale):
    """Warm-up checks: environment, checkpoint, raster contract, tile plan, API payload."""
    import rasterio

    phase = Phase("preflight")
    start = time.time()
    status = opensr.status(probe_device=False)
    checks = {"torch_installed": bool(status["torch_installed"]),
              "opensr_model_installed": bool(status["package_installed"]),
              "checkpoint_configured": opensr.checkpoint_path() is not None,
              "config_yaml_found": opensr.package_config_path() is not None}
    phase.values = {"source": str(source), "source_origin": origin, "sampling_steps": args.sampling_steps,
                    "production_sampling_steps": int(settings.opensr_sampling_steps)}
    try:
        with rasterio.open(source) as ds:
            checks["four_bands"] = ds.count == 4
            checks["has_crs"] = ds.crs is not None
            checks["pixels_ge_window"] = min(ds.width, ds.height) >= opensr.MIN_WINDOW
            window = opensr.configured_window_size()
            overlap = args.overlap if args.overlap >= 0 else int(settings.opensr_overlap)
            plan = opensr_pipeline.tile_plan(ds.height, ds.width, window, overlap)
            phase.tiles = len(plan)
            checks["tile_count_within_limit"] = len(plan) <= int(settings.opensr_max_tiles)
            checks["validate_source"] = (opensr_pipeline.validate_source(ds, settings.opensr_dn_scale) == dn_scale)
            phase.values.update({"size": [ds.width, ds.height], "bands": ds.count, "dtype": ds.dtypes[0],
                                 "crs": str(ds.crs), "resolution": list(ds.res), "nodata": ds.nodata,
                                 "window": window, "overlap": overlap, "tile_count": len(plan),
                                 "input_dn_scale": dn_scale})
    except Exception as exc:
        checks["raster_usable"] = False
        phase.detail = f"raster check failed: {type(exc).__name__}: {exc}"
    phase.checks = checks
    phase.seconds = time.time() - start
    if not phase.detail:
        phase.detail = "environment, checkpoint, raster contract and tile plan verified"
    phase.status = "passed" if all(checks.values()) else "failed"
    return phase


def run_tile(args, opensr, opensr_pipeline, work, source, dn_scale):
    """One-tile smoke test: read a single LR window and super-resolve it once."""
    import numpy as np
    import rasterio

    phase = Phase("tile")
    torch = torch_module_or_none()
    cuda = bool(torch is not None and torch.cuda.is_available())
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    chip, _ = crop_raster(source, work / "one_tile.tif", args.tile_size)
    checks = {}
    try:
        with rasterio.open(chip) as ds:
            block = opensr_pipeline.read_tile(ds, opensr_pipeline.Tile(0, 0, 0, 0), args.tile_size)
        checks["window_exact"] = (block.shape[-1], block.shape[-2]) == (args.tile_size, args.tile_size)
        checks["four_bands"] = block.shape[0] == 4
        start = time.time()
        super_resolved = opensr.super_resolve(block, dn_scale=dn_scale, sampling_steps=args.sampling_steps)
        phase.seconds = time.time() - start
        checks.update({
            "output_shape_4x": tuple(super_resolved.shape) == (4, args.tile_size * 4, args.tile_size * 4),
            "dtype_float32": super_resolved.dtype == np.float32,
            "finite": bool(np.isfinite(super_resolved).all()),
            "reflectance_0_1": bool(float(super_resolved.min()) >= 0.0 and float(super_resolved.max()) <= 1.0),
            "not_all_zero": bool(float(super_resolved.max()) > 0.0)})
        low_res = opensr.to_reflectance(block, dn_scale)
        nearest = np.repeat(np.repeat(low_res, 4, axis=1), 4, axis=2)
        correlations = [safe_correlation(nearest[band], super_resolved[band]) for band in range(4)]
        phase.values = {"sampling_steps": args.sampling_steps, "seconds": round(phase.seconds, 2),
                        "seconds_per_step": round(phase.seconds / max(args.sampling_steps, 1), 2),
                        "output_shape": list(super_resolved.shape),
                        "output_min": round(float(super_resolved.min()), 4),
                        "output_max": round(float(super_resolved.max()), 4),
                        "output_mean": round(float(super_resolved.mean()), 4),
                        "pearson_vs_nearest_neighbour_upscale":
                            [None if value is None else round(value, 3) for value in correlations],
                        "note": "correlation is an informational sanity signal, not an image-quality metric"}
    except Exception as exc:
        checks["inference_ran"] = False
        phase.detail = f"{type(exc).__name__}: {exc}"
    phase.checks = checks
    if not phase.detail:
        phase.detail = (f"one {args.tile_size}px tile at {args.sampling_steps} DDIM steps"
                        + ("" if cuda else " on CPU (NOT GPU VALIDATION)"))
    phase.status = "passed" if all(checks.values()) else "failed"
    phase.gpu_peak_allocated_mb, phase.gpu_peak_reserved_mb = peak_memory(torch)
    return phase


def multi_size(args, settings, opensr):
    """LR side length that yields multiple tiles: window + stride gives four of them."""
    if args.multi_size:
        return int(args.multi_size)
    window = opensr.configured_window_size()
    overlap = args.overlap if args.overlap >= 0 else int(settings.opensr_overlap)
    return window + max(window - overlap, 1)


def job_parameters(args, settings):
    """Per-job ``parameters`` sent to the pipeline/API (never production defaults)."""
    parameters = {"sampling_steps": args.sampling_steps}
    if args.overlap >= 0:
        parameters["overlap"] = args.overlap
    if args.batch_size:
        parameters["batch_size"] = args.batch_size
    return parameters


def run_pipeline(args, settings, opensr, opensr_pipeline, work, source):
    """Full pipeline test: tiled inference, GeoTIFF write, preview, temp cleanup."""
    phase = Phase("pipeline")
    torch = torch_module_or_none()
    cuda = bool(torch is not None and torch.cuda.is_available())
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    size = multi_size(args, settings, opensr)
    chip, _ = crop_raster(source, work / f"multi_{size}.tif", size)
    parameters = job_parameters(args, settings)
    job_id = f"gpuvalid{int(time.time())}"
    job = {"id": job_id, "method": opensr.MODEL_ID, "scale_factor": opensr.SCALE_FACTOR, "parameters": parameters}
    out_path = settings.data_dir / "outputs" / f"{job_id}.tif"
    preview_path = settings.data_dir / "previews" / f"{job_id}.png"
    temp_path = settings.data_dir / "temporary" / f"{job_id}.tif"
    checks = {}
    try:
        start = time.time()
        result = opensr_pipeline.process(chip, job)
        phase.seconds = time.time() - start
        phase.tiles = int(result["provenance"]["tiles"])
        checks["multiple_tiles"] = phase.tiles > 1
        checks["provenance_device"] = result["provenance"]["device"] == ("cuda" if cuda else "cpu")
        checks["provenance_steps"] = result["provenance"]["sampling_steps"] == args.sampling_steps
        checks["provenance_bands"] = result["provenance"]["input_bands"] == list(opensr.SENTINEL2_BAND_ORDER)
        checks["output_metadata"] = bool(result["output"]["crs"]) and result["output"]["bands"] == 4
        checks["disclaimers_present"] = bool(result.get("limitations")) and bool(result.get("provenance"))
        checks["output_path_matches"] = str(out_path) == result["output_path"]
        output_checks, output_values = verify_output(out_path, chip, settings.opensr_dn_scale, opensr)
        checks.update(output_checks)
        checks["preview_written"] = preview_path.is_file() and preview_path.stat().st_size > 0
        checks["temp_cleaned"] = not temp_path.exists()
        phase.values = {"tiles": phase.tiles, "seconds": round(phase.seconds, 2),
                        "seconds_per_tile": round(phase.seconds / max(phase.tiles, 1), 2),
                        "parameters": parameters, "output": output_values, "output_path": str(out_path),
                        "preview_path": str(preview_path),
                        "preview_bytes": preview_path.stat().st_size if preview_path.is_file() else 0,
                        "provenance": result["provenance"]}
    except Exception as exc:
        checks["pipeline_ran"] = False
        phase.detail = f"{type(exc).__name__}: {exc}"
    finally:
        if not args.keep_outputs:
            for path in (out_path, preview_path, temp_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
    phase.checks = checks
    if not phase.detail:
        phase.detail = (f"{phase.tiles or 0} tiles through opensr_pipeline.process"
                        + ("" if cuda else " on CPU (NOT GPU VALIDATION)"))
    phase.status = "passed" if all(checks.values()) else "failed"
    phase.gpu_peak_allocated_mb, phase.gpu_peak_reserved_mb = peak_memory(torch)
    return phase


def run_api(args, settings, opensr, work, source):
    """End-to-end test through the real FastAPI routes with an in-process TestClient."""
    from fastapi.testclient import TestClient
    from app.main import app

    phase = Phase("api")
    torch = torch_module_or_none()
    cuda = bool(torch is not None and torch.cuda.is_available())
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    size = multi_size(args, settings, opensr)
    chip, _ = crop_raster(source, work / f"api_{size}.tif", size)
    client = TestClient(app)
    checks = {}
    job_id = None
    try:
        start = time.time()
        with open(chip, "rb") as handle:
            uploaded = client.post("/api/imagery/upload",
                                   files={"file": (chip.name, handle.read(), "image/tiff")})
        checks["upload_ok"] = uploaded.status_code == 200
        image_id = uploaded.json().get("id", "") if checks["upload_ok"] else ""
        payload = api_request_example(image_id, args)
        started = client.post("/api/processing/start", json=payload)
        checks["start_accepted"] = started.status_code == 202
        job_id = started.json().get("id") if checks["start_accepted"] else None
        job = client.get(f"/api/jobs/{job_id}").json() if job_id else {}
        checks["job_completed"] = job.get("status") == "completed"
        if not checks["job_completed"]:
            phase.detail = f"job status={job.get('status')} error={job.get('error')}"
        if checks["job_completed"]:
            import rasterio
            from rasterio.io import MemoryFile
            downloaded = client.get(f"/api/jobs/{job_id}/download")
            checks["download_ok"] = downloaded.status_code == 200
            if checks["download_ok"]:
                with MemoryFile(downloaded.content) as mem, mem.open() as ds:
                    checks["output_size_4x"] = (ds.width, ds.height) == (size * 4, size * 4)
                    checks["output_crs_present"] = ds.crs is not None
                    checks["output_bands_dtype"] = ds.count == 4 and ds.dtypes[0] == "uint16"
            checks["preview_ok"] = client.get(f"/api/jobs/{job_id}/preview").status_code == 200
            body = client.get(f"/api/jobs/{job_id}/result").json()
            checks["result_has_provenance"] = "provenance" in body.get("result", {})
            checks["result_hides_paths"] = ("output_path" not in body.get("result", {})
                                            and "preview_path" not in body.get("result", {}))
            phase.tiles = body.get("result", {}).get("provenance", {}).get("tiles")
        phase.seconds = time.time() - start
        phase.values = {"request": payload, "tiles": phase.tiles,
                        "curl_equivalent": curl_example(image_id, args)}
    except Exception as exc:
        checks["api_flow_ran"] = False
        phase.detail = f"{type(exc).__name__}: {exc}"
    finally:
        if job_id:  # removes the job metadata, output and preview
            try:
                client.delete(f"/api/jobs/{job_id}")
            except Exception:
                pass
    phase.checks = checks
    if not phase.detail:
        phase.detail = ("upload -> processing/start -> download through the API"
                        + ("" if cuda else " on CPU (NOT GPU VALIDATION)"))
    phase.status = "passed" if all(checks.values()) else "failed"
    phase.gpu_peak_allocated_mb, phase.gpu_peak_reserved_mb = peak_memory(torch)
    return phase


def write_report(report, args):
    """Print a compact summary table and optionally write the JSON report."""
    print("\n--- summary ---")
    for phase in report.phases:
        line = f"{phase.name:<12} {phase.status:<9}"
        if phase.seconds:
            line += f" {phase.seconds:8.1f}s"
        if phase.tiles:
            line += f" tiles={phase.tiles}"
        if phase.gpu_peak_allocated_mb is not None:
            line += f" gpu_peak_alloc={phase.gpu_peak_allocated_mb}MB reserved={phase.gpu_peak_reserved_mb}MB"
        print(line)
    for note in report.notes:
        print(f"note: {note}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
        print(f"report written to {args.json}")


def main(argv=None):
    args = parse_args(argv)
    report = Report()
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir()) / "opensr_gpu_validation"
    work.mkdir(parents=True, exist_ok=True)
    applied = apply_environment(args)
    settings, opensr, opensr_pipeline = import_stack()
    environment = gpu_report(settings, opensr)
    environment["env_supplied_by_script"] = applied
    environment["work_dir"] = str(work)
    report.environment = environment

    print("=" * 78)
    print("OpenSR GPU validation (Phase 3)")
    print(DISCLAIMER)
    print("=" * 78)
    print(json.dumps({key: value for key, value in environment.items() if key != "opensr_status"},
                     indent=2, default=str))
    print("OpenSR status:", json.dumps(environment["opensr_status"], indent=2, default=str))
    cuda = bool(environment.get("cuda_available"))

    if args.synthetic:
        source = synthetic_sentinel2(work / f"synthetic_{args.source_size}.tif", args.source_size)
        origin = "synthetic"
        report.notes.append("SYNTHETIC input raster: validates mechanics only, not real Sentinel-2 imagery.")
    elif args.fetch_stac:
        source, scene, _ = fetch_sentinel2_chip(work / "stac_chip.tif", args.bbox,
                                                args.datetime_range, args.source_size)
        origin = f"STAC scene {scene}"
        report.notes.append("Chip built from public Sentinel-2 L2A COGs (scene-origin window, not bbox-centred).")
    elif args.raster:
        source = args.raster.resolve()
        origin = "operator-supplied file"
    else:
        print("ERROR: choose an input raster: --raster PATH, --fetch-stac or --synthetic")
        return 2
    if not Path(source).is_file():
        print(f"ERROR: raster not found: {source}")
        return 2

    try:
        import rasterio
        with rasterio.open(source) as dataset:
            dn_scale = opensr_pipeline.validate_source(dataset, settings.opensr_dn_scale)
    except Exception as exc:
        report.add(Phase("preflight", status="failed", detail=f"source raster rejected: {exc}"))
        write_report(report, args)
        return 1

    preflight = run_preflight(args, settings, opensr, opensr_pipeline, report, Path(source), origin, dn_scale)
    if args.dry_run and not preflight.ok():
        # Show the failing checks without calling the run a failure: nothing was attempted.
        preflight.status = "dry-run"
    report.add(preflight)
    if args.dry_run:
        print("\nDRY RUN - no inference was attempted. Planned job/API request:")
        print("  " + json.dumps(api_request_example("<uploaded-image-id>", args)))
        for line in curl_example("<uploaded-image-id>", args):
            print("  " + line)
        print(f"  direct call: opensr_pipeline.process({str(source)!r}, {{'id': '<job id>', "
              f"'method': 'opensr_ldsrs2', 'scale_factor': 4, 'parameters': "
              f"{json.dumps(job_parameters(args, settings))}}})")
        write_report(report, args)
        return 0
    if not cuda and not args.allow_cpu:
        report.add(Phase("cuda", status="not-run",
                         detail="CUDA is unavailable, so this is NOT a GPU validation. Re-run on a GPU host or "
                                "Colab, or pass --allow-cpu for a clearly-labelled CPU smoke test."))
        write_report(report, args)
        print("\nNOT RUN - no CUDA device. " + DISCLAIMER)
        return 2

    if not preflight.ok():
        write_report(report, args)
        print("\nFAILED preflight - fix the checks above before running inference.")
        return 1

    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        if mode == "tile":
            phase = report.add(run_tile(args, opensr, opensr_pipeline, work, Path(source), dn_scale))
        elif mode == "pipeline":
            phase = report.add(run_pipeline(args, settings, opensr, opensr_pipeline, work, Path(source)))
        elif mode == "api":
            phase = report.add(run_api(args, settings, opensr, work, Path(source)))
        else:
            print(f"WARNING: unknown mode '{mode}' (expected tile, pipeline, api)")
            continue
        if not phase.ok():
            print(f"\nABORT: phase '{mode}' failed; later phases depend on it.")
            break

    ran = [phase for phase in report.phases if phase.status in ("passed", "failed")]
    passed = bool(ran) and all(phase.ok() for phase in ran)
    write_report(report, args)
    print("\n" + "=" * 78)
    if cuda and passed:
        print("GPU VALIDATION PASSED - functional smoke test only")
    elif cuda:
        print("GPU VALIDATION FAILED - see the failing phase above")
    elif passed:
        print("CPU SMOKE TEST COMPLETED - THIS IS NOT A GPU VALIDATION")
    else:
        print("RUN FAILED - see the failing phase above")
    print(DISCLAIMER)
    print("Production defaults were unchanged (OPENSR_SAMPLING_STEPS stays "
          f"{environment.get('opensr_sampling_steps_default')}); nothing was deployed.")
    return 0 if passed else 1


def import_stack():
    """Import the application modules only after the environment is configured."""
    from app import opensr_pipeline
    from app.config import settings
    from app.ml import opensr
    return settings, opensr, opensr_pipeline


def gpu_report(settings, opensr):
    """Describe the environment: torch/CUDA/device plus resolved OpenSR configuration."""
    info = {"opensr_window": int(settings.opensr_window),
            "opensr_overlap_default": int(settings.opensr_overlap),
            "opensr_sampling_steps_default": int(settings.opensr_sampling_steps),
            "opensr_dn_scale": float(settings.opensr_dn_scale),
            "opensr_device_requested": settings.opensr_device,
            "opensr_max_tiles": int(settings.opensr_max_tiles),
            "opensr_threads": int(settings.opensr_torch_threads)}
    status = opensr.status(probe_device=True)
    info["opensr_status"] = {key: status[key] for key in
                             ("enabled", "available", "reason", "device", "cuda_available", "torch_installed",
                              "package_installed", "checkpoint", "checkpoint_exists", "checkpoint_size_bytes",
                              "config_path", "config_exists")}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu_name"] = properties.name
            info["gpu_total_memory_mb"] = round(properties.total_memory / 1048576, 1)
            info["gpu_compute_capability"] = f"{properties.major}.{properties.minor}"
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def peak_memory(torch_module):
    """(allocated, reserved) peak CUDA memory in MB, or (None, None) without CUDA."""
    if torch_module is None or not torch_module.cuda.is_available():
        return None, None
    return (round(torch_module.cuda.max_memory_allocated() / 1048576, 1),
            round(torch_module.cuda.max_memory_reserved() / 1048576, 1))


def torch_module_or_none():
    """Import torch for memory statistics only; never loads the model by itself."""
    try:
        import torch
        return torch
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
