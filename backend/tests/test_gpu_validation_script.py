"""Tests for scripts/gpu_validation.py (Phase 3 harness).

These never load the model, never run inference and never need CUDA: they exercise the
harness helpers and the ``--dry-run`` path, which must work even when torch and
opensr-model are absent.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = BACKEND / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpu_validation as gv  # noqa: E402
from app.ml import opensr  # noqa: E402

SCRIPT = SCRIPTS / "gpu_validation.py"


def build_output(source, path, source_size=128, scale=4, bands=4, descriptions=True, tagged=True):
    """Write a synthetic 'pipeline output' that should (or should not) pass verification."""
    size = source_size * scale
    ramp = np.rint(np.linspace(1000, 4000, size)).astype("uint16")
    data = np.tile(ramp, (bands, size, 1))
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(width=size, height=size, count=bands, dtype="uint16",
                       transform=src.transform @ Affine.scale(1 / scale))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = tuple(opensr.SENTINEL2_BAND_ORDER[:bands])
        if tagged:
            dst.update_tags(OPENSR_MODEL=opensr.MODEL_ID)
    return path


def test_cli_defaults_are_smoke_test_friendly():
    args = gv.parse_args([])
    assert args.sampling_steps == gv.DEFAULT_SAMPLING_STEPS == 2
    assert args.tile_size == gv.DEFAULT_TILE_SIZE == 128
    assert args.allow_cpu is False and args.dry_run is False
    assert [mode.strip() for mode in args.modes.split(",")] == ["tile", "pipeline", "api"]


def test_environment_only_supplies_missing_values(monkeypatch, tmp_path):
    for name in ("OPENSR_ENABLED", "OPENSR_CHECKPOINT", "OPENSR_DEVICE", "OPENSR_WINDOW"):
        monkeypatch.delenv(name, raising=False)
    checkpoint = tmp_path / "ckpt.ckpt"
    args = gv.parse_args(["--checkpoint", str(checkpoint), "--device", "cuda"])
    applied = gv.apply_environment(args)
    assert applied["OPENSR_ENABLED"] == "true" and applied["OPENSR_DEVICE"] == "cuda"
    assert os.environ["OPENSR_CHECKPOINT"] == str(checkpoint.resolve())
    # operator values win
    monkeypatch.setenv("OPENSR_ENABLED", "false")
    assert "OPENSR_ENABLED" not in gv.apply_environment(args)
    assert os.environ["OPENSR_ENABLED"] == "false"


def test_synthetic_raster_is_four_band_sentinel2_shaped(tmp_path):
    path = gv.synthetic_sentinel2(tmp_path / "synth.tif", 244)
    with rasterio.open(path) as ds:
        assert ds.count == 4 and ds.dtypes[0] == "uint16"
        assert (ds.width, ds.height) == (244, 244)
        assert ds.crs is not None and ds.crs.is_projected
        assert ds.res == (10.0, 10.0)
        assert ds.nodata == 0
        assert tuple(ds.descriptions) == opensr.SENTINEL2_BAND_ORDER
        data = ds.read()
    assert data.max() > 0 and data.min() >= 0


def test_crop_preserves_georeferencing(tmp_path):
    source = gv.synthetic_sentinel2(tmp_path / "synth.tif", 244)
    chip, size = gv.crop_raster(source, tmp_path / "chip.tif", 128)
    assert size == (128, 128)
    with rasterio.open(source) as src, rasterio.open(chip) as cropped:
        assert (cropped.width, cropped.height) == (128, 128)
        assert cropped.crs == src.crs
        assert cropped.transform.c == src.transform.c and cropped.transform.f == src.transform.f
        assert cropped.transform.a == src.transform.a and cropped.transform.e == src.transform.e
        assert cropped.nodata == src.nodata


def test_crop_at_offset_shifts_the_transform(tmp_path):
    source = gv.synthetic_sentinel2(tmp_path / "synth.tif", 244)
    chip, _ = gv.crop_raster(source, tmp_path / "chip.tif", 128, offset=(60, 0))
    with rasterio.open(source) as src, rasterio.open(chip) as cropped:
        assert cropped.transform.c == src.transform.c + 60 * src.transform.a


def test_verify_output_accepts_a_correct_output(tmp_path):
    source, _ = gv.crop_raster(gv.synthetic_sentinel2(tmp_path / "synth.tif", 244), tmp_path / "chip.tif", 128)
    output = build_output(source, tmp_path / "out.tif", 128)
    checks, values = gv.verify_output(output, source, 10000.0, opensr)
    assert all(checks.values()), checks
    assert values["max_dn"] == 4000 and values["shape"] == [4, 512, 512]


def test_verify_output_detects_contract_violations(tmp_path):
    source, _ = gv.crop_raster(gv.synthetic_sentinel2(tmp_path / "synth.tif", 244), tmp_path / "chip.tif", 128)
    checks, _ = gv.verify_output(build_output(source, tmp_path / "small.tif", source_size=64),
                                 source, 10000.0, opensr)
    assert checks["dimensions_4x"] is False
    checks, _ = gv.verify_output(build_output(source, tmp_path / "three.tif", bands=3),
                                 source, 10000.0, opensr)
    assert checks["band_count_4"] is False
    checks, _ = gv.verify_output(build_output(source, tmp_path / "plain.tif", descriptions=False, tagged=False),
                                 source, 10000.0, opensr)
    assert checks["band_descriptions"] is False and checks["model_tag"] is False
    checks, _ = gv.verify_output(build_output(source, tmp_path / "hot.tif", 128, scale=4),
                                 source, 100.0, opensr)  # dn_scale below the written DN
    assert checks["dn_range"] is False


def test_multi_size_produces_a_multi_tile_plan(monkeypatch):
    from app import opensr_pipeline
    from app.config import settings

    monkeypatch.setattr(settings, "opensr_window", 128)
    monkeypatch.setattr(settings, "opensr_overlap", 12)
    args = gv.parse_args(["--raster", "unused.tif"])
    size = gv.multi_size(args, settings, opensr)
    assert size == 244
    assert len(opensr_pipeline.tile_plan(size, size, 128, 12)) == 4
    assert len(opensr_pipeline.tile_plan(128, 128, 128, 12)) == 1


def test_job_parameters_use_low_steps_and_leave_production_defaults_alone():
    from app.config import Settings, settings

    assert Settings.model_fields["opensr_sampling_steps"].default == 100  # production default
    args = gv.parse_args(["--raster", "x.tif"])
    assert gv.job_parameters(args, settings) == {"sampling_steps": 2}
    args = gv.parse_args(["--raster", "x.tif", "--overlap", "4", "--batch-size", "2"])
    assert gv.job_parameters(args, settings) == {"sampling_steps": 2, "overlap": 4, "batch_size": 2}


def test_api_request_and_curl_examples_match_the_documented_contract():
    args = gv.parse_args(["--raster", "x.tif"])
    assert gv.api_request_example("img1", args) == {
        "image_id": "img1", "method": "opensr_ldsrs2", "scale_factor": 4,
        "parameters": {"sampling_steps": 2}}
    commands = gv.curl_example("img1", args)
    assert len(commands) == 4
    assert any("/api/processing/start" in command for command in commands)
    assert any("/api/jobs/<job_id>/download" in command for command in commands)


def run_script(tmp_path, *extra):
    env = dict(os.environ, OPENSR_ENABLED="false", OPENSR_CHECKPOINT="")
    return subprocess.run([sys.executable, str(SCRIPT), *extra, "--work-dir", str(tmp_path / "work")],
                          cwd=str(BACKEND), env=env, capture_output=True, text=True, timeout=300)


def test_dry_run_reports_the_plan_without_inference(tmp_path):
    done = run_script(tmp_path, "--dry-run", "--synthetic", "--source-size", "244")
    assert done.returncode == 0, done.stderr
    assert "DRY RUN" in done.stdout and "tiles=4" in done.stdout
    assert "GPU VALIDATION PASSED" not in done.stdout   # a dry run never claims a verdict
    assert "[  PASSED] tile" not in done.stdout         # no inference phase ran


def test_dry_run_writes_a_json_report(tmp_path):
    report_path = tmp_path / "report.json"
    done = run_script(tmp_path, "--dry-run", "--synthetic", "--source-size", "244", "--json", str(report_path))
    assert done.returncode == 0, done.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["disclaimer"].startswith("Smoke test only")
    assert payload["phases"][0]["name"] == "preflight"
    assert payload["phases"][0]["tiles"] == 4
    assert "env_supplied_by_script" in payload["environment"]


def test_missing_input_source_is_rejected(tmp_path):
    done = run_script(tmp_path)
    assert done.returncode == 2
    assert "choose an input raster" in done.stdout


def test_missing_raster_file_is_rejected(tmp_path):
    done = run_script(tmp_path, "--dry-run", "--raster", str(tmp_path / "absent.tif"))
    assert done.returncode == 2
    assert "raster not found" in done.stdout


def test_non_cuda_host_stops_without_allow_cpu(tmp_path):
    """Without CUDA and without --allow-cpu the harness must stop instead of using the CPU."""
    try:
        import torch
    except Exception:
        pytest.skip("torch is not importable in this environment")
    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; the no-CUDA gate only applies to GPU-less hosts")
    done = run_script(tmp_path, "--synthetic", "--source-size", "244")
    assert done.returncode == 2
    assert "NOT RUN" in done.stdout and "no CUDA device" in done.stdout
    assert "GPU VALIDATION PASSED" not in done.stdout
