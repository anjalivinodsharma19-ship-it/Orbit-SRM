"""Phase 1 tests for the optional OpenSR wrapper.

These run in the baseline environment, which has no PyTorch and no opensr-model:
that is exactly the "OpenSR is not installed / not configured / no CUDA" path the
wrapper must handle without breaking the existing API. Nothing here performs
inference, loads weights, or downloads a checkpoint.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.config import Settings, settings
from app.ml import opensr

BACKEND = Path(__file__).resolve().parents[1]


class _FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeTorch:
    """Stand-in for torch so device resolution can be tested without PyTorch."""

    def __init__(self, cuda_available):
        self.cuda = _FakeCuda(cuda_available)


@pytest.fixture(autouse=True)
def opensr_defaults(monkeypatch):
    """Force default (disabled) OpenSR settings and an empty model cache."""
    monkeypatch.setattr(settings, "opensr_enabled", False)
    monkeypatch.setattr(settings, "opensr_checkpoint", "")
    monkeypatch.setattr(settings, "opensr_device", "auto")
    monkeypatch.setattr(settings, "opensr_sampling_steps", 100)
    monkeypatch.setattr(settings, "opensr_max_sampling_steps", 100)
    monkeypatch.setattr(settings, "opensr_window", 128)
    monkeypatch.setattr(settings, "opensr_max_window_size", 256)
    monkeypatch.setattr(settings, "opensr_batch_size", 1)
    monkeypatch.setattr(settings, "opensr_max_batch_size", 4)
    monkeypatch.setattr(settings, "opensr_torch_threads", 0)
    monkeypatch.setitem(opensr._CACHE, "model", None)
    monkeypatch.setitem(opensr._CACHE, "device", None)
    monkeypatch.setitem(opensr._CACHE, "torch", None)


def test_contract_constants_match_the_audited_model():
    assert opensr.MODEL_ID == "opensr_ldsrs2"
    assert opensr.SENTINEL2_BAND_ORDER == ("B04", "B03", "B02", "B08")
    assert opensr.EXPECTED_BANDS == 4
    assert opensr.SCALE_FACTOR == 4
    assert opensr.MIN_WINDOW == 128
    assert opensr.DEFAULT_SAMPLING_STEPS == 100
    assert opensr.REFLECTANCE_SCALE == 10000.0
    assert opensr.PACKAGE_CONFIG == "configs/config_10m.yaml"

def test_resource_limit_defaults_preserve_existing_runtime_defaults():
    assert Settings.model_fields["opensr_batch_size"].default == 1
    assert Settings.model_fields["opensr_sampling_steps"].default == 100
    assert Settings.model_fields["opensr_max_batch_size"].default == 4
    assert Settings.model_fields["opensr_max_sampling_steps"].default == 100
    assert Settings.model_fields["opensr_window"].default == 128
    assert Settings.model_fields["opensr_max_window_size"].default == 256

def test_resource_limits_must_be_positive_integers():
    with pytest.raises(ValidationError):
        Settings(opensr_max_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(opensr_max_sampling_steps=0)

def test_maximum_window_must_cover_default_window():
    with pytest.raises(ValidationError):
        Settings(opensr_max_window_size=127)
    with pytest.raises(ValidationError):
        Settings(opensr_window=300,opensr_max_window_size=256)
    with pytest.raises(ValidationError):
        Settings.model_validate({"opensr_window":128.5})
    with pytest.raises(ValidationError):
        Settings.model_validate({"opensr_max_window_size":256.5})

@pytest.mark.parametrize("window_size",[128,256])
def test_configured_window_size_accepts_supported_boundaries(window_size):
    assert opensr.configured_window_size(window_size)==window_size

@pytest.mark.parametrize("window_size",[127,257,128.5])
def test_configured_window_size_rejects_invalid_explicit_values(window_size):
    with pytest.raises(opensr.OpenSRUnavailable):
        opensr.configured_window_size(window_size)

@pytest.mark.parametrize("window_size",[127,257,128.5])
def test_configured_window_size_rejects_invalid_defaults(monkeypatch,window_size):
    monkeypatch.setattr(settings,"opensr_window",window_size)
    with pytest.raises(opensr.OpenSRUnavailable):
        opensr.configured_window_size()

@pytest.mark.parametrize("steps",[1,100])
def test_resolve_sampling_steps_accepts_configured_boundaries(steps):
    assert opensr._resolve_steps(steps)==steps

@pytest.mark.parametrize("steps",[0,101,100.5])
def test_resolve_sampling_steps_rejects_out_of_range_values(steps):
    with pytest.raises(opensr.OpenSRInputError):
        opensr._resolve_steps(steps)

def test_super_resolve_batch_rejects_oversize_before_model_load(monkeypatch):
    monkeypatch.setattr(settings,"opensr_max_batch_size",1)
    monkeypatch.setattr(opensr,"load_model",lambda:pytest.fail("model must not load for oversized input batch"))
    blocks=np.zeros((2,4,128,128),dtype=np.float32)
    with pytest.raises(opensr.OpenSRInputError,match="batch size must be <= 1"):
        opensr.super_resolve_batch(blocks)

@pytest.mark.parametrize("steps",[0,101])
def test_super_resolve_rejects_invalid_steps_before_model_load(monkeypatch,steps):
    monkeypatch.setattr(opensr,"load_model",lambda:pytest.fail("model must not load for invalid steps"))
    block=np.zeros((4,128,128),dtype=np.float32)
    with pytest.raises(opensr.OpenSRInputError):
        opensr.super_resolve(block,sampling_steps=steps)

@pytest.mark.parametrize("window_size",[127,257])
def test_super_resolve_rejects_out_of_range_input_before_conversion_or_model_load(monkeypatch,window_size):
    monkeypatch.setattr(opensr,"to_reflectance",lambda *_args,**_kwargs:pytest.fail("conversion must not run"))
    monkeypatch.setattr(opensr,"load_model",lambda:pytest.fail("model must not load"))
    block=np.zeros((4,window_size,window_size),dtype=np.float32)
    with pytest.raises(opensr.OpenSRInputError):
        opensr.super_resolve(block)

def test_super_resolve_batch_rejects_oversized_window_before_conversion_or_model_load(monkeypatch):
    monkeypatch.setattr(opensr,"to_reflectance_batch",lambda *_args,**_kwargs:pytest.fail("conversion must not run"))
    monkeypatch.setattr(opensr,"load_model",lambda:pytest.fail("model must not load"))
    blocks=np.zeros((1,4,257,257),dtype=np.float32)
    with pytest.raises(opensr.OpenSRInputError):
        opensr.super_resolve_batch(blocks)


def test_status_is_disabled_by_default():
    report = opensr.status()
    assert report["enabled"] is False
    assert report["available"] is False
    assert report["loaded"] is False
    assert report["checkpoint_download"] is False
    assert "OPENSR_ENABLED" in report["reason"]
    assert json.loads(json.dumps(report)) == report


def test_status_requires_a_checkpoint(tmp_path):
    settings.opensr_enabled = True
    assert "OPENSR_CHECKPOINT" in opensr.status()["reason"]
    settings.opensr_checkpoint = str(tmp_path / "missing.ckpt")
    report = opensr.status()
    assert report["checkpoint_exists"] is False
    assert "not found" in report["reason"]


def test_status_reports_missing_dependencies_with_a_real_checkpoint(tmp_path):
    ckpt = tmp_path / "opensr-ldsrs2_v1_0_0.ckpt"
    ckpt.write_bytes(b"not a real checkpoint")
    settings.opensr_enabled = True
    settings.opensr_checkpoint = str(ckpt)
    report = opensr.status()
    assert report["checkpoint_exists"] is True
    assert report["checkpoint_size_bytes"] == ckpt.stat().st_size
    assert report["available"] is False
    # This environment has neither torch nor opensr-model; tell the operator which.
    assert ("PyTorch is not installed" in report["reason"]) or ("not installed" in report["reason"])


def test_status_never_raises_on_bad_settings():
    settings.opensr_enabled = True
    settings.opensr_window = 64
    report = opensr.status()
    assert report["available"] is False
    assert "OPENSR_WINDOW" in report["reason"]
    settings.opensr_window = 128
    settings.opensr_sampling_steps = 0
    assert "OPENSR_SAMPLING_STEPS" in opensr.status()["reason"]


@pytest.mark.parametrize("device,cuda_available,expected", [
    ("auto", False, None),
    ("auto", True, "cuda"),
    ("cuda", False, None),
    ("cuda:0", False, None),
    ("cuda:1", True, "cuda:1"),
    ("cpu", False, "cpu"),
    ("tpu", True, None),
])
def test_resolve_device_never_silently_falls_back_to_cpu(device, cuda_available, expected):
    settings.opensr_device = device
    resolved, reason = opensr.resolve_device(_FakeTorch(cuda_available))
    assert resolved == expected
    if expected is None:
        assert reason


def test_resolve_device_explains_the_cpu_rule():
    settings.opensr_device = "auto"
    resolved, reason = opensr.resolve_device(_FakeTorch(False))
    assert resolved is None
    assert "never falls back" in reason


def test_checkpoint_path_resolution(tmp_path):
    assert opensr.checkpoint_path() is None
    settings.opensr_checkpoint = "   "
    assert opensr.checkpoint_path() is None
    settings.opensr_checkpoint = str(tmp_path / "nope.ckpt")
    assert opensr.checkpoint_path() is None
    real = tmp_path / opensr.DEFAULT_CHECKPOINT_NAME
    real.write_bytes(b"x")
    settings.opensr_checkpoint = str(real)
    assert opensr.checkpoint_path() == real


def test_to_reflectance_scales_sentinel2_dn():
    dn = np.zeros((4, 2, 2), dtype=np.uint16)
    dn[0] = 10000
    dn[1] = 5000
    dn[3] = 20000
    out = opensr.to_reflectance(dn, dn_scale=opensr.REFLECTANCE_SCALE)
    assert out.dtype == np.float32 and out.shape == (4, 2, 2)
    assert float(out[0].max()) == pytest.approx(1.0)
    assert float(out[1].min()) == pytest.approx(0.5)
    assert float(out[2].max()) == 0.0
    assert float(out[3].max()) == pytest.approx(1.0)  # clipped


def test_to_reflectance_passes_through_float_reflectance():
    reflect = np.full((4, 3, 3), 0.25, dtype=np.float32)
    assert float(opensr.to_reflectance(reflect)[0, 0, 0]) == pytest.approx(0.25)

def test_to_reflectance_rejects_unmasked_nonfinite_values():
    for value in (np.nan,np.inf,-np.inf):
        reflect=np.full((4,128,128),0.25,dtype=np.float32)
        reflect[0,4,5]=value
        with pytest.raises(opensr.OpenSRInputError,match="unmasked non-finite"):
            opensr.to_reflectance(reflect)


def test_input_validation_rejects_bad_windows():
    with pytest.raises(opensr.OpenSRInputError) as exc:
        opensr.to_reflectance(np.zeros((3, 4, 4), dtype=np.uint16))
    assert "B04" in str(exc.value)
    with pytest.raises(opensr.OpenSRInputError):
        opensr.to_reflectance(np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(opensr.OpenSRInputError):
        opensr.to_reflectance(np.zeros((4, 4, 4), dtype=np.uint16), dn_scale=0)


def test_to_uint16_reflectance_restores_the_dn_scale():
    dn = np.array([[[0]], [[5000]], [[10000]], [[12000]]], dtype=np.uint16)
    out = opensr.to_uint16_reflectance(opensr.to_reflectance(dn, dn_scale=opensr.REFLECTANCE_SCALE))
    assert out.dtype == np.uint16
    assert out.tolist() == [[[0]], [[5000]], [[10000]], [[10000]]]


def test_super_resolve_refuses_when_opensr_is_disabled():
    with pytest.raises(opensr.OpenSRUnavailable):
        opensr.super_resolve(np.zeros((4, 128, 128), dtype=np.uint16), dn_scale=opensr.REFLECTANCE_SCALE)


def test_model_card_is_descriptive_and_reports_disabled():
    card = opensr.model_card()
    assert card["id"] == opensr.MODEL_ID
    assert card["input_bands"] == list(opensr.SENTINEL2_BAND_ORDER)
    assert card["scale_factors"] == [opensr.SCALE_FACTOR]
    assert card["scientific_evaluation_suitable"] is False
    assert card["available"] is False
    assert json.loads(json.dumps(card)) == card


def test_importing_the_module_neither_imports_torch_nor_loads_a_model():
    env = dict(os.environ, OPENSR_ENABLED="false", OPENSR_CHECKPOINT="")
    code = ("import sys; import app.ml.opensr as o; report = o.status(); "
            "print('torch' in sys.modules, report['loaded'], report['enabled'], report['available'])")
    done = subprocess.run([sys.executable, "-c", code], cwd=str(BACKEND), env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", "False", "False", "False"]


def test_phase_two_keeps_existing_models_and_adds_opensr_without_new_routes():
    from app.main import MODELS, app
    paths = {getattr(route, "path", "") for route in app.routes}
    # OpenSR is selected through the existing processing endpoint, not a new route.
    assert not any("opensr" in path.lower() for path in paths)
    assert [model["id"] for model in MODELS] == ["baseline", "trained_srcnn", "opensr_ldsrs2"]
    assert MODELS[0]["id"] == "baseline" and MODELS[1]["id"] == "trained_srcnn"
