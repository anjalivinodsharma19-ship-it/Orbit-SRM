"""Optional ESA OpenSR (LDSR-S2) Sentinel-2 super-resolution wrapper.

Verified provenance (inspected in this project, not assumed):

* Package ``opensr-model``: ``opensr_model.srmodel.SRLatentDiffusion(config, device)``,
  ``.load_pretrained(path)``, ``.forward(x, sampling_steps=..., histogram_matching=...)``.
* Config ``opensr_model/configs/config_10m.yaml`` ships inside the installed
  package (``sampling_steps: 100``, ``apply_normalization: False``,
  ``ckpt_version: opensr-ldsrs2_v1_0_0.ckpt``).
* Checkpoint: a local file configured by the operator. ``load_pretrained`` would
  download from Hugging Face when the file is missing, so the path is validated
  here first and always passed as an absolute path.

Input/output contract: four channels ordered B04, B03, B02, B08, reflectance in
[0, 1] in and out, with 4x output width/height.

Module rules:

* Nothing is imported or loaded at application startup.
* ``torch``/``opensr_model`` are imported lazily inside :func:`load_model`, which
  builds a lock-guarded process-wide singleton.
* The feature is inert unless ``OPENSR_ENABLED`` is true and a checkpoint is set.
* ``OPENSR_DEVICE=auto`` means CUDA only, and never falls back to CPU: CPU
  inference is impractically slow (measured here: ~79 s fixed cost plus ~6.7 s per
  DDIM step per 128x128 window, with roughly 2 GB resident for the weights alone).
  CPU must be requested explicitly with ``OPENSR_DEVICE=cpu``.

Not included in this module: GeoTIFF window tiling/stitching, API route wiring.
"""
import importlib.util
import logging
import threading
from pathlib import Path

import numpy as np

from ..config import settings

log = logging.getLogger(__name__)

MODEL_ID = "opensr_ldsrs2"
MODEL_NAME = "OpenSR LDSR-S2"
ARCHITECTURE = "Latent diffusion super-resolution (4x)"
PACKAGE_NAME = "opensr_model"
PACKAGE_CONFIG = "configs/config_10m.yaml"
DEFAULT_CHECKPOINT_NAME = "opensr-ldsrs2_v1_0_0.ckpt"
SENTINEL2_BAND_ORDER = ("B04", "B03", "B02", "B08")
EXPECTED_BANDS = len(SENTINEL2_BAND_ORDER)
SCALE_FACTOR = 4
MIN_WINDOW = 128
DEFAULT_WINDOW = 128
DEFAULT_SAMPLING_STEPS = 100
REFLECTANCE_SCALE = 10000.0

_CACHE = {}
_LOCK = threading.Lock()


class OpenSRError(RuntimeError):
    """Base class for OpenSR wrapper failures."""


class OpenSRUnavailable(OpenSRError):
    """OpenSR is disabled, incomplete, or has no usable device."""


class OpenSRInputError(OpenSRError):
    """The supplied window does not match the model input contract."""


class OpenSRInferenceError(OpenSRError):
    """The model raised while running inference, including memory exhaustion."""


def _module_present(name):
    """True when a module can be imported, without importing it (stdlib only)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def checkpoint_path():
    """Return the configured checkpoint as a Path, or None when unset/not a file."""
    raw = (settings.opensr_checkpoint or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def validate_band_descriptions(descriptions):
    """Require explicit metadata matching the model's fixed input channel order."""
    actual = tuple(descriptions or ())
    if actual != SENTINEL2_BAND_ORDER:
        raise OpenSRInputError("OpenSR requires GeoTIFF band descriptions in exact order "
                               + ", ".join(SENTINEL2_BAND_ORDER) + f"; found {actual}")


def package_config_path():
    """Absolute path of the config YAML inside the installed package, if present.

    Uses find_spec only, so it is safe to call before importing the package.
    """
    try:
        spec = importlib.util.find_spec(PACKAGE_NAME)
    except Exception:
        spec = None
    if not spec or not spec.origin:
        return None
    candidate = Path(spec.origin).resolve().parent / PACKAGE_CONFIG
    return candidate if candidate.is_file() else None


def configured_window_size(window_size: int | float | str | None = None):
    """Resolve and validate the configured or per-job LR window size."""
    is_default = window_size is None
    raw = settings.opensr_window if is_default else window_size
    name = "OPENSR_WINDOW" if is_default else "parameters.window_size"
    if isinstance(raw, bool):
        raise OpenSRUnavailable(f"{name} must be an integer")
    if not isinstance(raw, (int, float, str)):
        raise OpenSRUnavailable(f"{name} must be an integer; got {raw!r}")
    try:
        size = int(raw)
    except (TypeError, ValueError):
        raise OpenSRUnavailable(f"{name} must be an integer; got {raw!r}") from None
    if not isinstance(raw, str) and raw != size:
        raise OpenSRUnavailable(f"{name} must be an integer; got {raw!r}")
    if size < MIN_WINDOW:
        raise OpenSRUnavailable(f"{name} must be >= {MIN_WINDOW} (got {size}); the model pads smaller inputs to {MIN_WINDOW}")
    if size > settings.opensr_max_window_size:
        raise OpenSRUnavailable(f"{name} must be <= OPENSR_MAX_WINDOW_SIZE="
                                f"{settings.opensr_max_window_size} (got {size})")
    return size


def configured_sampling_steps():
    """Configured number of DDIM sampling steps."""
    steps = int(settings.opensr_sampling_steps)
    if steps < 1:
        raise OpenSRUnavailable(f"OPENSR_SAMPLING_STEPS must be >= 1 (got {steps})")
    if steps > settings.opensr_max_sampling_steps:
        raise OpenSRUnavailable(f"OPENSR_SAMPLING_STEPS must be <= OPENSR_MAX_SAMPLING_STEPS="
                                f"{settings.opensr_max_sampling_steps} (got {steps})")
    return steps


def resolve_device(torch_module):
    """Resolve OPENSR_DEVICE against a torch-like module.

    Returns ``(device, reason)``. ``device`` is None when no usable device exists.
    ``auto`` and ``cuda*`` require CUDA; CPU is only returned when it was requested
    explicitly, so inference never silently falls back to CPU.
    """
    requested = (settings.opensr_device or "auto").strip().lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda", None
        return None, ("CUDA is not available and OPENSR_DEVICE=auto never falls back to CPU; "
                      "set OPENSR_DEVICE=cpu to run on CPU (impractically slow) or provide a CUDA device")
    if requested.startswith("cuda"):
        if not torch_module.cuda.is_available():
            return None, f"OPENSR_DEVICE={requested} was requested but torch reports no CUDA device"
        return requested, None
    if requested == "cpu":
        return "cpu", None
    return None, f"Unsupported OPENSR_DEVICE '{requested}'; use auto, cpu, or cuda"


def status(probe_device=True):
    """Non-raising availability report for health checks, logs, and tooling.

    Deliberately avoids importing torch unless OpenSR is enabled and fully
    configured, so the default (disabled) path stays cheap. Pass
    ``probe_device=False`` to skip the device probe entirely.
    """
    info = {
        "model_id": MODEL_ID, "name": MODEL_NAME, "architecture": ARCHITECTURE,
        "enabled": bool(settings.opensr_enabled),
        "available": False, "reason": None, "loaded": _CACHE.get("model") is not None,
        "device": _CACHE.get("device"), "device_requested": settings.opensr_device,
        "cuda_available": None,
        "torch_installed": _module_present("torch"),
        "package_installed": _module_present(PACKAGE_NAME),
        "checkpoint_configured": bool((settings.opensr_checkpoint or "").strip()),
        "checkpoint": None, "checkpoint_exists": False, "checkpoint_size_bytes": None,
        "checkpoint_download": False,
        "config_path": None, "config_exists": False,
        "window_size": None, "sampling_steps": None, "scale_factor": SCALE_FACTOR,
        "input_bands": list(SENTINEL2_BAND_ORDER),
        "notes": ["Model output is inferred detail and is not scientifically validated in this project.",
                  "Checkpoints are never downloaded automatically.",
                  "OPENSR_DEVICE=auto requires CUDA and never falls back to CPU."],
    }
    try:
        info["sampling_steps"] = configured_sampling_steps()
        info["window_size"] = configured_window_size()
        if not info["enabled"]:
            info["reason"] = ("OpenSR is disabled; set OPENSR_ENABLED=true, OPENSR_CHECKPOINT=<absolute .ckpt path>, "
                              "and OPENSR_DEVICE=cuda (or cpu)")
            return info
        if not info["checkpoint_configured"]:
            info["reason"] = f"OPENSR_CHECKPOINT is not set; point it at a local {DEFAULT_CHECKPOINT_NAME}"
            return info
        info["checkpoint"] = str(Path(settings.opensr_checkpoint).expanduser())
        path = checkpoint_path()
        if path is None:
            info["reason"] = (f"OpenSR checkpoint not found at {info['checkpoint']} "
                              "(checkpoints are never downloaded automatically)")
            return info
        info["checkpoint_exists"] = True
        info["checkpoint_size_bytes"] = path.stat().st_size
        if not info["torch_installed"]:
            info["reason"] = "PyTorch is not installed in this environment; install backend/requirements-opensr.txt"
            return info
        if not info["package_installed"]:
            info["reason"] = f"The '{PACKAGE_NAME}' package is not installed; install backend/requirements-opensr.txt"
            return info
        config = package_config_path()
        info["config_path"] = str(config) if config else None
        info["config_exists"] = config is not None
        if config is None:
            info["reason"] = (f"The installed '{PACKAGE_NAME}' package does not provide {PACKAGE_CONFIG}; "
                              "reinstall opensr-model")
            return info
        if not probe_device:
            info["reason"] = "device probe skipped (probe_device=False)"
            return info
        import torch  # lazy: only reached when the feature is enabled and configured
        info["cuda_available"] = bool(torch.cuda.is_available())
        device, reason = resolve_device(torch)
        if device is None:
            info["reason"] = reason
            return info
        info["device"] = device
        info["available"] = True
        if device == "cpu":
            info["reason"] = "CPU was requested explicitly with OPENSR_DEVICE=cpu; inference is impractically slow"
        return info
    except OpenSRError as exc:
        info["reason"] = str(exc)
        return info
    except Exception as exc:  # reporting must never break the API
        log.warning("OpenSR status check failed: %s", exc)
        info["reason"] = f"OpenSR status check failed: {exc}"
        return info


def is_available():
    """True when OpenSR is enabled, configured, installed, and has a usable device."""
    return bool(status()["available"])


def model_card():
    """Descriptor shaped like the existing /api/models entries.

    Availability reported here is configuration-level only: probing the device
    would import torch in the request path, so the device is checked when a job
    actually executes.
    """
    report = status(probe_device=False)
    configured = bool(report["enabled"] and report["checkpoint_exists"] and report["torch_installed"]
                      and report["package_installed"] and report["config_exists"])
    return {
        "id": MODEL_ID, "name": MODEL_NAME, "architecture": ARCHITECTURE,
        "input_channels": [EXPECTED_BANDS], "input_bands": list(SENTINEL2_BAND_ORDER),
        "scale_factors": [SCALE_FACTOR], "device_support": ["cuda", "cpu"],
        "requires_explicit_cpu_opt_in": True,
        "trained_weights_available": report["checkpoint_exists"],
        "configured": configured, "available": configured,
        "availability_scope": "configuration only; the device probe runs when a job executes",
        "reason": report["reason"],
        "scientific_evaluation_suitable": False,
        "description": ("Optional ESA OpenSR LDSR-S2 latent-diffusion checkpoint requiring a local checkpoint and, "
                        "by default, a CUDA device. Output is model-inferred detail, not validated reference data."),
    }


def _as_float_array(block, batch=False):
    """Validate one four-band window (or a batch of them) as a float32 numpy array."""
    if hasattr(block, "detach"):  # torch tensor
        block = block.detach().to("cpu").numpy()
    array = np.asarray(block)
    expected_ndim = 4 if batch else 3
    hint = "a batch (N, 4, H, W)" if batch else "a single 3-D window of shape (4, H, W)"
    if array.ndim != expected_ndim:
        raise OpenSRInputError(f"OpenSR expects {hint}; got shape {array.shape}")
    if array.shape[-3] != EXPECTED_BANDS:
        raise OpenSRInputError("OpenSR expects exactly 4 channels ordered "
                               + ", ".join(SENTINEL2_BAND_ORDER) + f"; got {array.shape[-3]}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise OpenSRInputError("OpenSR input contains unmasked non-finite pixel values")
    return array


def _scale_to_reflectance(array, dn_scale):
    """Divide DN by ``dn_scale`` when one is given (None means already reflectance)."""
    if dn_scale is None:
        return array
    scale = float(dn_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise OpenSRInputError(f"dn_scale must be a positive number; got {dn_scale!r}")
    return array / scale


def to_reflectance(block, dn_scale=None):
    """Return float32 reflectance in [0, 1] ordered B04, B03, B02, B08.

    ``dn_scale`` is the divisor for integer Sentinel-2 L2A DN (use 10000.0 for
    the standard 1e-4 reflectance scaling); pass None when the data is already
    reflectance in [0, 1].
    """
    array = _scale_to_reflectance(_as_float_array(block), dn_scale)
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def to_reflectance_batch(blocks, dn_scale=None):
    """Like :func:`to_reflectance` but for a (N, 4, H, W) stack."""
    array = _scale_to_reflectance(_as_float_array(blocks, batch=True), dn_scale)
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def to_uint16_reflectance(block, dn_scale=REFLECTANCE_SCALE):
    """Convert reflectance in [0, 1] back to Sentinel-2 DN for uint16 GeoTIFF output.

    Use this on model output before writing a uint16 raster so the result keeps the
    same reflectance scale as the input (Sentinel-2 L2A convention: 10000 = 1.0).
    """
    reflectance = to_reflectance(block)
    return np.rint(np.clip(reflectance * float(dn_scale), 0.0, 65535.0)).astype(np.uint16)


def load_model():
    """Build and cache the OpenSR model. Never downloads weights.

    Raises OpenSRUnavailable when the feature is disabled, incomplete, or has no
    usable device, and OpenSRInferenceError when the model cannot be initialised.
    """
    model = _CACHE.get("model")
    if model is not None:
        return model, _CACHE["device"]
    with _LOCK:
        model = _CACHE.get("model")
        if model is not None:
            return model, _CACHE["device"]
        if not settings.opensr_enabled:
            raise OpenSRUnavailable("OpenSR is disabled; set OPENSR_ENABLED=true and configure OPENSR_CHECKPOINT")
        steps = configured_sampling_steps()
        configured_window_size()
        path = checkpoint_path()
        if path is None:
            raise OpenSRUnavailable(f"OpenSR checkpoint is not configured or not found (OPENSR_CHECKPOINT); automatic "
                                    f"downloads are disabled, so a local {DEFAULT_CHECKPOINT_NAME} is required")
        try:
            import torch
        except Exception as exc:
            raise OpenSRUnavailable("PyTorch is not installed in this environment; "
                                    "install backend/requirements-opensr.txt") from exc
        try:
            from omegaconf import OmegaConf
            import opensr_model
        except Exception as exc:
            raise OpenSRUnavailable(f"The '{PACKAGE_NAME}' package and omegaconf must be installed; "
                                    "install backend/requirements-opensr.txt") from exc
        config_path = Path(opensr_model.__file__).resolve().parent / PACKAGE_CONFIG
        if not config_path.is_file():
            raise OpenSRUnavailable(f"The installed '{PACKAGE_NAME}' package does not provide {PACKAGE_CONFIG}; "
                                    "reinstall opensr-model")
        device, reason = resolve_device(torch)
        if device is None:
            raise OpenSRUnavailable(reason)
        threads = int(settings.opensr_torch_threads or 0)
        if threads > 0:
            torch.set_num_threads(threads)
        try:
            config = OmegaConf.load(str(config_path))
            model = opensr_model.SRLatentDiffusion(config, device=device)
            model.load_pretrained(str(path.resolve()))  # absolute path, existence checked: no download attempt
            model.eval()
        except Exception as exc:
            raise OpenSRInferenceError(f"Could not initialise the OpenSR model: {exc}") from exc
        _CACHE.update(model=model, device=device, torch=torch, sampling_steps=steps)
        log.info("OpenSR model ready on %s (window=%s, steps=%d, checkpoint=%s)",
                 device, settings.opensr_window, steps, path)
        return model, device


def _validate_window(height, width):
    """Both axes must be square and at least MIN_WINDOW so the model never pads."""
    if height != width:
        raise OpenSRInputError(f"OpenSR requires square windows in this phase; got {height}x{width}")
    if min(height, width) < MIN_WINDOW:
        raise OpenSRInputError(f"OpenSR requires windows of at least {MIN_WINDOW} pixels; got {height}x{width}")
    if max(height, width) > settings.opensr_max_window_size:
        raise OpenSRInputError(f"OpenSR windows must be <= {settings.opensr_max_window_size} pixels; got {height}x{width}")


def _validate_input_window(block, batch=False):
    shape = getattr(block, "shape", None)
    if shape is None:
        return
    expected_ndim = 4 if batch else 3
    if len(shape) != expected_ndim:
        return
    _validate_window(int(shape[-2]), int(shape[-1]))


def _resolve_steps(sampling_steps):
    if sampling_steps is None:
        steps = configured_sampling_steps()
    else:
        if isinstance(sampling_steps, bool):
            raise OpenSRInputError("sampling_steps must be an integer")
        try:
            steps = int(sampling_steps)
        except (TypeError, ValueError):
            raise OpenSRInputError(f"sampling_steps must be an integer; got {sampling_steps!r}") from None
        if not isinstance(sampling_steps, str) and sampling_steps != steps:
            raise OpenSRInputError(f"sampling_steps must be an integer; got {sampling_steps!r}")
    if steps < 1:
        raise OpenSRInputError(f"sampling_steps must be >= 1; got {sampling_steps!r}")
    if steps > settings.opensr_max_sampling_steps:
        raise OpenSRInputError(f"sampling_steps must be <= {settings.opensr_max_sampling_steps}; got {steps}")
    return steps


def _infer(torch, model, device, data, steps, histogram_matching):
    """Run one model call on a normalised (N, 4, H, W) float32 stack."""
    height, width = data.shape[-2], data.shape[-1]
    size = f"{height}x{width}"
    tensor = torch.from_numpy(np.ascontiguousarray(data)).to(device)
    try:
        with torch.no_grad():
            output = model.forward(tensor, sampling_steps=steps, histogram_matching=bool(histogram_matching))
    except OpenSRError:
        raise
    except MemoryError as exc:
        raise OpenSRInferenceError(f"OpenSR ran out of host memory for a {size} window; "
                                   "reduce OPENSR_WINDOW") from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise OpenSRInferenceError(f"OpenSR ran out of device memory for a {size} window; "
                                       "reduce OPENSR_WINDOW or OPENSR_BATCH_SIZE") from exc
        raise OpenSRInferenceError(f"OpenSR inference failed for a {size} window: {exc}") from exc
    array = output.detach().to("cpu", torch.float32).numpy()
    if array.shape[-3] != EXPECTED_BANDS:
        raise OpenSRInferenceError(f"OpenSR returned {array.shape[-3]} channels; expected {EXPECTED_BANDS}")
    if array.shape[-2:] != (height * SCALE_FACTOR, width * SCALE_FACTOR):
        log.warning("OpenSR returned shape %s for a %s window (expected 4x)", array.shape, size)
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def super_resolve_batch(blocks, dn_scale=None, sampling_steps=None, histogram_matching=True):
    """Super-resolve several windows in one model call.

    ``blocks`` is (N, 4, H, W) ordered B04, B03, B02, B08; returns (N, 4, 4H, 4W)
    reflectance in [0, 1]. Batched execution was verified in this project, not
    assumed: a batched call returns one item per input, and the model's per-image
    spectral correction matches per-tile calls because ``hq_histogram_matching``
    applies ``match_histograms(..., channel_axis=0)`` to each band slice, matching
    every batch entry independently against its own low-resolution input.
    """
    configured_window_size()
    _validate_input_window(blocks, batch=True)
    data = to_reflectance_batch(blocks, dn_scale)
    if data.shape[0] < 1:
        raise OpenSRInputError("OpenSR batch must contain at least one window")
    if data.shape[0] > settings.opensr_max_batch_size:
        raise OpenSRInputError(f"batch size must be <= {settings.opensr_max_batch_size}; got {data.shape[0]}")
    _validate_window(data.shape[-2], data.shape[-1])
    steps = _resolve_steps(sampling_steps)
    model, device = load_model()
    torch = _CACHE["torch"]
    return _infer(torch, model, device, data, steps, histogram_matching)


def super_resolve(block, dn_scale=None, sampling_steps=None, histogram_matching=True):
    """Super-resolve one four-band Sentinel-2 window by 4x.

    ``block`` is a (4, H, W) array or torch tensor ordered B04, B03, B02, B08.
    ``dn_scale`` is 10000.0 for Sentinel-2 L2A DN, or None when already reflectance.
    Returns float32 reflectance in [0, 1] with shape (4, H*4, W*4).

    No tiling and no file writing happen here. Availability, checkpoint, device,
    input, and memory failures all raise OpenSRError subclasses.
    """
    configured_window_size()
    _validate_input_window(block)
    data = to_reflectance(block, dn_scale)
    _validate_window(data.shape[-2], data.shape[-1])
    steps = _resolve_steps(sampling_steps)
    model, device = load_model()
    torch = _CACHE["torch"]
    return _infer(torch, model, device, data[None], steps, histogram_matching)[0]
