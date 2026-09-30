"""OrbitSRM readiness reporting.

Kept separate from the liveness endpoint in ``main.py`` so orchestrators can tell
"the process is up" apart from "the process can serve requests".

Design notes:

* Every check is cheap: no model loading, no network, no device probing.
* The response never contains credentials, environment values, or absolute
  filesystem paths - only booleans, fixed labels and short reasons.
* Optional capabilities (OpenSR) are reported but never make the service
  unready, because the baseline and ``trained_srcnn`` pipelines work without them.
"""
import importlib.util
import logging
import os
import tempfile
from datetime import datetime, timezone

log = logging.getLogger("orbitsrm.readiness")

#: (check name, reason used when the probe fails without its own message).
HARD_CHECKS = (("storage_writable", "unwritable"),
               ("rasterio", "not importable"),
               ("scikit_image", "not importable"))


def module_present(name):
    """True when a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def storage_writable(data_dir):
    """``(ok, reason)`` for "a temporary file can be created and removed" in the data dir."""
    probe = data_dir / "temporary"
    try:
        probe.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".ready-", dir=probe)
        os.close(handle)
        os.unlink(name)
        return True, None
    except Exception as exc:
        # Only the exception type is surfaced: messages can embed absolute paths.
        log.warning("Readiness storage probe failed: %s", type(exc).__name__)
        return False, type(exc).__name__


def opensr_summary(opensr):
    """Configuration-level OpenSR status; device probing is deliberately skipped."""
    try:
        report = opensr.status(probe_device=False)
    except Exception as exc:
        log.warning("Readiness OpenSR probe failed: %s", type(exc).__name__)
        return {"enabled": None, "configured": False, "available": None, "reason": type(exc).__name__}
    configured = bool(report["enabled"] and report["checkpoint_exists"] and report["torch_installed"]
                      and report["package_installed"] and report["config_exists"])
    return {"enabled": bool(report["enabled"]), "configured": configured,
            "available": bool(report["available"]), "reason": report["reason"]}


def check(settings, opensr):
    """Build the readiness report; ``status`` is ``ready`` only if every hard check passes."""
    storage_ok, storage_reason = storage_writable(settings.data_dir)
    probes = {
        "storage_writable": (storage_ok, storage_reason),
        "rasterio": (module_present("rasterio"), None),
        "scikit_image": (module_present("skimage"), None),
    }
    failures = {}
    for name, fallback in HARD_CHECKS:
        ok, reason = probes[name]
        if not ok:
            failures[name] = reason or fallback
    checks = {name: probes[name][0] for name, _ in HARD_CHECKS}
    checks.update({
        # Honest capability labels: no durable storage or queue is configured yet.
        "storage": "local_filesystem",
        "database": "not_configured",
        "background_executor": "in_process",
        "opensr": opensr_summary(opensr),
    })
    report = {"status": "ready" if not failures else "not_ready",
              "application": {"name": settings.app_name, "version": settings.app_version},
              "checks": checks,
              "generated_at": datetime.now(timezone.utc).isoformat()}
    if failures:
        report["failures"] = failures
    return report
