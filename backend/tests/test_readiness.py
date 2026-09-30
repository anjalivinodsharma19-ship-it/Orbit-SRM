"""Tests for the storage-independent Stage 2 slice: OrbitSRM branding, liveness vs
readiness, and the guarantee that probes leak neither paths nor environment values."""
import json

import pytest
from fastapi.testclient import TestClient

from app import readiness
from app.config import Settings, settings
from app.main import app
from app.ml import opensr as opensr_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    for folder in ("uploads", "outputs", "previews", "metadata", "temporary"):
        (data_dir / folder).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "data_dir", data_dir)


def test_liveness_keeps_legacy_shape_and_adds_branding():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["application"] == {"name": "OrbitSRM", "version": settings.app_version}
    assert body["dependencies"]["rasterio"] is True
    assert "rasterio_version" in body["dependencies"]
    assert body["opensr"]["model_id"] == opensr_module.MODEL_ID


def test_openapi_metadata_uses_orbitsrm_branding():
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "OrbitSRM API"
    assert schema["info"]["version"] == settings.app_version
    assert "OrbitSRM" in schema["info"]["description"]
    assert "/api/ready" in schema["paths"]
    assert "/api/health" in schema["paths"]


def test_config_defaults_carry_the_branding():
    fields = Settings.model_fields
    assert fields["app_name"].default == "OrbitSRM"
    assert fields["app_version"].default == settings.app_version


def test_ready_reports_ready_with_honest_capability_labels():
    response = client.get("/api/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["storage_writable"] is True
    assert body["checks"]["rasterio"] is True
    assert body["checks"]["scikit_image"] is True
    assert body["checks"]["storage"] == "local_filesystem"
    assert body["checks"]["database"] == "not_configured"
    assert body["checks"]["background_executor"] == "in_process"
    assert "failures" not in body
    assert body["application"]["name"] == "OrbitSRM"
    assert body["generated_at"]


def test_optional_opensr_never_makes_the_service_unready():
    body = client.get("/api/ready").json()
    opensr_status = body["checks"]["opensr"]
    assert set(opensr_status) == {"enabled", "configured", "available", "reason"}
    assert opensr_status["enabled"] is False  # disabled by default
    assert body["status"] == "ready"


def test_ready_returns_503_and_names_the_failing_check(monkeypatch):
    monkeypatch.setattr(readiness, "storage_writable", lambda data_dir: (False, "PermissionError"))
    response = client.get("/api/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["failures"] == {"storage_writable": "PermissionError"}
    assert body["checks"]["storage_writable"] is False


def test_ready_survives_a_broken_opensr_probe(monkeypatch):
    def boom(probe_device=True):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(opensr_module, "status", boom)
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["opensr"]["enabled"] is None


def test_probes_do_not_expose_the_data_directory_or_environment_dump():
    for path in ("/api/health", "/api/ready"):
        text = client.get(path).text
        assert str(settings.data_dir) not in text
        assert "DATA_DIR" not in text
        assert "C:\\\\" not in text and "/data/" not in text
    assert json.loads(client.get("/api/ready").text)["checks"]["storage"] == "local_filesystem"


def test_startup_and_shutdown_are_logged_with_branding(caplog):
    with caplog.at_level("INFO", logger="orbitsrm"):
        with TestClient(app):
            pass
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "OrbitSRM" in messages
    assert "starting" in messages
    assert "shutting down" in messages
    assert settings.app_version in messages
