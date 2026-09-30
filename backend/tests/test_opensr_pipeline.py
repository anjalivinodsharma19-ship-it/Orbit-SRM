"""Phase 2 tests for tiled OpenSR processing and the OpenSR API surface.

Model inference is always mocked: these tests need no GPU, no torch and no
checkpoint, and they assert the tiling/blending/IO behaviour of the pipeline plus
the API contracts (method selection, validation, health, model registry).
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from app.config import settings
from app import main as main_module
from app.main import app
from app.ml import opensr
from app.opensr_pipeline import (Tile, blend_left, blend_top, blend_weights, output_profile, process,
                                 read_tile, tile_offsets, tile_plan, validate_block, validate_source)
from fastapi.testclient import TestClient

client = TestClient(app)

RASTER_SIZE = 244  # window 128 + stride 116: exactly two tiles per axis, 12 px overlap


def write_raster(path, height=RASTER_SIZE, width=RASTER_SIZE, bands=4, dtype="uint16",
                 crs="EPSG:32643", nodata=0, ramp=(500, 9000), descriptions=None):
    """Four-band Sentinel-2-like GeoTIFF with a two-axis gradient (B04, B03, B02, B08)."""
    columns = np.linspace(ramp[0], ramp[1], width)
    rows = np.linspace(ramp[0], ramp[1], height)
    plane = (rows[:, None] + columns[None, :]) / 2.0
    data = np.repeat(plane[None, :, :], bands, axis=0)
    if np.dtype(dtype).kind != "f":
        data = np.rint(data)
    data = data.astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=bands,
                       dtype=dtype, crs=crs, transform=from_origin(500000, 2000000, 10, 10),
                       nodata=nodata) as ds:
        ds.write(data)
        if descriptions is None and bands == opensr.EXPECTED_BANDS:
            descriptions = opensr.SENTINEL2_BAND_ORDER
        if descriptions:
            ds.descriptions = descriptions
    return path


def block_mean(path, x, y, window=128, bands=4):
    """Mean DN of a source window, which is what the fake model returns."""
    with rasterio.open(path) as ds:
        return float(ds.read(window=Window(x, y, window, window)).reshape(bands, -1).mean())


def _raise_unavailable():
    raise opensr.OpenSRUnavailable("CUDA is not available and OPENSR_DEVICE=auto never falls back to CPU")


@pytest.fixture
def job_id(request):
    """Unique job id per test, with its output/temp/preview files removed afterwards."""
    name = "t" + "".join(char if char.isalnum() else "_" for char in request.node.name)[:40]
    yield name
    for folder, suffix in (("outputs", ".tif"), ("temporary", ".tif"), ("previews", ".png")):
        try:
            (settings.data_dir / folder / f"{name}{suffix}").unlink(missing_ok=True)
        except OSError:
            pass


@pytest.fixture
def fake_opensr(monkeypatch):
    """Replace model loading and inference with deterministic CPU stand-ins."""
    calls = {"single": 0, "batch": 0, "steps": []}

    def _predict(block, dn_scale=None):
        value = float(opensr.to_reflectance(block, dn_scale).mean())
        return np.full((4, block.shape[-2] * 4, block.shape[-1] * 4), value, dtype=np.float32)

    def fake_super_resolve(block, dn_scale=None, sampling_steps=None, histogram_matching=True):
        calls["single"] += 1
        calls["steps"].append(sampling_steps)
        return _predict(block, dn_scale)

    def fake_super_resolve_batch(blocks, dn_scale=None, sampling_steps=None, histogram_matching=True):
        calls["batch"] += 1
        calls["steps"].append(sampling_steps)
        return np.stack([_predict(block, dn_scale) for block in blocks])

    monkeypatch.setattr(opensr, "load_model", lambda: (object(), "cpu"))
    monkeypatch.setattr(opensr, "super_resolve", fake_super_resolve)
    monkeypatch.setattr(opensr, "super_resolve_batch", fake_super_resolve_batch)
    return calls


def test_process_accepts_batch_and_sampling_step_boundaries(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "src.tif")
    for parameters in ({"batch_size": 1, "sampling_steps": 1},
                       {"batch_size": settings.opensr_max_batch_size,
                        "sampling_steps": settings.opensr_max_sampling_steps}):
        result = process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                               "parameters": parameters})
        assert result["provenance"]["batch_size"] == parameters["batch_size"]
        assert result["provenance"]["sampling_steps"] == parameters["sampling_steps"]


@pytest.mark.parametrize("window_size",[128,256])
def test_process_accepts_supported_window_boundaries(tmp_path, fake_opensr, job_id, window_size):
    src = write_raster(tmp_path / "src.tif")
    result = process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                           "parameters": {"window_size": window_size, "sampling_steps": 1}})
    assert result["provenance"]["window"] == window_size


@pytest.mark.parametrize("window_size",[127,257,128.5,None])
def test_process_rejects_invalid_window_before_opening_raster(tmp_path, monkeypatch, window_size):
    monkeypatch.setattr(rasterio,"open",lambda *_args,**_kwargs:pytest.fail("raster must not be opened"))
    with pytest.raises(opensr.OpenSRError):
        process(tmp_path / "not-opened.tif", {"id": "invalid-window", "method": "opensr_ldsrs2",
                                               "scale_factor": 4,
                                               "parameters": {"window_size": window_size}})


@pytest.mark.parametrize("window_size",[127,257,128.5])
def test_process_rejects_invalid_configured_window_before_opening_raster(tmp_path, monkeypatch, window_size):
    monkeypatch.setattr(settings,"opensr_window",window_size)
    monkeypatch.setattr(rasterio,"open",lambda *_args,**_kwargs:pytest.fail("raster must not be opened"))
    with pytest.raises(opensr.OpenSRUnavailable):
        process(tmp_path / "not-opened.tif", {"id": "invalid-config-window", "method": "opensr_ldsrs2",
                                               "scale_factor": 4, "parameters": {}})


@pytest.mark.parametrize("parameters",[
    {"batch_size": 0}, {"batch_size": 5}, {"batch_size": 4.5},
    {"sampling_steps": 0}, {"sampling_steps": 101}, {"sampling_steps": 100.5},
])
def test_process_rejects_out_of_range_resources_before_inference(tmp_path, fake_opensr, job_id, parameters):
    src = write_raster(tmp_path / "src.tif")
    with pytest.raises(ValueError):
        process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                      "parameters": parameters})
    assert fake_opensr["single"] == 0 and fake_opensr["batch"] == 0
    assert not (settings.data_dir / "outputs" / f"{job_id}.tif").exists()


def test_tile_offsets_cover_the_raster_without_short_windows():
    for length, window, overlap in ((128, 128, 12), (244, 128, 12), (300, 128, 12), (1024, 128, 16)):
        offsets = tile_offsets(length, window, overlap)
        assert offsets[0] == 0
        assert all(offset + window <= length for offset in offsets)
        assert offsets == sorted(offsets)
        assert offsets[-1] == length - window
        for previous, current in zip(offsets, offsets[1:]):
            assert current - previous <= window - overlap


def test_tile_offsets_for_rasters_smaller_than_a_tile():
    assert tile_offsets(40, 128, 12) == [0]
    assert tile_offsets(128, 128, 12) == [0]


def test_tile_plan_small_raster_is_a_single_tile():
    plan = tile_plan(30, 40, 128, 12)
    assert plan == [Tile(0, 0, 0, 0)]


def test_tile_plan_overlap_is_the_shared_band_including_the_clamped_last_tile():
    plan = tile_plan(300, 300, 128, 12)
    offsets = tile_offsets(300, 128, 12)
    assert offsets == [0, 116, 172]
    assert len(plan) == 9
    first_row = plan[:3]
    assert [tile.overlap_x for tile in first_row] == [0, 12, 128 - (172 - 116)]
    assert all(tile.y == 0 and tile.overlap_y == 0 for tile in first_row)
    assert plan[3].overlap_y == 12


def test_tile_plan_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        tile_plan(300, 300, 128, 128)
    with pytest.raises(ValueError):
        tile_plan(300, 300, 128, -1)


def test_blend_weights_ramp_only_inside_the_shared_band():
    horizontal = blend_weights(48, 48, 0, 48)
    assert horizontal[0, 0] == pytest.approx(0.0)
    assert horizontal[0, -1] == pytest.approx(1.0)
    assert np.all(np.diff(horizontal[0, :]) >= 0)
    bordered = blend_weights(128, 128, 48, 48)
    assert np.allclose(bordered[48:, 48:], 1.0)   # outside both bands
    assert bordered[0, 0] == pytest.approx(0.0)   # inside both ramps
    assert bordered[0, 48] == pytest.approx(0.0)  # top band, beyond the left ramp
    assert bordered[48, 0] == pytest.approx(0.0)  # left band, beyond the top ramp


def test_blend_top_and_left_feather_against_the_neighbour_bands():
    tile = np.full((4, 4, 4), 1.0, dtype=np.float32)
    blend_top(tile, np.zeros((4, 2, 4), dtype=np.float32), 2)
    assert tile[0, 0, 0] == pytest.approx(0.0) and tile[0, 1, 0] == pytest.approx(1.0)
    assert np.allclose(tile[:, 2:, :], 1.0)
    tile = np.full((4, 4, 4), 1.0, dtype=np.float32)
    blend_left(tile, np.zeros((4, 4, 2), dtype=np.float32), 2)
    assert tile[0, 0, 0] == pytest.approx(0.0) and tile[0, 0, 1] == pytest.approx(1.0)
    assert np.allclose(tile[:, :, 2:], 1.0)


def test_output_profile_preserves_georeferencing_and_scales_geometry(tmp_path):
    src = write_raster(tmp_path / "src.tif", height=40, width=30, nodata=0)
    with rasterio.open(src) as ds:
        profile = output_profile(ds, opensr.SCALE_FACTOR)
    assert profile["count"] == 4 and profile["dtype"] == "uint16"
    assert (profile["width"], profile["height"]) == (120, 160)
    assert profile["crs"] is not None and profile["nodata"] == 0
    assert profile["transform"].a == pytest.approx(2.5)
    assert profile["transform"].e == pytest.approx(-2.5)
    assert profile["transform"].c == pytest.approx(500000)
    assert profile["transform"].f == pytest.approx(2000000)
    assert "photometric" not in profile
    assert profile["compress"] == "deflate" and profile["tiled"] is True


def test_validate_source_rejects_unusable_rasters(tmp_path):
    with rasterio.open(write_raster(tmp_path / "three.tif", bands=3)) as ds:
        with pytest.raises(ValueError) as exc:
            validate_source(ds, 10000.0)
        assert "B04" in str(exc.value)
    with rasterio.open(write_raster(tmp_path / "five.tif", bands=5)) as ds:
        with pytest.raises(ValueError,match="exactly 4 bands"):
            validate_source(ds,10000.0)
    with rasterio.open(write_raster(tmp_path / "eight.tif", dtype="uint8", ramp=(1, 200))) as ds:
        with pytest.raises(ValueError) as exc:
            validate_source(ds, 10000.0)
        assert "8-bit" in str(exc.value)
    with rasterio.open(write_raster(tmp_path / "nocrs.tif", crs=None)) as ds:
        with pytest.raises(ValueError) as exc:
            validate_source(ds, 10000.0)
        assert "CRS" in str(exc.value)
    with rasterio.open(write_raster(tmp_path / "float.tif", dtype="float32", ramp=(0.05, 0.9))) as ds:
        assert validate_source(ds, 10000.0) is None  # float input is already reflectance


@pytest.mark.parametrize("descriptions",[(),("B04","B02","B03","B08")])
def test_validate_source_rejects_missing_or_ambiguous_band_descriptions(tmp_path,descriptions):
    src=write_raster(tmp_path/"ambiguous.tif",descriptions=descriptions)
    with rasterio.open(src) as ds,pytest.raises(opensr.OpenSRInputError,match="band descriptions"):
        validate_source(ds,10000.0)


def test_read_tile_maps_declared_nodata_to_model_zero_and_output_metadata(tmp_path):
    data=np.full((4,128,128),5000,dtype=np.uint16)
    data[0,7,9]=65535
    src=tmp_path/"nonzero-nodata.tif"
    with rasterio.open(src,"w",driver="GTiff",width=128,height=128,count=4,dtype="uint16",
                       crs="EPSG:32643",transform=from_origin(500000,2000000,10,10),nodata=65535) as ds:
        ds.write(data)
        ds.descriptions=opensr.SENTINEL2_BAND_ORDER
    with rasterio.open(src) as ds:
        block=read_tile(ds,Tile(0,0,0,0),128)
        validate_block(block,10000.0)
        profile=output_profile(ds,opensr.SCALE_FACTOR)
    assert block[0,7,9]==0
    assert block[1,7,9]==5000
    assert profile["nodata"]==0


def test_read_tile_masks_declared_nan_and_rejects_unmasked_nonfinite(tmp_path):
    data=np.full((4,128,128),0.4,dtype=np.float32)
    data[0,3,4]=np.nan
    masked_src=tmp_path/"nan-nodata.tif"
    with rasterio.open(masked_src,"w",driver="GTiff",width=128,height=128,count=4,dtype="float32",
                       crs="EPSG:32643",transform=from_origin(500000,2000000,10,10),nodata=np.nan) as ds:
        ds.write(data)
        ds.descriptions=opensr.SENTINEL2_BAND_ORDER
    with rasterio.open(masked_src) as ds:
        block=read_tile(ds,Tile(0,0,0,0),128)
    assert block[0,3,4]==0
    validate_block(block,None)

    data[0,3,4]=np.inf
    unmasked_src=tmp_path/"unmasked-infinity.tif"
    with rasterio.open(unmasked_src,"w",driver="GTiff",width=128,height=128,count=4,dtype="float32",
                       crs="EPSG:32643",transform=from_origin(500000,2000000,10,10)) as ds:
        ds.write(data)
        ds.descriptions=opensr.SENTINEL2_BAND_ORDER
    with rasterio.open(unmasked_src) as ds:
        block=read_tile(ds,Tile(0,0,0,0),128)
    with pytest.raises(ValueError,match="unmasked non-finite"):
        validate_block(block,None)


def test_invalid_pixel_data_is_rejected_before_model_loading(tmp_path,monkeypatch,job_id):
    data=np.full((4,128,128),0.4,dtype=np.float32)
    data[0,2,3]=np.nan
    src=tmp_path/"unmasked-nan.tif"
    with rasterio.open(src,"w",driver="GTiff",width=128,height=128,count=4,dtype="float32",
                       crs="EPSG:32643",transform=from_origin(500000,2000000,10,10)) as ds:
        ds.write(data)
        ds.descriptions=opensr.SENTINEL2_BAND_ORDER
    monkeypatch.setattr(opensr,"load_model",lambda:pytest.fail("invalid raster must fail before model load"))
    with pytest.raises(ValueError,match="unmasked non-finite"):
        process(src,{"id":job_id,"method":"opensr_ldsrs2","scale_factor":4,"parameters":{}})


def test_validate_block_rejects_dn_like_floats():
    validate_block(np.full((4, 4, 4), 0.9, dtype=np.float32), None)
    validate_block(np.full((4, 4, 4), 5000.0, dtype=np.float32), 10000.0)
    with pytest.raises(ValueError):
        validate_block(np.full((4, 4, 4), 5000.0, dtype=np.float32), None)


def test_read_tile_pads_rasters_smaller_than_the_window(tmp_path):
    src = write_raster(tmp_path / "small.tif", height=30, width=40)
    with rasterio.open(src) as ds:
        block = read_tile(ds, Tile(0, 0, 0, 0), 128)
    assert block.shape == (4, 128, 128) and block.dtype == np.float32
    assert np.all(np.isfinite(block))


def test_process_can_batch_tiles(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "src.tif")
    result = process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                           "parameters": {"sampling_steps": 2, "batch_size": 2}})
    assert fake_opensr["batch"] == 2 and fake_opensr["single"] == 0
    assert result["provenance"]["batch_size"] == 2


def test_process_enforces_the_tile_limit(tmp_path, fake_opensr, job_id, monkeypatch):
    monkeypatch.setattr(settings, "opensr_max_tiles", 2)
    src = write_raster(tmp_path / "src.tif")
    with pytest.raises(ValueError) as exc:
        process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4, "parameters": {}})
    assert "OPENSR_MAX_TILES" in str(exc.value)


def test_process_rejects_wrong_band_count_without_leaving_files(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "three.tif", bands=3)
    with pytest.raises(ValueError):
        process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4, "parameters": {}})
    assert not (settings.data_dir / "outputs" / f"{job_id}.tif").exists()
    assert not (settings.data_dir / "temporary" / f"{job_id}.tif").exists()


def test_process_rejects_invalid_parameters(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "src.tif")
    for parameters in ({"overlap": 128}, {"overlap": -1}, {"batch_size": 0}, {"sampling_steps": 0},
                       {"sampling_steps": "many"}):
        with pytest.raises(ValueError):
            process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4, "parameters": parameters})


def test_process_cleans_up_when_inference_fails(tmp_path, fake_opensr, job_id, monkeypatch):
    def boom(*args, **kwargs):
        raise opensr.OpenSRInferenceError("OpenSR ran out of device memory for a 128x128 window")

    monkeypatch.setattr(opensr, "super_resolve", boom)
    src = write_raster(tmp_path / "src.tif")
    with pytest.raises(opensr.OpenSRInferenceError):
        process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                      "parameters": {"sampling_steps": 2}})
    for folder, suffix in (("outputs", ".tif"), ("temporary", ".tif"), ("previews", ".png")):
        assert not (settings.data_dir / folder / f"{job_id}{suffix}").exists()


def test_process_reports_unavailable_model_without_writing_files(tmp_path, job_id, monkeypatch):
    monkeypatch.setattr(opensr, "load_model", _raise_unavailable)
    src = write_raster(tmp_path / "src.tif")
    with pytest.raises(opensr.OpenSRUnavailable):
        process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4, "parameters": {}})
    assert not (settings.data_dir / "temporary" / f"{job_id}.tif").exists()


def test_opensr_method_validates_scale_bands_and_parameters(tmp_path, configured_opensr):
    image = _upload(client, tmp_path)
    three_band = _upload(client, tmp_path, "three.tif", bands=3)
    missing_descriptions = _upload(client, tmp_path, "missing-descriptions.tif", descriptions=())
    ambiguous_descriptions = _upload(client, tmp_path, "ambiguous-descriptions.tif",
                                     descriptions=("B04", "B02", "B03", "B08"))
    base = {"image_id": image["id"], "method": "opensr_ldsrs2", "scale_factor": 4}
    assert client.post("/api/processing/start", json={**base, "scale_factor": 2}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "image_id": three_band["id"]}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "image_id": missing_descriptions["id"]}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "image_id": ambiguous_descriptions["id"]}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "parameters": {"unknown": 1}}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "parameters": {"overlap": 999}}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "parameters": {"sampling_steps": 0}}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "parameters": {"batch_size": "x"}}).status_code == 422
    assert client.post("/api/processing/start", json={**base, "parameters": {}}).status_code == 202


@pytest.mark.parametrize("parameters",[
    {"batch_size": 0}, {"batch_size": 5}, {"batch_size": 4.5},
    {"sampling_steps": 0}, {"sampling_steps": 101}, {"sampling_steps": 100.5},
])
def test_opensr_api_rejects_out_of_range_resources_before_scheduling(tmp_path, configured_opensr,
                                                                     monkeypatch, parameters):
    image = _upload(client, tmp_path)
    job_id = "resource-limit-rejected"
    monkeypatch.setattr(main_module, "ident", lambda: job_id)
    response = client.post("/api/processing/start", json={
        "image_id": image["id"], "method": "opensr_ldsrs2", "scale_factor": 4,
        "parameters": parameters})
    assert response.status_code == 422
    assert not (settings.data_dir / "metadata" / f"job_{job_id}.json").exists()


@pytest.mark.parametrize("parameters",[
    {"batch_size": 1}, {"batch_size": 4},
    {"sampling_steps": 1}, {"sampling_steps": 100},
    {"window_size": 128}, {"window_size": 256},
])
def test_opensr_api_accepts_resource_limit_boundaries(tmp_path, configured_opensr, parameters):
    image = _upload(client, tmp_path)
    response = client.post("/api/processing/start", json={
        "image_id": image["id"], "method": "opensr_ldsrs2", "scale_factor": 4,
        "parameters": parameters})
    assert response.status_code == 202, response.text
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"


def test_opensr_job_runs_through_the_existing_job_workflow(tmp_path, configured_opensr):
    image = _upload(client, tmp_path)
    started = client.post("/api/processing/start", json={
        "image_id": image["id"], "method": "opensr_ldsrs2", "scale_factor": 4,
        "parameters": {"sampling_steps": 2, "overlap": 12}})
    assert started.status_code == 202, started.text
    job = client.get(f"/api/jobs/{started.json()['id']}").json()
    assert job["status"] == "completed", job
    assert job["method"] == "opensr_ldsrs2" and job["scale_factor"] == 4
    body = client.get(f"/api/jobs/{job['id']}/result").json()
    assert body["result"]["output"]["width"] == 976 and body["result"]["output"]["height"] == 976
    assert body["result"]["output"]["resolution"] == [2.5, 2.5]
    assert body["result"]["provenance"]["tiles"] == 4
    assert "output_path" not in body["result"] and "preview_path" not in body["result"]
    assert client.get(f"/api/jobs/{job['id']}/preview").status_code == 200
    downloaded = client.get(f"/api/jobs/{job['id']}/download")
    assert downloaded.status_code == 200
    with rasterio.MemoryFile(downloaded.content) as mem, mem.open() as ds:
        assert (ds.width, ds.height) == (976, 976)
        assert ds.transform.a == pytest.approx(2.5) and ds.crs.to_string() == "EPSG:32643"
        assert ds.count == 4 and ds.dtypes[0] == "uint16"
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_opensr_job_fails_cleanly_when_cuda_is_unavailable(tmp_path, configured_opensr, monkeypatch):
    monkeypatch.setattr(opensr, "load_model", _raise_unavailable)
    image = _upload(client, tmp_path)
    started = client.post("/api/processing/start",
                          json={"image_id": image["id"], "method": "opensr_ldsrs2", "scale_factor": 4})
    assert started.status_code == 202, started.text
    job = client.get(f"/api/jobs/{started.json()['id']}").json()
    assert job["status"] == "failed"
    assert "CUDA" in job["error"] and "never falls back to CPU" in job["error"]
    assert not (settings.data_dir / "temporary" / f"{job['id']}.tif").exists()
    assert not (settings.data_dir / "outputs" / f"{job['id']}.tif").exists()
    assert client.get("/api/jobs").status_code == 200  # the API stays healthy


def _upload(client, tmp_path, name="opensr.tif", bands=4, descriptions=None):
    raster = write_raster(tmp_path / name, bands=bands, descriptions=descriptions)
    with open(raster, "rb") as handle:
        response = client.post("/api/imagery/upload", files={"file": (name, handle.read(), "image/tiff")})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def configured_opensr(monkeypatch, tmp_path, fake_opensr):
    """Pretend OpenSR is fully configured so the API path can run without torch."""
    checkpoint = tmp_path / opensr.DEFAULT_CHECKPOINT_NAME
    checkpoint.write_bytes(b"fake-checkpoint")
    report = {"model_id": opensr.MODEL_ID, "name": opensr.MODEL_NAME, "architecture": opensr.ARCHITECTURE,
              "enabled": True, "available": True, "reason": None, "loaded": False, "device": "cpu",
              "device_requested": "cpu", "cuda_available": False, "torch_installed": True,
              "package_installed": True, "checkpoint_configured": True, "checkpoint": str(checkpoint),
              "checkpoint_exists": True, "checkpoint_size_bytes": checkpoint.stat().st_size,
              "checkpoint_download": False, "config_path": "configs/config_10m.yaml", "config_exists": True,
              "window_size": 128, "sampling_steps": 100, "scale_factor": opensr.SCALE_FACTOR,
              "input_bands": list(opensr.SENTINEL2_BAND_ORDER), "notes": []}
    monkeypatch.setattr(opensr, "status", lambda probe_device=True: dict(report))
    return report


def test_health_keeps_existing_keys_and_adds_an_opensr_block():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["dependencies"]["rasterio"] is True and "rasterio_version" in body["dependencies"]
    assert body["opensr"]["model_id"] == opensr.MODEL_ID
    assert body["opensr"]["available"] is False and body["opensr"]["reason"]


def test_models_registry_keeps_existing_entries_and_lists_opensr():
    items = client.get("/api/models").json()["items"]
    assert [item["id"] for item in items] == ["baseline", "trained_srcnn", "opensr_ldsrs2"]
    assert items[0]["scale_factors"] == [2, 3, 4]
    detail = client.get("/api/models/opensr_ldsrs2").json()
    assert detail["input_bands"] == list(opensr.SENTINEL2_BAND_ORDER)
    assert detail["scale_factors"] == [4] and detail["scientific_evaluation_suitable"] is False
    assert detail["available"] is False  # not configured in the test environment


def test_opensr_method_is_rejected_while_disabled(tmp_path):
    image = _upload(client, tmp_path)
    response = client.post("/api/processing/start",
                           json={"image_id": image["id"], "method": "opensr_ldsrs2", "scale_factor": 4})
    assert response.status_code == 422
    assert "OpenSR is not available" in response.text


def test_process_scales_output_and_blends_tile_seams(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "src.tif")
    result = process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                           "parameters": {"sampling_steps": 2, "overlap": 12}})
    assert fake_opensr["single"] == 4 and fake_opensr["steps"] == [2, 2, 2, 2]
    output = result["output"]
    assert (output["width"], output["height"]) == (976, 976)
    assert output["crs"] == "EPSG:32643" and output["resolution"] == [2.5, 2.5]
    assert output["bands"] == 4 and output["dtype"] == "uint16" and output["nodata"] == 0
    provenance = result["provenance"]
    assert provenance["tiles"] == 4 and provenance["blended"] is True
    assert provenance["scale_factor"] == 4 and provenance["device"] == "cpu"
    assert provenance["input_bands"] == list(opensr.SENTINEL2_BAND_ORDER)
    assert provenance["input_scale"] == 10000.0 and provenance["output_dn_scale"] == 10000.0
    out_path = settings.data_dir / "outputs" / f"{job_id}.tif"
    assert (settings.data_dir / "previews" / f"{job_id}.png").is_file()
    assert not (settings.data_dir / "temporary" / f"{job_id}.tif").exists()
    with rasterio.open(out_path) as ds:
        assert ds.crs.to_string() == "EPSG:32643"
        assert ds.transform.a == pytest.approx(2.5) and ds.transform.e == pytest.approx(-2.5)
        assert tuple(ds.descriptions) == opensr.SENTINEL2_BAND_ORDER
        assert ds.tags()["OPENSR_MODEL"] == opensr.MODEL_ID
        band = ds.read(1).astype(float)
    horizontal = band[0, 464:512]
    assert horizontal[0] == pytest.approx(block_mean(src, 0, 0), abs=2)
    assert horizontal[-1] == pytest.approx(block_mean(src, 116, 0), abs=2)
    assert np.all(np.diff(horizontal) >= -1.0)
    assert len(np.unique(horizontal)) > 3  # a real ramp, not a hard overwrite
    vertical = band[464:512, 0]
    assert vertical[0] == pytest.approx(block_mean(src, 0, 0), abs=2)
    assert vertical[-1] == pytest.approx(block_mean(src, 0, 116), abs=2)
    assert np.all(np.diff(vertical) >= -1.0)


def test_process_without_overlap_disables_blending(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "src.tif")
    result = process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                           "parameters": {"sampling_steps": 2, "overlap": 0}})
    assert result["provenance"]["blended"] is False
    with rasterio.open(settings.data_dir / "outputs" / f"{job_id}.tif") as ds:
        band = ds.read(1).astype(float)
    assert band[0, 464] == pytest.approx(block_mean(src, 116, 0), abs=1)
    assert band[0, 463] == pytest.approx(block_mean(src, 0, 0), abs=1)


def test_process_handles_rasters_smaller_than_a_window(tmp_path, fake_opensr, job_id):
    src = write_raster(tmp_path / "small.tif", height=30, width=40)
    result = process(src, {"id": job_id, "method": "opensr_ldsrs2", "scale_factor": 4,
                           "parameters": {"sampling_steps": 2}})
    assert result["provenance"]["tiles"] == 1 and fake_opensr["single"] == 1
    assert (result["output"]["width"], result["output"]["height"]) == (160, 120)
    with rasterio.open(settings.data_dir / "outputs" / f"{job_id}.tif") as ds:
        assert ds.read(1).shape == (120, 160)
