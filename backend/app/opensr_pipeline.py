"""Tiled OpenSR (LDSR-S2) GeoTIFF processing.

Phase 2 of the optional OpenSR integration. ``app/ml/opensr.py`` owns the model
wrapper; this module owns raster IO, tiling and blending, and is only imported
from ``services.process`` when an OpenSR job actually runs.

Implementation notes, checked against the OpenSR reference deployment shipped in
``opensr-model-demo/deployment/opensr_hpc`` rather than assumed:

* ``opensr-utils`` (the vendor's tiled runner) is **not installed**, so tiling and
  stitching are implemented here.
* The vendor patch planner clamps patch centres so every patch is a full square
  inside the raster bounds (``patching.py: clamp_center``). This module clamps tile
  offsets the same way, so the model never receives a short window; only rasters
  smaller than one tile are padded, and then to exactly the configured window size.
* The vendor default runtime is ``window_size=[128,128]``, ``factor=4``,
  ``overlap=12``, ``eliminate_border_px=2`` with a ``batch_size`` driven by an
  opensr-utils ``PredictionDataModule`` (``deployment/configs/runtime.default.yaml``).
* Tiles are read with context overlap and written with a feathered blend against the
  already written pixels, so memory stays bounded by one tile instead of the whole
  4x output. There is no full-raster accumulator.

Outputs are uint16 Sentinel-2 DN scaled by ``OPENSR_DN_SCALE`` so the model's
reflectance output keeps the same radiometry as the input.
"""
import logging
import os
from typing import NamedTuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from .config import settings
from .ml import opensr
from .services import inspect_raster, make_preview, now

log = logging.getLogger(__name__)

MAX_FLOAT_REFLECTANCE = 1.5  # float rasters above this are DN, not reflectance


class Tile(NamedTuple):
    """One LR window placement.

    ``x``/``y`` are LR pixel offsets, ``overlap_x``/``overlap_y`` are the LR pixels
    actually shared with the previous column/row tile (0 when there is none or when
    blending is disabled).
    """

    x: int
    y: int
    overlap_x: int
    overlap_y: int


def tile_offsets(length, window, overlap):
    """Ordered tile offsets along one axis, each sized so a full window fits.

    Mirrors the vendor ``clamp_center`` behaviour: the final offset is pulled back
    inside the raster instead of emitting a short edge window (which the model
    cannot consume). Returns ``[0]`` when the raster is smaller than one window.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if length < window:
        return [0]
    stride = window - overlap
    offsets = list(range(0, length - window + 1, stride))
    last = length - window
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def tile_plan(height, width, window, overlap):
    """Full tile plan for a raster, in row-major order.

    Each tile records its real overlap with the previous tile on each axis so the
    blend band always covers the whole shared area, including the clamped last
    column/row where the overlap is larger than the configured value.
    """
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= window:
        raise ValueError(f"overlap ({overlap}) must be smaller than the window size ({window})")
    blend = overlap > 0
    xs = tile_offsets(width, window, overlap)
    ys = tile_offsets(height, window, overlap)
    plan = []
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            # Shared band with the previous tile: window - (offset delta). Interior
            # tiles share exactly `overlap` pixels; the clamped last tile shares more.
            overlap_x = window - (xs[column] - xs[column - 1]) if column > 0 and blend else 0
            overlap_y = window - (ys[row] - ys[row - 1]) if row > 0 and blend else 0
            plan.append(Tile(x, y, overlap_x, overlap_y))
    return plan


def validate_source(ds, dn_scale):
    """Validate the source raster and return the DN scale to normalise with.

    Returns ``None`` for float rasters (already reflectance) or the configured
    ``dn_scale`` for integer rasters (Sentinel-2 DN). Raises ValueError with an
    operator-facing message for anything OpenSR cannot consume.
    """
    if ds.count != opensr.EXPECTED_BANDS:
        raise ValueError("OpenSR requires exactly 4 bands ordered "
                         + ", ".join(opensr.SENTINEL2_BAND_ORDER) + f"; this raster has {ds.count}")
    opensr.validate_band_descriptions(ds.descriptions)
    if ds.width < 1 or ds.height < 1:
        raise ValueError("Raster contains no pixels")
    if ds.crs is None:
        raise ValueError("OpenSR requires a georeferenced raster; this GeoTIFF has no CRS")
    dtypes = set(ds.dtypes)
    if len(dtypes) != 1:
        raise ValueError(f"All bands must share one data type; found {sorted(dtypes)}")
    dtype = np.dtype(ds.dtypes[0])
    if dtype.kind not in ("u", "i", "f"):
        raise ValueError(f"Unsupported raster data type '{ds.dtypes[0]}'")
    if dtype.kind == "u" and dtype.itemsize == 1:
        raise ValueError("8-bit rasters are not supported for OpenSR; supply Sentinel-2 L2A "
                         "reflectance as uint16 DN (10000 = 1.0)")
    if dtype.kind == "f":
        return None
    return float(dn_scale)


def output_profile(ds, scale):
    """Output profile: same CRS/nodata/metadata, 4x size, quarter pixel size, uint16."""
    profile = ds.profile.copy()
    profile.update(width=ds.width * scale, height=ds.height * scale, count=opensr.EXPECTED_BANDS,
                   dtype="uint16", transform=ds.transform * Affine.scale(1 / scale),
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    has_source_mask = ds.nodata is not None or any(
        "all_valid" not in {flag.name for flag in flags} for flags in ds.mask_flag_enums)
    if has_source_mask:
        profile.update(nodata=0)
    # A source photometric (for example RGB) is invalid for a 4-band output.
    profile.pop("photometric", None)
    return profile


def read_tile(ds, tile, window):
    """Read one LR tile as (4, window, window) float32.

    Offsets are clamped by :func:`tile_plan`, so windows are full size except when
    the raster is smaller than one tile. In that case the block is padded with edge
    values up to exactly ``window`` so the model never sees an invalid window; the
    padding is cropped away again when the output is written.
    """
    height = min(window, ds.height - tile.y)
    width = min(window, ds.width - tile.x)
    block = ds.read(window=Window(tile.x, tile.y, width, height), masked=True).filled(0).astype(np.float32)
    if (height, width) != (window, window):
        block = np.pad(block, ((0, 0), (0, window - height), (0, window - width)), mode="edge")
    return block


def validate_block(block, dn_scale):
    """Reject pixel data that cannot be Sentinel-2 reflectance."""
    finite = np.isfinite(block)
    if not finite.all():
        raise ValueError("OpenSR raster contains unmasked non-finite pixel values")
    if dn_scale is None and finite.any():
        peak = float(block[finite].max())
        if peak > MAX_FLOAT_REFLECTANCE:
            raise ValueError("Float raster values exceed "
                             f"{MAX_FLOAT_REFLECTANCE} (peak {peak:.3g}), so they are not reflectance "
                             "in [0, 1]; provide Sentinel-2 DN or scaled reflectance")


def output_window(ds, tile, height, width):
    """Clip one HR tile footprint to the scaled raster bounds."""
    scale = opensr.SCALE_FACTOR
    return Window(tile.x * scale, tile.y * scale,
                  min(width, (ds.width - tile.x) * scale),
                  min(height, (ds.height - tile.y) * scale))


def write_tile(dst, target, reflectance, output_scale):
    """Write one HR reflectance tile as uint16 DN."""
    dst.write(opensr.to_uint16_reflectance(reflectance, output_scale), window=target)


def blend_top(tile_reflectance, existing, band_rows):
    """Feather the top ``band_rows`` against the previous tile row's matching band."""
    band_rows = min(int(band_rows), tile_reflectance.shape[-2], existing.shape[-2])
    if band_rows <= 0:
        return
    weights = blend_weights(band_rows, tile_reflectance.shape[-1], band_rows, 0)
    incoming = tile_reflectance[:, :band_rows, :]
    tile_reflectance[:, :band_rows, :] = incoming * weights + existing[:, :band_rows, :] * (1.0 - weights)


def blend_left(tile_reflectance, existing, band_columns):
    """Feather the left ``band_columns`` against the previous tile's right band."""
    band_columns = min(int(band_columns), tile_reflectance.shape[-1], existing.shape[-1])
    if band_columns <= 0:
        return
    weights = blend_weights(tile_reflectance.shape[-2], band_columns, 0, band_columns)
    incoming = tile_reflectance[:, :, :band_columns]
    tile_reflectance[:, :, :band_columns] = (incoming * weights
                                             + existing[:, :, -band_columns:] * (1.0 - weights))


def _int_param(parameters, name, default, minimum, maximum=None):
    """Read an integer override from the job's ``parameters`` dict, with validation."""
    value = parameters.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"parameters.{name} must be an integer, not a boolean")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"parameters.{name} must be an integer; got {value!r}") from None
    if not isinstance(value, str) and value != number:
        raise ValueError(f"parameters.{name} must be an integer; got {value!r}")
    if number < minimum:
        raise ValueError(f"parameters.{name} must be >= {minimum}; got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"parameters.{name} must be <= {maximum}; got {number}")
    return number


def infer_tiles(ds, plan, window, batch_size, sampling_steps, input_scale):
    """Yield ``(tile, reflectance, target_window)`` for every tile in the plan.

    Tiles are read through a bounded buffer of ``batch_size`` windows and handed to
    the model in one call, so host memory stays proportional to the window size
    instead of the raster size.
    """
    pending = []
    for tile in plan:
        block = read_tile(ds, tile, window)
        validate_block(block, input_scale)
        pending.append((tile, block))
        if len(pending) >= batch_size:
            yield from _run_pending(ds, pending, sampling_steps, input_scale)
            pending = []
    if pending:
        yield from _run_pending(ds, pending, sampling_steps, input_scale)


def validate_tiles(ds, plan, window, input_scale):
    """Scan bounded source windows before model loading to reject invalid pixels."""
    for tile in plan:
        block = read_tile(ds, tile, window)
        validate_block(block, input_scale)
        del block


def _run_pending(ds, pending, sampling_steps, input_scale):
    if len(pending) == 1:
        tile, block = pending[0]
        results = [(tile, opensr.super_resolve(block, dn_scale=input_scale, sampling_steps=sampling_steps))]
    else:
        stacked = np.stack([block for _, block in pending])
        batch = opensr.super_resolve_batch(stacked, dn_scale=input_scale, sampling_steps=sampling_steps)
        results = [(pending[index][0], batch[index]) for index in range(len(pending))]
    return [(tile, reflectance, output_window(ds, tile, *reflectance.shape[-2:]))
            for tile, reflectance in results]


def row_overlaps(plan, window):
    """LR overlap the row below each row offset needs (0 for the last row)."""
    offsets = []
    for tile in plan:
        if not offsets or offsets[-1] != tile.y:
            offsets.append(tile.y)
    return {y: (window - (offsets[index + 1] - y) if index + 1 < len(offsets) else 0)
            for index, y in enumerate(offsets)}


def write_plan(dst, tiles, plan, output_scale, window):
    """Write every tile, feathering seams, with memory bounded by a few narrow bands.

    No read-back from the output dataset is used: rasterio exposes no flush on a
    writer, so reading a partially written file back would be unreliable. The only
    neighbour state kept is the previous tile's right band, plus one bottom band per
    column for the row below - each at most one window wide, so the cost does not grow
    with raster size.
    """
    scale = opensr.SCALE_FACTOR
    needed = row_overlaps(plan, window)
    bottom_bands = {}
    previous_band = None
    current_row = None
    column = 0
    written = 0
    for tile, reflectance, target in tiles:
        height, width = int(target.height), int(target.width)
        cropped = np.ascontiguousarray(reflectance[:, :height, :width])
        if tile.y != current_row:
            current_row = tile.y
            column = 0
            previous_band = None
        existing = bottom_bands.get(column)
        if tile.overlap_y and existing is not None and existing.shape[-1] == width:
            blend_top(cropped, existing, tile.overlap_y * scale)
        if tile.overlap_x and previous_band is not None and previous_band.shape[-2] == height:
            blend_left(cropped, previous_band, tile.overlap_x * scale)
        write_tile(dst, target, cropped, output_scale)
        previous_band = cropped
        keep_rows = min(needed.get(tile.y, 0) * scale, height)
        if keep_rows > 0:
            bottom_bands[column] = np.array(cropped[:, height - keep_rows:, :], dtype=np.float32, copy=True)
        else:
            bottom_bands.pop(column, None)
        column += 1
        written += 1
        if written % 25 == 0 or written == len(plan):
            log.info("OpenSR: wrote %d/%d tiles", written, len(plan))


def _set_output_metadata(dst, ds, device, window, overlap, batch_size, sampling_steps, input_scale):
    """Keep source band descriptions (or label the Sentinel-2 order) and tag provenance."""
    descriptions = tuple(ds.descriptions or ())
    if len(descriptions) == opensr.EXPECTED_BANDS and all(descriptions):
        dst.descriptions = descriptions
    else:
        dst.descriptions = opensr.SENTINEL2_BAND_ORDER
    dst.update_tags(OPENSR_MODEL=opensr.MODEL_ID, OPENSR_DEVICE=str(device),
                    OPENSR_WINDOW=str(window), OPENSR_OVERLAP=str(overlap),
                    OPENSR_BATCH_SIZE=str(batch_size), OPENSR_SAMPLING_STEPS=str(sampling_steps),
                    OPENSR_OUTPUT_DN_SCALE=str(settings.opensr_dn_scale),
                    OPENSR_INPUT_SCALE=("reflectance" if input_scale is None else str(input_scale)),
                    OPENSR_LIMITATIONS=("Model-inferred detail from ESA OpenSR LDSR-S2; not scientifically "
                                        "validated and not a source of newly observed ground detail."))


def process(src, job):
    """Tiled OpenSR processing for one uploaded GeoTIFF; returns the job result dict.

    Raises ValueError / OpenSRError for any unusable input, and removes its temporary
    output, preview and output file if anything fails.
    """
    parameters = job.get("parameters") or {}
    scale = opensr.SCALE_FACTOR
    window = opensr.configured_window_size()
    if "window_size" in parameters:
        if parameters["window_size"] is None:
            raise opensr.OpenSRInputError("parameters.window_size must be an integer")
        window = opensr.configured_window_size(parameters["window_size"])
    overlap = _int_param(parameters, "overlap", settings.opensr_overlap, 0)
    batch_size = _int_param(parameters, "batch_size", settings.opensr_batch_size, 1,
                            settings.opensr_max_batch_size)
    sampling_steps = _int_param(parameters, "sampling_steps", opensr.configured_sampling_steps(), 1,
                                settings.opensr_max_sampling_steps)
    if overlap >= window:
        raise ValueError(f"overlap ({overlap}) must be smaller than the window size ({window})")
    output_scale = float(settings.opensr_dn_scale)
    out_path = settings.data_dir / "outputs" / f"{job['id']}.tif"
    temp_path = settings.data_dir / "temporary" / f"{job['id']}.tif"
    preview_path = settings.data_dir / "previews" / f"{job['id']}.png"
    with rasterio.open(src) as ds:
        input_scale = validate_source(ds, settings.opensr_dn_scale)
        plan = tile_plan(ds.height, ds.width, window, overlap)
        max_tiles = int(settings.opensr_max_tiles)
        if len(plan) > max_tiles:
            raise ValueError(f"OpenSR job needs {len(plan)} tiles which exceeds OPENSR_MAX_TILES={max_tiles}; "
                             "raise the limit or crop the raster")
        validate_tiles(ds, plan, window, input_scale)
        profile = output_profile(ds, scale)
        _, device = opensr.load_model()  # fail fast, before any output file exists
        if batch_size > 1:
            log.warning("OpenSR batch_size=%s: batched execution is documented by the model and the per-image "
                        "spectral correction was verified equivalent, but a full batched forward pass was not "
                        "verified on this host", batch_size)
        try:
            with rasterio.open(temp_path, "w", **profile) as dst:
                _set_output_metadata(dst, ds, device, window, overlap, batch_size, sampling_steps, input_scale)
                write_plan(dst, infer_tiles(ds, plan, window, batch_size, sampling_steps, input_scale),
                           plan, output_scale, window)
            make_preview(temp_path, preview_path)
            os.replace(temp_path, out_path)
        except Exception:
            # Never leave a partial output, preview or temp file behind.
            for path in (temp_path, out_path, preview_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    log.warning("Could not remove %s after a failed OpenSR job", path)
            raise
    return {"output_path": str(out_path), "preview_path": str(preview_path), "output": inspect_raster(out_path),
            "created_at": now(), "method": job["method"], "trained_model_used": True, "inferred_details": True,
            "quantitative_uncertainty": "unavailable", "validation": "not performed; no reference image supplied",
            "limitations": ["Output pixels are model-inferred and do not represent newly observed ground detail.",
                            "No calibrated uncertainty estimate is available.",
                            "Model output is not scientifically validated in this project.",
                            "Tiles are blended with a feathered overlap; blending is approximate and seams may remain."],
            "provenance": {"model_id": opensr.MODEL_ID, "scale_factor": scale, "tiles": len(plan),
                           "window": window, "overlap": overlap, "batch_size": batch_size,
                           "sampling_steps": sampling_steps, "device": device,
                           "input_bands": list(opensr.SENTINEL2_BAND_ORDER),
                           "input_scale": input_scale, "output_dn_scale": output_scale,
                           "blended": bool(overlap > 0),
                           "checkpoint": opensr.status(probe_device=False)["checkpoint"]}}


def blend_weights(rows, columns, overlap_rows=0, overlap_columns=0):
    """Feather weights for a (rows, columns) edge region.

    The ramp rises from 0 to 1 across the shared band on the left edge and/or the top
    edge, and stays 1 everywhere else, so an incoming tile fades in over the pixels a
    neighbour already contributed.
    """
    weights = np.ones((rows, columns), dtype=np.float32)
    if overlap_columns > 0:
        band = min(int(overlap_columns), columns)
        weights[:, :band] = np.linspace(0.0, 1.0, band, endpoint=True, dtype=np.float32)[None, :]
    if overlap_rows > 0:
        band = min(int(overlap_rows), rows)
        weights[:band, :] *= np.linspace(0.0, 1.0, band, endpoint=True, dtype=np.float32)[:, None]
    return weights
