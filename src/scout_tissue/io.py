"""Whole-slide reading at a fixed physical resolution, window reads, and the unscanned-padding rule.

Thumbnail (``read_at_mpp_info``): an ``H x W x 3`` uint8 RGB raster at ``target_mpp`` um/px and the
:class:`~scout_tissue.grid.Grid` that places it on the slide's level-0 frame.

1. Level-0 MPP from ``openslide.mpp-x/y``. Missing, non-positive, out-of-range or anisotropic values raise
   :class:`MPPError` (the resolution is never guessed). For generic TIFFs without a ``ResolutionUnit`` tag
   OpenSlide silently assumes inches, so that case raises too.
2. Grid over the scanned area: origin ``(openslide.bounds-x, openslide.bounds-y)`` when present (else
   ``(0, 0)``), area ``bounds-width x bounds-height`` when present (else the level-0 dimensions).
3. Level geometry per axis (:func:`axis_pitch`): the true integer level-0 extent of a level pixel when the
   pyramid was built by floor/ceil halving, else the per-axis ratio.
4. Source level: the coarsest level whose MPP (``mpp0 * max(pitch_x, pitch_y)``) is ``<= target_mpp``
   (relative tolerance ``LEVEL_RTOL``); level 0 if none is fine enough.
5. Exact area resampling, tile by tile (overlap-weighted averages, two sparse matrix products per tile).
6. Transparent pixels are composited onto white (glass) before averaging.

Windows (``read_window_rgb``): a level-0 square read at the coarsest level whose MPP is ``<=`` the window's
target MPP, composited on white and resized with OpenCV (``INTER_AREA`` when downsampling).

Privacy: only pyramid levels and the numeric properties needed here are read; associated images (label,
macro, thumbnail) are never accessed.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from scout_tissue.grid import Grid

LEVEL_RTOL = 1e-3  # a level at 8.0025 um/px counts as "<= 8 um/px"
DEFAULT_TILE_PX = 1024


class MPPError(ValueError):
    """The slide's level-0 MPP is missing or unusable; the slide must be flagged, not guessed."""


# ---------------------------------------------------------------------------------------- metadata
def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _prop_float(props: Any, key: str) -> float | None:
    raw = props.get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _prop_int(props: Any, key: str) -> int | None:
    raw = props.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def mpp_status(mpp_x: float | None, mpp_y: float | None, lo: float = 0.05, hi: float = 2.0,
               rel_tol: float = 0.01) -> str:
    """Classify a slide's level-0 resolution: ok / missing / out_of_range / anisotropic."""
    if mpp_x is None or mpp_y is None or not (mpp_x > 0 and mpp_y > 0):
        return "missing"
    if not (lo <= mpp_x <= hi and lo <= mpp_y <= hi):
        return "out_of_range"
    if abs(mpp_x - mpp_y) > rel_tol * max(mpp_x, mpp_y):
        return "anisotropic"
    return "ok"


def _rational(value: Any) -> float | None:
    try:
        num, den = value
        return float(num) / float(den) if den else None
    except (TypeError, ValueError):
        return _to_float(str(value))


def tiff_resolution_tags(path: str) -> dict[str, Any]:
    """Presence and values of the page-0 TIFF resolution tags (numeric tags only, no pixel data)."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        xres = tags.get("XResolution")
        yres = tags.get("YResolution")
        unit = tags.get("ResolutionUnit")
        return {
            "tiff_res_unit_present": unit is not None,
            "tiff_res_unit_value": None if unit is None else int(unit.value),
            "tiff_xres": None if xres is None else _rational(xres.value),
            "tiff_yres": None if yres is None else _rational(yres.value),
        }


def slide_mpp0(props: Any, path: str | None = None) -> float:
    """Isotropic level-0 MPP from OpenSlide properties, or raise :class:`MPPError`."""
    mpp_x = _prop_float(props, "openslide.mpp-x")
    mpp_y = _prop_float(props, "openslide.mpp-y")
    status = mpp_status(mpp_x, mpp_y)
    if status != "ok":
        raise MPPError(f"level-0 MPP unusable (status={status}, mpp_x={mpp_x}, mpp_y={mpp_y}); "
                       "the slide must be flagged, not read with a guessed resolution")
    if (path is not None and props.get("openslide.vendor") == "generic-tiff"
            and not tiff_resolution_tags(path)["tiff_res_unit_present"]):
        raise MPPError("generic TIFF without ResolutionUnit: OpenSlide assumed inches, the MPP may be "
                       "off by 2.54x (status=unit_implicit); flag the slide instead of guessing")
    return 0.5 * (mpp_x + mpp_y)


# ---------------------------------------------------------------------------------------- geometry
def choose_level(mpp0: float, level_downsamples: list[float] | tuple[float, ...], target_mpp: float,
                 rtol: float = LEVEL_RTOL) -> int:
    """Index of the coarsest level whose MPP is ``<= target_mpp`` (the finest level if none is)."""
    level_mpps = [mpp0 * float(ds) for ds in level_downsamples]
    eligible = [i for i, m in enumerate(level_mpps) if m <= target_mpp * (1.0 + rtol)]
    if not eligible:  # target finer than every level: the finest one
        return min(range(len(level_mpps)), key=lambda i: level_mpps[i])
    return max(eligible, key=lambda i: level_mpps[i])


def composite_on_white(rgba: np.ndarray) -> np.ndarray:
    """Non-premultiplied ``H x W x 4`` uint8 RGBA -> ``H x W x 3`` float32 RGB on a white background."""
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected an H x W x 3/4 image, got shape {arr.shape}")
    rgb = arr[..., :3].astype(np.float32)
    if arr.shape[2] == 3:
        return rgb
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    return rgb * alpha + 255.0 * (1.0 - alpha)


def axis_weights(out_start: float, out_step: float, n_out: int,
                 in_start: float, in_step: float, n_in: int) -> sp.csr_matrix:
    """Sparse ``n_out x n_in`` area-resampling weights along one axis (overlap lengths, rows sum to 1)."""
    k = np.arange(n_out, dtype=np.float64)
    a = out_start + k * out_step
    b = a + out_step
    j_lo = np.floor((a - in_start) / in_step).astype(np.int64)
    m = math.ceil(out_step / in_step) + 2  # max input pixels one output pixel can touch
    j = j_lo[:, None] + np.arange(m)[None, :]
    s = in_start + j * in_step
    overlap = np.minimum(b[:, None], s + in_step) - np.maximum(a[:, None], s)
    keep = (overlap > 0) & (j >= 0) & (j < n_in)
    rows = np.broadcast_to(np.arange(n_out)[:, None], j.shape)[keep]
    w = sp.csr_matrix((overlap[keep], (rows, j[keep])), shape=(n_out, n_in))
    row_sum = np.asarray(w.sum(axis=1)).ravel()
    if n_out and np.any(row_sum <= 0):
        raise ValueError("source region does not cover every output pixel")
    w = sp.diags(1.0 / row_sum) @ w if n_out else w
    return sp.csr_matrix(w, dtype=np.float32)


def _resample_block(src: np.ndarray, wy: sp.csr_matrix, wx: sp.csr_matrix) -> np.ndarray:
    """``out[r, c, :] = sum_ij wy[r, i] * wx[c, j] * src[i, j, :]`` with sparse products."""
    n_y, n_x, ch = src.shape
    tmp = wy @ src.reshape(n_y, n_x * ch)  # (h, n_x * ch)
    h = tmp.shape[0]
    tmp = np.ascontiguousarray(tmp.reshape(h, n_x, ch).transpose(1, 0, 2)).reshape(n_x, h * ch)
    out = wx @ tmp  # (w, h * ch)
    return out.reshape(wx.shape[0], h, ch).transpose(1, 0, 2)


def _block_ranges(n: int, step: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def axis_pitch(n0: int, n_level: int, ds_openslide: float) -> tuple[float, str]:
    """True level-0 px per level px along one axis, and the rule that gave it.

    Candidates ``2 ** round(log2(ds))`` then ``round(ds)``: the first ``t`` with ``0 <= n0 - t * n_level < t``
    (floor halving) or ``0 <= t * n_level - n0 < t`` (ceil halving) gives ``(t, "integer")``, else
    ``(n0 / n_level, "ratio")``.
    """
    if n0 <= 0 or n_level <= 0 or not ds_openslide > 0:
        raise ValueError(f"level sizes and downsample must be positive, got {n0!r}, {n_level!r}, {ds_openslide!r}")
    for t in (2 ** max(0, round(math.log2(ds_openslide))), max(1, round(ds_openslide))):
        if 0 <= n0 - t * n_level < t or 0 <= t * n_level - n0 < t:
            return float(t), "integer"
    return n0 / n_level, "ratio"


def level_geometry(slide: Any) -> list[dict[str, Any]]:
    """Per-level ``ds`` (OpenSlide), ``pitch_x/y`` and ``rule_x/y`` (see :func:`axis_pitch`)."""
    w0, h0 = slide.dimensions
    out = []
    for ds, (wl, hl) in zip(slide.level_downsamples, slide.level_dimensions):
        px, rx = axis_pitch(int(w0), int(wl), float(ds))
        py, ry = axis_pitch(int(h0), int(hl), float(ds))
        out.append({"ds": float(ds), "pitch_x": px, "pitch_y": py, "rule_x": rx, "rule_y": ry})
    return out


def _source_span(start0: float, end0: float, pitch: float, ds_openslide: float) -> tuple[float, int, int, float]:
    """Source pixels covering level-0 ``[start0, end0)`` on a level: ``(in_start, n, loc, phase)``."""
    k = math.floor(start0 / pitch + 1e-9)
    n = max(1, math.ceil(end0 / pitch - 1e-9) - k)
    loc = round(k * ds_openslide)
    return k * pitch, n, loc, loc / ds_openslide - k


# ---------------------------------------------------------------------------------------- thumbnail
def read_slide_at_mpp(slide: Any, target_mpp: float, tile_px: int = DEFAULT_TILE_PX,
                      path: str | None = None) -> tuple[np.ndarray, Grid, dict[str, Any]]:
    """``(rgb, grid, info)`` of an opened OpenSlide-like object at ``target_mpp`` um/px."""
    if not target_mpp > 0:
        raise ValueError(f"target_mpp must be positive, got {target_mpp!r}")
    if tile_px < 4:
        raise ValueError(f"tile_px must be >= 4, got {tile_px!r}")
    props = slide.properties
    mpp0 = slide_mpp0(props, path)

    w0, h0 = slide.dimensions
    bx = _prop_int(props, "openslide.bounds-x")
    by = _prop_int(props, "openslide.bounds-y")
    bw = _prop_int(props, "openslide.bounds-width")
    bh = _prop_int(props, "openslide.bounds-height")
    origin = (float(bx or 0), float(by or 0))
    grid = Grid.for_slide(bw if bw is not None else w0, bh if bh is not None else h0,
                          mpp0, target_mpp, origin=origin)

    geometry = level_geometry(slide)
    pitches = [max(g["pitch_x"], g["pitch_y"]) for g in geometry]
    level = choose_level(mpp0, pitches, target_mpp)
    geo = geometry[level]
    ds, px, py = geo["ds"], geo["pitch_x"], geo["pitch_y"]
    f = grid.scale / min(px, py)  # output pixel size in source pixels (largest axis)
    block = max(1, int((tile_px - 2) // f))  # output pixels per block side

    rgb = np.empty((grid.height, grid.width, 3), dtype=np.uint8)
    t_read = t_resample = 0.0
    n_reads = 0

    col_blocks = []
    for c0, c1 in _block_ranges(grid.width, block):
        x_start = grid.origin_x + c0 * grid.scale
        sx, n_x, x_loc, _ = _source_span(x_start, grid.origin_x + c1 * grid.scale, px, ds)
        col_blocks.append((c0, c1, x_loc, n_x, axis_weights(x_start, grid.scale, c1 - c0, sx, px, n_x)))

    for r0, r1 in _block_ranges(grid.height, block):
        y_start = grid.origin_y + r0 * grid.scale
        sy, n_y, y_loc, _ = _source_span(y_start, grid.origin_y + r1 * grid.scale, py, ds)
        wy = axis_weights(y_start, grid.scale, r1 - r0, sy, py, n_y)
        for c0, c1, x_loc, n_x, wx in col_blocks:
            t0 = time.perf_counter()
            region = np.asarray(slide.read_region((x_loc, y_loc), level, (n_x, n_y)))
            t1 = time.perf_counter()
            if region.shape[:2] != (n_y, n_x):
                raise RuntimeError(f"read_region returned shape {region.shape}, expected ({n_y}, {n_x})")
            out = _resample_block(composite_on_white(region), wy, wx)
            rgb[r0:r1, c0:c1] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
            t_read += t1 - t0
            t_resample += time.perf_counter() - t1
            n_reads += 1

    info = {
        "mpp0": mpp0,
        "target_mpp": float(target_mpp),
        "level": level,
        "level_mpp": mpp0 * max(px, py),
        "vendor": props.get("openslide.vendor"),
        "n_reads": n_reads,
        "read_seconds": round(t_read, 4),
        "resample_seconds": round(t_resample, 4),
    }
    return rgb, grid, info


def read_at_mpp_info(path: str, target_mpp: float,
                     tile_px: int = DEFAULT_TILE_PX) -> tuple[np.ndarray, Grid, dict[str, Any]]:
    """Open ``path`` with OpenSlide and read it at ``target_mpp`` um/px: ``(rgb, grid, info)``."""
    import openslide

    slide = openslide.OpenSlide(path)
    try:
        rgb, grid, info = read_slide_at_mpp(slide, target_mpp, tile_px=tile_px, path=path)
    finally:
        slide.close()
    info["format"] = os.path.splitext(path)[1].lower().lstrip(".")
    return rgb, grid, info


# ---------------------------------------------------------------------------------------- padding
def padding_mask(rgb: np.ndarray, mpp: float, max_per_channel: int = 10, min_area_mm2: float = 1.0) -> np.ndarray:
    """Unscanned-area padding: 8-connected components of pixels with every channel ``<= max_per_channel`` and
    area ``>= min_area_mm2`` (opaque black stored outside the scanned region by some scanners/converters)."""
    from scipy import ndimage as ndi

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    black = np.all(rgb[..., :3] <= max_per_channel, axis=-1)
    if not black.any():
        return black
    lab, n = ndi.label(black, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    min_px = min_area_mm2 / (float(mpp) / 1000.0) ** 2
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def whiten(rgb: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Copy of ``rgb`` with padding pixels set to white (glass)."""
    out = np.array(rgb, copy=True)
    out[pad] = 255
    return out


# ---------------------------------------------------------------------------------------- windows
def read_window_rgb(slide: Any, geom: list[dict[str, Any]], mpp0: float, x0: float, y0: float, size0: float,
                    target_mpp: float, out_px: int) -> np.ndarray:
    """RGB of the level-0 square ``[x0, x0+size0) x [y0, y0+size0)`` at ``target_mpp``, ``out_px`` side."""
    import cv2

    pitches = [max(g["pitch_x"], g["pitch_y"]) for g in geom]
    lvl = choose_level(mpp0, pitches, target_mpp)
    pitch = pitches[lvl]
    n = max(1, math.ceil(size0 / pitch))
    reg = slide.read_region((round(x0), round(y0)), lvl, (n, n))
    rgb = composite_on_white(np.asarray(reg))
    interp = cv2.INTER_AREA if n >= out_px else cv2.INTER_LINEAR
    out = cv2.resize(rgb, (out_px, out_px), interpolation=interp)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


_READER: dict[str, Any] = {}


def reader_init(path: str) -> None:
    """Process-pool initializer: open the slide once per reader process."""
    import openslide

    sl = openslide.OpenSlide(path)
    _READER.update(slide=sl, geom=level_geometry(sl), mpp0=slide_mpp0(sl.properties, path))


def read_batch(job: tuple[list[tuple[float, float]], float, float, int]) -> np.ndarray:
    """``[n, px, px, 3]`` uint8 windows with top-left level-0 corners ``xy``, side ``size_um``, at ``mpp_in``."""
    xy, size_um, mpp_in, px = job
    sl, geom, mpp0 = _READER["slide"], _READER["geom"], _READER["mpp0"]
    return np.stack([read_window_rgb(sl, geom, mpp0, x, y, size_um / mpp0, mpp_in, px) for x, y in xy])


def timed_read(job: tuple[list[tuple[float, float]], float, float, int]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    rgb = read_batch(job)
    return rgb, time.perf_counter() - t0
