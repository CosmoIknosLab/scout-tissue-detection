"""Classical base detector (thresholded tissue score + morphology, all in physical units) and the hole
harmonization applied to the base mask before it enters the audit network.

Base pipeline (order fixed): score (``tissue_score``) -> Gaussian blur (skipped when ``blur_sigma_um == 0``) ->
``score > t`` (``t`` = Otsu of the score, or the given number) -> disk closing -> disk opening -> removal of
8-connected components ``< min_object_mm2`` -> filling of holes with shortest diameter ``< hole_fill_mm``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi

BASE_DEFAULTS: dict[str, Any] = {
    "channel": "saturation",
    "threshold": "otsu",
    "blur_sigma_um": 0.0,
    "close_radius_um": 0.0,
    "open_radius_um": 0.0,
    "min_object_mm2": 0.0,
    "hole_fill_mm": 0.0,
}
CHANNELS = ("gray", "saturation", "od_sum")


# ----------------------------------------------------------------------------- checks and units
def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected a uint8 HxWx3 RGB image, got dtype={rgb.dtype} shape={rgb.shape}")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"empty image, shape={rgb.shape}")
    return np.ascontiguousarray(rgb)


def _check_mpp(mpp: float, name: str = "mpp") -> float:
    mpp = float(mpp)
    if not math.isfinite(mpp) or mpp <= 0:
        raise ValueError(f"{name} must be a positive finite number of um/px, got {mpp!r}")
    return mpp


def um_to_px(length_um: float, mpp: float) -> int:
    """Length in um -> whole pixels at ``mpp`` (rounded, never negative)."""
    return max(0, round(float(length_um) / mpp))


def mm2_to_px(area_mm2: float, mpp: float) -> float:
    """Area in mm^2 -> pixels at ``mpp`` (not rounded)."""
    return float(area_mm2) / (mpp * 1e-3) ** 2


# ----------------------------------------------------------------------------- colour features
def estimate_glass(rgb: np.ndarray, chroma_max: int = 20, top_quantile: float = 0.9, min_pixels: int = 50,
                   near_white_min: int = 250) -> np.ndarray:
    """Per-slide glass colour: per-channel median of the brightest low-chroma, not near-white pixels.

    Fallbacks: fewer than ``min_pixels`` candidates -> low-chroma pixels including near-white ones -> all pixels
    that are not pure white; none -> ``(255, 255, 255)``. Returns float64 ``(3,)`` in ``[1, 255]``.
    """
    rgb = _check_rgb(rgb)
    flat = rgb.reshape(-1, 3)
    not_white = ~np.all(flat == 255, axis=1)
    i16 = flat.astype(np.int16)
    chroma = i16.max(axis=1) - i16.min(axis=1)
    low_chroma = not_white & (chroma <= chroma_max)
    cand = low_chroma & (i16.min(axis=1) < near_white_min)
    if cand.sum() < min_pixels:
        cand = low_chroma
    if cand.sum() < min_pixels:
        cand = not_white
    if not cand.any():
        return np.full(3, 255.0)
    bright = flat.mean(axis=1)
    thr = np.quantile(bright[cand], top_quantile)
    sel = cand & (bright >= thr)
    glass = np.median(flat[sel].astype(np.float64), axis=0)
    return np.clip(glass, 1.0, 255.0)


def od_sum(rgb: np.ndarray, glass: np.ndarray | None = None) -> np.ndarray:
    """Optical density summed over RGB, relative to the glass colour (float32, >= 0)."""
    rgb = _check_rgb(rgb)
    g = estimate_glass(rgb) if glass is None else np.asarray(glass, dtype=np.float64)
    od = -np.log10((rgb.astype(np.float32) + 1.0) / (g.astype(np.float32) + 1.0))
    return np.clip(od, 0.0, None).sum(axis=2, dtype=np.float32)


def tissue_score(rgb: np.ndarray, channel: str) -> np.ndarray:
    """Per-pixel score, higher = more tissue-like (float32): ``255 - luma``, HSV S, or summed OD."""
    if channel == "gray":
        return (255 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).astype(np.float32)
    if channel == "saturation":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    if channel == "od_sum":
        return od_sum(rgb)
    raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold (scikit-image, 256 bins); foreground is ``values > t``. Constant input -> its value."""
    try:
        from skimage.filters import threshold_otsu
    except ImportError as exc:  # not needed by the released configuration (numeric threshold)
        raise ImportError("an 'otsu' base threshold needs scikit-image (pip install scikit-image)") from exc

    v = np.asarray(values)
    vmin, vmax = v.min(), v.max()
    if vmin == vmax:
        return float(vmax)
    return float(threshold_otsu(v, nbins=256))


# ----------------------------------------------------------------------------- morphology in physical units
def _disk_kernel(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def _morph(mask: np.ndarray, op: int, radius_px: int) -> np.ndarray:
    """OpenCV morphology with a disk; OpenCV's default border is neutral (no edge erosion)."""
    if radius_px <= 0:
        return mask
    out = cv2.morphologyEx(mask.astype(np.uint8), op, _disk_kernel(radius_px))
    return out.astype(bool)


def remove_small_objects_mm2(mask: np.ndarray, mpp: float, min_area_mm2: float) -> np.ndarray:
    """Drop 8-connected tissue components with area ``< min_area_mm2``."""
    mask = np.asarray(mask, dtype=bool)
    if min_area_mm2 <= 0 or not mask.any():
        return mask.copy()
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= mm2_to_px(min_area_mm2, mpp)
    keep[0] = False
    return keep[lab]


def fill_holes_smaller_than(mask: np.ndarray, mpp: float, max_diameter_mm: float) -> np.ndarray:
    """Fill holes (4-connected background not touching the border) whose shortest diameter, estimated as the
    diameter of the largest inscribed disk, is ``< max_diameter_mm`` (ambiguous widths re-measured at 2x)."""
    mask = np.asarray(mask, dtype=bool)
    if max_diameter_mm <= 0 or not mask.any():
        return mask.copy()
    bg = ~mask
    lab, n = ndi.label(bg)  # 4-connectivity for background
    if n == 0:
        return mask.copy()
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    dist = ndi.distance_transform_edt(bg)
    dmax_all = np.zeros(n + 1, dtype=np.float64)
    sel = lab > 0
    np.maximum.at(dmax_all, lab[sel], dist[sel])
    dmax = dmax_all[1:]
    thr_px = round(max_diameter_mm * 1e3 / mpp, 9)  # rounded: 0.072 mm / 8 um must be exactly 9 px
    width_lo, width_hi = 2.0 * dmax - 1.0, 2.0 * dmax  # px, bounds on the true shortest width
    fill = np.zeros(n + 1, dtype=bool)
    fill[1:] = width_hi < thr_px
    is_border = np.zeros(n + 1, dtype=bool)
    is_border[border] = True
    ambiguous = np.flatnonzero((width_lo < thr_px) & (width_hi >= thr_px)) + 1
    ambiguous = ambiguous[~is_border[ambiguous]]
    if ambiguous.size:
        objs = ndi.find_objects(lab)
        for lbl in ambiguous:
            rs, cs = objs[lbl - 1]
            sub = lab[rs.start - 1 : rs.stop + 1, cs.start - 1 : cs.stop + 1] == lbl
            up = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
            fill[lbl] = float(ndi.distance_transform_edt(up).max()) < thr_px
    fill[is_border] = False
    fill[0] = False
    return mask | fill[lab]


# ----------------------------------------------------------------------------- base detector
def validate_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Merge with ``BASE_DEFAULTS`` and check keys and values."""
    unknown = set(params) - set(BASE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown base parameters: {sorted(unknown)}")
    p = {**BASE_DEFAULTS, **params}
    if p["channel"] not in CHANNELS:
        raise ValueError(f"unknown channel {p['channel']!r}")
    t = p["threshold"]
    if not (t == "otsu" or (isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t))):
        raise ValueError(f"threshold must be 'otsu' or a finite number, got {t!r}")
    for key in ("blur_sigma_um", "close_radius_um", "open_radius_um", "min_object_mm2", "hole_fill_mm"):
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{key} must be a finite number >= 0, got {v!r}")
    return p


def base_mask(rgb: np.ndarray, mpp: float, params: Mapping[str, Any]) -> np.ndarray:
    """Base tissue mask (bool ``H x W``) of the classical detector with ``params``."""
    rgb = _check_rgb(rgb)
    mpp = _check_mpp(mpp)
    p = validate_params(params)
    score = tissue_score(rgb, p["channel"])
    sigma_px = float(p["blur_sigma_um"]) / mpp
    if sigma_px > 0:
        score = cv2.GaussianBlur(score, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px, borderType=cv2.BORDER_REFLECT)
    thr = _otsu(score) if p["threshold"] == "otsu" else float(p["threshold"])
    mask = score > thr
    mask = _morph(mask, cv2.MORPH_CLOSE, um_to_px(p["close_radius_um"], mpp))
    mask = _morph(mask, cv2.MORPH_OPEN, um_to_px(p["open_radius_um"], mpp))
    mask = remove_small_objects_mm2(mask, mpp, p["min_object_mm2"])
    return fill_holes_smaller_than(mask, mpp, p["hole_fill_mm"])


# ----------------------------------------------------------------------------- hole harmonization
_FOUR = ndi.generate_binary_structure(2, 1)


def hole_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label image of the holes of ``mask`` (4-connected background not touching the border) and their count."""
    mask = np.asarray(mask, dtype=bool)
    bg_labels, n = ndi.label(~mask, structure=_FOUR)
    if n == 0:
        return bg_labels, 0
    border = np.unique(np.concatenate([bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]]))
    is_hole = np.ones(n + 1, dtype=bool)
    is_hole[0] = False
    is_hole[border] = False
    hole_ids = np.nonzero(is_hole)[0]
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[hole_ids] = np.arange(1, len(hole_ids) + 1)
    return remap[bg_labels], len(hole_ids)


def _min_width(points: np.ndarray) -> float:
    """Minimum caliper width of a 2-D point set (rotating calipers over the convex-hull edges)."""
    from scipy.spatial import ConvexHull

    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    edges /= np.hypot(edges[:, 0], edges[:, 1])[:, None]
    normals = np.c_[-edges[:, 1], edges[:, 0]]
    proj = hull @ normals.T
    return float((proj.max(axis=0) - proj.min(axis=0)).min())


def hole_feret_widths_um(mask: np.ndarray, mpp: float, below_um: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Label image of the holes of ``mask`` and the minimum Feret width (um) of each hole (pixel squares).

    With ``below_um``, holes whose shorter bounding-box side is already ``< below_um`` get that side (an upper
    bound on the width) instead of the exact width.
    """
    labels, n = hole_labels(mask)
    widths = np.zeros(n)
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    for k, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        side_um = min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * float(mpp)
        if below_um is not None and side_um < below_um:
            widths[k - 1] = side_um
            continue
        rr, cc = np.nonzero(labels[sl] == k)
        pts = (np.c_[cc, rr][:, None, :] + corners[None]).reshape(-1, 2)
        widths[k - 1] = _min_width(np.unique(pts, axis=0)) * float(mpp)
    return labels, widths


def harmonize(mask: np.ndarray, mpp: float, hole_max_mm: float, size: str = "feret") -> np.ndarray:
    """``mask`` with holes of minimum Feret width ``< hole_max_mm`` filled (``hole_max_mm <= 0``: ``mask``)."""
    if size != "feret":
        raise ValueError(f"only the 'feret' hole size is supported, got {size!r}")
    mask = np.asarray(mask, dtype=bool)
    if float(hole_max_mm) <= 0 or mask.size == 0 or not mask.any():
        return mask
    thr = float(hole_max_mm) * 1000.0
    labels, diam = hole_feret_widths_um(mask, mpp, below_um=thr)
    if diam.size == 0:
        return mask
    return mask | np.r_[False, diam < thr][labels]
