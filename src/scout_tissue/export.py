"""Output formats: the bit-packed npz mask (with its grid), a PNG mask and GeoJSON tissue polygons.

npz members: ``mask`` (bit-packed, ``np.unpackbits(mask, count=H*W).reshape(H, W)``), ``shape`` [H, W], ``mpp``
(um per mask px), ``origin_x``, ``origin_y`` (level-0 px), ``scale`` (level-0 px per mask px), ``grid`` (JSON),
``system``, ``meta`` (JSON). Mask pixel (row r, col c) covers level-0 x in
``[origin_x + c*scale, origin_x + (c+1)*scale)`` and y in ``[origin_y + r*scale, origin_y + (r+1)*scale)``.

GeoJSON: one ``Polygon`` feature per 8-connected tissue component, traced along pixel edges (exact: the polygon
area equals the component's pixel area) and mapped to level-0 pixel coordinates with the same convention, i.e.
the grid corner ``(col, row)`` becomes ``(origin_x + col*scale, origin_y + row*scale)``. Exterior rings are
counter-clockwise and holes clockwise in the numeric (x, y) plane (RFC 7946); a ring may touch itself at an
isolated vertex where two tissue pixels meet only diagonally.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from scout_tissue.grid import Grid


# ----------------------------------------------------------------------------- npz
def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=bool).ravel())


def unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    shape = tuple(int(s) for s in shape)
    return np.unpackbits(np.asarray(packed, np.uint8), count=shape[0] * shape[1]).reshape(shape).astype(bool)


def save_npz(path: str | os.PathLike[str], mask: np.ndarray, grid: Grid | dict[str, Any], system: str = "final",
             meta: dict[str, Any] | None = None) -> None:
    """Write the mask npz atomically (temporary file + rename)."""
    g = grid.to_dict() if isinstance(grid, Grid) else dict(grid)
    mask = np.asarray(mask, dtype=bool)
    if tuple(mask.shape) != (int(g["height"]), int(g["width"])):
        raise ValueError(f"mask shape {mask.shape} does not match the grid ({g['height']}, {g['width']})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, mask=pack_mask(mask), shape=np.asarray(mask.shape, np.int64),
                            mpp=np.asarray(float(g["mpp"])), origin_x=np.asarray(float(g["origin_x"])),
                            origin_y=np.asarray(float(g["origin_y"])), scale=np.asarray(float(g["scale"])),
                            grid=np.asarray(json.dumps(g)), system=np.asarray(system),
                            meta=np.asarray(json.dumps(meta or {}, default=str)))
    os.replace(tmp, path)


def load_npz(path: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """``(mask [H, W] bool, grid dict)`` of an output npz."""
    with np.load(path) as z:
        shape = tuple(int(s) for s in z["shape"])
        return unpack_mask(z["mask"], shape), json.loads(str(z["grid"]))


# ----------------------------------------------------------------------------- PNG
def save_png(path: str | os.PathLike[str], mask: np.ndarray, mpp: float | None = None) -> None:
    """8-bit PNG (255 = tissue) on the mask grid; ``mpp`` is stored as the PNG resolution (pHYs, dots per inch)."""
    from PIL import Image

    img = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L")
    kw = {"dpi": (25400.0 / mpp, 25400.0 / mpp)} if mpp else {}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)


# ----------------------------------------------------------------------------- polygons
def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _simplify(ring: list[tuple[int, int]]) -> np.ndarray:
    """Drop vertices where the boundary goes straight on (the ring is given without the closing vertex)."""
    a = np.asarray(ring, dtype=np.int64)
    prev, nxt = np.roll(a, 1, axis=0), np.roll(a, -1, axis=0)
    d1, d2 = a - prev, nxt - a
    turn = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) != 0
    return a[turn] if turn.any() else a


def mask_polygons(mask: np.ndarray) -> list[list[np.ndarray]]:
    """Polygons of the 8-connected components of ``mask`` in grid corner coordinates ``(col, row)``.

    Returns one ``[exterior, hole, ...]`` list per component (label order of ``scipy.ndimage.label``); each ring is
    an ``(N, 2)`` integer array without the closing vertex.
    """
    from scipy import ndimage as ndi

    m = np.asarray(mask, dtype=bool)
    H, W = m.shape
    if not m.any():
        return []
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    P = np.zeros((H + 2, W + 2), bool)
    P[1:-1, 1:-1] = m
    starts, ends, owners = [], [], []

    def add(sel: np.ndarray, start: Any, end: Any, owner: Any) -> None:
        r, c = np.nonzero(sel)
        starts.append(np.c_[start(r, c)])
        ends.append(np.c_[end(r, c)])
        orr, occ = owner(r, c)
        owners.append(orr * W + occ)

    # horizontal edges at y = r between rows r-1 (above) and r (below), columns c
    above, below = P[:-1, 1:-1], P[1:, 1:-1]
    # top edge of pixel (r, c): left -> right; bottom edge of pixel (r-1, c): right -> left
    add(below & ~above, lambda r, c: (c, r), lambda r, c: (c + 1, r), lambda r, c: (r, c))
    add(above & ~below, lambda r, c: (c + 1, r), lambda r, c: (c, r), lambda r, c: (r - 1, c))
    # vertical edges at x = c between columns c-1 (left) and c (right), rows r
    left, right = P[1:-1, :-1], P[1:-1, 1:]
    # left edge of pixel (r, c): bottom -> top; right edge of pixel (r, c-1): top -> bottom
    add(right & ~left, lambda r, c: (c, r + 1), lambda r, c: (c, r), lambda r, c: (r, c))
    add(left & ~right, lambda r, c: (c, r), lambda r, c: (c, r + 1), lambda r, c: (r, c - 1))
    s = np.concatenate(starts)
    e = np.concatenate(ends)
    own = np.concatenate(owners)
    key_s = s[:, 1] * (W + 1) + s[:, 0]
    key_e = e[:, 1] * (W + 1) + e[:, 0]
    out_edges: dict[int, list[int]] = {}
    for i, k in enumerate(key_s.tolist()):
        out_edges.setdefault(k, []).append(i)
    used = np.zeros(len(s), bool)
    comp_label = lab.ravel()[own]
    rings: dict[int, list[np.ndarray]] = {}
    key_e_l, own_l, s_l = key_e.tolist(), own.tolist(), s.tolist()
    for start in range(len(s)):
        if used[start]:
            continue
        ring = []
        cur = start
        while True:
            used[cur] = True
            ring.append(s_l[cur])
            nxt = out_edges[key_e_l[cur]]
            if len(nxt) == 1:
                cur = nxt[0]
            else:  # saddle (two pixels meet diagonally): stay with the other pixel, i.e. 8-connected tissue
                cur = nxt[0] if own_l[nxt[0]] != own_l[cur] else nxt[1]
            if cur == start:
                break
            if used[cur]:
                raise RuntimeError("inconsistent boundary while tracing the mask")
        rings.setdefault(int(comp_label[start]), []).append(_simplify(ring))
    polys = []
    for k in range(1, n + 1):
        rs = rings.get(k, [])
        ext = [q for q in rs if _signed_area(q) > 0]
        holes = [q for q in rs if _signed_area(q) <= 0]
        if len(ext) != 1:
            raise RuntimeError(f"component {k}: expected one exterior ring, got {len(ext)}")
        polys.append([ext[0], *holes])
    return polys


def to_geojson(mask: np.ndarray, grid: Grid | dict[str, Any], properties: dict[str, Any] | None = None
               ) -> dict[str, Any]:
    """GeoJSON ``FeatureCollection`` of the tissue components in level-0 pixel coordinates."""
    g = grid if isinstance(grid, Grid) else Grid.from_dict(grid)
    px_mm2 = g.pixel_area_mm2
    feats = []
    for poly in mask_polygons(mask):
        rings = []
        area_px = 0.0
        for q in poly:
            area_px += _signed_area(q)
            xy = g.grid_to_level0(np.vstack([q, q[:1]]).astype(np.float64))
            rings.append(xy.tolist())
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings},
                      "properties": {"objectType": "annotation", "classification": {"name": "Tissue"},
                                     "area_mm2": area_px * px_mm2}})
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": feats}
    fc["properties"] = {"coordinates": "level-0 pixels (x right, y down)", "mpp": g.mpp, **(properties or {})}
    return fc


def save_geojson(path: str | os.PathLike[str], mask: np.ndarray, grid: Grid | dict[str, Any],
                 properties: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_geojson(mask, grid, properties)))
