"""Raster grid at a fixed physical resolution, anchored in slide level-0 pixel coordinates.

Coordinate convention:

- Level-0 coordinates are OpenSlide level-0 pixel coordinates ``(x, y)``: origin at the top-left
  corner of the level-0 image, ``x`` to the right (columns), ``y`` downwards (rows). A level-0
  pixel ``(i, j)`` covers ``[i, i + 1) x [j, j + 1)``. For formats with a bounds offset (``.mrxs``,
  some ``.ndpi``) the grid origin is ``(openslide.bounds-x, openslide.bounds-y)``, still expressed
  in level-0 coordinates.
- Grid pixel ``(r, c)`` (row, column) covers level-0
  ``x in [origin_x + c * scale, origin_x + (c + 1) * scale)`` and
  ``y in [origin_y + r * scale, origin_y + (r + 1) * scale)``,
  with ``scale = mpp / mpp0`` level-0 pixels per grid pixel (any positive float).
- Continuous grid coordinates use the same corner convention: the level-0 point ``(x, y)`` maps to
  ``((x - origin_x) / scale, (y - origin_y) / scale)`` in ``(col, row)`` order, so grid pixel
  ``(r, c)`` spans ``[c, c + 1) x [r, r + 1)``.
- Arrays on the grid have shape ``(height, width[, channels])``. The last row/column may extend past
  the scanned area (filled as glass).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

# Guards ceil() against floating-point noise: 1000 / (8 / 0.5) must give 63, not 64.
_CEIL_DECIMALS = 9


def _ceil(value: float) -> int:
    return math.ceil(round(value, _CEIL_DECIMALS))


@dataclass(frozen=True)
class Grid:
    """A raster of ``height x width`` pixels of ``mpp`` um, placed on the slide's level-0 frame."""

    origin_x: float  # level-0 px
    origin_y: float  # level-0 px
    scale: float  # level-0 px per grid px (= mpp / mpp0)
    height: int
    width: int
    mpp: float  # um per grid px

    def __post_init__(self) -> None:
        for name in ("origin_x", "origin_y", "scale", "mpp"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"Grid.{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.scale <= 0 or self.mpp <= 0:
            raise ValueError(f"Grid.scale and Grid.mpp must be positive, got {self.scale!r}, {self.mpp!r}")
        for name in ("height", "width"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"Grid.{name} must be a non-negative integer, got {value!r}")
            object.__setattr__(self, name, int(value))

    @classmethod
    def for_slide(cls, width0: float, height0: float, mpp0: float, target_mpp: float,
                  origin: tuple[float, float] = (0.0, 0.0)) -> Grid:
        """Grid covering a ``width0 x height0`` level-0 area at ``target_mpp`` um/px (last row/column: ``ceil``)."""
        if not (mpp0 > 0 and target_mpp > 0):
            raise ValueError(f"mpp0 and target_mpp must be positive, got {mpp0!r}, {target_mpp!r}")
        if width0 < 0 or height0 < 0:
            raise ValueError(f"width0 and height0 must be non-negative, got {width0!r}, {height0!r}")
        scale = float(target_mpp) / float(mpp0)
        return cls(origin_x=float(origin[0]), origin_y=float(origin[1]), scale=scale,
                   height=_ceil(height0 / scale), width=_ceil(width0 / scale), mpp=float(target_mpp))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def mpp0(self) -> float:
        """Level-0 um/px implied by the grid (``mpp / scale``)."""
        return self.mpp / self.scale

    @property
    def pixel_area_mm2(self) -> float:
        return (self.mpp * 1e-3) ** 2

    def level0_to_grid(self, xy: np.ndarray) -> np.ndarray:
        """Level-0 points ``(N, 2)`` in ``(x, y)`` order -> continuous grid coordinates ``(col, row)``."""
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"expected an (N, 2) array of (x, y) points, got shape {pts.shape}")
        return (pts - np.array([self.origin_x, self.origin_y])) / self.scale

    def grid_to_level0(self, cr: np.ndarray) -> np.ndarray:
        """Grid ``(col, row)`` points ``(N, 2)`` -> level-0 ``(x, y)``."""
        pts = np.asarray(cr, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"expected an (N, 2) array of (col, row) points, got shape {pts.shape}")
        return pts * self.scale + np.array([self.origin_x, self.origin_y])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (plain floats and ints)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Grid:
        names = [f.name for f in fields(cls)]
        missing = [n for n in names if n not in d]
        if missing:
            raise KeyError(f"Grid dict is missing {missing}")
        return cls(**{n: d[n] for n in names})
