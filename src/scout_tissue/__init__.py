"""SCOUT-open: risk-guided foundation-model escalation for tissue detection in whole-slide images.

    from scout_tissue import Scout
    scout = Scout.from_pretrained(device="cuda")
    res = scout.predict("slide.svs")
    res.mask, res.mpp, res.grid, res.to_geojson()
"""

from scout_tissue.config import load_config
from scout_tissue.export import load_npz, to_geojson
from scout_tissue.grid import Grid
from scout_tissue.pipeline import Scout, ScoutResult

__version__ = "1.0.0"

__all__ = ["Grid", "Scout", "ScoutResult", "__version__", "load_config", "load_npz", "to_geojson"]
