"""The frozen parameters of the released SCOUT-open configuration (``config.json``, shipped with the package)."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """The packaged configuration, or a JSON file with the same structure."""
    if path is None:
        text = resources.files("scout_tissue").joinpath("config.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)
