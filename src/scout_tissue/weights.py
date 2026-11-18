"""The two released SCOUT-open heads: download from the Hugging Face hub (pinned revision) or local files,
always checked against the SHA-256 stored in ``config.json``."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

HEADS = ("audit", "fm_head")


class WeightsError(RuntimeError):
    """A downloaded head does not match the released file."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weights(cfg: dict[str, Any], audit_ckpt: str | Path | None = None, fm_head: str | Path | None = None,
                    revision: str | None = None, cache_dir: str | Path | None = None) -> dict[str, dict[str, str]]:
    """``{"audit": {"path", "sha256"}, "fm_head": {...}}``.

    Heads not given as local paths are fetched with ``huggingface_hub.hf_hub_download`` from the released
    repository at the pinned revision (the hub's cache, offline mode and token handling apply). A downloaded file
    must match the stored SHA-256; a local file that does not is accepted with a warning (e.g. a retrained head).
    """
    w = cfg["weights"]
    local = {"audit": audit_ckpt, "fm_head": fm_head}
    out: dict[str, dict[str, str]] = {}
    for name in HEADS:
        expected = w[name]["sha256"]
        if local[name] is not None:
            path = Path(local[name]).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{name} checkpoint not found: {path}")
            sha = sha256_file(path)
            if sha != expected:
                warnings.warn(f"{name} checkpoint {path} is not the released file (sha256 {sha[:12]}..., "
                              f"released {expected[:12]}...)", stacklevel=2)
        else:
            from huggingface_hub import hf_hub_download

            rev = revision or w["revision"]
            path = Path(hf_hub_download(repo_id=w["repo_id"], filename=w[name]["filename"], revision=rev,
                                        cache_dir=None if cache_dir is None else str(cache_dir)))
            sha = sha256_file(path)
            if sha != expected:
                msg = (f"{name}: {w['repo_id']}/{w[name]['filename']}@{rev} has sha256 {sha}, "
                       f"expected {expected}")
                if rev == w["revision"]:
                    raise WeightsError(msg)
                warnings.warn(msg, stacklevel=2)
        out[name] = {"path": str(path), "sha256": sha}
    return out
