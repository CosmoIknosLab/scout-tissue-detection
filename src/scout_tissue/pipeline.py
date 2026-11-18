"""SCOUT-open on one whole-slide image, end to end.

1. Thumbnail at 8 um/px on the slide's metric grid (``io.read_at_mpp_info``); unscanned black padding is
   detected and treated as glass.
2. Base mask: thresholded optical density + morphology (``base.base_mask``), padding removed.
3. Audit tier: the base mask with small holes filled (``base.harmonize``) and the thumbnail form the 10-channel
   input of the audit U-Net, which predicts per-pixel miss / false-tissue risk (stored as float16).
4. Candidate windows (448 um cells of the grid within 0.5 mm of the base mask) are ranked by summed risk (stable,
   descending); the top half (``budget = "f0.5"``, ceil) is escalated.
5. FM tier on the escalated windows only: the window re-read at 2 um/px, frozen H-optimus-0 tokens, the decoder
   head, threshold ``pr >= thr`` (float).
6. Fusion inside the escalated windows: ``fm_bin & ((q >= 0.5) | base)`` with ``q`` the probability quantized to
   uint8 (``rint(255 p) / 255``); then padding removed and 8-connected components ``< 0.01 mm^2`` dropped.

The output mask is the raw (un-harmonized) tissue mask on the 8 um/px grid.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scout_tissue import audit as A
from scout_tissue import base as B
from scout_tissue import export as X
from scout_tissue import io
from scout_tissue.config import load_config
from scout_tissue.grid import Grid

STAGES = ("open_thumbnail", "classical", "audit_inputs", "audit_head", "candidates_gating", "window_reads",
          "fm_forward", "fusion")


# ----------------------------------------------------------------------------- pure helpers
def fuse(base: np.ndarray, windows: list[tuple[int, int]], fm_bin: np.ndarray, fm_q: np.ndarray | None, p: int,
         cut: float | None) -> np.ndarray:
    """Paste FM decisions of ``windows`` (top-left corners on the grid) into a copy of ``base``.

    ``fm_bin`` / ``fm_q`` are ``[n, p, p]``, aligned with ``windows``; windows cut by the slide border keep their
    top-left part. ``cut=None``: FM binary only; otherwise ``fm_bin & ((fm_q >= cut) | base)``.
    """
    final = base.copy()
    for i, (rr, cc) in enumerate(windows):
        v = final[rr:rr + p, cc:cc + p]
        h, w = v.shape
        fb, bb = fm_bin[i][:h, :w], base[rr:rr + h, cc:cc + w]
        v[...] = fb if cut is None else fb & ((fm_q[i][:h, :w] >= cut) | bb)
    return final


def quantize_prob(pr: np.ndarray) -> np.ndarray:
    """Probability quantized to uint8 (``np.rint(pr * 255)``) and read back as float32 / 255."""
    return np.rint(pr * 255).astype(np.uint8).astype(np.float32) / 255.0


def n_threads() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


class Timer:
    """Wall-clock seconds per stage; CUDA work is synchronized at stage ends."""

    def __init__(self, device: str) -> None:
        self.cuda = str(device).startswith("cuda")
        self.t: dict[str, float] = {}

    def sync(self) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()

    def add(self, stage: str, seconds: float) -> None:
        self.t[stage] = self.t.get(stage, 0.0) + float(seconds)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.add(name, time.perf_counter() - t0)


# ----------------------------------------------------------------------------- result
@dataclass
class ScoutResult:
    """Tissue mask of one slide on its 8 um/px grid, with the grid and what the pipeline did."""

    slide: str
    mask: np.ndarray  # bool [H, W]
    grid: Grid
    n_candidates: int
    n_escalated: int
    windows: list[tuple[int, int]]  # escalated windows (row, col) on the grid, in risk order
    window_px: int
    threshold: float
    timing: dict[str, float]
    n_padding_px: int
    meta: dict[str, Any] = field(default_factory=dict)
    thumbnail: np.ndarray | None = None  # uint8 [H, W, 3] (padding whitened), with keep_intermediates=True
    risk: np.ndarray | None = None  # float32 [H, W], max(pfn, pfp), with keep_intermediates=True

    @property
    def mpp(self) -> float:
        return self.grid.mpp

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def tissue_area_mm2(self) -> float:
        return float(self.mask.sum()) * self.grid.pixel_area_mm2

    def to_geojson(self) -> dict[str, Any]:
        """Tissue polygons in level-0 pixel coordinates (see ``scout_tissue.export``)."""
        return X.to_geojson(self.mask, self.grid, {"slide": Path(self.slide).name})

    def save_npz(self, path: str | os.PathLike[str]) -> None:
        X.save_npz(path, self.mask, self.grid, "final", self.meta)

    def save_png(self, path: str | os.PathLike[str]) -> None:
        X.save_png(path, self.mask, self.grid.mpp)

    def save_geojson(self, path: str | os.PathLike[str]) -> None:
        X.save_geojson(path, self.mask, self.grid, {"slide": Path(self.slide).name})

    def save(self, out_dir: str | os.PathLike[str], formats: tuple[str, ...] = ("npz",),
             stem: str | None = None) -> dict[str, Path]:
        """Write ``<stem>.npz`` / ``.png`` / ``.geojson`` in ``out_dir``; returns the paths."""
        stem = stem or slide_stem(self.slide)
        out = {}
        for f in formats:
            p = Path(out_dir) / f"{stem}.{f}"
            {"npz": self.save_npz, "png": self.save_png, "geojson": self.save_geojson}[f](p)
            out[f] = p
        return out


def slide_stem(path: str | os.PathLike[str]) -> str:
    """Output file stem of a slide: its file name without the extension."""
    return Path(path).name.rsplit(".", 1)[0] if "." in Path(path).name else Path(path).name


# ----------------------------------------------------------------------------- model
class Scout:
    """SCOUT-open with its networks loaded once on ``device``; ``predict`` runs one slide."""

    def __init__(self, cfg: dict[str, Any], device: str, audit: Any, fm: Any, head: Any, threshold: float,
                 weights: dict[str, dict[str, str]], load_seconds: float) -> None:
        self.cfg, self.device = cfg, device
        self.audit, self.fm, self.head = audit, fm, head
        self.threshold = float(threshold)
        self.weights = weights
        self.load_seconds = load_seconds

    @classmethod
    def from_pretrained(cls, device: str | None = None, audit_ckpt: str | os.PathLike[str] | None = None,
                        fm_head: str | os.PathLike[str] | None = None, revision: str | None = None,
                        cache_dir: str | os.PathLike[str] | None = None, warmup: bool = True,
                        config: str | os.PathLike[str] | None = None) -> Scout:
        """Load the released heads (downloaded from the hub unless local paths are given) and the frozen
        H-optimus-0 backbone (gated on the hub: accept its terms and log in first)."""
        import torch

        from scout_tissue.fm import build_fm, make_head
        from scout_tissue.weights import resolve_weights

        cfg = load_config(config)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        w = resolve_weights(cfg, audit_ckpt, fm_head, revision, cache_dir)
        t0 = time.perf_counter()
        ck = torch.load(w["audit"]["path"], map_location=device, weights_only=True)
        audit = A.make_audit(cfg["audit"]).to(device)
        audit.load_state_dict(ck["state"])
        audit.eval()
        hk = torch.load(w["fm_head"]["path"], map_location=device, weights_only=True)
        head = make_head(hk["token_dim"]).to(device)
        head.load_state_dict(hk["state"])
        head.eval()
        fm, _ = build_fm(cfg["fm"], device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        s = cls(cfg, device, audit, fm, head, float(hk["threshold"]), w, time.perf_counter() - t0)
        if warmup:
            s.warmup()
        return s

    def warmup(self) -> float:
        """One dummy forward of the audit network and of the FM tier (CUDA context, kernels); returns seconds."""
        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            tile = int(self.cfg["audit"]["infer_tile_px"])
            self.audit(torch.zeros((1, A.N_IN, tile, tile), device=self.device))
            s = int(self.cfg["window"]["input_px"])
            x = torch.zeros((2, 3, s, s), device=self.device)
            self.head(x, self.fm(x).half())
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def environment(self) -> dict[str, Any]:
        import torch

        from scout_tissue import __version__

        env = {"scout_tissue": __version__, "torch": torch.__version__, "python": sys.version.split()[0],
               "device": self.device, "audit_sha256": self.weights["audit"]["sha256"],
               "fm_head_sha256": self.weights["fm_head"]["sha256"]}
        if str(self.device).startswith("cuda"):
            env["gpu"] = torch.cuda.get_device_name(torch.device(self.device))
        return env

    def predict(self, slide: str | os.PathLike[str], readers: int | None = None,
                keep_intermediates: bool = False) -> ScoutResult:
        """Tissue mask of one slide (any format OpenSlide reads, with level-0 MPP in its metadata)."""
        from scout_tissue.fm import fm_probs

        cfg, dev = self.cfg, self.device
        wsi = str(slide)
        timer = Timer(dev)
        t_total = time.perf_counter()
        with timer.stage("open_thumbnail"):
            rgb, grid, info = io.read_at_mpp_info(wsi, float(cfg["mpp"]))
        mpp = float(grid.mpp)
        if abs(mpp - float(cfg["mpp"])) > 1e-9:
            raise RuntimeError(f"thumbnail at {mpp} um/px, expected {cfg['mpp']}")
        g = grid.to_dict()
        pad = io.padding_mask(rgb, mpp, int(cfg["padding"]["max_per_channel"]),
                              float(cfg["padding"]["min_area_mm2"]))
        if pad.any():
            rgb = io.whiten(rgb, pad)
        with timer.stage("classical"):
            base = B.base_mask(rgb, mpp, cfg["base"])
            base &= ~pad
        cheap: list[np.ndarray] = []  # v1.4: no cheap-detector ensemble (vote channel 0)
        with timer.stage("audit_inputs"):
            hz = cfg["harmonization"]
            base_h = B.harmonize(base, mpp, float(hz["hole_max_mm"]), hz["size"]) & ~pad
            x = A.audit_inputs(rgb, base_h, cheap, mpp, cfg["audit"])
        with timer.stage("audit_head"):
            pr2 = A.predict_full(self.audit, x, int(cfg["audit"]["infer_tile_px"]),
                                 int(cfg["audit"]["infer_margin_px"]), dev)
            risk = {"pfn": pr2[0].astype(np.float16), "pfp": pr2[1].astype(np.float16)}
            del pr2
        p = int(cfg["window"]["px_metric"])
        with timer.stage("candidates_gating"):
            cand = A.candidate_region(base, cheap, mpp, float(cfg["candidates"]["near_mm"])) & ~pad
            rc = A.candidates(cand, p)
            if cfg["gate"] != "risk":
                raise ValueError(f"unsupported gate {cfg['gate']!r}")
            scores = A.risk_scores(rc, p, risk)
            chosen = A.chosen_windows(scores, cfg["budget"])
        run_idx = sorted(int(i) for i in chosen)
        pr = fm_probs(self.fm, self.head, dev, cfg["window"], int(cfg["fm"]["batch"]), wsi, g,
                      [rc[i] for i in run_idx], readers or n_threads(), timer)
        with timer.stage("fusion"):
            pos = {i: k for k, i in enumerate(run_idx)}
            sel = [pos[int(i)] for i in chosen]
            fm_bin = pr[sel] >= self.threshold  # float threshold
            fm_q = quantize_prob(pr[sel])
            win = [rc[int(i)] for i in chosen]
            fused = fuse(base, win, fm_bin, fm_q, p, float(cfg["fusion_cut"])) & ~pad
            final = B.remove_small_objects_mm2(fused, mpp, float(cfg["min_component_mm2"]))
        t = {k: round(timer.t.get(k, 0.0), 4) for k in STAGES}
        t["total"] = round(time.perf_counter() - t_total, 4)
        meta = {"system": "final", "variant": cfg["name"], "config_version": cfg["version"],
                "n_candidates": len(rc), "n_escalated": len(chosen), "fm_threshold": self.threshold,
                "fusion_cut": float(cfg["fusion_cut"]), "budget": cfg["budget"],
                "min_component_mm2": float(cfg["min_component_mm2"]), "n_padding_px": int(pad.sum()),
                "read_info": {k: info.get(k) for k in ("mpp0", "level", "level_mpp", "format", "vendor")},
                **self.environment()}
        res = ScoutResult(slide=wsi, mask=final, grid=grid, n_candidates=len(rc), n_escalated=len(chosen),
                          windows=win, window_px=p, threshold=self.threshold, timing=t,
                          n_padding_px=int(pad.sum()), meta=meta)
        if keep_intermediates:
            res.thumbnail = np.ascontiguousarray(rgb)
            res.risk = np.maximum(risk["pfn"], risk["pfp"]).astype(np.float32)
        return res
