"""Browser demo (Gradio): one whole-slide image in, tissue mask, escalated windows and audit risk out.

    scout-tissue demo [--device cuda] [--host 0.0.0.0] [--port 7860]

The slide is given as a path on the machine running the demo (any format OpenSlide reads, including multi-file
formats such as .mrxs) or uploaded (single-file formats). Only pyramid levels are read; label and macro images are
never opened.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scout_tissue.pipeline import Scout, ScoutResult, slide_stem

MAX_SIDE = 1400  # display size of the thumbnails
MASK_RGB = (0, 114, 178)  # mask outline and tint
WIN_RGB = (230, 159, 0)  # escalated windows
UPLOAD_TYPES = [".svs", ".ndpi", ".tif", ".tiff", ".scn", ".bif", ".svslide", ".vms", ".vmu"]


def _small(a: np.ndarray, interp: int) -> np.ndarray:
    s = min(1.0, MAX_SIDE / max(a.shape[:2]))
    if s == 1.0:
        return a
    return cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))), interpolation=interp)


def overlay(res: ScoutResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thumbnail, mask overlay with escalated windows, audit risk) as RGB uint8 at display size."""
    rgb = res.thumbnail
    s = min(1.0, MAX_SIDE / max(rgb.shape[:2]))
    thumb = _small(rgb, cv2.INTER_AREA)
    m = _small(res.mask.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
    tint = thumb.astype(np.float32)
    tint[m] = 0.75 * tint[m] + 0.25 * np.array(MASK_RGB, np.float32)
    img = np.ascontiguousarray(tint.astype(np.uint8))
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, MASK_RGB, 2)
    p = res.window_px
    for rr, cc in res.windows:
        cv2.rectangle(img, (round(cc * s), round(rr * s)), (round((cc + p) * s) - 1, round((rr + p) * s) - 1),
                      WIN_RGB, 1)
    risk = _small(res.risk, cv2.INTER_AREA)
    heat = cv2.applyColorMap(np.clip(np.rint(risk * 255), 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    return thumb, img, (0.45 * thumb.astype(np.float32) + 0.55 * heat).astype(np.uint8)


def summary(res: ScoutResult, name: str) -> str:
    t = res.timing
    return "\n".join([
        f"**SCOUT-open** on {name}",
        f"- Tissue: {res.tissue_area_mm2:.1f} mm² ({res.mask.shape[1]} x {res.mask.shape[0]} px at {res.mpp:g} µm/px)",
        f"- Candidate windows: {res.n_candidates}, sent to the foundation model: {res.n_escalated}",
        f"- Time: {t['total']:.1f} s (foundation model {t['fm_forward']:.1f} s)",
    ])


def build(scout: Scout) -> Any:
    import gradio as gr

    def run(path: str, upload: str | None) -> tuple[Any, ...]:
        wsi = upload or (path or "").strip()
        if not wsi:
            raise gr.Error("Give the path of a slide or upload one")
        if not Path(wsi).is_file():
            raise gr.Error(f"Slide not found: {wsi}")
        try:
            res = scout.predict(wsi, keep_intermediates=True)
        except Exception as exc:  # shown to the user instead of a stack trace
            raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
        thumb, img, heat = overlay(res)
        d = Path(tempfile.mkdtemp(prefix="scout_"))
        files = res.save(d, ("npz", "geojson"), slide_stem(wsi))
        return thumb, img, heat, summary(res, Path(wsi).name), [str(files["npz"]), str(files["geojson"])]

    with gr.Blocks(title="SCOUT tissue detection", delete_cache=(3600, 3600)) as app:
        gr.Markdown("# SCOUT-open: tissue detection for whole-slide images\nA cheap detector covers the slide, an "
                    "audit network finds where it is likely wrong, and only those windows (orange) go to a frozen "
                    "pathology foundation model (H-optimus-0). The blue outline is the final tissue mask at 8 µm/px.")
        with gr.Row():
            with gr.Column(scale=1):
                path = gr.Textbox(label="Slide path on this machine", placeholder="/data/slides/slide.svs")
                upload = gr.File(label="…or upload a slide", type="filepath", file_types=UPLOAD_TYPES)
                go = gr.Button("Detect tissue", variant="primary")
                info = gr.Markdown()
                files = gr.File(label="Mask (.npz) and tissue polygons (.geojson, level-0 pixels)", file_count="multiple")
            with gr.Column(scale=2):
                with gr.Tab("Mask"):
                    img = gr.Image(label="Tissue mask (blue) and escalated windows (orange)", type="numpy")
                with gr.Tab("Thumbnail"):
                    thumb = gr.Image(label="Slide at 8 µm/px", type="numpy")
                with gr.Tab("Audit risk"):
                    heat = gr.Image(label="Where the cheap detector is likely wrong", type="numpy")
        go.click(run, [path, upload], [thumb, img, heat, info, files], concurrency_limit=1)
    return app


def main(args: argparse.Namespace) -> int:
    scout = Scout.from_pretrained(device=args.device, audit_ckpt=args.audit_ckpt, fm_head=args.fm_head,
                                  revision=args.revision, cache_dir=args.cache_dir)
    app = build(scout)
    kw: dict[str, Any] = {"share": args.share, "max_file_size": os.environ.get("SCOUT_DEMO_MAX_FILE", "5gb")}
    if args.host:
        kw["server_name"] = args.host
    if args.port:
        kw["server_port"] = args.port
    app.queue(default_concurrency_limit=1).launch(**kw)
    return 0
