"""Command line: ``scout-tissue predict | download | demo``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

FORMATS = ("npz", "png", "geojson")


def _formats(text: str) -> tuple[str, ...]:
    fs = tuple(dict.fromkeys(f.strip().lower() for f in text.split(",") if f.strip()))
    bad = [f for f in fs if f not in FORMATS]
    if bad or not fs:
        raise argparse.ArgumentTypeError(f"formats must be a comma-separated subset of {','.join(FORMATS)}")
    return fs


def _weights_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("weights")
    g.add_argument("--audit-ckpt", default=None, help="local audit head (default: download the released one)")
    g.add_argument("--fm-head", default=None, help="local H-optimus-0 decoder head (default: download)")
    g.add_argument("--revision", default=None, help="hub revision of the released heads (default: pinned commit)")
    g.add_argument("--cache-dir", default=None, help="cache directory for the heads (default: Hugging Face cache)")
    g.add_argument("--offline", action="store_true", help="use cached files only (sets HF_HUB_OFFLINE=1)")


def build_parser() -> argparse.ArgumentParser:
    from scout_tissue import __version__

    ap = argparse.ArgumentParser(prog="scout-tissue", description="SCOUT-open tissue detection for whole-slide images")
    ap.add_argument("--version", action="version", version=f"scout-tissue {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="tissue masks of one or more slides (models loaded once)")
    p.add_argument("slides", nargs="+", help="whole-slide images (any format OpenSlide reads)")
    p.add_argument("--out", required=True, help="output directory (<slide name>.npz / .png / .geojson)")
    p.add_argument("--device", default=None, help="cuda, cpu or a torch device (default: cuda if available)")
    p.add_argument("--format", dest="formats", type=_formats, default=("npz",),
                   help="comma-separated: npz,png,geojson (default: npz)")
    p.add_argument("--readers", type=int, default=None,
                   help="window-reader processes (default: $SLURM_CPUS_PER_TASK or the CPU count)")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: as --readers)")
    p.add_argument("--timing", action="store_true", help="also write <slide name>.timing.json")
    p.add_argument("--skip-existing", action="store_true", help="skip slides whose outputs all exist")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip the dummy forward pass")
    _weights_args(p)

    d = sub.add_parser("download", help="fetch the released heads into the cache (for later offline use)")
    _weights_args(d)

    w = sub.add_parser("demo", help="browser demo (needs: pip install 'scout-tissue[demo]')")
    w.add_argument("--device", default=None)
    w.add_argument("--host", default=None, help="server address (default: Gradio's, 127.0.0.1)")
    w.add_argument("--port", type=int, default=None, help="server port (default: Gradio's, 7860)")
    w.add_argument("--share", action="store_true", help="create a public Gradio share link")
    _weights_args(w)
    return ap


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_predict(args: argparse.Namespace) -> int:
    import torch

    from scout_tissue.pipeline import Scout, n_threads, slide_stem

    stems = [slide_stem(s) for s in args.slides]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        _log(f"error: several slides would write the same output name: {', '.join(dup)}")
        return 2
    missing = [s for s in args.slides if not Path(s).exists()]
    if missing:
        _log(f"error: slide not found: {', '.join(missing)}")
        return 2
    torch.set_num_threads(args.threads or n_threads())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    scout = Scout.from_pretrained(device=args.device, audit_ckpt=args.audit_ckpt, fm_head=args.fm_head,
                                  revision=args.revision, cache_dir=args.cache_dir, warmup=args.warmup)
    _log(f"[scout-tissue] models loaded on {scout.device} in {time.perf_counter() - t0:.1f} s")
    failed = 0
    for i, (slide, stem) in enumerate(zip(args.slides, stems), 1):
        if args.skip_existing and all((out / f"{stem}.{f}").exists() for f in args.formats):
            _log(f"[scout-tissue] {i}/{len(stems)} {stem}: outputs exist, skipped")
            continue
        try:
            res = scout.predict(slide, readers=args.readers)
            res.save(out, args.formats, stem)
            if args.timing:
                info = {"timing_s": res.timing, "n_candidates": res.n_candidates, "n_escalated": res.n_escalated,
                        "shape": list(res.shape), **scout.environment()}
                (out / f"{stem}.timing.json").write_text(json.dumps(info, indent=1))
            _log(f"[scout-tissue] {i}/{len(stems)} {stem}: {res.tissue_area_mm2:.1f} mm2 tissue, "
                 f"{res.n_escalated}/{res.n_candidates} windows escalated, {res.timing['total']:.1f} s")
        except Exception as exc:  # noqa: BLE001  (one unreadable slide must not stop the batch)
            failed += 1
            _log(f"[scout-tissue] {i}/{len(stems)} {stem}: FAILED ({type(exc).__name__}: {exc})")
            if os.environ.get("SCOUT_TISSUE_DEBUG"):
                traceback.print_exc()
    if failed:
        _log(f"[scout-tissue] {failed} of {len(stems)} slides failed")
    return 1 if failed else 0


def cmd_download(args: argparse.Namespace) -> int:
    from scout_tissue.config import load_config
    from scout_tissue.weights import resolve_weights

    w = resolve_weights(load_config(), args.audit_ckpt, args.fm_head, args.revision, args.cache_dir)
    for k, v in w.items():
        print(f"{k}: {v['path']} (sha256 {v['sha256']})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    try:
        from scout_tissue import demo
    except ImportError as exc:
        _log(f"error: the demo needs extra packages: pip install 'scout-tissue[demo]' ({exc})")
        return 2
    return demo.main(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"  # before huggingface_hub / timm are imported
    return {"predict": cmd_predict, "download": cmd_download, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
