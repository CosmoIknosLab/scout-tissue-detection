# scout-tissue

Tissue detection for whole-slide images (WSI) with **SCOUT-open**. A cheap classical detector covers the whole slide, a small audit network predicts where it is likely wrong, and only those windows are re-segmented by a frozen pathology foundation model (H-optimus-0). The output is a binary tissue mask at 8 µm/px, placed exactly on the slide's level-0 pixel frame.

```
slide ─► thumbnail 8 µm/px ─► base mask (optical density + morphology)
                           └─► audit U-Net: per-pixel risk ─► rank 448 µm windows near tissue by risk
                                                              └─► top half re-read at 2 µm/px
                                                                  ─► H-optimus-0 + decoder head
      final mask = base, with the escalated windows replaced by the fused FM decision,
                   components < 0.01 mm² removed
```

## Install

```bash
pip install scout-tissue                 # once published on PyPI
pip install git+https://github.com/CosmoIknosLab/scout-tissue-detection
pip install "scout-tissue[demo]"         # with the browser demo (Gradio)
```

Python ≥ 3.10. OpenSlide comes with the `openslide-bin` wheel; no system library is needed. Install PyTorch first if you need a specific CUDA build (see pytorch.org).

**H-optimus-0 is gated.** Accept its terms at <https://huggingface.co/bioptimus/H-optimus-0> and log in (`huggingface-cli login` or `HF_TOKEN`). The SCOUT heads (~7 MB) and H-optimus-0 (~4.5 GB) are downloaded on first use into your Hugging Face cache (`HF_HOME`); afterwards `--offline` (or `HF_HUB_OFFLINE=1`) works without network.

## Quick start

```bash
scout-tissue predict slide1.svs slide2.ndpi --out masks/ --format npz,png,geojson
```

Models are loaded once for all slides. Options: `--device cuda|cpu`, `--timing` (per-stage times), `--skip-existing`, `--readers N` (window-reader processes), `--audit-ckpt` / `--fm-head` (local head files instead of the download), `--cache-dir`, `--offline`. `scout-tissue download` fetches the heads in advance.

```python
from scout_tissue import Scout

scout = Scout.from_pretrained(device="cuda")
res = scout.predict("slide.svs")
res.mask            # bool array [H, W] on the 8 µm/px grid
res.mpp, res.grid   # 8.0, Grid(origin_x, origin_y, scale, height, width, mpp)
res.to_geojson()    # tissue polygons in level-0 pixel coordinates
res.save("masks/", ("npz", "png", "geojson"))
```

Browser demo: `scout-tissue demo [--host 0.0.0.0] [--port 7860]` (slide path on the server or upload; shows thumbnail, mask, escalated windows and audit risk).

## Output and coordinates

`<slide>.npz` (read with `scout_tissue.load_npz(path)` → `(mask, grid)`):

| member | content |
|---|---|
| `mask` | bit-packed mask: `np.unpackbits(z["mask"], count=H*W).reshape(H, W)` |
| `shape` | `[H, W]` |
| `mpp` | µm per mask pixel (8.0) |
| `origin_x`, `origin_y` | level-0 pixel coordinates of the mask's top-left corner |
| `scale` | level-0 pixels per mask pixel (`mpp / mpp0`, not rounded) |
| `grid` | the same as JSON; `system` (`"final"`); `meta` (JSON: counts, thresholds, versions, weight hashes) |

Mask pixel (row `r`, column `c`) covers level-0 `x ∈ [origin_x + c·scale, origin_x + (c+1)·scale)` and `y ∈ [origin_y + r·scale, origin_y + (r+1)·scale)`. Level-0 coordinates are OpenSlide's (origin at the top-left corner of level 0, x right, y down); the origin is the slide's `openslide.bounds-x/y` (0 if absent, e.g. non-zero for many `.mrxs`). The last row/column may extend past the scanned area (treated as glass).

`<slide>.png`: the same mask, 255 = tissue, one pixel per mask pixel (the PNG resolution field stores the 8 µm/px). `<slide>.geojson`: one `Polygon` feature per 8-connected tissue component, traced along pixel edges and converted with the convention above, so vertices are level-0 pixel coordinates (features carry a `Tissue` classification, so QuPath can import them as annotations).

The slide's level-0 MPP must be in its metadata; slides without a usable MPP are rejected, never read at a guessed resolution. Only pyramid levels are read; label and macro images are never opened.

## Hardware

A GPU is strongly recommended: about 30–60 s per slide on an RTX 3090, depending on the amount of tissue (window reads at 2 µm/px use one process per CPU core). The CPU path works but the foundation model then takes minutes per slide. GPU memory: ~6 GB.

## Performance

On the Tissues-with-Issues TEST set (113 case groups, 153 slides), SCOUT-open fails on 19.5 % of case groups (95 % CI 12.6–28.0) under the paper's failure rule. See the paper for the protocol and comparisons.

## Licences

- Code: Apache-2.0 (`LICENSE`).
- SCOUT heads (downloaded from [Cristiandieti/scout-tissue-detection](https://huggingface.co/Cristiandieti/scout-tissue-detection), also on Zenodo, doi:10.5281/zenodo.23664072): CC BY-NC 4.0 (`LICENSE-WEIGHTS.md`).
- H-optimus-0 (Bioptimus): Apache-2.0, gated on Hugging Face; not redistributed.
- The paper's main variant uses Virchow2 as the foundation model; its decoder head is not redistributed because Virchow2's licence (CC-BY-NC-ND 4.0) restricts models trained on its outputs. This package runs SCOUT-open only.

## Citation

Cristian Tommasino and Antonio M. Rinaldi (University of Naples Federico II). *SCOUT: Risk-Guided Foundation-Model Escalation for Robust Tissue Detection in Whole-Slide Images.* Preprint, under review. See `CITATION.cff`.
