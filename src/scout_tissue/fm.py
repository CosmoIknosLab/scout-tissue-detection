"""Foundation-model tier: frozen H-optimus-0 patch tokens + a small decoder head, on the escalated windows only.

Each window is a ``px_metric x px_metric`` cell of the 8 um/px grid (448 um), re-read from the slide at
``mpp_input`` (2 um/px, 224 px). The backbone returns 16 x 16 patch tokens; the head projects them, upsamples to
56 x 56, concatenates the RGB average-pooled to 8 um/px and predicts tissue logits on the grid.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from scout_tissue import io


def build_fm(fcfg: dict[str, Any], device: str) -> tuple[Any, int]:
    """Frozen FM (eval mode) returning patch tokens ``[B, 256, D]`` for ``[0, 1]`` images ``[B, 3, 224, 224]``.

    Loaded with timm from the Hugging Face hub (``hf-hub:<repo>``); gated models need an accepted licence and a
    token (``HF_TOKEN`` or ``huggingface-cli login``).
    """
    import timm
    import torch

    kw = dict(fcfg["timm_kwargs"])
    if kw.get("mlp_layer") == "SwiGLUPacked":
        kw["mlp_layer"] = timm.layers.SwiGLUPacked
    if kw.get("act_layer") == "SiLU":
        kw["act_layer"] = torch.nn.SiLU
    net = timm.create_model(f"hf-hub:{fcfg['repo']}", pretrained=True, num_classes=0, **kw).eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    m = torch.tensor(fcfg["mean"], device=device).view(1, 3, 1, 1)
    s = torch.tensor(fcfg["std"], device=device).view(1, 3, 1, 1)
    n_prefix = int(getattr(net, "num_prefix_tokens", 1))

    @torch.no_grad()
    def tokens(x: torch.Tensor) -> torch.Tensor:
        return net.forward_features((x - m) / s)[:, n_prefix:]

    return tokens, int(net.num_features)


def make_head(token_dim: int) -> Any:
    """Decoder head: 16x16 tokens -> 1x1 conv -> upsample to 56x56, concat RGB at 8 um/px -> 3 convs -> logits."""
    import torch
    from torch import nn

    class FMHead(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(d, 128, 1), nn.GELU())
            self.dec = nn.Sequential(nn.Conv2d(128 + 3, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))

        def forward(self, rgb: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
            b, n, d = tok.shape
            g = round(n ** 0.5)
            f = self.proj(tok.float().transpose(1, 2).reshape(b, d, g, g))
            f = nn.functional.interpolate(f, size=(56, 56), mode="bilinear", align_corners=False)
            r8 = nn.functional.avg_pool2d(rgb, 4) * 2 - 1
            return self.dec(torch.cat([f, r8], 1))[:, 0]

    return FMHead(int(token_dim))


def window_origins(grid: dict[str, Any], windows: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """Level-0 ``(x, y)`` of the top-left corner of each ``(row, col)`` window."""
    return [(grid["origin_x"] + c * grid["scale"], grid["origin_y"] + r * grid["scale"]) for r, c in windows]


def fm_probs(fm: Any, head: Any, device: str, wcfg: dict[str, Any], batch: int, wsi: str, grid: dict[str, Any],
             windows: list[tuple[int, int]], readers: int, timer: Any) -> np.ndarray:
    """FM probabilities ``[n, p, p]`` (float32) for ``windows``, in batches of ``batch`` consecutive windows;
    window reads run in a process pool and overlap with inference."""
    import torch

    p = int(wcfg["px_metric"])
    size_um = float(wcfg["px_metric"]) * float(wcfg["mpp_metric"])
    org = window_origins(grid, windows)
    jobs = [(org[b:b + batch], size_um, float(wcfg["mpp_input"]), int(wcfg["input_px"]))
            for b in range(0, len(org), batch)]
    probs: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_fwd = 0.0
    if jobs:
        with ProcessPoolExecutor(readers, initializer=io.reader_init, initargs=(wsi,)) as ex:
            for rgb, _ in ex.map(io.timed_read, jobs):  # ordered
                tb = time.perf_counter()
                with torch.no_grad():
                    x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255
                    tok = fm(x).half()
                    probs.append(torch.sigmoid(head(x, tok)).float().cpu().numpy())  # .cpu() synchronizes
                t_fwd += time.perf_counter() - tb
    timer.add("fm_forward", t_fwd)
    timer.add("window_reads", time.perf_counter() - t_start - t_fwd)
    return np.concatenate(probs) if probs else np.zeros((0, p, p), np.float32)
