"""Audit tier: inputs of the audit network, its tiled full-slide prediction, candidate windows and the risk gate.

The audit network (a small U-Net) predicts, per pixel of the 8 um/px grid, the probability that the base mask
misses tissue (``pfn``) or calls glass tissue (``pfp``). Candidate windows are the ``p x p`` cells of the grid
within ``near_mm`` of the base mask; they are ranked by the summed risk ``pfn + pfp`` (stable, descending) and the
first ``budget_n(budget, n)`` are escalated to the foundation-model tier.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

N_IN = 10  # RGB, OD x3, base, vote fraction, signed distance, OD floor


def audit_inputs(rgb: np.ndarray, base_h: np.ndarray, cheap: list[np.ndarray], mpp: float,
                 acfg: dict[str, Any]) -> np.ndarray:
    """uint8 ``[H, W, 10]``: RGB, OD x3 (0..2 -> 0..255), base, vote fraction (0 without cheap detectors), signed
    boundary distance, OD floor."""
    from scipy import ndimage as ndi

    od = -np.log10(np.clip((rgb.astype(np.float32) + 1.0) / 256.0, 1e-6, 1.0))
    vote = np.mean(np.stack(cheap), 0) if cheap else np.zeros(base_h.shape, np.float32)
    clip_px = float(acfg["dist_clip_mm"]) * 1000.0 / mpp
    if base_h.all() or not base_h.any():  # no boundary: everything inside / outside, at the clip distance
        sd = np.full(base_h.shape, clip_px if base_h.all() else -clip_px, np.float32)
    else:
        sd = np.where(base_h, ndi.distance_transform_edt(base_h), -ndi.distance_transform_edt(~base_h))
    x = np.empty((*base_h.shape, N_IN), np.uint8)
    x[..., :3] = rgb
    x[..., 3:6] = np.clip(od / 2.0 * 255.0, 0, 255).astype(np.uint8)
    x[..., 6] = base_h * 255
    x[..., 7] = np.rint(vote * 255).astype(np.uint8)
    x[..., 8] = np.rint((np.clip(sd, -clip_px, clip_px) / clip_px + 1.0) * 127.5).astype(np.uint8)
    x[..., 9] = (od.sum(-1) > float(acfg["od_floor"])) * 255
    return x


def make_audit(acfg: dict[str, Any]) -> Any:
    """The audit U-Net (untrained; weights are loaded from the released checkpoint)."""
    import segmentation_models_pytorch as smp

    return smp.Unet(acfg["encoder"], encoder_weights=None, in_channels=int(acfg["in_channels"]),
                    classes=int(acfg["classes"]), decoder_channels=tuple(acfg["decoder_channels"]))


def predict_full(net: Any, x: np.ndarray, tile: int, margin: int, device: str) -> np.ndarray:
    """Sigmoid maps ``[2, H, W]`` by overlapping tiles (centre kept). Outside the slide the input is padded with
    white RGB and 0 in the other channels."""
    import torch

    if tile % 32 or tile <= 2 * margin:
        raise ValueError("tile must be a multiple of 32 and larger than 2 x margin")
    H, W = x.shape[:2]
    step = tile - 2 * margin
    Hp, Wp = math.ceil(H / step) * step, math.ceil(W / step) * step
    xp = np.zeros((Hp + 2 * margin, Wp + 2 * margin, N_IN), np.uint8)
    xp[..., :3] = 255
    xp[margin:margin + H, margin:margin + W] = x
    out = np.zeros((2, Hp, Wp), np.float32)
    net.eval()
    with torch.no_grad():
        for r0 in range(0, Hp, step):
            for c0 in range(0, Wp, step):
                t = torch.from_numpy(np.ascontiguousarray(xp[r0:r0 + tile, c0:c0 + tile])).to(device)
                p = torch.sigmoid(net(t.permute(2, 0, 1)[None].float() / 255.0))[0].cpu().numpy()
                out[:, r0:r0 + step, c0:c0 + step] = p[:, margin:margin + step, margin:margin + step]
    return out[:, :H, :W]


def candidate_region(base: np.ndarray, cheap: list[np.ndarray], mpp: float, near_mm: float) -> np.ndarray:
    """Pixels within ``near_mm`` of the union of the base mask and the cheap detectors (none in v1.4)."""
    from scipy import ndimage as ndi

    u = base | np.logical_or.reduce(list(cheap))
    if not u.any():  # edt of an all-True array measures to a virtual corner
        return np.zeros_like(u)
    return ndi.distance_transform_edt(~u) * mpp <= near_mm * 1000.0


def window_grid(shape: tuple[int, int], p: int) -> list[tuple[int, int]]:
    """Top-left ``(row, col)`` of the non-overlapping ``p x p`` windows of a grid (row-major)."""
    return [(r, c) for r in range(0, shape[0], p) for c in range(0, shape[1], p)]


def candidates(cand: np.ndarray, p: int) -> list[tuple[int, int]]:
    """Windows that intersect the candidate region (row-major order)."""
    return [(r, c) for r, c in window_grid(cand.shape, p) if cand[r:r + p, c:c + p].any()]


def risk_scores(rc: list[tuple[int, int]], p: int, risk: dict[str, np.ndarray]) -> np.ndarray:
    """Summed ``pfn + pfp`` (float32 maps) per window; higher = escalate first."""
    rm = risk["pfn"].astype(np.float32) + risk["pfp"].astype(np.float32)
    return np.array([rm[r:r + p, c:c + p].sum() for r, c in rc])


def budget_n(b: Any, n: int) -> int:
    """Windows escalated for budget ``b``: 'all', an absolute cap, or 'f<frac>' (fraction of candidates, ceil)."""
    if b == "all":
        return n
    if isinstance(b, str) and b.startswith("f"):
        return min(n, math.ceil(float(b[1:]) * n))
    return min(int(b), n)


def chosen_windows(scores: np.ndarray, budget: Any) -> np.ndarray:
    """Candidate indices escalated for ``budget``: stable descending order of ``scores``, first ``budget_n``."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    return order[: budget_n(budget, len(order))]
