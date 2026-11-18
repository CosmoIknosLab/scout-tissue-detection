"""Synthetic tests (no WSI, no weights, CPU only)."""

from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pytest

from scout_tissue import audit as A
from scout_tissue import base as B
from scout_tissue import export as X
from scout_tissue import io
from scout_tissue.config import load_config
from scout_tissue.grid import Grid
from scout_tissue.pipeline import fuse, quantize_prob, slide_stem


# ----------------------------------------------------------------------------- config
def test_config_loads_and_is_complete():
    c = load_config()
    assert c["mpp"] == 8.0
    assert c["budget"] == "f0.5" and c["gate"] == "risk"
    assert c["fusion_cut"] == 0.5 and c["min_component_mm2"] == 0.01
    assert c["window"] == {"px_metric": 56, "mpp_metric": 8.0, "mpp_input": 2.0, "input_px": 224}
    assert c["window"]["px_metric"] * c["window"]["mpp_metric"] == c["window"]["input_px"] * c["window"]["mpp_input"]
    assert B.validate_params(c["base"]) == c["base"]
    assert c["audit"]["in_channels"] == A.N_IN
    assert c["fm"]["repo"] == "bioptimus/H-optimus-0"
    w = c["weights"]
    assert re.fullmatch(r"[0-9a-f]{40}", w["revision"])
    for k in ("audit", "fm_head"):
        assert re.fullmatch(r"[0-9a-f]{64}", w[k]["sha256"])


def test_local_weights_sha_check(tmp_path):
    from scout_tissue.weights import resolve_weights, sha256_file

    c = load_config()
    a, h = tmp_path / "a.pt", tmp_path / "h.pt"
    a.write_bytes(b"audit")
    h.write_bytes(b"head")
    with pytest.warns(UserWarning, match="not the released file"):
        w = resolve_weights(c, a, h)
    assert w["audit"]["sha256"] == sha256_file(a)
    with pytest.raises(FileNotFoundError):
        resolve_weights(c, tmp_path / "missing.pt", h)


# ----------------------------------------------------------------------------- fusion and quantization
def test_quantize_prob_matches_uint8_storage():
    p = np.array([0.0, 0.25, 0.4990, 0.4995, 0.5, 0.9999, 1.0], np.float32)
    q = quantize_prob(p)
    assert q.dtype == np.float32
    np.testing.assert_allclose(q * 255, np.rint(p * 255), atol=1e-4)
    # 0.4990 -> 127/255 < 0.5; 0.4995 -> 127.37 -> 127 < 0.5; 0.5 -> 127.5 -> 128 (round half to even) >= 0.5
    assert list(q >= 0.5) == [False, False, False, False, True, True, True]


def test_fuse_rule_and_border_windows():
    p = 4
    base = np.zeros((6, 6), bool)
    base[0:2, 0:4] = True
    windows = [(0, 0), (4, 4)]  # the second is cut by the border (2 x 2 inside)
    fm_bin = np.ones((2, p, p), bool)
    fm_q = np.zeros((2, p, p), np.float32)
    fm_q[0, 3, 3] = 0.6  # confident FM pixel outside the base
    out = fuse(base, windows, fm_bin, fm_q, p, 0.5)
    exp = base.copy()
    exp[3, 3] = True  # FM & (q >= 0.5 | base): base kept, the confident pixel added
    np.testing.assert_array_equal(out, exp)
    assert not base[3, 3]  # input untouched
    out_v1 = fuse(base, windows, fm_bin, None, p, None)  # FM binary only
    assert out_v1[:4, :4].all() and out_v1[4:, 4:].all() and not out_v1[4:, :4].any()
    fm_bin[0, 0, 0] = False  # the FM can remove base tissue
    assert not fuse(base, windows, fm_bin, fm_q, p, 0.5)[0, 0]


def test_budget_and_stable_order():
    assert A.budget_n("f0.5", 7) == 4 and A.budget_n("f0.5", 0) == 0
    assert A.budget_n("all", 5) == 5 and A.budget_n(3, 10) == 3
    scores = np.array([1.0, 3.0, 3.0, 2.0, 0.5])
    np.testing.assert_array_equal(A.chosen_windows(scores, "f0.5"), [1, 2, 3])  # ties keep index order


def test_candidates_and_risk_scores():
    base = np.zeros((10, 13), bool)
    base[1, 1] = True
    cand = A.candidate_region(base, [], 8.0, 0.016)  # 2 px
    assert cand[1, 3] and not cand[1, 4]
    assert not A.candidate_region(np.zeros((4, 4), bool), [], 8.0, 0.5).any()
    rc = A.candidates(cand, 4)
    assert rc == [(0, 0)]
    assert A.window_grid((10, 13), 4)[:4] == [(0, 0), (0, 4), (0, 8), (0, 12)]
    risk = {"pfn": np.full((10, 13), 0.25, np.float16), "pfp": np.zeros((10, 13), np.float16)}
    np.testing.assert_allclose(A.risk_scores([(0, 0), (8, 12)], 4, risk), [4.0, 0.5])


def test_audit_inputs_channels():
    rgb = np.full((20, 30, 3), 255, np.uint8)
    rgb[5:15, 5:15] = (120, 60, 150)
    base_h = np.zeros((20, 30), bool)
    base_h[5:15, 5:15] = True
    x = A.audit_inputs(rgb, base_h, [], 8.0, {"dist_clip_mm": 0.5, "od_floor": 0.05})
    assert x.shape == (20, 30, 10) and x.dtype == np.uint8
    np.testing.assert_array_equal(x[..., :3], rgb)
    assert (x[..., 6] == base_h * 255).all() and (x[..., 7] == 0).all()
    assert x[10, 10, 8] > 128 > x[0, 29, 8]
    assert x[10, 10, 9] == 255 and x[0, 0, 9] == 0


def test_predict_full_tiles_cover_the_slide():
    torch = pytest.importorskip("torch")

    class Probe(torch.nn.Module):  # logit = channel 6 (base) rescaled: output reproduces the input
        def forward(self, t):
            return torch.cat([t[:, 6:7] * 20 - 10, -(t[:, 6:7] * 20 - 10)], 1)

    x = np.zeros((70, 90, 10), np.uint8)
    x[10:50, 20:80, 6] = 255
    out = A.predict_full(Probe(), x, 64, 8, "cpu")
    assert out.shape == (2, 70, 90)
    np.testing.assert_array_equal(out[0] > 0.5, x[..., 6] > 0)


# ----------------------------------------------------------------------------- base detector
def test_base_mask_disk_and_holes():
    rgb = np.full((120, 120, 3), 242, np.uint8)
    yy, xx = np.mgrid[:120, :120]
    disk = (yy - 60) ** 2 + (xx - 60) ** 2 < 40 ** 2
    rgb[disk] = (170, 90, 160)
    hole = (yy - 60) ** 2 + (xx - 60) ** 2 < 4 ** 2  # 64 um wide hole: filled (< 1 mm)
    rgb[hole] = 242
    c = load_config()
    m = B.base_mask(rgb, 8.0, c["base"])
    assert m[hole].all() and abs(int(m.sum()) - int(disk.sum())) <= 0.02 * disk.sum()
    assert not m[:10].any() and not m[:, :10].any()
    assert B.remove_small_objects_mm2(m, 8.0, 1.0).sum() == 0  # the disk is ~0.32 mm^2
    assert B.remove_small_objects_mm2(m, 8.0, 0.01).sum() == m.sum()


def test_harmonize_fills_narrow_holes_only():
    m = np.ones((100, 100), bool)
    m[10:13, 10:60] = False  # 3 px = 24 um wide crack: filled at 0.25 mm
    m[40:80, 40:80] = False  # 320 um hole: kept
    h = B.harmonize(m, 8.0, 0.25)
    assert h[10:13, 10:60].all() and not h[40:80, 40:80].any()
    np.testing.assert_array_equal(B.harmonize(m, 8.0, 0.0), m)


def test_padding_rule():
    rgb = np.full((300, 300, 3), 240, np.uint8)
    rgb[:, :150] = 0  # 150 x 300 px at 8 um = 2.88 mm^2: padding
    rgb[200:205, 200:205] = 0  # small black mark: kept
    pad = io.padding_mask(rgb, 8.0, 10, 1.0)
    assert pad[:, :150].all() and not pad[:, 150:].any()
    assert (io.whiten(rgb, pad)[:, :150] == 255).all()


# ----------------------------------------------------------------------------- reading (fake slide)
class FakeSlide:
    """OpenSlide-like object over an in-memory RGBA level 0 and a 4x-downsampled level 1."""

    def __init__(self, rgba: np.ndarray, mpp: float | None = 0.5, bounds: tuple[int, int] | None = None):
        self.l0 = rgba
        h, w = rgba.shape[:2]
        self.l1 = rgba.reshape(h // 4, 4, w // 4, 4, 4).mean((1, 3)).round().astype(np.uint8)
        self.dimensions = (w, h)
        self.level_dimensions = [(w, h), (w // 4, h // 4)]
        self.level_downsamples = [1.0, 4.0]
        self.properties = {"openslide.vendor": "fake"}
        if mpp is not None:
            self.properties.update({"openslide.mpp-x": str(mpp), "openslide.mpp-y": str(mpp)})
        if bounds is not None:
            self.properties.update({"openslide.bounds-x": str(bounds[0]), "openslide.bounds-y": str(bounds[1])})

    def read_region(self, loc, level, size):
        img = self.l0 if level == 0 else self.l1
        ds = self.level_downsamples[level]
        x, y = round(loc[0] / ds), round(loc[1] / ds)
        out = np.zeros((size[1], size[0], 4), np.uint8)
        src = img[max(0, y):y + size[1], max(0, x):x + size[0]]
        out[max(0, -y):max(0, -y) + src.shape[0], max(0, -x):max(0, -x) + src.shape[1]] = src
        return out


def _fake(color=(200, 100, 50)):
    rgba = np.zeros((512, 768, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = 255
    return rgba


def test_read_slide_at_mpp_uniform_and_grid():
    sl = FakeSlide(_fake())
    rgb, grid, info = io.read_slide_at_mpp(sl, 8.0)
    assert grid.scale == 16.0 and grid.shape == (32, 48) and info["level"] == 1
    assert (rgb == (200, 100, 50)).all()
    sl = FakeSlide(_fake(), bounds=(32, 16))
    _, grid, _ = io.read_slide_at_mpp(sl, 8.0)
    assert (grid.origin_x, grid.origin_y) == (32.0, 16.0)


def test_missing_mpp_is_rejected():
    with pytest.raises(io.MPPError):
        io.read_slide_at_mpp(FakeSlide(_fake(), mpp=None), 8.0)
    with pytest.raises(io.MPPError):
        io.slide_mpp0({"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.5"})


def test_read_window_rgb_size_and_white_outside():
    sl = FakeSlide(_fake())
    geom = io.level_geometry(sl)
    w = io.read_window_rgb(sl, geom, 0.5, 0, 0, 448 / 0.5, 2.0, 224)
    assert w.shape == (224, 224, 3) and w.dtype == np.uint8
    assert (w[:100, :100] == (200, 100, 50)).all()  # inside the fake slide
    assert (w[-10:, -10:] == 255).all()  # transparent outside -> white


def test_axis_pitch_and_choose_level():
    assert io.axis_pitch(1001, 500, 2.002) == (2.0, "integer")
    assert io.choose_level(0.25, [1, 4, 16, 64], 8.0) == 2
    assert io.choose_level(0.25, [1, 4], 0.1) == 0


# ----------------------------------------------------------------------------- grid, npz, png
def test_grid_transforms():
    g = Grid.for_slide(1000, 500, 0.5, 8.0, origin=(10.0, 20.0))
    assert g.shape == (32, 63) and g.scale == 16.0
    pts = np.array([[10.0, 20.0], [26.0, 52.0]])
    np.testing.assert_allclose(g.level0_to_grid(pts), [[0, 0], [1, 2]])
    np.testing.assert_allclose(g.grid_to_level0(g.level0_to_grid(pts)), pts)
    assert Grid.from_dict(json.loads(json.dumps(g.to_dict()))) == g


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (17, 9), (64, 64)])
def test_pack_roundtrip(shape):
    rng = np.random.default_rng(0)
    m = rng.random(shape) > 0.5
    np.testing.assert_array_equal(X.unpack_mask(X.pack_mask(m), shape), m)


def test_npz_format(tmp_path):
    g = Grid(origin_x=5.0, origin_y=7.0, scale=32.0, height=9, width=11, mpp=8.0)
    m = np.random.default_rng(1).random(g.shape) > 0.3
    p = tmp_path / "s.npz"
    X.save_npz(p, m, g, "final", {"n_escalated": 3})
    with np.load(p) as z:
        assert sorted(z.files) == sorted(["mask", "shape", "mpp", "origin_x", "origin_y", "scale", "grid", "system",
                                          "meta"])
        assert str(z["system"]) == "final" and json.loads(str(z["meta"]))["n_escalated"] == 3
        assert float(z["scale"]) == 32.0 and list(z["shape"]) == [9, 11]
        np.testing.assert_array_equal(np.unpackbits(z["mask"], count=99).reshape(9, 11).astype(bool), m)
    m2, g2 = X.load_npz(p)
    np.testing.assert_array_equal(m2, m)
    assert Grid.from_dict(g2) == g
    with pytest.raises(ValueError):
        X.save_npz(p, m[:5], g)


def test_png(tmp_path):
    from PIL import Image

    m = np.zeros((5, 7), bool)
    m[1:3, 2:6] = True
    X.save_png(tmp_path / "m.png", m, 8.0)
    with Image.open(tmp_path / "m.png") as im:
        a = np.asarray(im)
        assert round(im.info["dpi"][0]) == 3175
    np.testing.assert_array_equal(a, m.astype(np.uint8) * 255)


def test_slide_stem():
    assert slide_stem("/a/b/slide_01.svs") == "slide_01"
    assert slide_stem("x.ome.tiff") == "x.ome"


# ----------------------------------------------------------------------------- GeoJSON
def _rasterize(polys, shape):
    """Even-odd rasterization of grid-coordinate polygons at pixel centres (test helper)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]] + 0.5
    out = np.zeros(shape, bool)
    for poly in polys:
        inside = np.zeros(shape, bool)
        for ring in poly:
            r = np.vstack([ring, ring[:1]]).astype(float)
            for (x1, y1), (x2, y2) in itertools.pairwise(r):
                if y1 == y2:
                    continue
                cross = ((y1 <= yy) & (yy < y2)) | ((y2 <= yy) & (yy < y1))
                xi = x1 + (yy - y1) * (x2 - x1) / (y2 - y1)
                inside ^= cross & (xx < xi)
        out |= inside
    return out


def test_single_pixel_polygon_in_level0():
    m = np.zeros((3, 4), bool)
    m[1, 2] = True
    g = Grid(origin_x=100.0, origin_y=50.0, scale=16.0, height=3, width=4, mpp=8.0)
    fc = X.to_geojson(m, g)
    assert len(fc["features"]) == 1
    ring = fc["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) == 5
    assert sorted(map(tuple, ring[:-1])) == [(132.0, 66.0), (132.0, 82.0), (148.0, 66.0), (148.0, 82.0)]
    assert fc["features"][0]["properties"]["area_mm2"] == pytest.approx(64e-6)
    assert X._signed_area(np.asarray(ring)) > 0  # exterior counter-clockwise in the numeric plane


def test_polygons_holes_diagonals_islands():
    m = np.zeros((12, 12), bool)
    m[1:10, 1:10] = True
    m[3:8, 3:8] = False  # hole
    m[5, 5] = True  # island in the hole
    m[10, 10] = True  # diagonal neighbour of the frame: same 8-connected component
    polys = X.mask_polygons(m)
    assert len(polys) == 2  # frame (+ diagonal pixel) with one hole, island
    frame = max(polys, key=lambda p: X._signed_area(p[0]))
    assert len(frame) == 2 and X._signed_area(frame[1]) == -25
    assert sum(X._signed_area(q) for p in polys for q in p) == m.sum()


@pytest.mark.parametrize("seed", range(5))
def test_polygons_are_exact_on_random_masks(seed):
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    m = rng.random((40, 50)) > 0.55
    polys = X.mask_polygons(m)
    assert len(polys) == ndi.label(m, structure=np.ones((3, 3), bool))[1]
    assert sum(X._signed_area(q) for p in polys for q in p) == m.sum()
    np.testing.assert_array_equal(_rasterize(polys, m.shape), m)


def test_empty_mask_geojson():
    g = Grid(origin_x=0.0, origin_y=0.0, scale=16.0, height=3, width=3, mpp=8.0)
    assert X.to_geojson(np.zeros((3, 3), bool), g)["features"] == []


# ----------------------------------------------------------------------------- CLI
def test_cli_parser():
    from scout_tissue.cli import build_parser

    a = build_parser().parse_args(["predict", "a.svs", "b.svs", "--out", "o", "--format", "npz,geojson"])
    assert a.formats == ("npz", "geojson") and a.slides == ["a.svs", "b.svs"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "a.svs", "--out", "o", "--format", "tiff"])


def test_cli_rejects_duplicate_names(tmp_path):
    pytest.importorskip("torch")
    from scout_tissue.cli import main

    for d in ("x", "y"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "s.svs").write_bytes(b"")
    assert main(["predict", str(tmp_path / "x/s.svs"), str(tmp_path / "y/s.svs"), "--out", str(tmp_path)]) == 2
