import numpy as np
import pytest

from atelier import Canvas, Scene, Mat, sphere, plane, circle, gradient
from atelier.stylize import (toon, outlines, render_hatch, shadow_dots, shading, kuwahara, edge_smooth, xdog,
                             watercolorize, pencil, linocut, stained_glass, mosaic)


@pytest.fixture(scope="module")
def scene_render():
    s = Scene()
    s.add(plane(0.0), Mat("#9a9a9a"))
    ball = s.add(sphere(0.5).move(0, 0.5, 0), Mat("#c03020"))
    s.sun((-1, 1, 0.6), power=3, soft=0.02)
    s.camera((0, 1.0, 3.0), (0, 0.45, 0), fov=40)
    c = Canvas(96, 64)
    r = c.render(s)
    return c, r, s, ball


def test_shading_range_and_sky(scene_render):
    c, r, s, ball = scene_render
    sh = shading(r)
    assert sh.shape == r.id.shape and sh.min() >= 0 and sh.max() <= 1
    assert np.all(sh[r.id < 0] == 1)
    m = r.id == ball
    assert sh[m].max() > 0.9 and sh[m].min() < 0.5          # lit side and shadow side


def test_toon_quantises_object(scene_render):
    c, r, s, ball = scene_render
    img = c.flatten()
    out = toon(r, bands=3, soft=0, spec=None, tint=None)(img, c.s)
    m = (r.id == ball)
    core = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    levels = np.unique(np.round(out[core][:, 0], 3))
    assert len(levels) <= 4
    assert np.allclose(out[:4, :, :3], img[:4, :, :3])                # sky kept
    sky = toon(r, sky="#ffffff")(img, c.s)
    assert np.allclose(sky[0, 0, :3], 1)


def test_outlines_on_silhouette_only(scene_render):
    c, r, s, ball = scene_render
    img = np.ones(r.id.shape + (4,), np.float32)
    out = outlines(r, width=2, crease=0, depth=0)(img, c.s)
    ink = 1 - out[..., 0]
    m = r.id == ball
    edge = m & ~(np.roll(m, 3, 0) & np.roll(m, -3, 0) & np.roll(m, 3, 1) & np.roll(m, -3, 1))
    assert ink[edge].mean() > 0.3
    inner = m & np.roll(m, 6, 0) & np.roll(m, -6, 0) & np.roll(m, 6, 1) & np.roll(m, -6, 1)
    assert ink[inner].max() < 0.05


@pytest.mark.parametrize("with_scene", [True, False])
def test_render_hatch_follows_darkness(scene_render, with_scene):
    c, r, s, ball = scene_render
    img = np.ones(r.id.shape + (4,), np.float32)
    out = render_hatch(r, s if with_scene else None, spacing=3, width=1, value=0)(img, c.s)
    ink = 1 - out[..., 0]
    sh = shading(r)
    m = r.id == ball
    assert ink[m & (sh < 0.4)].mean() > ink[m & (sh > 0.9)].mean() + 0.1
    assert ink[r.id < 0].max() == 0


def test_shadow_dots_only_in_shadow(scene_render):
    c, r, s, ball = scene_render
    img = np.ones(r.id.shape + (4,), np.float32)
    ink = 1 - shadow_dots(r, size=3, lo=0.3)(img, c.s)[..., 0]
    sh = shading(r)
    assert ink[sh > 0.95].mean() < 0.02 and ink[sh < 0.3].mean() > 0.1


def _test_image():
    c = Canvas(80, 60)
    c.fill(gradient(["#204070", "#e0b060"]))
    c.draw(circle(40, 30, 16), fill="#c03020")
    return c.flatten(), c


FILTERS = [kuwahara(4), edge_smooth(3), xdog(), watercolorize(seed=1), pencil(seed=1), linocut(seed=1),
           stained_glass(10, seed=1), mosaic(6, seed=1), watercolorize(ink=0.8, seed=1)]


@pytest.mark.parametrize("fx", FILTERS)
def test_filters_shape_range_determinism(fx):
    img, c = _test_image()
    a, b = fx(img.copy(), 1.0), fx(img.copy(), 1.0)
    assert a.shape == img.shape and np.isfinite(a).all()
    assert a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, b)
    assert np.abs(a[..., :3] - img[..., :3]).mean() > 1e-3 or fx is FILTERS[1]


def test_kuwahara_keeps_flat_colour_and_edges():
    img = np.zeros((40, 40, 4), np.float32)
    img[..., 3] = 1
    img[:, 20:, :3] = 1
    out = kuwahara(5)(img, 1.0)
    assert np.allclose(out[:, :14, :3], 0, atol=1e-3) and np.allclose(out[:, 26:, :3], 1, atol=1e-3)
    assert out[20, 18, 0] < 0.2 and out[20, 21, 0] > 0.8        # edge stays sharp


def test_xdog_inks_edges_not_flats():
    img, c = _test_image()
    out = xdog(paper="#ffffff")(img, 1.0)
    ink = 1 - out[..., 0]
    assert ink[30, 24:26].max() > 0.5                           # disc border
    assert ink[30, 38:42].max() < 0.1                           # disc interior


def test_segmentation_filters_draw_joints():
    img, c = _test_image()
    g = stained_glass(10, lead=2, lead_color="#000000", seed=3)(img, 1.0)
    assert (g[..., :3].max(-1) < 0.05).mean() > 0.05            # lead lines
    assert not np.array_equal(g, stained_glass(10, lead=2, seed=4)(img, 1.0))
    m = mosaic(8, grout=2, grout_color="#ffffff", seed=3)(img, 1.0)
    assert (m[..., :3].min(-1) > 0.95).mean() > 0.05             # grout
