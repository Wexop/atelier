import numpy as np
import pytest

from atelier import *


NOISES = [perlin(5), simplex(5), value(5), fbm(5), ridged(3), turbulence(4), worley(6),
          worley(6, "F2-F1"), worley(6, "id"), clouds(), terrain_height(), wood(), marble(),
          paper(), fabric(), water_ripples(), stars()]


@pytest.mark.parametrize("f", NOISES)
def test_range_and_finite(f):
    a = f.render(64, 96)
    assert a.shape == (64, 96) and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= -0.01 and a.max() <= 1.21


def test_seed_determinism():
    for mk in (lambda s: fbm(6, seed=s), lambda s: worley(8, seed=s), lambda s: domain_warp(fbm(3), 0.2, seed=s),
               lambda s: stars(seed=s), lambda s: water_ripples(seed=s)):
        a, b, c = mk(1).render(50, 80), mk(1).render(50, 80), mk(2).render(50, 80)
        assert np.array_equal(a, b)
        assert not np.allclose(a, c)


def test_resolution_independence():
    # same field at 2 resolutions: downsampled big ~= small (low-frequency content)
    import cv2
    f = fbm(3, octaves=2)
    big, small = f.render(200, 300), f.render(100, 150)
    assert np.abs(cv2.resize(big, (150, 100), interpolation=cv2.INTER_AREA) - small).mean() < 0.01


def test_box_eval_matches_full():
    f = fbm(4) * 0.5 + X * 0.5
    full = f.render(40, 60)
    part = f.render(40, 60, (10, 5, 30, 25))
    assert np.allclose(full[5:25, 10:30], part, atol=1e-5)


def test_arithmetic_and_ufuncs():
    f = np.sin(X * np.pi) * 0.5 + Y
    a = f.render(10, 10)
    x, y = grid(10, 10)
    assert np.allclose(a, np.sin(x * np.pi) * 0.5 + y, atol=1e-6)
    assert set(np.unique((X > 0.5).render(4, 4))) == {0.0, 1.0}


def test_ramp_gradient_palettes():
    assert len(palette()) >= 10
    g = gradient(["#000000", "#ffffff"], angle=0).render(1, 64)
    assert g.shape == (1, 64, 3) and g[0, 0, 0] < 0.05 and g[0, -1, 0] > 0.95
    assert np.all(np.diff(g[0, :, 0]) >= -1e-6)  # monotonic
    r = ramp(np.array([0.0, 1.0]), [(0, "red"), (1, "blue")], space="srgb")
    assert np.allclose(r, [[1, 0, 0], [0, 0, 1]], atol=1e-3)
    assert cosine_palette((.5,) * 3, (.5,) * 3, (1,) * 3, (0, .33, .67), n=5).shape == (5, 3)
    for c in ([0.2, 0.5, 0.8], [1, 0, 0]):
        assert np.allclose(oklab2rgb(rgb2oklab(np.array(c))), c, atol=1e-3)
        assert np.allclose(hsv2rgb(rgb2hsv(np.array(c))), c, atol=1e-5)


def test_shade_flat_is_neutral():
    s = shade(Field(lambda x, y: x * 0 + 0.3)).render(8, 8)
    assert np.allclose(s, 0.5, atol=1e-4)


def test_curve_fill_below():
    c = Canvas(100, 100)
    c.fill_below(curve(lambda x: 0.505 + 0 * x), "red")
    a = c.flatten()[..., 3]
    assert a[10, 50] == 0 and a[90, 50] == 1 and 0.3 < a[50, 50] < 0.7


def test_aastep_is_antialiased():
    a = aastep(X, 0.5).render(1, 101)
    mid = a[0, 45:56]
    assert ((mid > 0.05) & (mid < 0.95)).sum() >= 1 and a[0, 0] == 0 and a[0, -1] == 1
