import numpy as np
import pytest

from atelier import *


def img(h=40, w=60):
    x, y = grid(w, h)
    a = np.stack([x, y, x * y, np.ones_like(x)], -1).astype(np.float32)
    a[10:20, 10:30, :3] = 1.0  # bright patch
    return a


EFFECTS = [blur(3), motion_blur(8, 30), sharpen(), bloom(), glow(4), drop_shadow(), inner_shadow(), vignette(),
           grain(), chromatic_aberration(2), lens_distortion(0.1), exposure(0.5), contrast(0.3), levels(0.1, 0.9, 1.2),
           curves([(0.3, 0.2), (0.7, 0.8)]), lut(lambda c: c ** 2), hue_sat(30, 1.2, 0.02), saturation(0.5),
           color_balance((0, 0, .1), (0, .05, 0), (.1, 0, 0)), gradient_map("sunset"), invert(), tone_map(),
           tone_map("filmic"), displace(fbm(4), 5), posterize(3), pixelate(4, "retro16", dither="bayer"),
           pixelate(3, colors=8, dither="fs"), halftone(4), halftone(4, cmyk=True)]


@pytest.mark.parametrize("f", EFFECTS)
def test_effect_runs(f):
    a = img()
    out = f(a, 1.0)
    assert out.shape == a.shape and np.isfinite(out).all()
    assert out.min() >= -1e-4 and out.max() <= 1 + 1e-4


def test_blur_premultiplied_no_dark_fringe():
    a = np.zeros((30, 30, 4), np.float32)
    a[10:20, 10:20] = [1, 0, 0, 1]
    out = blur(3)(a)
    edge = out[15, 8]
    assert edge[3] > 0.01 and edge[0] > 0.99 and edge[1] < 0.01  # pure red, not darkened


def test_grain_deterministic_and_scale_aware():
    a = img()
    assert np.array_equal(grain(seed=3)(a), grain(seed=3)(a))
    assert not np.array_equal(grain(seed=3)(a), grain(seed=4)(a))


def test_pixelate_uses_palette():
    pal = palette("green4")
    out = pixelate(2, pal)(img())
    cols = {tuple(np.round(c, 3)) for c in out[..., :3].reshape(-1, 3)}
    allowed = {tuple(np.round(color(p)[:3], 3)) for p in pal}
    assert cols <= allowed


def test_tonemap_neutral_and_monotonic():
    v = np.linspace(0, 4, 50, dtype=np.float32)[:, None].repeat(3, 1)
    for kind in ("agx", "filmic", "reinhard"):
        out = tonemap_linear(v, kind, look=None)
        assert np.all(np.diff(out[:, 0]) >= -1e-4)
        assert np.allclose(out[:, 0], out[:, 1], atol=2e-3) and np.allclose(out[:, 1], out[:, 2], atol=2e-3)


def test_depth_of_field():
    a = np.zeros((40, 80, 4), np.float32); a[..., 3] = 1
    a[:, ::4, :3] = 1  # stripes
    depth = np.where(np.arange(80)[None, :] < 40, 1.0, 5.0).repeat(40, 0).astype(np.float32)
    out = depth_of_field(depth, focus=1.0, aperture=6)(a)
    sharp, soft = out[:, :36, 0].std(), out[:, 44:, 0].std()
    assert sharp > 0.4 and soft < 0.15


def test_masked_apply_and_canvas_apply():
    c = Canvas(60, 40, bg="white")
    c.layer().fill("black", mask=left(0.5))
    c.apply(invert(), mask=top(0.5))
    a = c.flatten()
    assert a[5, 5, 0] > 0.99 and a[35, 5, 0] < 0.01 and a[5, 55, 0] < 0.01 and a[35, 55, 0] > 0.99
    assert len(c.layers) == 1
