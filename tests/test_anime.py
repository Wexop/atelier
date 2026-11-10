import numpy as np
from atelier import Canvas, CLOUD_PALETTES, TREE_PALETTES, god_rays, lens_flare, gradient, disc, MOUNTAIN_PALETTES


def test_anime_cloud_draws_and_is_deterministic():
    def draw(seed):
        c = Canvas(240, 160)
        c.layer().anime_cloud(120, 120, 160, seed=seed)
        return c.flatten()
    a, b = draw(1), draw(1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, draw(2))
    cov = (a[..., 3] > 0.5).mean()
    assert 0.1 < cov < 0.7
    assert a[150:, :, 3].max() < 0.01          # nothing below the flat base (y=120, plus margin)


def test_anime_cloud_palettes_and_offscreen():
    c = Canvas(200, 120)
    L = c.layer()
    for i, p in enumerate(CLOUD_PALETTES):
        L.anime_cloud(40 + i * 40, 100, 60, palette=p, bands=2 + i % 3, seed=i)
    L.anime_cloud(-500, -500, 50)               # fully off-canvas: no error
    assert c.flatten()[..., 3].max() > 0.9


def test_anime_tree_kinds_palettes_deterministic():
    def draw(seed, **kw):
        c = Canvas(300, 200)
        c.layer().anime_tree(150, 190, 160, seed=seed, **kw)
        return c.flatten()
    assert np.array_equal(draw(3), draw(3))
    for k in ("round", "tall", "umbrella"):
        a = draw(1, kind=k)
        assert 0.05 < (a[..., 3] > 0.5).mean() < 0.8
    for p in TREE_PALETTES:
        assert draw(2, palette=p)[..., 3].max() > 0.9


def test_light_effects_brighten_and_sparkles():
    c = Canvas(200, 120)
    c.fill(gradient(["#202040", "#ffcc88"]))
    c.layer().rect(0, 60, 200, 60, fill="#101010")      # dark ground occludes
    before = c.flatten().copy()
    c.apply(god_rays(100, 55, seed=1), lens_flare(100, 55, seed=1))
    after = c.flatten()
    assert after[..., :3].mean() > before[..., :3].mean()
    assert after[100, 20, :3].mean() < 0.35             # ground stays dark
    L = c.layer()
    L.sparkles(10, region=disc(0.5, 0.5, 0.3), seed=2)
    assert L.px[..., 3].max() > 0.9


def test_anime_mountains():
    def draw(seed, **kw):
        c = Canvas(320, 160)
        c.layer().anime_mountains(150, 120, seed=seed, **kw)
        return c.flatten()
    a = draw(1)
    assert np.array_equal(a, draw(1))
    assert a[155:, :, 3].max() < 0.01 and a[140, :, 3].mean() > 0.5   # nothing below the foot, solid near it
    for p in MOUNTAIN_PALETTES:
        assert draw(2, palette=p, peaks=2)[..., 3].max() > 0.9
