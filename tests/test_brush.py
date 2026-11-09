import numpy as np
import pytest

from atelier import *

PTS = [(20, 50), (60, 30), (100, 60), (140, 40), (180, 50)]


@pytest.mark.parametrize("b", brushes())
def test_every_brush_paints_deterministically(b):
    imgs = []
    for _ in range(2):
        c = Canvas(200, 100, bg="white")
        c.stroke(PTS, b, 16, "#2050c0", pressure=[0.5, 1, 1, 1, 0.5], seed=4)
        imgs.append(c.flatten())
    assert np.array_equal(imgs[0], imgs[1])
    diff = np.abs(imgs[0][..., :3] - 1).sum(-1)
    assert (diff > 0.05).sum() > 150          # something visible was painted
    assert diff[:, :5].max() < 1e-3           # and only near the stroke


def test_seed_changes_texture():
    a, b = Canvas(200, 100), Canvas(200, 100)
    a.stroke(PTS, "bristle", 20, "red", seed=1)
    b.stroke(PTS, "bristle", 20, "red", seed=2)
    assert not np.array_equal(a.flatten(), b.flatten())


def test_round_stroke_width():
    c = Canvas(200, 60)
    c.stroke([(20, 30), (180, 30)], "round", 10, "black")
    col = c.flatten()[:, 100, 3]
    assert 9 <= col.sum() <= 11


def test_km_mix_blue_yellow_is_green():
    g = km_mix("#1f4fbf", "#ffd500", 0.5)
    assert g[1] > g[0] and g[1] > g[2]
    assert np.allclose(km_mix("#ff0000", "#00ff00", 0.0), [1, 0, 0], atol=0.01)


def test_watercolor_glazes_darker():
    c = Canvas(120, 80, bg="white")
    c.stroke([(20, 40), (100, 40)], "watercolor", 30, "#e07a5f", seed=1)
    one = c.flatten()[40, 60, :3].mean()
    c.stroke([(20, 40), (100, 40)], "watercolor", 30, "#e07a5f", seed=2)
    two = c.flatten()[40, 60, :3].mean()
    assert two < one < 1


def test_oil_writes_height():
    c = Canvas(200, 100)
    c.stroke(PTS, "oil", 24, "#c08040", seed=0)
    assert c.cur.height is not None and c.cur.height.max() > 0


def test_scatter_respects_region_and_seed():
    c = Canvas(200, 100)
    c.scatter("dot", 300, region=left(0.5), size=(3, 3), seed=5)
    a = c.flatten()[..., 3]
    assert a[:, :95].sum() > 500 and a[:, 110:].sum() == 0
    d = Canvas(200, 100)
    d.scatter("dot", 300, region=left(0.5), size=(3, 3), seed=5)
    assert np.array_equal(a, d.flatten()[..., 3])


@pytest.mark.parametrize("kind", ["dot", "soft", "star", "grass", "leaf", "petal", "square", "person"])
def test_scatter_kinds(kind):
    c = Canvas(100, 100)
    c.scatter(kind, 20, seed=1)
    assert c.flatten()[..., 3].sum() > 10


def test_hatch_inside_region_and_tone():
    c = Canvas(200, 100)
    c.hatch(rect(50, 20, 100, 60), angle=45, spacing=6)
    a = c.flatten()[..., 3]
    assert a[:15].sum() == 0 and a[20:80, 50:150].mean() > 0.1
    d = Canvas(200, 100)
    d.hatch(None, spacing=6, tone=X)              # dark on the left -> more ink on the left
    b = d.flatten()[..., 3]
    assert b[:, :60].mean() > b[:, 140:].mean() * 2


def test_paint_like_small():
    ref = Canvas(160, 100)
    ref.fill(gradient(["#204080", "#f0c060"]))
    c = Canvas(160, 100, bg="white")
    n = c.paint_like(ref, "round", sizes=(20, 10), seed=1)
    out = c.flatten()[..., :3]
    assert n > 10 and np.abs(out - ref.flatten()[..., :3]).mean() < 0.12
