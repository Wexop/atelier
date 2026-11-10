import numpy as np
from PIL import Image

from atelier import Canvas
from atelier.core import to_u8
from atelier.inspect import (metrics, contact_sheet, compare, guides, composition, analyze_reference,
                             match_palette, transfer_tone)

N = 192
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
GRAD = np.repeat(((xx + yy) / (2 * N) * 0.6 + 0.2)[..., None], 3, -1)


def disc(aa):
    d = np.hypot(xx - 96.3, yy - 91.7)
    m = np.clip(60.5 - d, 0, 1) if aa else (d < 60).astype(np.float32)
    return np.repeat(m[..., None], 3, -1) * 0.8 + 0.1


def test_noise():
    clean = metrics(to_u8(GRAD))["noise"]
    noisy = metrics(np.clip(GRAD + np.random.default_rng(0).normal(0, 8 / 255, GRAD.shape), 0, 1))["noise"]
    assert clean < 1 and 5 < noisy < 11


def test_aliasing():
    hard, soft = metrics(disc(False))["aliasing"], metrics(disc(True))["aliasing"]
    assert hard > 80 and soft < 25
    assert any("aliasing" in f for f in metrics(disc(False))["flags"])


def test_banding():
    assert metrics(to_u8(GRAD))["banding"] < 0.5
    assert metrics(np.floor(GRAD * 32) / 32)["banding"] > 20
    assert metrics(disc(True))["banding"] < 0.5          # flat shapes are not banding


def test_clipping_and_values():
    m = metrics(np.clip(GRAD * 2, 0, 1))
    assert m["clip_highlights"] > 30 and m["clip_shadows"] == 0
    assert metrics(np.zeros((32, 32, 3)))["clip_shadows"] == 100
    m = metrics(GRAD * 0.3 + 0.6)
    assert any("no darks" in f for f in m["flags"])
    assert abs(sum(m["values"]) - 100) < 0.5


def test_colorfulness_sharpness():
    grey = metrics(np.full((32, 32, 3), 0.5))
    red = metrics(np.dstack([np.ones((32, 32)), np.zeros((32, 32)), np.zeros((32, 32))]))
    assert red["colorfulness"] > grey["colorfulness"] + 50 and red["saturation"] > 20
    assert metrics(disc(False))["sharpness"] > metrics(GRAD)["sharpness"]


def test_pixel_art_detected():
    small = (np.random.default_rng(1).random((30, 40, 3)) > 0.5).astype(np.float32)
    m = metrics(np.repeat(np.repeat(small, 4, 0), 4, 1))
    assert m["pixel_grid"] == 4 and m["pixel_art"]
    assert not any("aliasing" in f for f in m["flags"])


def canvas():
    c = Canvas(240, 160, bg="#203040")
    c.circle(150, 70, 40, fill="#f0c060")
    c.rect(20, 110, 200, 20, fill="#a03020")
    return c


def test_contact_sheet_and_guides(tmp_path):
    c = canvas()
    r = contact_sheet(c, str(tmp_path / "s.jpg"))
    im = Image.open(r["path"])
    assert im.size[0] == 1280 and im.size[1] > 600
    assert 1 <= len(r["crops"]) <= 4 and r["palette"]
    for _, (x, y, w, h), _ in r["crops"]:
        assert 0 <= x and 0 <= y and x + w <= 240 and y + h <= 160
    fx, fy = composition(c)["focal"]
    assert 100 < fx < 200 and 20 < fy < 120          # the bright disc is the focal point
    arr, comp = guides(c, str(tmp_path / "g.png"), width=300)
    assert arr.shape == (200, 300, 3) and (tmp_path / "g.png").exists() and "tilt" in comp


def test_compare(tmp_path):
    a = canvas()
    b = canvas()
    b.circle(40, 40, 10, fill="white")
    r = compare(a, b, str(tmp_path / "c.jpg"), labels=("a", "b"))
    assert Image.open(r["path"]).size[0] == 1280
    x, y, w, h = r["bbox"]
    assert 25 <= x <= 32 and 25 <= y <= 32 and w <= 25 and r["changed"] < 5
    assert compare(a, a, str(tmp_path / "d.jpg"))["mean_diff"] == 0


def test_reference_and_transfer():
    ref = np.zeros((60, 90, 3), np.float32)
    ref[:, :45] = (0.9, 0.3, 0.1)
    ref[:, 45:] = (0.1, 0.2, 0.6)
    ref[:20] = ref[:20] * 0.5 + 0.5                         # lighter top -> light from top
    r = analyze_reference(ref, k=4)
    assert len(r["palette"]) >= 2 and abs(sum(w for _, w in r["palette"]) - 1) < 0.02
    assert r["values"].dtype == np.uint8 and ("top" in r["light"] or "upper" in r["light"])
    img = np.dstack([GRAD[..., 0]] * 3 + [np.ones((N, N), np.float32)])
    for fx in (match_palette(ref), transfer_tone(ref)):
        out = fx(img, 1.0)
        assert out.shape == img.shape and np.allclose(out[..., 3], 1)
        assert metrics(out[..., :3])["colorfulness"] > metrics(img[..., :3])["colorfulness"] + 10
    assert np.allclose(match_palette(ref, 0)(img)[..., :3], img[..., :3], atol=2e-3)
