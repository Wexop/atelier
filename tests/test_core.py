import numpy as np
import pytest
from PIL import Image

from atelier import Canvas, blend, color, BLENDS, to_linear, to_srgb


def px(c):
    return np.full((1, 1, 4), [*c, 1.0], np.float32)


@pytest.mark.parametrize("mode,b,s,expected", [
    ("normal", 0.2, 0.7, 0.7),
    ("multiply", 0.5, 0.5, 0.25),
    ("screen", 0.5, 0.5, 0.75),
    ("overlay", 0.25, 0.5, 0.25),       # 2*b*s
    ("overlay", 0.75, 0.5, 0.75),       # 1-2(1-b)(1-s)
    ("soft_light", 0.5, 0.5, 0.5),
    ("soft_light", 0.25, 1.0, 0.5),     # D(0.25)=0.25*((4-12)*.25+4)=0.5 ... b+(2s-1)(D-b)
    ("add", 0.6, 0.6, 1.0),
    ("color_dodge", 0.25, 0.5, 0.5),
    ("color_burn", 0.75, 0.5, 0.5),
    ("darken", 0.3, 0.6, 0.3),
    ("lighten", 0.3, 0.6, 0.6),
    ("difference", 0.3, 0.8, 0.5),
])
def test_separable_blends(mode, b, s, expected):
    out = blend(px([b] * 3), px([s] * 3), mode)
    assert np.allclose(out[0, 0, :3], expected, atol=1e-5)


def test_nonseparable_blends():
    red, gray = px([1, 0, 0]), px([0.5, 0.5, 0.5])
    lum = lambda c: 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    # color: hue+sat of source, luminosity of backdrop
    out = blend(gray, red, "color")[0, 0, :3]
    assert abs(lum(out) - 0.5) < 1e-4 and out[0] > out[1] and np.isclose(out[1], out[2])
    # luminosity of gray onto red -> lum 0.5
    out = blend(red, gray, "luminosity")[0, 0, :3]
    assert abs(lum(out) - 0.5) < 1e-4
    # hue/saturation of a gray source keeps backdrop luminance, removes saturation
    out = blend(red, gray, "saturation")[0, 0, :3]
    assert np.allclose(out, lum([1, 0, 0]), atol=1e-4)
    out = blend(gray, red, "hue")[0, 0, :3]
    assert np.allclose(out, 0.5, atol=1e-4)  # gray backdrop has 0 saturation


def test_opacity_and_alpha():
    out = blend(px([0, 0, 0]), px([1, 1, 1]), "normal", opacity=0.25)
    assert np.allclose(out[0, 0], [0.25, 0.25, 0.25, 1], atol=1e-5)
    transparent = np.zeros((1, 1, 4), np.float32)
    out = blend(transparent, px([0.2, 0.4, 0.6]), "multiply")
    assert np.allclose(out[0, 0], [0.2, 0.4, 0.6, 1], atol=1e-5)  # nothing to multiply with


def test_all_blends_run():
    c = Canvas(16, 16, bg="#336699")
    for m in BLENDS:
        L = c.layer(blend=m, opacity=0.7)
        L.fill(lambda x, y: np.stack([x, y, x * y], -1))
    img = c.flatten()
    assert img.shape == (16, 16, 4) and np.isfinite(img).all()


def test_color_parse():
    assert np.allclose(color("#ff8000"), [1, 128 / 255, 0, 1])
    assert np.allclose(color((255, 0, 0)), [1, 0, 0, 1])
    assert np.allclose(color((0.5, 0.5, 0.5, 0.5)), [0.5] * 4)
    assert np.allclose(color("white"), [1, 1, 1, 1])
    assert np.allclose(to_srgb(to_linear(np.linspace(0, 1, 11))), np.linspace(0, 1, 11), atol=1e-5)


@pytest.mark.parametrize("ext", ["png", "jpg", "webp"])
def test_export_formats(tmp_path, ext):
    c = Canvas(40, 30, bg="#204080")
    c.layer().fill("red", mask=lambda x, y: x > 0.5)
    p = c.save(tmp_path / f"a.{ext}")
    im = Image.open(p)
    assert im.size == (40, 30)
    r, g, b = im.convert("RGB").getpixel((35, 15))
    assert r > 240 and g < 15
    p2 = c.save(tmp_path / f"b.{ext}", scale=2)
    assert Image.open(p2).size == (80, 60)


def test_transparent_png_keeps_alpha(tmp_path):
    c = Canvas(10, 10)
    c.fill("blue", mask=lambda x, y: x < 0.5)
    im = Image.open(c.save(tmp_path / "t.png"))
    assert im.mode == "RGBA" and im.getpixel((9, 5))[3] == 0 and im.getpixel((0, 5))[3] == 255


def test_preview_and_crop(tmp_path):
    c = Canvas(1000, 500, bg="gray")
    assert Image.open(c.preview(tmp_path / "p.jpg")).size == (512, 256)
    assert Image.open(c.crop_preview(tmp_path / "c.jpg", 100, 100, 50, 25)).size == (512, 256)


def test_pixel_canvas_nearest(tmp_path):
    c = Canvas(4, 2, pixel=True)
    c.fill(lambda x, y: np.where(x < 0.5, 0.0, 1.0))
    im = np.asarray(Image.open(c.save(tmp_path / "p.png", scale=4)).convert("L"))
    assert im.shape == (8, 16) and set(np.unique(im)) <= {0, 1, 254, 255}


def test_place_image():
    c = Canvas(100, 100, bg="black")
    img = np.zeros((10, 10, 3), np.float32); img[..., 0] = 1
    c.place(img, x=50, y=50, scale=2)
    out = c.flatten()
    assert out[50, 50, 0] > 0.99 and out[5, 5, 0] < 0.01


def test_named_layers_transform_and_stack():
    import numpy as np
    from atelier import Canvas
    c = Canvas(200, 100, bg="black")
    sun = c.layer(name="sun")
    c.circle(50, 50, 10, fill="white")
    assert c["sun"] is sun and "sun" in c and "moon" not in c
    assert sun.bounds == (40, 40, 20, 20)
    sun.move(30, -10)
    assert sun.bounds == (70, 30, 20, 20)
    sun.move_to(150, 90, anchor="bottom")
    assert sun.bounds == (140, 70, 20, 20)
    sun.transform(scale=2)
    assert all(abs(a - b) <= 2 for a, b in zip(sun.bounds, (130, 60, 40, 40)))  # AA edge may grow
    sh = sun.duplicate("hero/shadow")
    c.layer(name="hero/body")
    assert [L.name for L in c.group("hero")] == ["hero/shadow", "hero/body"]
    sh.back()
    assert c.layers[0] is sh
    sh.above("sun")
    assert c.layers.index(sh) == c.layers.index(sun) + 1
    before = c.flatten()[80, 150, 0]
    sun.hide(); sh.hide()
    assert before > 0.9 and c.flatten()[80, 150, 0] < 0.1
    sun.show()
    assert c.use("sun") is sun and c.cur is sun
    sun.remove()
    assert "sun" not in c and c.cur is c.layers[-1]
    try:
        c["nope"]
        assert False
    except KeyError as e:
        assert "hero/body" in str(e)
    flip = c.layer(name="f")
    c.rect(10, 10, 20, 10, fill="red")
    flip.transform(flip="x", center=(100, 50))
    assert flip.bounds == (170, 10, 20, 10)
    assert np.isfinite(c.flatten()).all()


def test_reflect_mirrors_below_waterline():
    c = Canvas(200, 200)
    c.layer(name="sea").fill("#204060")
    c.layer(name="boat/hull").rect(80, 80, 40, 20, fill="#ff0000")
    L = c.reflect("boat", 100, ripple=0, breakup=0, dim=1, seed=0)
    assert L.name == "boat/reflection" and c.layers.index(L) == c.layers.index(c["boat/hull"]) - 1
    a = L.alpha
    assert a[:int(100 * c.s) - 1].max() == 0                              # nothing above the waterline
    assert a[int(105 * c.s), int(100 * c.s)] > 0.5                       # mirrored just below it
    Lb = c.reflect("boat", 100, bands=4, seed=1, name="flat")
    assert Lb.alpha.max() > 0
