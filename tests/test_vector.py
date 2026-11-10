import math
import numpy as np

from atelier import *


def area(c):
    return c.flatten()[..., 3].sum()


def test_circle_area_and_aa():
    c = Canvas(200, 200)
    c.circle(100, 100, 50, fill="red")
    assert abs(area(c) - math.pi * 50 ** 2) / (math.pi * 50 ** 2) < 0.01
    a = c.flatten()[..., 3]
    assert ((a > 0.05) & (a < 0.95)).sum() > 100  # anti-aliased edge


def test_pixel_canvas_is_aliased():
    c = Canvas(64, 64, pixel=True)
    c.circle(32, 32, 20, fill="red")
    assert set(np.unique(c.flatten()[..., 3])) <= {0.0, 1.0}


def test_booleans():
    a, b = rect(0, 0, 100, 100), rect(50, 0, 100, 100)
    for sh, expect in [(a | b, 15000), (a & b, 5000), (a - b, 5000), (a ^ b, 10000)]:
        c = Canvas(200, 100)
        c.draw(sh)
        assert abs(area(c) - expect) < 60


def test_shape_as_mask_and_scale():
    for scale in (None, 2):
        CONFIG["scale"] = scale
        try:
            c = Canvas(100, 100)
            c.fill("blue", mask=rect(10, 10, 30, 30))
            a = c.flatten()[..., 3]
            assert abs(a.sum() / c.s ** 2 - 900) < 5
            m = (fbm(3) * circle(50, 50, 20)).render(c.H, c.W, None, c.s)
            assert m[0, 0] == 0 and m.max() > 0.3
        finally:
            CONFIG["scale"] = None


def test_svg_parser():
    s = svg("M10 10 h80 v30 a10 10 0 01 -10 10 H10 z M20 20 l5 5 c1 1 2 2 3 3 s1 1 2 2 q1 1 2 2 t1 1")
    x0, y0, x1, y1 = s.bounds
    assert abs(x0 - 10) < 1e-3 and abs(y0 - 10) < 1e-3 and abs(x1 - 90) < 1e-3 and abs(y1 - 50) < 1e-3


def test_text():
    t = text_shape("Hello", 0, 0, 40)
    x0, y0, x1, y1 = t.bounds
    assert x1 - x0 > 60 and y0 < -20 and y1 <= 1
    w, cap = text_size("Hello", 40)
    t2 = text_shape("Hello", 100, 50, 40, align="center", valign="middle")
    bx0, by0, bx1, by1 = t2.bounds
    assert abs((bx0 + bx1) / 2 - 100) < 3 and abs((by0 + by1) / 2 - 50) < 3
    f = text_shape("Wide title", 0, 0, fit=500)
    assert abs((f.bounds[2] - f.bounds[0]) - 500) < 15
    c = Canvas(300, 80)
    c.text("Bold", 10, 60, 50, weight="bold", font="mono", stroke="red", width=2)
    assert area(c) > 500


def test_stroke_dash_spline_points():
    c = Canvas(200, 50)
    c.line(0, 25, 200, 25, width=4, dash=(10, 10), cap="butt")
    assert abs(area(c) - 400) < 30  # half of 200*4
    s = spline([(0, 0), (50, 40), (100, 0)])
    pts = s.points(20)
    assert pts.shape == (20, 2) and abs(pts[0, 0]) < 1e-3 and abs(pts[-1, 0] - 100) < 1e-3
    assert abs(circle(0, 0, 10).length - 2 * math.pi * 10) < 0.5


def test_within():
    s = rect(100, 0, 100, 100)
    g = X.within(s).render(100, 300)
    assert abs(g[50, 100] - 0.005) < 0.01 and abs(g[50, 199] - 0.995) < 0.01


def test_kerning():
    for pair in ("AV", "To", "Ty", "LT"):
        assert text_size(pair, 100)[0] < text_size(pair, 100, kern=False)[0] - 2
    assert abs(text_size("HH", 100)[0] - text_size("HH", 100, kern=False)[0]) < 1e-3
    a = text_shape("AV", 0, 0, 100).bounds
    b = text_shape("AV", 0, 0, 100, kern=False).bounds
    assert a[2] - a[0] < b[2] - b[0]
    tf = typeface("sans", 400)[0]
    assert kern_pairs(tf) is kern_pairs(tf)  # cached


def test_text_on_path():
    arcp = arc(200, 200, 150, 180, 180)
    t = text_on_path("HELLO WORLD", arcp, 30, align="center")
    x0, y0, x1, y1 = t.bounds
    assert 60 < x0 and x1 < 340 and y0 > 20 and y1 < 210
    flat = text_on_path("HELLO", [(0, 100), (200, 100), (400, 100)], 30)
    assert abs(flat.bounds[3] - 100) < 2 and flat.bounds[1] < 85
    lifted = text_on_path("HELLO", [(0, 100), (400, 100)], 30, lift=10)
    assert abs(lifted.bounds[3] - 90) < 2
    c = Canvas(400, 260)
    c.text_on_path("ROUND", arcp, 28, fill="red")
    assert area(c) > 300


def test_text_box():
    txt = "The quick brown fox jumps over the lazy dog. " * 6
    b = text_box(txt, 10, 10, 200, size=16)
    assert b.lines > 4 and b.bounds[2] <= 211 and b.bounds[0] >= 9
    j = text_box(txt, 10, 10, 200, size=16, align="justify")
    assert j.lines == b.lines
    f = text_box(txt, 0, 0, 200, h=60, size=40, fit=True)
    assert f.size < 40 and f.height <= 60 and f.bounds[3] <= 62
    r = text_box("plain **bold** [#ff0000]red words[/] end", 0, 0, 400, size=20)
    assert len(r.runs) == 2 and "#ff0000" in [k for k, _ in r.runs]
    c = Canvas(300, 200)
    c.text_box(txt, 10, 10, 280, size=14, align="justify")
    for st in ("shadow", "extrude", "outline", "inline"):
        c.title("Title", 20, 150, 60, style=st, fill="#223344")
    assert area(c) > 2000
