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


def test_grid_view_and_layer_boxes(tmp_path):
    from atelier.inspect import grid_view, layer_boxes
    c = Canvas(400, 200, bg="#204060")
    c.layer(name="dot")
    c.circle(300, 60, 20, fill="white")
    c.layer(name="none")
    b = {n: box for _, n, box in layer_boxes(c)}
    assert b["dot"] == (280, 40, 40, 40) and b["none"] is None
    r = grid_view(c, tmp_path / "g.jpg", width=400, marks=[(300, 60), (10, 10, 50, 20)], boxes=True)
    assert r["step"] == 50 and (tmp_path / "g.jpg").exists()
    z = grid_view(c, None, width=200, region=(270, 30, 60, 60))
    assert z["region"] == (270, 30, 60, 60) and z["step"] == 5


def test_critique_finds_layout_problems(tmp_path):
    from atelier.inspect import critique
    c = Canvas(600, 400, bg="#335577")
    c.layer(name="title"); c.rect(100, 40, 200, 40, fill="#3a5c7e")        # blends into the background
    c.layer(name="a"); c.circle(300, 250, 50, fill="#ffcc44")
    c.layer(name="b"); c.circle(300 + 50 + 23, 250, 20, fill="#ffffff")    # 3 px gap -> tangent
    c.layer(name="low"); c.rect(450, 100, 60, 60, fill="#aa3333")
    c.layer(name="cover"); c.rect(440, 90, 80, 80, fill="#33aa33")        # hides 'low'
    c.layer(name="edgy"); c.rect(5, 300, 40, 40, fill="#ffffff")           # 5 px from the left edge
    c.layer(name="n1"); c.rect(100, 200, 40, 40, fill="#000000")
    c.layer(name="n2"); c.rect(103, 300, 40, 20, fill="#000000")           # left edges 3 px apart
    r = critique(c, tmp_path / "cr.jpg")
    kinds = {(t["kind"], tuple(t["layers"])) for t in r["issues"]}
    assert ("contrast", ("title",)) in kinds
    assert ("tangent", ("a", "b")) in kinds
    assert ("hidden", ("low",)) in kinds
    assert ("edge", ("edgy",)) in kinds
    assert any(k == "align" and set(l) == {"n1", "n2"} for k, l in kinds)
    assert r["issues"][0]["sev"] == "high" and [t["n"] for t in r["issues"]] == list(range(1, len(r["issues"]) + 1))
    assert (tmp_path / "cr.jpg").exists()


def test_critique_clean_layout_has_no_layer_issues():
    from atelier.inspect import critique
    c = Canvas(600, 400, bg="#f2efe8")
    c.layer(name="disc"); c.circle(300, 200, 80, fill="#1d2b44")
    issues = critique(c)["issues"]
    assert not [t for t in issues if t["layers"]]


def _hier_scene(busy):
    from atelier import fbm, band
    c = Canvas(600, 400, bg="#7a9cc0")
    c.layer(name="ground")
    c.fill(fbm(18, seed=3).ramp(["#1d3b1d", "#c9e07a", "#0c1a0c"]) if busy else "#5d7a52", mask=band(0.45, 1.0))
    c.layer(name="hero")
    c.rect(280, 200, 40, 140, fill="#f2b233")
    return c


def test_hierarchy_flags_busy_background():
    from atelier.inspect import hierarchy
    busy = hierarchy(_hier_scene(True), focus="hero")
    calm = hierarchy(_hier_scene(False), focus="hero")
    assert busy["scores"]["detail"]["ratio"] < 0.9 and busy["scores"]["detail"]["best_other"] == "'ground'"
    assert any("texture detail" in t[1] for t in busy["issues"])
    assert calm["scores"]["detail"]["ratio"] > 1 and not any("texture detail" in t[1] for t in calm["issues"])
    assert "squint" in calm and calm["focus"] == "'hero'"


def test_critique_survives_apply_and_ignores_attached_and_fx_layers():
    from atelier import vignette
    from atelier.inspect import critique
    c = Canvas(600, 400, bg="#335577")
    c.layer(name="tower"); c.rect(300, 100, 30, 200, fill="#111111")
    c.layer(name="lamp"); c.rect(300, 80, 30, 20, fill="#ffee99")          # attached on purpose (0 px gap)
    c.layer(name="halo", blend="screen"); c.circle(100, 100, 60, fill="#553322")
    c.layer(name="hero/rim"); c.rect(448, 248, 24, 64, fill="#ffaa66")
    c.layer(name="hero/body"); c.rect(450, 250, 20, 60, fill="#000000")    # covers its own rim: fine
    c.apply(vignette(.2))                                                   # merges every layer
    assert [n for n, *_ in c.merged][:2] == ["tower", "lamp"]
    r = critique(c, focus="hero")
    layers = {n for t in r["issues"] for n in t["layers"]}
    assert "halo" not in layers
    assert not [t for t in r["issues"] if t["kind"] in ("tangent", "hidden")]
    assert r["hierarchy"]["focus"] == "'hero'"


def test_variant_sheet_scores(tmp_path):
    from atelier.inspect import variant_sheet
    res = variant_sheet([_hier_scene(True), _hier_scene(False)], tmp_path / "v.jpg", labels=["busy", "calm"], focus="hero")
    assert [r["label"] for r in res] == ["busy", "calm"] and res[1]["score"] > res[0]["score"]
    assert (tmp_path / "v.jpg").exists()


def test_critique_flags_placeholders_and_floating_subject():
    from atelier import critique, svg
    c = Canvas(400, 300)
    c.layer(name="sky").fill("#5a4a80")
    c.layer(name="sea").rect(0, 180, 400, 120, fill="#2a2440")
    c.layer(name="sun").circle(120, 120, 40, fill="#ffe0a0")             # bare flat disc
    c.layer(name="island").draw(svg("M 20 182 C 60 160 110 160 150 182 Z"), fill="#30283f")
    c.layer(name="hero").draw(svg("M 250 230 L 270 170 L 290 230 Z"), fill="#100c18")
    kinds = lambda r: {(t["kind"], tuple(t["layers"])) for t in r["issues"]}
    r = critique(c, focus="hero")
    assert ("simple", ("sun",)) in kinds(r)
    assert ("grounding", ("hero",)) in kinds(r)
    c.reflect("hero", 230)                                                # water contact anchors it
    assert ("grounding", ("hero",)) not in kinds(critique(c, focus="hero"))


def test_critique_flags_style_mix():
    from atelier import critique, fbm
    c = Canvas(400, 300)
    c.layer(name="ground").fill(fbm(60, seed=1).ramp(["#203040", "#b0c0d0"]))   # textured, full frame
    c.layer(name="house").rect(150, 100, 100, 120, fill="#804030")
    assert any(t["kind"] == "style" for t in critique(c)["issues"])
