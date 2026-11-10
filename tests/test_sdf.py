import numpy as np
from atelier import *


def test_minimal_scene_and_determinism():
    def make():
        s = Scene()
        s.add(plane(), Mat("#888888"))
        oid = s.add(sphere(0.5).move(0, 0.5, 0), Mat("#c03030", rough=0.3))
        s.sun((-1, 2, 1)); s.camera((0, 1, 3), (0, 0.5, 0), fov=40)
        return s, oid
    s, oid = make()
    r = s.render(64, 40, threads=1)
    r2 = make()[0].render(64, 40, threads=2)
    assert r.rgb.shape == (40, 64, 3) and np.isfinite(r.rgb).all()
    assert np.allclose(r.rgb, r2.rgb, atol=1e-5)
    assert r.id[20, 32] == oid and r.rgb[20, 32, 0] > r.rgb[20, 32, 2]
    assert np.isfinite(r.depth[20, 32]) and r.mask(oid).sum() > 50


def test_primitives_distances():
    p = np.array([[0, 0, 0], [2, 0, 0]], np.float32)
    assert np.allclose(sphere(1)(p), [-1, 1], atol=1e-5)
    assert np.allclose(box(2, 2, 2)(p), [-1, 1], atol=1e-5)
    assert np.allclose((sphere(1) - sphere(0.5))(p)[0], 0.5, atol=1e-5)
    assert abs(lathe([(0.5, -1), (0.5, 1)], smooth=False)(p)[1] - 1.5) < 1e-4


def test_object_space_textures_follow_transforms():
    tex = lambda p, n: np.stack([np.clip(p[:, 0] + 0.5, 0, 1)] * 3, 1)
    base = box(1, 1, 1)
    moved = base.rotate(y=90).move(2, 0, 0)
    p = np.array([[-0.3, 0.1, 0.5]], np.float32)
    n = np.array([[0, 0, 1]], np.float32)
    q = (p @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float32)) + np.array([2, 0, 0], np.float32)
    assert np.allclose(moved.local(q), p, atol=1e-5)
    a = Mat(tex, local=True).eval(p, n, base)["alb"]
    b = Mat(tex, local=True).eval(q, n, moved)["alb"]
    assert np.allclose(a, b, atol=1e-5)


def test_noise3_footprint_and_wear():
    f = noise3((40, 40, 40), 4, seed=1)
    p = np.random.default_rng(0).random((4000, 3)).astype(np.float32)
    assert f(p, fw=np.full(4000, 0.5, np.float32)).std() < 0.2 * f(p).std()
    n = np.tile(np.array([[0, 1, 0]], np.float32), (4000, 1))
    m = Mat("#808080", rough=0.3, bump=0.5, wear=[smudges(), scratches(), dust(), edge_wear()])
    r = m.eval(p, n, box(1, 1, 1), np.full(4000, 1e-3, np.float32))
    assert np.isfinite(r["alb"]).all() and (r["rough"] >= 0.3 - 1e-6).all() and r["n"].shape == (4000, 3)
    assert not np.allclose(r["n"], n)
