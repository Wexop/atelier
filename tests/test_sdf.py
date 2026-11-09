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
