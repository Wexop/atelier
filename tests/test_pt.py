import numpy as np
from atelier import Scene, Mat, sphere, plane, box, Canvas
from atelier.pt import pathtrace, denoise


def _scene(glass=False):
    s = Scene()
    s.add(plane(), Mat("#808080", rough=0.7))
    s.add(sphere(0.5).move(-0.55, 0.5, 0), Mat("#c03030", rough=0.3))
    s.add(sphere(0.5).move(0.55, 0.5, 0), Mat("#d0d0d0", rough=0.1, metal=1))
    if glass:
        s.add(sphere(0.3).move(0, 0.3, 0.7), Mat("#ffffff", glass=1, ior=1.5, rough=0))
    s.area((-2, 3, 2), (1.2, 1.2), (1, -1.5, -1), power=30, pattern="window")
    s.sky(power=0.5)
    s.camera((0, 1.3, 3.4), (0, 0.4, 0), fov=40)
    return s


def test_determinism_seed_and_threads():
    s = _scene()
    a = pathtrace(s, 32, 20, spp=4, seed=3, threads=1, grid=2e4)
    b = pathtrace(s, 32, 20, spp=4, seed=3, threads=2, grid=2e4)
    c = pathtrace(s, 32, 20, spp=4, seed=4, threads=2, grid=2e4)
    assert a.rgb.shape == (20, 32, 3) and np.isfinite(a.linear).all()
    assert np.array_equal(a.linear, b.linear)
    assert not np.array_equal(a.linear, c.linear)
    assert a.id[10, 9] == 1 and a.rgb[10, 9, 0] > a.rgb[10, 9, 2]      # red sphere
    assert np.isfinite(a.depth[10, 9]) and a.spp.mean() > 3


def test_white_furnace_energy():
    for mat in (Mat("#ffffff", rough=0.5), Mat("#ffffff", metal=1, rough=0.3), Mat("#ffffff", glass=1, rough=0)):
        s = Scene()
        s.add(sphere(0.5), mat)
        s.sky("#ffffff", "#ffffff", "#ffffff", power=1.0)
        s.point((5, 5, 5), power=0)
        s.camera((0, 0, 3), (0, 0, 0), fov=25)
        r = pathtrace(s, 32, 32, spp=6, bounces=6, denoise=False, adaptive=False, threads=2, grid=0)
        m = r.id >= 0
        assert 0.9 < r.linear[m].mean() < 1.05, (mat.metal, mat.glass, r.linear[m].mean())


def test_glass_metal_smoke_and_canvas():
    s = _scene(glass=True)
    c = Canvas(40, 25)
    r = c.pathtrace(s, spp=4, bounces=4, grid=2e4, threads=2)
    assert c.cur.rgba.shape == (25, 40, 4) and np.isfinite(r.linear).all()
    gid = 3
    m = r.id == gid
    assert m.sum() > 3 and r.linear[m].mean() > 0.01           # glass shows its surroundings
    assert r.linear[r.id == 2].mean() > 0.01                    # metal reflects


def test_denoiser_reduces_variance_keeps_edges():
    rng = np.random.default_rng(0)
    H, W = 48, 64
    alb = np.ones((H, W, 3), np.float32) * 0.8
    alb[:, W // 2:] = 0.2                                      # texture edge in the albedo guide
    clean = alb * 0.5
    noisy = (clean * (1 + 0.5 * rng.standard_normal((H, W, 1)))).astype(np.float32)
    nrm = np.zeros((H, W, 3), np.float32); nrm[..., 2] = 1
    dep = np.ones((H, W), np.float32)
    out = denoise(noisy, alb, nrm, dep)
    err0 = np.abs(noisy - clean).mean()
    err1 = np.abs(out - clean).mean()
    assert err1 < err0 / 3
    assert out[:, W // 2 - 3, 0].mean() > 3 * out[:, W // 2 + 2, 0].mean()   # edge preserved
