import numpy as np
from atelier import *
from atelier.nature import *
from atelier.terrain import heightfield, erode, thermal, normals, shadow_map, materials


def test_heightfield_determinism_and_range():
    a = heightfield(64, "alpine", seed=3)
    b = heightfield(64, "alpine", seed=3)
    c = heightfield(64, "alpine", seed=4)
    assert a.shape == (64, 64) and a.dtype == np.float32
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert abs(a.min()) < 1e-6 and abs(a.max() - 1) < 1e-6
    for k in ("hills", "mesa", "island", "dunes"):
        h = heightfield(48, k, seed=1)
        assert np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1


def test_thermal_conserves_mass_and_limits_slope():
    h = heightfield(64, "alpine", seed=1)
    t = thermal(h, 60, talus=0.02)
    assert abs(float(t.sum()) - float(h.sum())) < 1e-2
    steep = lambda x: np.abs(np.diff(x, axis=1)).max()
    assert steep(t) < steep(h)


def test_hydraulic_erosion_moves_material_sensibly():
    h = heightfield(64, "alpine", seed=2)
    e, flow = erode(h, 1.0, seed=5, flow=True)
    e2 = erode(h, 1.0, seed=5)
    assert np.array_equal(e, e2)                              # seeded
    moved = np.abs(e - h).sum()
    assert moved > 1.0                                        # something happened
    assert abs(float(e.sum() - h.sum())) < 0.5 * moved        # only sediment carried off the edges is lost
    lo = h < np.percentile(h, 20); hi = h > np.percentile(h, 80)
    assert (e - h)[lo].mean() > (e - h)[hi].mean()            # valleys filled relative to peaks
    assert flow.shape == h.shape and 0 <= flow.min() and flow.max() <= 1


def test_terrain_maps():
    h = heightfield(48, "hills", seed=0)
    n = normals(h, 1000, 200)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1, atol=1e-4) and (n[..., 1] > 0).all()
    s = shadow_map(h, sun_direction(10, 45), 1000, 200)
    assert s.shape == h.shape and 0 <= s.min() and s.max() <= 1
    m = materials(h, 1000, 200, water=0.2)
    assert m.shape == (48, 48, 3) and np.isfinite(m).all()


def test_sky_physics():
    up = np.array([[0, 1, 0]], np.float32)
    noon = sky_radiance(up, 60)[0]
    assert noon[2] > noon[1] > noon[0]                       # blue zenith (Rayleigh)
    assert (sun_color(60) > sun_color(2)).all()
    sc = sun_color(2)
    assert sc[0] > sc[1] > sc[2]                             # reddened low sun
    assert sky_radiance(up, -8)[0].sum() < noon.sum() * 0.2  # dusk is dark
    E = sky_irradiance(np.array([[0, 1, 0], [0, -1, 0]], np.float32), 60)
    assert E[0, 2] > E[1, 2] and E[0].sum() > 0              # blue skylight from above
    L = np.ones((2, 3), np.float32)
    D = np.array([[0, 0.05, 1], [0, 0.05, 1]], np.float32)
    out = aerial(L, D, np.array([10.0, 80000.0]), np.zeros(2), np.zeros(2), 60)
    assert np.abs(out[0] - 1).max() < np.abs(out[1] - 1).max()   # far objects fade into haze


def test_sky_sea_landscape_render():
    c = Canvas(96, 64)
    c.sky(sun_elev=5, clouds=0.4, seed=1)
    a = c.flatten()
    assert np.isfinite(a).all() and a[..., :3].std() > 0.01
    c2 = Canvas(96, 64)
    c2.sky(sun_elev=5, clouds=0.4, seed=1)
    assert np.allclose(a, c2.flatten())
    c3 = Canvas(96, 64)
    v = c3.sea(sun_elev=5, clouds=0.2)
    assert v.water[-1].mean() > 0.9 and v.water[0].mean() == 0
    c4 = Canvas(96, 64)
    h = heightfield(48, "alpine", seed=1)
    v = c4.landscape(h, water=0.3, clouds=0, sun_elev=5)
    assert np.isfinite(v.depth).any() and np.isinf(v.depth).any()
    px = v.project(8000, v.ground(8000, 12000), 12000)
    assert px is None or len(px) == 3


def test_waves():
    w = wave_field(64, seed=2)
    assert w.shape == (64, 64) and abs(w.std() - 1) < 1e-3
    x, y, z = gerstner(np.linspace(0, 10, 50), np.zeros(50), seed=1)
    assert y.shape == (50,) and np.abs(y).max() > 0


def test_tree_skeletons_and_drawing():
    for k in TREES:
        p, par, r = tree_skeleton(k, seed=1)
        assert len(p) == len(par) == len(r) and par[0] == -1
        assert (par[1:] < np.arange(1, len(par))).all()      # parents come first
        assert r[0] >= r.max() * 0.5 and p[:, 1].max() <= 1.2
    p1 = tree_skeleton("oak", seed=1)[0]
    assert np.array_equal(p1, tree_skeleton("oak", seed=1)[0])
    c = Canvas(160, 120, bg="white")
    c.tree(50, 110, 90, "oak", seed=1)
    c.tree(120, 110, 90, "pine", seed=1, season="winter")
    c.grass(80, 115, 40, 15); c.flower(20, 115, 20, "poppy"); c.fern(140, 118, 30)
    img = c.flatten()
    assert (img[..., :3].mean(-1) < 0.6).sum() > 500


def test_tree3d_sdf():
    trunk, leaves = tree3d("oak", 4.0, seed=2)
    p = np.array([[0, 0.5, 0], [0, 30, 0]], np.float32)
    d = trunk(p)
    assert d[0] < 0 and d[1] > 20
    assert leaves is not None and leaves.bounds is not None
    dl = leaves(np.array([leaves.bounds[0]], np.float32))
    assert np.isfinite(dl).all()
    assert tree3d("dead", 3.0)[1] is None
