import numpy as np
from atelier import *


def _draw(method, *args, seed=1, **kw):
    c = Canvas(160, 120, bg="#202430")
    c.layer(name="fx")
    getattr(c, method)(*args, seed=seed, **kw)
    return c["fx"].rgba


CALLS = {
    "smoke": (80, 110, 90),
    "rain": (),
    "snowfall": (),
    "mist": (60, 110),
    "lightning": (100, 0, 70, 100),
}


def test_effects_paint_something_and_are_deterministic():
    for name, args in CALLS.items():
        a = _draw(name, *args)
        b = _draw(name, *args)
        c = _draw(name, *args, seed=2)
        assert np.isfinite(a).all(), name
        assert a[..., 3].max() > 0.3, name
        assert np.array_equal(a, b), name
        assert not np.array_equal(a, c), name


def test_smoke_stays_above_its_source():
    al = _draw("smoke", *CALLS["smoke"])[..., 3]
    rows = np.where(al.max(1) > 0.05)[0]
    assert rows.max() <= 112 and rows.min() < 80


def test_rain_region_masks():
    a = _draw("rain", region=zone(0, 0, 0.5, 1))[..., 3]
    assert a[:, 85:].max() < 0.02 and a[:, :75].max() > 0.2


def test_mist_band():
    a = _draw("mist", 60, 100, wisps=0)[..., 3]
    assert a[:50].max() < 1e-3 and a[95:100].mean() > 0.5
