import numpy as np
import pytest
from PIL import Image

from atelier import *
from atelier.anim import motion_metrics, continuity_probes, render_frames, save_anim
from atelier.cli import main


def test_ease_key_wave_orbit():
    for k in ("linear", "in", "out", "in_out", "sine", "back", "bounce"):
        assert ease(0, k) == pytest.approx(0, abs=1e-6) and ease(1, k) == pytest.approx(1, abs=1e-6)
    assert ease(0.25, "in") < 0.25 < ease(0.25, "out")
    assert key(0.5, [(0, 0), (1, 10)], ease="linear") == pytest.approx(5)
    assert key(-1, [(0.2, 3), (1, 10)]) == 3 and key(2, [(0, 0), (1, 10)]) == 10       # holds outside
    assert key(0.5, [(0, (0, 0)), (1, (10, 20))], ease="linear") == pytest.approx((5, 10))
    assert key(1, [(0, "#000000"), (1, "#ffffff")]) == pytest.approx((1, 1, 1))
    assert wave(0) == pytest.approx(wave(1)) and wave(0.5) == pytest.approx(1)
    assert orbit(0) == pytest.approx(orbit(1)) and pingpong(0.5) == pytest.approx(1)


def _frames(fn, n=8, h=40, w=60):
    return [np.dstack([fn(i, n) * np.ones((h, w), np.float32)] * 3 + [np.ones((h, w), np.float32)]) for i in range(n)]


def test_motion_metrics_flags():
    still = motion_metrics(_frames(lambda i, n: 0.5))
    assert any("static" in f for f in still["flags"])
    loop = motion_metrics(_frames(lambda i, n: wave(i / n)))
    assert loop["flags"] == [] and loop["seam"] < 2
    ramp = motion_metrics(_frames(lambda i, n: i / n))                 # never comes back: the loop jumps
    assert any("seam" in f for f in ramp["flags"])
    jump = motion_metrics(_frames(lambda i, n: 0.1 * i / n + (0.5 if i >= 5 else 0)), loop=False)
    assert any("jump at frame" in f for f in jump["flags"])


def test_rain_and_snow_loop_in_time():
    for name in ("rain", "snowfall"):
        out = []
        for t in (0.0, 0.25, 1.0):
            c = Canvas(120, 80, bg="#203040")
            c.layer(name="fx")
            getattr(c, name)(120, t=t, seed=3)
            out.append(c["fx"].rgba)
        assert not np.allclose(out[0], out[1]), name                  # it moves
        assert np.abs(out[0] - out[2]).mean() < 0.02, name            # and t = 1 is t = 0 again


def test_animate_cli_writes_gif_and_detects_flicker(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("from atelier import *\nc = Canvas(80, 50, bg='#223')\n"
                    "c.layer().circle(40 + 20 * orbit(T, 1)[0], 25, 8, fill='#fc6')\n")
    bad = tmp_path / "bad.py"
    bad.write_text("from atelier import *\nc = Canvas(80, 50)\nc.layer().fill(fbm(5, seed=FRAME))\n")
    assert main(["animate", str(good), "--frames", "6", "--width", "80", "--check"]) == 0
    gif = Image.open(tmp_path / "good.gif")
    assert gif.n_frames == 6 and (tmp_path / "good_frames.jpg").exists()
    frames = render_frames(str(bad), 6, width=80)
    m = motion_metrics(frames, probes=continuity_probes(str(bad), 6, width=80))
    assert any("flicker" in f for f in m["flags"])
    frames = render_frames(str(good), 6, width=80)
    assert motion_metrics(frames, probes=continuity_probes(str(good), 6, width=80))["flags"] == []
    for ext in ("webp", "mp4"):
        p = tmp_path / f"a.{ext}"
        save_anim(frames, p, 6)
        assert p.exists() and p.stat().st_size > 0
