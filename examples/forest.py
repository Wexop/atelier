# Forest edge on a summer morning: physical sky, hazy wooded ridges, meadow, oaks, birches and pines.
import numpy as np
from atelier import *
from atelier.nature import *

W, H = 1500, 1000
rng = np.random.default_rng(SEED)
LIGHT = (150, 28)                                   # 2D light: from the upper left
SUN = dict(sun_elev=11, sun_az=-38)


def ground(y0, amp, seed, k=3):
    """A rolling ground line y(x) in canvas px."""
    r = np.random.default_rng(seed)
    ph, fr = r.uniform(0, 6.3, k), r.uniform(1.0, 3.5, k)
    return lambda x: y0 + amp * sum(np.sin(x / W * f * np.pi + p) / (i + 1) for i, (p, f) in enumerate(zip(ph, fr)))


c = Canvas(W, H)
c.sky(**SUN, horizon=0.56, fov=50, clouds=0.3, haze=1.3, seed=SEED + 7)
sky_haze = "#c9cfd2"

# far wooded ridges: each band = hazy fill + a dense row of small conifers along its crest
bands = [(540, 34, 0.5, 34), (585, 30, 0.38, 44), (635, 26, 0.26, 56)]
for i, (y0, amp, hz, th) in enumerate(bands):
    g = ground(y0, amp, SEED + 10 + i)
    xs = np.arange(0, W + 1, 4.0)
    c.layer()
    col = np.array(color("#22382a")[:3]) * (1 - hz) + np.array(color(sky_haze)[:3]) * hz
    c.polygon([(0, H)] + [(x, g(x) + th * 0.35) for x in xs] + [(W, H)], fill=tuple(col))
    for x in np.sort(rng.uniform(-20, W + 20, int(W / th * 1.6))):
        c.tree(x, g(x) + th * 0.45, th * rng.uniform(0.8, 1.4), "pine", seed=int(rng.integers(1e6)),
               light=LIGHT, haze=hz, haze_color=sky_haze, shadow=0, leaves=0.7)

# meadow: lit gold-green with darker undulations, soft shadow under the near ridge
c.layer()
mg = ground(690, 14, SEED + 20)
meadow = polygon([(0, H)] + [(x, mg(x)) for x in np.arange(0, W + 1, 5.0)] + [(W, H)])
c.draw(meadow, fill=gradient([(0, "#9aa457"), (0.35, "#7f9444"), (1, "#45612a")], angle=90).within(meadow))
c.fill("#566f2e", mask=meadow * smoothstep(0.45, 0.7, fbm(5, seed=SEED + 3)).scale(1, 0.3) * 0.5)
c.fill("#d4c27a", mask=meadow * (left(0.45, 0.3) * band(0.68, 0.8, 0.05) * 0.25))      # warm light on the left

# mid-distance trees along the meadow edge, back to front
mid = [(60, "pine", 190), (420, "oak", 150), (520, "pine", 210), (610, "birch", 170), (700, "oak", 140),
       (880, "pine", 230), (980, "oak", 160), (1080, "birch", 150), (1250, "pine", 200), (1360, "oak", 150)]
c.layer()
for x, k, hgt in mid:
    c.tree(x, mg(x) + 6, hgt, k, seed=int(x), light=LIGHT, haze=0.22, haze_color=sky_haze, shadow=0.3)
for x in rng.uniform(0, W, 40):
    c.grass(x, mg(x) + 8, 50, 12, n=18, seed=int(x * 7), light=LIGHT, haze=0.2, haze_color=sky_haze, dry=0.4)
c.scatter("grass", 2500, region=meadow * bottom(0.72, 0.05), size=(6, 16), color=["#5d7a34", "#7b9142", "#a3a65a",
          "#4c662b"], color_jitter=0.15, seed=SEED + 4)

# foreground: big oak (left), birch pair (right), undergrowth
c.layer()
fy = 965
c.draw(polygon([(0, H), (0, 900), (300, 880), (700, 915), (1100, 895), (1500, 870), (W, H)]),
       fill=gradient(["#4b6427", "#2a3d19"], angle=90).within(rect(0, 860, W, 140)))
c.tree(230, fy, 700, "oak", seed=SEED + 5, light=LIGHT, shadow=0.45, width=1.1)
c.tree(1180, fy - 10, 700, "birch", seed=SEED + 6, light=LIGHT, shadow=0.35)
c.tree(1320, fy + 5, 600, "birch", seed=SEED + 8, light=LIGHT, shadow=0.35)
c.tree(1460, fy + 15, 760, "pine", seed=SEED + 9, light=LIGHT, shadow=0.35)
for i, x in enumerate(np.linspace(-20, W + 20, 46)):
    c.grass(x + rng.normal(0, 12), 1000 - rng.uniform(0, 70), 90, rng.uniform(30, 60), n=40, seed=i,
            light=LIGHT, dry=0.15)
for x in (520, 640, 1020):
    c.fern(x, 990, rng.uniform(80, 120), seed=int(x), light=LIGHT)
for i in range(40):
    x = rng.uniform(380, 1100)
    c.flower(x, rng.uniform(930, 995), rng.uniform(22, 38), ["daisy", "poppy", "buttercup", "bell"][i % 4],
             seed=i, light=LIGHT)

# morning light: warm glow from the sun side, gentle haze, then grade
c.layer(blend="screen", opacity=0.35).fill(gradient(["#ffcf8a", "#000000"], kind="radial", center=(0.05, 0.45),
                                                    radius=0.9))
c.apply(bloom(0.8, 14, 0.25), curves([(0.1, 0.08), (0.85, 0.9)]), vignette(0.28), grain(0.012, seed=SEED))
c.save(__file__[:-3] + ".png")
