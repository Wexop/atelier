# Path-traced still life: window light with bounce light, glass of white wine, brass jug, glazed bowl,
# fruit with blemishes, linen napkin (sheen), oak table with object-space grain. Uses c.pathtrace.
import numpy as np
from atelier import *

s = Scene()
f32 = np.float32
warp, pores, fleck = noise3((0.8, 3, 3), 3, seed=SEED + 1), noise3((2, 60, 60), 2, seed=SEED + 2), noise3(9, 3, seed=SEED + 3)


def oak(p, n):                          # rings around an axis below the board -> long flowing grain (object space)
    r = np.hypot(p[:, 1] + 0.42, p[:, 2] * 0.55 + 0.8) * 34 + (warp(p) - 0.5) * 7
    ring = (0.5 + 0.5 * np.sin(r * 2 * np.pi)) ** 3
    t = np.clip(0.25 + 0.55 * ring + (pores(p) - 0.5) * 0.9 + (fleck(p) - 0.5) * 0.35, 0, 1)[:, None]
    return np.array([0.50, 0.33, 0.19], f32) * (1 - t) + np.array([0.27, 0.15, 0.07], f32) * t


oak_rough = lambda p, n: 0.32 + 0.22 * pores(p)
s.add(round_box(3.2, 0.08, 1.5, 0.012).move(0, -0.04, 0.05),
      Mat(oak, rough=oak_rough, local=True, bump=0.25, bump_scale=90, wear=[scratches(0.25, 25, SEED), dust(0.15)]))
s.add(plane(-0.75, (0, 0, 1)), Mat("#6f6253", rough=0.95, bump=0.1, bump_scale=12))          # plaster wall

jug = lathe([(0.15, 0), (0.19, 0.1), (0.185, 0.3), (0.12, 0.45), (0.1, 0.56), (0.125, 0.62)], thickness=0.016)
handle = torus(0.105, 0.016).rotate(x=90).scale((0.8, 1.4, 1)).move(0.19, 0.36, 0)
s.add((jug | handle).rotate(y=-25).move(-0.62, 0, -0.38),
      Mat("#c9a466", rough=0.26, metal=1, local=True, wear=[smudges(0.35, 5, SEED), scratches(0.25, 70, SEED)]))


def bowl_r(u):                          # tulip wine-glass bowl centre line, u = 0 (bottom) .. 1 (rim)
    return 0.064 * np.sin(np.pi / 2 * np.minimum(1, u / 0.6)) ** 0.55 * (1 - 0.22 * np.maximum(0, u - 0.6) / 0.4)


u = np.linspace(0, 1, 70)
y0, y1, th = 0.112, 0.27, 0.0035
foot = [(0.043, 0.0015), (0.041, 0.0045), (0.012, 0.008), (0.005, 0.016), (0.0038, 0.06), (0.0042, 0.1), (0.002, y0)]
glass = lathe(foot + [(bowl_r(v) + 0.002, y0 + v * (y1 - y0)) for v in u[1:]], thickness=th, smooth=False)
fill = 0.19
uw = u[u * (y1 - y0) + y0 <= fill]
wine = lathe([(max(bowl_r(v) - th / 2 - 0.002, 0.0), y0 + 0.003 + v * (y1 - y0)) for v in uw[1:]], smooth=False)
gx, gz = 0.08, -0.22
glass, wine = glass.scale(1.35), wine.scale(1.35)
s.add(glass.move(gx, 0, gz), Mat("#f6faf8", glass=1, rough=0.0, ior=1.5))
s.add(wine.move(gx, 0, gz), Mat("#f0c860", glass=1, rough=0.0, ior=1.34))

bowl = lathe([(0.055, 0.0), (0.11, 0.015), (0.19, 0.06), (0.235, 0.13)], thickness=0.014)
s.add(bowl.move(0.6, 0, -0.2), Mat("#ece5d6", rough=0.1, sss=0.15, bump=0.1, bump_scale=30,
                                   wear=[edge_wear(0.25, 0.006, "#c9b9a0", SEED)]))
rng = np.random.default_rng(SEED)
grapes = None
for i in range(14):
    a, r = rng.uniform(0, 6.28), rng.uniform(0, 0.11)
    g = sphere(0.03).scale((1, 1.12, 1)).move(0.6 + r * np.cos(a), 0.1 + rng.uniform(0, 0.06) + (0.11 - r) * 0.45,
                                             -0.2 + r * np.sin(a))
    grapes = g if grapes is None else grapes | g
s.add(grapes, Mat("#a3ab4e", rough=0.18, sss=0.55, sss_color="#c8d060", wear=[dust(0.5, seed=SEED)]))

blush, streak, dots = noise3(5, 3, seed=SEED + 7), noise3((40, 5, 40), 2, seed=SEED + 8), noise3(140, 1, seed=SEED + 9)


def apple(p, n):                        # yellow-green base, red blush, vertical streaks, pale lenticels
    t = np.clip(0.55 + (blush(p) - 0.5) * 2.4 + (streak(p) - 0.5) * 0.9, 0, 1)[:, None]
    c = np.array([0.80, 0.70, 0.28], f32) * (1 - t) + np.array([0.55, 0.06, 0.05], f32) * t
    return c + np.clip((dots(p) - 0.72) * 5, 0, 1)[:, None] * 0.25


for (x, z, sc, ry) in [(-0.2, 0.2, 1.0, 0), (0.25, 0.3, 0.9, 70)]:
    ap = sphere(0.13 * sc).scale((1, 0.88, 1)).sub(sphere(0.04 * sc).move(0, 0.125 * sc, 0), k=0.045 * sc)
    s.add(ap.rotate(y=ry).move(x, 0.114 * sc, z), Mat(apple, rough=0.32, sss=0.3, sss_color="#d0401c", local=True,
                                                         bump=0.12, bump_scale=50, wear=[smudges(0.4, 4, SEED)]))
    s.add(capsule((0, 0, 0), (0.014, 0.055, 0.004), 0.0055).move(x, 0.2 * sc, z), Mat("#4a3420", rough=0.7))
lemon = ellipsoid(0.1, 0.07, 0.07).rotate(y=25)        # pores via bump (cheaper than displacement)
s.add(lemon.move(0.56, 0.07, 0.36), Mat("#e8b10c", rough=0.38, sss=0.3, bump=0.6, bump_scale=110, local=True))

nb = round_box(0.5, 0.012, 0.36, 0.005)                 # gentle analytic folds (cheap to trace)
napkin = SDF(lambda p: nb.f(p) + 0.005 * np.sin(p[:, 0] * 13 + 1) * np.sin(p[:, 2] * 10), nb.bounds, 0.8).rotate(y=14)
s.add(napkin.move(0.5, 0.006, 0.27), Mat(fabric(90).ramp(["#cfc6b4", "#e4dccb"]), proj="xz", tex_scale=0.3,
                                         rough=0.85, sheen=0.7, sheen_color="#fff6e8", local=True))
knife = round_box(0.3, 0.005, 0.026, 0.0025).move(0.06, 0, 0) | round_box(0.12, 0.018, 0.024, 0.008).move(-0.16, 0.006, 0)
s.add(knife.rotate(y=-18).move(-0.42, 0.004, 0.42), Mat("#c8ccd0", rough=0.16, metal=1, wear=[scratches(0.7, 90, SEED)]))

s.studio(key=(-1.9, 1.5, 1.5), target=(0.05, 0.2, -0.1), size=(1.0, 1.4), power=26, window=True, fill=0.08,
         bg="#2a241e")
s.camera((0.12, 0.56, 2.05), (0.06, 0.19, -0.1), fov=27)

c = Canvas(1200, 800)
r = c.pathtrace(s, spp=8, bounces=4, exposure=0.45, look="punchy", seed=SEED, threads=2, verbose=True)
c.apply(depth_of_field(r.depth, focus=r.focus_at(560, 470, c.s), aperture=2.2), bloom(0.85, 10, 0.2),
        vignette(0.4), grain(0.01, seed=SEED))
c.save(__file__[:-3] + ".png")
