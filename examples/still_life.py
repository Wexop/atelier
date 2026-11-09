# Photoreal still life (3D SDF): chiaroscuro window light, pewter, glass, ceramic, fruit, wood, DOF.
import numpy as np
from atelier import *

s = Scene()
s.add(box(4, 0.1, 1.6).move(0, -0.05, 0.1), Mat(wood(rings=22, seed=4).ramp(["#2a170b", "#4e2e17", "#6e4524"]),
      rough=0.4, proj="xz", tex_scale=0.4))                                                  # table top
s.add(plane(-0.9, (0, 0, 1)), Mat("#3a3128", rough=0.95))                                   # dark wall
jug = lathe([(0.17, 0), (0.21, 0.12), (0.2, 0.32), (0.13, 0.48), (0.11, 0.6), (0.13, 0.66)], thickness=0.02)
handle = torus(0.11, 0.018).rotate(x=90).scale((0.8, 1.4, 1)).move(0.2, 0.38, 0)
s.add((jug | handle).move(-0.55, 0, -0.35), Mat("#a7a9ab", rough=0.32, metal=1))           # pewter jug
glass = lathe([(0.0, 0.0), (0.12, 0.005), (0.125, 0.04), (0.11, 0.2), (0.13, 0.34)], thickness=0.01)
s.add(glass.move(0.05, 0, -0.25), Mat("#e8f0ea", glass=1, rough=0.02, ior=1.5))
bowl = lathe([(0.06, 0.0), (0.12, 0.02), (0.2, 0.07), (0.26, 0.15)], thickness=0.018)
s.add(bowl.move(0.55, 0, -0.15), Mat("#f1ece2", rough=0.15, sss=0.3))                      # glazed white bowl
rng = np.random.default_rng(SEED)
grapes = None
for i in range(16):
    a, r = rng.uniform(0, 6.28), rng.uniform(0, 0.13)
    g = sphere(0.035).move(0.55 + r * np.cos(a), 0.11 + rng.uniform(0, 0.08) + (0.13 - r) * 0.5, -0.15 + r * np.sin(a))
    grapes = g if grapes is None else grapes | g
s.add(grapes, Mat("#2a1c3a", rough=0.22, sss=0.45, sss_color="#6a2348"))
streak, blush = noise3(1, 2, seed=SEED + 7), noise3(6, 2, seed=SEED + 8)
def apple_skin(p, n):                         # red blush + fine vertical streaks over yellow-green
    q = p * np.array([40, 6, 40], np.float32)
    t = np.clip(0.6 + (blush(p) - 0.5) * 2.0 + (streak(q) - 0.5) * 0.7, 0, 1)[:, None]
    return np.array([0.86, 0.72, 0.30]) * (1 - t) + np.array([0.62, 0.09, 0.06]) * t
for (x, z, sc) in [(-0.15, 0.22, 1.0), (0.22, 0.3, 0.9)]:                                    # apples + stems
    ap = sphere(0.13 * sc).scale((1, 0.88, 1)).sub(sphere(0.04 * sc).move(0, 0.125 * sc, 0), k=0.045 * sc)
    s.add(ap.move(x, 0.115 * sc, z), Mat(apple_skin, rough=0.3, sss=0.3))
    s.add(capsule((0, 0, 0), (0.012, 0.06, 0), 0.007).move(x, 0.2 * sc, z), Mat("#4a3420", rough=0.7))
lemon = ellipsoid(0.11, 0.075, 0.075).displace(noise3(70, 2, seed=SEED + 2), 0.004).rotate(y=25)
s.add(lemon.move(0.62, 0.075, 0.28), Mat("#f5bf00", rough=0.35, sss=0.35))
s.add(round_box(0.36, 0.008, 0.035, 0.003).rotate(y=-20).move(-0.45, 0.004, 0.42), Mat("#c9ccd0", rough=0.2, metal=1))
s.area(pos=(-2.4, 1.9, 1.2), size=(1.2, 1.6), normal=(1, -0.55, -0.4), color="#ffe7c4", power=48)
s.add(box(0.05, 1.6, 1.2).rotate(y=-30).move(-3.3, 2.4, 1.6), Mat("#000000", emit="#fff1dc", power=6))  # window seen in reflections
s.sky("#202026", "#2c2722", "#120d09", power=0.15)
s.camera((0.15, 0.62, 2.0), (0.05, 0.2, -0.05), fov=30)

c = Canvas(1200, 800)
r = c.render(s, aa=2, exposure=0.1, look="punchy")
c.apply(depth_of_field(r.depth, focus=r.focus_at(727, 540, c.s), aperture=3), bloom(0.85, 10, 0.25),
        vignette(0.45), grain(0.012, seed=SEED))
c.save(__file__[:-3] + ".png")
