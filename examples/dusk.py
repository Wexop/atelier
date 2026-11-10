# Flat vector dusk: banded sky, layered hills with rim light, lone tree, lake reflection, fireflies and petals.
import numpy as np
from atelier import *

W, H = 900, 1600
HZ = 1180                                     # shoreline (px)
SUN = (300, 880)
c = Canvas(W, H)

# sky: purple -> peach, with flat horizontal bands
c.fill(gradient([(0, "#2a2350"), (0.18, "#4b3f86"), (0.36, "#7a5fa8"), (0.48, "#e0795f"),
                 (0.56, "#ffb070"), (0.62, "#ffd9a0"), (0.74, "#c9a6d6")]))
bands = c.layer(blend="soft_light", opacity=0.6)
rng = np.random.default_rng(SEED)
for y in np.cumsum(rng.uniform(40, 110, 14)):
    t = rng.uniform(0.3, 1)
    bands.rect(-20, y, W + 40, rng.uniform(14, 40), r=20,
               fill="#ffffff" if rng.random() > 0.45 else "#20184a", opacity=0.35 * t)
c.layer(blend="screen").fill("#ffd59a", mask=disc(SUN[0] / W, SUN[1] / H, 0.08, soft=0.35))
c.layer(blend="screen", opacity=0.5).fill("#ff9d6e", mask=disc(SUN[0] / W, SUN[1] / H, 0.25, soft=0.8))

# distant mountains, hazy lavender
far = c.layer()
far.path("M -20 980 C 100 960 200 1000 300 1010 C 420 930 500 900 560 880 C 620 860 660 900 720 870 C 790 835 860 850 920 830 L 920 1200 L -20 1200 Z",
         fill="#8f8fcf")
far.path("M -20 1080 C 150 1050 280 1090 380 1060 C 480 1010 560 990 640 975 C 720 960 820 985 920 960 L 920 1200 L -20 1200 Z", fill="#a3a8de")

# distant tree silhouettes on the right
def blob_tree(L, x, y, s, col):
    L.rect(x - s * 0.04, y - s * 0.55, s * 0.08, s * 0.55, fill=col)
    L.draw(union(ellipse(x, y - s * 0.7, s * 0.42, s * 0.22), ellipse(x - s * 0.3, y - s * 0.6, s * 0.25, s * 0.16),
                 ellipse(x + s * 0.32, y - s * 0.62, s * 0.26, s * 0.15), ellipse(x + s * 0.05, y - s * 0.86, s * 0.25, s * 0.15)),
           fill=col)

trees = c.layer()
for x, y, s in [(520, 1100, 150), (640, 1090, 120), (790, 1080, 190), (880, 1090, 130)]:
    blob_tree(trees, x, y, s, "#7d82c4")

# rolling hills, back to front, each with a warm rim on its crest
hills = [("M -20 1000 C 120 940 280 960 420 1060 C 520 1120 600 1140 700 1150 L 700 1220 L -20 1220 Z", "#3b3a78"),
         ("M 330 1150 C 450 1090 600 1090 760 1120 C 840 1135 900 1140 920 1140 L 920 1220 L 330 1220 Z", "#4d4f92"),
         ("M -20 1080 C 160 1060 300 1120 380 1180 L -20 1200 Z", "#2a2a5e")]
for d, col in hills:
    L = c.layer()
    s = svg(d)
    L.draw(s, fill=col)
    L.draw(s.outline(4), fill="#ffb87a", opacity=0.55, feather=1.5)
c.layer().rect(0, HZ - 14, W, 14, fill="#2a2a5e")

# lone tree: dark trunk + layered canopy, rim-lit on the sun side
tree = c.layer()
trunk = svg("M 80 1080 C 95 960 70 860 120 760 C 140 720 170 690 210 660 L 225 675 C 180 720 150 780 145 860 "
            "C 140 950 150 1020 150 1080 Z")
branch = svg("M 150 780 C 220 720 300 700 380 690 L 382 702 C 300 715 230 740 160 800 Z") | \
         svg("M 120 820 C 70 760 40 720 20 640 L 32 636 C 55 710 90 760 130 805 Z")
tree.draw(trunk | branch, fill="#1c1a3c")
canopy = union(ellipse(170, 610, 190, 70), ellipse(70, 560, 120, 55), ellipse(260, 560, 150, 50),
               ellipse(420, 640, 130, 32), ellipse(520, 665, 70, 18), ellipse(140, 500, 110, 45))
tree.draw(canopy, fill="#1f1d44")
tree.draw(canopy & canopy.move(-18, -14), fill="#2b2958")
tree.draw(canopy.outline(3), fill="#f6b27c", opacity=0.5)

# lake: mirrored sky/land, darkened, with flat ripple strokes
img = c.flatten()
hz = int(HZ * c.s)
h = img.shape[0] - hz
src = img[hz - h:hz][::-1].copy()
fade = np.linspace(0.85, 0.45, h)[:, None, None]
src[..., :3] = src[..., :3] * fade + np.array([0.12, 0.12, 0.3]) * (1 - fade)
src[..., 3] = 1
lake = c.layer()
buf = np.zeros_like(img)
buf[hz:] = src
lake.rgba = buf
rip = c.layer(opacity=0.8)
h = H - HZ
for k in range(45):
    y = HZ + 8 + (k / 45) ** 1.6 * h
    x = rng.uniform(-100, W)
    w = rng.uniform(60, 260) * (0.5 + (y - HZ) / h)
    near_sun = abs(x + w / 2 - SUN[0]) < 200
    rip.line(x, y, x + w, y, stroke="#ffcf9a" if near_sun else "#8a86d0", width=1.5 + 2.5 * (y - HZ) / h)
c.layer().rect(0, HZ - 2, W, 4, fill="#ffc38a", opacity=0.6)

# fireflies and falling petals
ff = c.layer()
ff.scatter("dot", 120, region=band(0.05, 0.95), size=(5, 11), color=["#fff1b0", "#ffd36e"], seed=SEED + 1)
ff.apply(glow(18, 1.6, "#ffd36e"))
c.layer().scatter("petal", 70, region=band(0.0, 0.9), size=(14, 24), color=["#f5b8c8", "#ffd0dc", "#e89ab8"],
                  rot=(0, 360), seed=SEED + 2)

c.layer().sparkles(60, region=zone(0, HZ / H, 1, 1) * disc(SUN[0] / W, 0.85, 0.25, soft=0.6), size=(6, 18), seed=SEED + 5)
c.apply(god_rays(*SUN, rays=16, strength=0.5), lens_flare(*SUN, strength=0.5), bloom(0.7, 18, 0.6), grain(0.015), vignette(0.25))
c.save(__file__[:-3] + ".png")
