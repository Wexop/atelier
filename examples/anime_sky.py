# Anime summer sky: cel-shaded cumulus over snowy peaks and a bright meadow.
import numpy as np
from atelier import *

W, H = 1600, 1000
c = Canvas(W, H)
c.fill(gradient([(0, "#1f5fd0"), (0.35, "#3f86e6"), (0.62, "#8fc6f5"), (0.75, "#cfe8fb")]))

# clouds: big tower left-centre, smaller ones trailing, faint far row near the horizon
sky = c.layer()
for x, y, w, sd in [(300, 640, 300, 11), (1350, 600, 280, 12), (800, 650, 220, 13)]:
    sky.anime_cloud(x, y, w, h=w * 0.35, palette=("#b8cbe8", "#dce8f7", "#ffffff", "#ffffff"), wisps=0.2, seed=sd)
sky.anime_cloud(1060, 470, 760, h=420, light=(130, 40), seed=4)
sky.anime_cloud(420, 330, 420, h=200, light=(130, 40), seed=7)
sky.anime_cloud(1480, 250, 260, h=120, light=(130, 40), seed=9)

# mountains: far pale range, then the main snowy range (cel-shaded facets radiating from the summits)
rng = np.random.default_rng(SEED)
c.layer().anime_mountains(760, 330, peaks=5, light=(150, 30), snow=0.25, haze=0.8, seed=5,
                          palette=("#7d9fd2", "#93b3e0", "#a9c5ea", "#f6f9ff", "#cddbf0", "#d6e8f8"))
c.layer().anime_mountains(780, 430, peaks=3, x0=250, x1=1450, light=(150, 30), snow=0.35, seed=8)
c.layer(opacity=0.4).fill("#d4e7f8", mask=band(0.66, 0.82, soft=0.1))     # aerial haze at the foot

# rolling meadows, far to near: flat cel bands, a lighter lit strip on each crest
fields = [(0.70, 0.025, "#8cc95e", "#b5e07a", 21), (0.76, 0.04, "#6db74a", "#9ad65e", 22), (0.86, 0.06, "#4c9a3a", "#7cc64d", 23)]
for y, amp, dark, light, sd in fields:
    crest = fbm(1.2, 2, seed=sd).ridge_line(y, amp)
    L = c.layer()
    L.fill_below(crest, light)
    L2 = c.layer()
    L2.fill_below(fbm(1.2, 2, seed=sd).ridge_line(y + 0.025, amp * 0.8), dark)
for x, y, hgt, kind, sd in [(110, 770, 260, "round", 31), (270, 778, 180, "tall", 32), (1400, 768, 270, "round", 33),
                            (1545, 780, 170, "round", 34), (700, 745, 70, "round", 35), (760, 748, 55, "tall", 36)]:
    c.layer().anime_tree(x, y, hgt, kind=kind, light=(130, 40), seed=sd)

# foreground: bright grass tufts and small flowers
fg = c.layer()
for k in range(26):
    gx = k * 64 + rng.uniform(-20, 20)
    fg.grass(gx, 1005, 120, rng.uniform(50, 90), 70, color=["#3f8f34", "#6cbf45", "#a6e06a"], seed=SEED + k, bend=0.4)
fg.scatter("dot", 260, region=band(0.88, 1.0), size=(3, 7), color=["#ffffff", "#fff3a0", "#ffd84a"], seed=SEED + 3)

c.apply(curves([(0, 0.03), (0.5, 0.52), (1, 1)]), saturation(1.08), grain(0.01), vignette(0.15))
c.save(__file__[:-3] + ".png")
