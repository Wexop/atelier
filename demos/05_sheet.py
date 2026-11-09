# Step 5 check: each brush on paper, plus scatter / hatch / KM mixing.
import numpy as np
from atelier import *

c = Canvas(1400, 1000, bg="#f4efe4")
wave = lambda y, a=30, n=7: [(80 + i * 140, y + a * np.sin(i * 1.3)) for i in range(n)]
names = ["round", "round_soft", "airbrush", "ink", "chalk", "bristle", "oil", "watercolor"]
cols = ["#1d3557", "#e63946", "#457b9d", "#111111", "#6a4c93", "#2a9d8f", "#e9c46a", "#3a86ff"]
for i, (b, col) in enumerate(zip(names, cols)):
    y = 70 + i * 82
    c.stroke(wave(y), brush=b, size=34, color=col, pressure=[0.6, 1, 0.8, 1, 0.7, 1, 0.5], seed=i)
    c.text(b, 1000, y + 8, 22, fill="#333")
# watercolor glazing: overlapping washes darken
for k, col in enumerate(["#f4a261", "#2a9d8f", "#e76f51"]):
    c.stroke([(1060 + k * 40, 120), (1120 + k * 40, 250), (1080 + k * 40, 380)], "watercolor", 60, col, seed=20 + k)
# KM mix swatches: blue+yellow
for k, t in enumerate(np.linspace(0, 1, 6)):
    c.rect(1040 + k * 50, 430, 46, 46, fill=tuple(km_mix("#1f4fbf", "#ffd500", t)))
    c.rect(1040 + k * 50, 480, 46, 46, fill=tuple(np.array(color("#1f4fbf")[:3]) * (1 - t) + color("#ffd500")[:3] * t))
c.scatter("grass", 400, region=zone(0.74, 0.62, 0.99, 0.78), seed=1)
c.scatter("person", 60, region=zone(0.74, 0.8, 0.99, 0.84), seed=2)
c.layer().hatch(circle(1180, 920, 60), angle=30, spacing=6, cross=True, tone=X.within(rect(1120, 860, 120, 120)))
c.save("demos/out/05_sheet.png")
c.preview("demos/out/05_sheet_prev.jpg", 1400)
