# Flat graphic poster: geometric composition, halftone-shaded planet, typographic grid.
import numpy as np
from atelier import *

W, H = 1200, 1700
navy, cream, gold, coral, teal = "#14213d", "#f4ecd8", "#fca311", "#e85d4a", "#2a9d8f"
c = Canvas(W, H, bg=navy)
c.scatter("dot", 260, region=top(0.62), size=(1.5, 4), color=cream, seed=SEED + 1)    # stars

cx, cy, R = 700, 620, 330
sky = rect(0, 0, W, 1110)
for i, r in enumerate([430, 520, 610]):                                               # orbits, clipped above the rule
    c.draw(circle(cx, cy, r).outline(2, dash=(2, 12) if i % 2 else None) & sky, fill=cream, opacity=0.45)
planet = circle(cx, cy, R)
c.draw(planet, fill=gradient([gold, coral], angle=120).within(planet))

def lambert(u, v):                       # sphere lit from the upper left, in the planet's local [0,1] box
    nx, ny = u * 2 - 1, v * 2 - 1
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    return np.clip((-0.55 * nx - 0.45 * ny + 0.7 * nz) * 1.4, 0, 1) ** 1.2
shadow = c.layer(blend="multiply")
shadow.fill(Field(lambert).within(planet), mask=planet)
shadow.apply(halftone(9, 30, ink="#1b2440", paper="white"))
c.layer()                                                # back to a normal layer for the rest

c.circle(cx - 470, cy + 330, 46, fill=teal)                                           # moons on orbits
c.circle(cx + 398, cy - 334, 22, fill=cream)
c.circle(cx - 573, cy - 209, 14, fill=coral)

c.line(80, 1135, W - 80, 1135, stroke=cream, width=3)
c.text("ORBITE", 80, 1400, fit=W - 160, weight="bold", fill=cream, spacing=-0.01)
c.text("NUIT DES ÉTOILES", 80, 1490, 38, weight="semibold", fill=gold, spacing=0.18)
c.text("Observatoire du Pic\nSamedi 14 août — 21 h", 80, 1565, 30, fill=cream, line_height=1.4)
c.text("Télescopes ouverts\nau public · gratuit", W - 80, 1565, 30, fill=cream, align="right", line_height=1.4)
c.text("N° 07", 80, 130, 34, weight="semibold", fill=cream, spacing=0.1)
c.apply(grain(0.02, 1.5, seed=SEED))
c.save(__file__[:-3] + ".png")
