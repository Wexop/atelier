# Filter showcase: one 3D still life re-imagined by the stylize filters (each panel is its own canvas).
import os
from atelier import *
from atelier.stylize import kuwahara, xdog, watercolorize, pencil, linocut, stained_glass, mosaic

SRC = os.path.join(os.path.dirname(__file__), "still_life.png")
PW, PH, G, TOP = 560, 373, 26, 44
PANELS = [
    ("ORIGINAL RENDER", []),
    ("KUWAHARA  oil", [kuwahara(9, sharp=10)]),
    ("XDOG  ink", [xdog(1.2, eps=0.9, fill=0.1, paper="#f6f1e6")]),
    ("WATERCOLOUR", [watercolorize(bleed=2.5, seed=SEED)]),
    ("WATERCOLOUR + INK", [watercolorize(bleed=2.5, wash=0.6, ink=0.9, seed=SEED)]),
    ("PENCIL", [pencil(seed=SEED)]),
    ("LINOCUT", [linocut(angle=-15, seed=SEED)]),
    ("STAINED GLASS", [stained_glass(22, seed=SEED)]),
    ("MOSAIC", [mosaic(9, seed=SEED)]),
]
c = Canvas(3 * PW + 4 * G, 3 * (PH + TOP) + 2 * G + G // 2, bg="#ece6da")
for i, (name, fx) in enumerate(PANELS):
    px, py = G + (i % 3) * (PW + G), G // 2 + (i // 3) * (PH + TOP + G)
    p = Canvas(PW, PH)
    p.place(SRC)
    if fx:
        p.apply(*fx)
    img = p.flatten()
    c.place(img, px + PW / 2, py + TOP + PH / 2, scale=PW / img.shape[1])
    c.rect(px, py + TOP, PW, PH, fill=None, stroke="#2a241e", width=2)
    c.text(name, px, py + TOP - 12, size=20, fill="#2a241e", font="sans", weight="bold", spacing=0.12)
c.save(__file__[:-3] + ".png")
