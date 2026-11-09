# Oil landscape: a field-built colour study repainted with oil strokes (coarse to fine).
import numpy as np
from atelier import *

W, H = 1500, 1000
ref = Canvas(W, H)
ref.fill(gradient([(0, "#4f78b0"), (0.35, "#86a6cc"), (0.55, "#f0d2a6")]))                  # sky
cl = clouds(1.8, 0.42, sharp=3, seed=SEED + 3).scale(2.2, 0.8) * top(0.48, 0.08)
ref.fill(mix(gradient(["#fff6e8", "#cdbfc4"], angle=90), "#b7aebd", fbm(4, seed=SEED + 4).remap(0.4, 0.8)),
         mask=cl)                                                                              # lit cumulus, grey bellies
ref.fill_below(fbm(2, seed=SEED + 5).ridge_line(0.55, 0.08), "#8f95b8")                       # far hills
ref.fill_below(fbm(3, seed=SEED + 6).ridge_line(0.6, 0.05), "#6d8566")                        # mid hills
ref.fill_below(curve(lambda x: 0.62 + 0 * x), fbm(4, seed=SEED + 7).ramp(["#a98f3c", "#d1b25a", "#e4c878"]))
ref.fill("#7f8f45", mask=smoothstep(0.5, 0.75, fbm(3, seed=SEED + 8)) * bottom(0.63, 0.02) * 0.5)            # green patches
ref.fill("#6b6040", mask=clouds(1.5, 0.35, seed=SEED + 9).scale(2.5, 0.6) * bottom(0.66, 0.03) * 0.45)  # cloud shadows
river = svg("M 300 1000 C 520 900 980 860 820 760 S 640 650 760 620 L 790 620 C 700 660 900 760 940 790"
            " S 760 920 560 1000 Z")
ref.draw(river, fill=gradient(["#b9cde4", "#e9dcc2"], angle=-90).within(river))
for x, h in [(1040, 230), (1095, 270), (1150, 250), (1205, 210), (1255, 240)]:                # poplars
    tree = ellipse(x, 620 - h / 2, 26, h / 2)
    ref.draw(tree, fill=gradient(["#2f4a2a", "#5f7a3a"], angle=0).within(tree))
    ref.draw(ellipse(x + 60, 640, 70, 9), fill="#7d7140", opacity=0.7)                        # cast shadows
ref.draw(ellipse(260, 560, 120, 95), fill=gradient(["#3d5530", "#6e8a45"], angle=-20).within(rect(140, 465, 240, 190)))
ref.rect(250, 600, 14, 70, fill="#4a3b2a")                                                    # lone tree trunk
ref.fill("#6a5a2e", mask=band(0.86, 1.0, 0.08) * 0.35)                                        # darker foreground

c = Canvas(W, H, bg="#e9dfcc")
c.paint_like(ref, "oil", sizes=(64, 32, 16, 9), tol=0.08, seed=SEED)
c.cur.relief = 0.6                                                                  # gentler impasto
c.apply(curves([(0.2, 0.17), (0.8, 0.84)]), vignette(0.25), grain(0.015, seed=SEED))
c.save(__file__[:-3] + ".png")
ref.preview(__file__[:-3] + "_ref.jpg", 768)
