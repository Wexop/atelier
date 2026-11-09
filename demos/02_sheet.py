# Step 2 check: contact sheet of every noise / pattern (4 x 4 tiles).
from atelier import *
c = Canvas(1600, 1000, bg="#111")
tiles = [
    ("perlin", perlin(6)), ("simplex", simplex(6)), ("value", value(6)), ("fbm", fbm(6)),
    ("ridged", ridged(4)), ("turbulence", turbulence(5)), ("worley F1", worley(10)), ("F2-F1", worley(10, "F2-F1")),
    ("cells", worley(8, "id").ramp("pastel")), ("warp", domain_warp(fbm(4), 0.25).ramp("neon")),
    ("wood", wood().ramp("wood")), ("marble", marble().ramp("marble")),
    ("clouds", mix("#3d7ac2", "white", clouds())), ("terrain", shade(terrain_height(3), color=terrain_height(3).ramp("earth"), strength=2)),
    ("fabric", shade(fabric(), color="#7a2f3a", strength=0.5)), ("paper+stars", mix(paper().ramp(["#d8d0c0", "#fffaf0"]), "#ffffff", stars(40))),
]
for i, (name, f) in enumerate(tiles):
    x0, y0 = (i % 4) / 4, (i // 4) / 4
    c.fill(f, mask=zone(x0 + .004, y0 + .006, x0 + .246, y0 + .244))
c.save("demos/out/02_sheet.png"); c.preview("demos/out/02_sheet_prev.jpg", 1024)
