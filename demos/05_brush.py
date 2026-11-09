# Step 5 demo: oil landscape painted from a field-built reference.
from atelier import *

W, H = 1500, 1000
ref = Canvas(W, H)                                                   # quick colour sketch
ref.fill(gradient([(0, "#5d7fb3"), (0.45, "#a9c2d9"), (0.6, "#f0d9b5")]))
ref.fill("#f7e7c8", mask=clouds(2.2, 0.35, seed=2).scale(2.5, 0.7) * top(0.45, 0.1))
ref.fill_below(fbm(2.5, seed=4).ridge_line(0.55, 0.12), "#6f7f9e")        # far hills (bluer, darker than sky)
ref.fill_below(fbm(3, seed=5).ridge_line(0.63, 0.08), fbm(6, seed=1).ramp(["#56703f", "#7b8f4a"]))
ref.fill_below(curve(lambda x: 0.74 + 0.03 * x), fbm(8, seed=3).ramp(["#c9a447", "#e0c070", "#a8893a"]))  # wheat
ref.fill(fbm(5, seed=8).ramp(["#2c4423", "#3f5e2c", "#5d7a39"]),
         mask=(worley(9, seed=6) < 0.45) * band(0.57, 0.72, 0.03) * right(0.55, 0.05))   # trees
ref.draw(svg("M 700 1000 C 760 900 640 840 760 760 L 790 760 C 700 840 860 900 860 1000 Z"), fill="#9a7a45")  # path

c = Canvas(W, H, bg="#efe6d4")
c.paint_like(ref, "oil", sizes=(56, 28, 14), seed=1)
c.cur.relief = 1.0
c.apply(vignette(0.25), grain(0.02))
c.save("demos/out/05_brush.png")
c.preview("demos/out/05_brush_prev.jpg")
c.crop_preview("demos/out/05_brush_crop.jpg", 900, 520, 360, 240)
ref.preview("demos/out/05_ref_prev.jpg")
