# Step 4 demo: grade the step-2 landscape (left: raw, right: graded).
import runpy
from atelier import *

raw = runpy.run_path("demos/02_field.py")["c"]
before = raw.flatten()
raw.apply(
    bloom(0.7, 18, 1.2, tint="#ffd9b0"),                       # moon & stars bloom
    color_balance(shadows=(-.02, 0, .05), highlights=(.06, .03, -.03)),
    curves([(0.2, 0.16), (0.8, 0.86)]),                       # gentle S
    vignette(0.35, 0.7),
    grain(0.035, 1.2, seed=7),
)
raw.save("demos/out/04_fx.png")
raw.preview("demos/out/04_fx_prev.jpg")
raw.crop_preview("demos/out/04_fx_crop.jpg", 1000, 120, 320, 200)
cmp = Canvas(1600, 500)
cmp.place(before, 400, 250, scale=0.5, layer=cmp.cur)
cmp.place(raw.flatten(), 1200, 250, scale=0.5, layer=cmp.cur)
cmp.preview("demos/out/04_fx_compare.jpg", 1024)
