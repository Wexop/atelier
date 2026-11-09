# Step 1 demo: core only (layers, blend modes, masks, place, export).
import numpy as np
from atelier import *

c = Canvas(1200, 750)
# background: vertical gradient written as a plain f(x, y) field
top, bot = color("#10163a")[:3], color("#e8875a")[:3]
c.fill(lambda x, y: top + (bot - top) * (y ** 1.6)[..., None])

# sun: radial falloff on a screen layer
disc = lambda cx, cy, r, soft=0.004: lambda x, y: np.clip((r - np.hypot((x - cx) * 1.6, y - cy)) / soft, 0, 1)
sun = c.layer(blend="screen")
sun.fill("#ffd38a", mask=lambda x, y: np.exp(-((np.hypot((x - .5) * 1.6, y - .62)) / .18) ** 2))
sun.fill("#fff1c9", mask=disc(.5, .62, .07))

# sea: lower band, multiply-darkened, with sine ripples reflecting the sun
sea = c.layer(blend="multiply")
sea.fill(lambda x, y: np.stack([0.35 + 0.1 * np.sin(y * 300 + np.sin(x * 40) * 2)] * 3, -1),
         mask=lambda x, y: y > 0.7)
glint = c.layer(blend="add", opacity=0.8)
glint.fill("#ffcf8a", mask=lambda x, y: (y > 0.7) * np.exp(-((x - .5) / (.02 + (y - .7) * .5)) ** 2)
           * np.clip((np.sin(y * 420 + np.sin(x * 60) * 3) - 0.6) * 4, 0, 1))

# blend-mode swatches: same orange disc over a black->white strip, 6 modes
strip = c.layer()
strip.fill(lambda x, y: np.stack([(x - .02) / .96] * 3, -1),
           mask=lambda x, y: (abs(y - .2) < .1) * (abs(x - .5) < .48))
for i, m in enumerate(["multiply", "overlay", "soft_light", "color_dodge", "difference", "hue"]):
    c.layer(blend=m).fill("#ff6a00", mask=disc(0.1 + i * 0.16, 0.2, 0.08))

# place an imported image (here a generated checker), rotated & scaled, half opacity
chk = ((np.indices((8, 8)).sum(0) % 2) * 0.8 + 0.1).astype(np.float32)
c.place(chk, x=1080, y=560, scale=12, rot=20, opacity=0.7)

c.save("demos/out/01_core.png")
c.preview("demos/out/01_core_prev.jpg")
c.crop_preview("demos/out/01_core_crop.jpg", 500, 520, 200, 120)
