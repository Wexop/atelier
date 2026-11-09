# Build examples/gallery.jpg (README banner) from the five example renders.
import os
from atelier import *

here = os.path.dirname(os.path.abspath(__file__))
g = Canvas(1800, 1080, bg="#111111")
items = [("still_life", 20, 20, 1000, 667), ("landscape", 1040, 20, 740, 493), ("poster", 1040, 533, 370, 527),
         ("pixel", 1430, 533, 350, 197), ("abstract", 20, 707, 1000, 353)]
for name, x, y, w, h in items:
    img = load(os.path.join(here, f"{name}.png"))
    ih, iw = img.shape[:2]
    tile = Canvas(w, h)
    tile.place(img, w / 2, h / 2, scale=max(w / iw, h / ih), layer=tile.cur)    # cover-fit crop
    g.place(tile.flatten(), x + w / 2, y + h / 2, scale=1, layer=g.cur)
g.text("atelier", 1605, 790, 46, fill="#eeeeee", weight="bold", align="center")
g.text("5 images, 0 generative AI", 1605, 840, 20, fill="#aaaaaa", align="center")
g.save(os.path.join(here, "gallery.jpg"), quality=88)
