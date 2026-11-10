# Build examples/gallery.jpg (README / landing banner) from the example renders: a 4 x 3 mosaic.
import os
from atelier import *

here = os.path.dirname(os.path.abspath(__file__))
names = ["still_life_pt", "mountains", "comic", "people",
         "landscape", "typography", "forest", "engraving",
         "poster", "seascape", "pixel", "abstract"]
anchor = {"typography": 0.18, "poster": 0.33}          # vertical crop position for portrait images (0 top, 1 bottom)
cols, tw, th, gap = 4, 450, 300, 10
rows = -(-len(names) // cols)
g = Canvas(cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap, bg="#111111")
for i, name in enumerate(names):
    img = load(os.path.join(here, f"{name}.png"))
    ih, iw = img.shape[:2]
    k = max(tw / iw, th / ih)                        # cover-fit
    y_off = (ih * k - th) * (anchor.get(name, 0.5) - 0.5)
    tile = Canvas(tw, th)
    tile.place(img, tw / 2, th / 2 - y_off, scale=k, layer=tile.cur)
    x, y = gap + (i % cols) * (tw + gap), gap + (i // cols) * (th + gap)
    g.place(tile.flatten(), x + tw / 2, y + th / 2, scale=1, layer=g.cur)
g.save(os.path.join(here, "gallery.jpg"), quality=86)
