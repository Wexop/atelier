# Step 3 demo: flat graphic poster (shapes, booleans, text).
from atelier import *

W, H = 1000, 1400
ink, paper, sun, sea, rose = "#1d1a2f", "#f2e8d5", "#f05a28", "#2a6f97", "#f4a6a0"
c = Canvas(W, H, bg=paper)
panel = rect(60, 60, W - 120, 960, r=6)                       # art is clipped to this panel
c.draw(panel, fill="#f7d9b0")

disc_ = circle(500, 560, 290)                                 # sun cut by slits
slits = union(rect(150, 610 + i * 34, 700, 5 + i * 3) for i in range(7))
c.draw(disc_ - slits, fill=gradient(["#f7a541", sun]).within(disc_))

for d, col in [("M0 760 C 180 640 320 700 480 760 S 800 690 1000 740 V 1100 H 0 Z", rose),
               ("M0 860 C 220 760 380 840 560 880 S 860 820 1000 860 V 1100 H 0 Z", sea),
               ("M0 960 C 260 900 520 990 1000 930 V 1100 H 0 Z", ink)]:
    c.draw(svg(d) & panel, fill=col)
c.draw(star(860, 170, 46, n=8), fill=paper)
c.text("2026", 110, 190, 72, fill=ink, weight="bold")

c.text("SOLSTICE", W / 2, 1190, fit=W - 120, weight="bold", align="center", fill=ink)
c.line(60, 1225, W - 60, 1225, stroke=ink, width=3, dash=(1, 9))
c.text("MUSIC FESTIVAL", 60, 1290, 30, fill=ink, weight="semibold", spacing=0.12)
c.text("JUNE 21 — 23", W - 60, 1290, 30, fill=sun, weight="semibold", spacing=0.12, align="right")
c.text("Prado Beach, Marseille", 60, 1335, 26, fill=ink)
c.text("free entry", W - 60, 1335, 26, fill=ink, align="right", italic=True)

c.save("demos/out/03_vector.png")
c.preview("demos/out/03_vector_prev.jpg", 640)
c.crop_preview("demos/out/03_vector_crop.jpg", 60, 1250, 420, 110)
