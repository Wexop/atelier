# Editorial poster: kerned display type, text on curved paths, a justified rich-text column, title presets.
import math
from atelier import *

W, H = 1200, 1600
ink, paper, red, blue, gold = "#1b1a1f", "#f3ede1", "#d9412b", "#23466b", "#e8a93a"
M = 80
c = Canvas(W, H, bg=paper)
c.layer(blend="multiply", opacity=.18).fill(fabric(90) * .15 + .85)
c.layer()

# masthead
c.text("No. 12", M, 110, 26, weight="semibold", fill=ink, spacing=.08)
c.text("THE KERNING ISSUE", W - M, 110, 26, weight="semibold", fill=red, align="right", spacing=.22)
c.line(M, 132, W - M, 132, stroke=ink, width=3)

# display type: kerned (default) — AV, YA, To, Ty pairs tuck in
head = c.text("VOYAGE", M, 400, fit=W - 2 * M, weight="bold", fill=ink, spacing=-.015)
c.text("TYPE", M - 6, 640, fit=560, weight="bold", fill=red, spacing=-.02)

# kerning specimen: unkerned (grey ghost) vs kerned
sx, sy = 690, 470
c.text("kern=False", sx, sy - 6, 17, fill="#8a857b", spacing=.05)
c.text("kern=True", sx + 260, sy - 6, 17, fill=red, spacing=.05)
for i, pair in enumerate(["AV", "To", "Ty", "LT"]):
    y = sy + 52 + i * 46
    c.text(pair, sx, y, 44, weight="semibold", fill="#a9a398", kern=False)
    c.text(pair, sx + 260, y, 44, weight="semibold", fill=ink)
    c.line(sx + 120, y - 15, sx + 230, y - 15, stroke="#c9c1b2", width=1.5, dash=(4, 5))

# emblem: text around a circle
cx, cy, R = 930, 960, 170
c.circle(cx, cy, R, fill=gold)
c.circle(cx, cy, R - 46, fill=blue)
c.text_on_path("LETTERS IN ORBIT  ·  TYPE ON A PATH  ·  " * 2, circle(cx, cy, R - 34), 21, weight="semibold",
               fill=ink, lift=-9, fit=True)
c.text("Aa", cx, cy + 34, 110, weight="bold", fill=paper, align="center")

# wave with text riding on it
pts = [(M + i * 26, 1215 + 30 * math.sin(i * .21)) for i in range(0, 41)]
wave = spline(pts)
c.draw(wave, fill=None, stroke=blue, width=2.5)
c.text_on_path("Words can follow any curve you draw — kerned, spaced and upright along the tangent.", wave,
               25, fill=blue, lift=10, align="center", weight="semibold")

# justified rich-text column
body = ("**Good spacing is invisible.** Kerning adjusts the gap between specific pairs of letters, so that "
        "*AV*, *To* and *Ty* sit as close as their shapes allow while straight-sided pairs keep their "
        "natural rhythm. The values come straight from the font's own [#d9412b]GPOS pair tables[/], read by a "
        "tiny parser and cached per typeface.\n"
        "Paragraphs wrap word by word into a column; justified lines share the leftover space between words, "
        "and the last line of each paragraph stays flush left. Give the box a height and **fit=True**, and the "
        "size shrinks until everything fits.")
col = c.text_box(body, M, 780, 500, h=310, size=21, align="justify", line_height=1.45, fill=ink, fit=True)
c.text(f"set in Inter {col.size:.1f} px, justified", M, 780 + col.height + 26, 15, fill="#8a857b", spacing=.04)

# footer: title presets
c.line(M, 1300, W - M, 1300, stroke=ink, width=3)
c.title("SHADOW", M, 1400, 64, style="shadow", fill=red, color2=ink, depth=5)
c.title("INLINE", 440, 1400, 64, style="inline", fill=blue, color2=paper)
c.title("BLOCK", 790, 1400, 64, style="extrude", fill=gold, color2=ink, depth=7)
c.text("atelier · typography · kerning · text on path · paragraph layout", W / 2, 1520, 18, fill="#6f6a60",
       align="center", spacing=.06)
c.apply(grain(.02, 1.3, seed=SEED))
c.save(__file__[:-3] + ".png")
