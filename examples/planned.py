"""Planned scene (see styles/planning.md): thumbnails compared with `atelier sheet`, then built in passes."""
from atelier import *

PLAN = """
intent: a lighthouse keeper rows home at dusk; calm, a little lonely
read order: 1 keeper in the boat (darkest shape on the brightest light)  2 lit lighthouse  3 sky / sea
values: sky light-mid, sea mid-dark, light path lightest, island + lighthouse dark, keeper darkest
light: sun just above the horizon, left; warm light, cool violet shadows; accent = lamp + sun only
composition (thumbnail C, moved off-centre): sun + light path on the left third, keeper inside the path,
  lighthouse on the right third, horizon at 0.55
layers (back -> front): sky, sun, clouds, sea, path, island, lighthouse, lamp, keeper/boat, keeper/body, keeper/oar, ripples
"""
FOCUS = "keeper"

W, H = 1200, 800
HZ = 440                      # horizon y
SUN = (400, 395)
c = Canvas(W, H)

# ---- pass 1-3: masses, light, colour
c.layer(name="sky")
c.fill(gradient([(0, "#25234a"), (.45, "#6e5283"), (.78, "#d9806c"), (.93, "#f4ae73"), (1, "#ffd49a")]),
       mask=zone(0, 0, 1, HZ / H))
c.layer(name="glow", blend="screen")                     # warm halo around the low sun
c.fill(gradient([(0, "#b8643c"), (.5, "#3a1c1a"), (1, "#000000")], kind="radial",
                center=(SUN[0] / W, SUN[1] / H), radius=.55), mask=zone(0, 0, 1, (HZ + 40) / H))
c.layer(name="sun")
c.circle(*SUN, 46, fill="#fff1c9")
c.cur.apply(glow(40, .9, "#ffb36b"))
c.layer(name="clouds")                                   # cool tops, sun-lit warm undersides
for x, y, w, h in [(110, 250, 270, 16), (540, 300, 330, 13), (830, 205, 290, 18), (290, 342, 170, 9)]:
    c.rect(x, y, w, h, r=h / 2, fill="#4c3a66")
    c.rect(x + h, y + h * .55, w - 2 * h, h * .45, r=h / 4, fill="#e0927e")
c.layer(name="sea")
c.fill(gradient([(0, "#4a3f68"), (1, "#16142a")]), mask=zone(0, HZ / H, 1, 1))
c.layer(name="path", blend="screen", opacity=.55)           # soft reflection cone
path = polygon([(SUN[0] - 26, HZ), (SUN[0] + 26, HZ), (SUN[0] + 200, H), (SUN[0] - 180, H)])
c.fill(gradient([(0, "#ffcf94"), (1, "#7a3c3c")]), mask=path.soft(40))
c.layer(name="pool", blend="screen")                      # light behind the keeper: the value contrast lives here
c.draw(ellipse(SUN[0], 600, 105, 170), fill="#b77656", feather=70)
c.layer(name="island")
c.draw(svg(f"M 760 {HZ + 6} C 820 {HZ - 30} 990 {HZ - 34} 1070 {HZ + 6} Z"), fill="#1c1830")
c.layer(name="lighthouse")
c.polygon([(894, HZ - 22), (902, 190), (930, 190), (938, HZ - 22)], fill="#2a2440")
c.polygon([(916, 190), (930, 190), (938, HZ - 22), (920, HZ - 22)], fill="#3b3157")     # lit side
c.rect(890, 176, 52, 14, fill="#1c1830")
c.layer(name="lamp")
c.rect(904, 152, 24, 24, fill="#ffe7a8")
c.polygon([(898, 152), (916, 132), (934, 152)], fill="#1c1830")
c.cur.apply(glow(24, .8, "#ffcf7a"))

# ---- pass 4: shimmer = the detail budget, spent around the focus
import numpy as np
rng = np.random.default_rng(SEED if "SEED" in globals() else 0)
c.layer(name="ripples", blend="screen")
for i in range(150):
    t = (i / 150) ** 1.6                                   # denser near the horizon
    y = HZ + 4 + t * (H - HZ)
    half = 24 + t * 190                                    # cone widens toward the viewer
    w = (6 + t * 70) * rng.uniform(.4, 1)
    x = SUN[0] + rng.normal(0, .45) * half
    fade = 1 - .6 * max(0, t - .55) / .45                  # calmer foreground: detail stays near the keeper
    c.rect(x - w / 2, y, w, 1.2 + t * 4, r=2, fill="#ffe2b0",
           opacity=float(np.clip(1 - abs(x - SUN[0]) / half, .15, 1) * fade))

# ---- the keeper: darkest shape on the brightest area
bx, by = 418, 650
INK = "#0b0914"
c.layer(name="keeper/boat")
c.draw(svg(f"M {bx - 95} {by - 22} L {bx + 105} {by - 24} C {bx + 80} {by + 10} {bx - 70} {by + 12} {bx - 95} {by - 22} Z"),
       fill=INK)
c.layer(name="keeper/body")
c.draw(svg(f"M {bx - 6} {by - 22} C {bx - 10} {by - 70} {bx + 6} {by - 105} {bx + 30} {by - 112} "
           f"L {bx + 40} {by - 98} C {bx + 26} {by - 80} {bx + 26} {by - 50} {bx + 34} {by - 22} Z"), fill=INK)
c.circle(bx + 38, by - 124, 13, fill=INK)                                       # head
c.ellipse(bx + 38, by - 134, 21, 5, fill=INK)                                   # cap brim
c.line([(bx + 30, by - 98), (bx + 2, by - 72), (bx - 14, by - 66)], stroke=INK, width=8)  # arms to the oar
rim = c["keeper/body"].duplicate("keeper/rim")          # warm rim light on the side facing the sun
rim.fill("#ffb27a", mask=rim.alpha > 0)
rim.move(-3, -2)
rim.below("keeper/body")
c.layer(name="keeper/oar")
c.line(bx - 10, by - 70, bx - 140, by + 24, stroke=INK, width=5)
c.layer(name="splash", blend="screen")
c.ellipse(bx - 140, by + 26, 26, 4, fill="#ffe2b0", opacity=.7)
c.apply(vignette(.3), grain(.015, seed=1))
c.save(__file__[:-3] + ".png")
