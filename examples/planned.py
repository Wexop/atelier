"""Planned scene (see styles/planning.md): flat vector dusk, built in passes and checked with `atelier critique`."""
import numpy as np
from atelier import *

PLAN = """
intent: a lighthouse keeper rows home at dusk; calm, a little lonely; flat vector illustration
read order: 1 keeper in the boat (darkest shape on the brightest water)  2 lit lighthouse + cottage  3 sun, sky, sea
values: sky light-mid, sea mid-dark, light path lightest, island + lighthouse dark, keeper darkest
light: sun on the horizon, left; warm rims on every edge facing it, cool violet shadows; accents = sun, lamp, window
composition: sun + light path on the left third, keeper inside the path, lighthouse on the right third, horizon 0.55
depth: far headlands paler and hazier; detail (rims, ripples, reflections) concentrated around the keeper
layers (back -> front): sky, stars, halo, sun, clouds/*, headland/*, sea, waves, path, island, cottage, lighthouse,
  lamp, beam, gulls, keeper/* (boat, body, rim, oar, reflection), ripples, splash
"""
FOCUS = "keeper"

W, H = 1200, 800
HZ = 440                                     # horizon y
SUN = (400, 430)                             # sun centre, sitting on the horizon
WARM, PALE, INK = "#f6a77a", "#fff1c9", "#0d0a16"
rng = np.random.default_rng(SEED if "SEED" in globals() else 0)
c = Canvas(W, H)

# ---- sky: smooth dusk gradient + a few soft flat bands + first stars
c.layer(name="sky")
c.fill(gradient([(0, "#1d1b3d"), (.35, "#3c2f63"), (.62, "#7a5486"), (.84, "#c77a8a"), (1, "#f3a98a")]),
       mask=zone(0, 0, 1, HZ / H))
c.layer(name="afterglow", blend="screen")                     # warm air low over the sun
c.fill(gradient([(0, "#c8704f"), (.5, "#3a1c26"), (1, "#000000")], kind="radial",
                center=(SUN[0] / W, SUN[1] / H), radius=.6), mask=zone(0, 0, 1, HZ / H))
c.layer(name="stars")
for _ in range(45):
    x, y = rng.uniform(0, W), rng.uniform(0, 210) ** 1.0
    c.circle(x, y, rng.uniform(.6, 1.6), fill="#fff4e0", opacity=float(rng.uniform(.25, .8) * (1 - y / 260)))

# ---- sun: flat halo rings (stepped glow), pale disc with a warm lower half, cut by the horizon
c.layer(name="halo")
for r, o in [(150, .05), (108, .07), (78, .11)]:
    c.circle(*SUN, r, fill="#ffc29a", opacity=o)
c.layer(name="sun")
disc_ = circle(*SUN, 52) & rect(0, 0, W, HZ)
c.draw(disc_, fill=gradient([(0, PALE), (.5, "#ffeab8"), (1, "#ffc48e")]).within(rect(0, SUN[1] - 52, W, HZ - SUN[1] + 52)))

# ---- clouds: lobed flat shapes, cool body, warm sun-lit undersides, smaller and paler toward the horizon
def mix_col(a, b, t):
    a, b = np.array(rgba(np.array(color(a)))[..., :3]).ravel(), np.array(rgba(np.array(color(b)))[..., :3]).ravel()
    return tuple(float(v) for v in a * (1 - t) + b * t)


def cloud(x, y, w, h, lobes, body, lit, name):
    base = rect(x + w * .08, y - h * .42, w * .84, h * .42 + 1, r=h * .21)
    tops = [ellipse(x + w * fx, y - h * .45, w * fw, h * fh) for fx, fw, fh in lobes]
    shape = union(base, *tops) & rect(x - w, y - 3 * h, 3 * w, 3 * h)          # flat bottom at y
    c.layer(name=f"clouds/{name}")
    top = y - 1.6 * h
    c.draw(shape, fill=gradient([(0, body), (.55, body), (1, mix_col(body, lit, .45))]).within(rect(x, top, w, y - top)))
    c.draw(shape - shape.move(0, -max(2.5, h * .1)), fill=lit)                              # lit underside
    c.draw(shape - shape.move(-h * .12, h * .12), fill="#ffd9b8", opacity=.55)    # rim facing the sun
    return shape


cloud(70, 262, 320, 52, [(.22, .15, .6), (.42, .17, 1.1), (.64, .15, .8), (.82, .1, .45)], "#5d4479", "#e39a8c", "a")
cloud(560, 312, 290, 40, [(.25, .13, .7), (.48, .17, 1.0), (.72, .12, .6)], "#6e5185", "#ec9f8e", "b")
cloud(980, 140, 200, 34, [(.25, .14, .6), (.5, .18, .95), (.78, .1, .45)], "#4d3a68", "#d98f88", "c")
cloud(250, 352, 140, 14, [(.5, .3, .5)], "#8a5f8a", "#f5ae92", "d")
cloud(700, 380, 220, 10, [(.4, .25, .5)], "#9a6a8e", "#f2b096", "e")

# ---- far headlands on the left horizon: paler with distance
c.layer(name="headland/far")
c.path(f"M -10 {HZ} C 60 {HZ-16} 130 {HZ-22} 200 {HZ-12} C 240 {HZ-6} 270 {HZ-4} 300 {HZ} Z", fill="#9c6b8c")
c.layer(name="headland/near")
c.path(f"M -10 {HZ} C 30 {HZ-10} 80 {HZ-12} 140 {HZ-4} L 160 {HZ} Z", fill="#7d5578")
c.line(-10, HZ - 1, 300, HZ - 1, stroke=WARM, width=1, opacity=.4)

# ---- sea: flat bands that tighten toward the horizon, sun-tinted near it
c.layer(name="sea")
c.fill(gradient([(0, "#6b4a7a"), (.12, "#4a3866"), (.5, "#2c2549"), (1, "#16132a")]), mask=zone(0, HZ / H, 1, 1))
c.line(0, HZ + .5, W, HZ + .5, stroke="#f0a88c", width=1.2, opacity=.6)
c.layer(name="waves")                                                            # wave dashes, perspective spaced
for i in range(170):
    t = (i / 170) ** 1.7
    y = HZ + 3 + t * (H - HZ - 3)
    w = (14 + t * 150) * rng.uniform(.4, 1)
    x = rng.uniform(-50, W)
    c.rect(x, y, w, .8 + t * 3.2, r=2, fill="#8a6aa0" if rng.random() < .55 else "#0f0c1e",
           opacity=float(.25 + .3 * rng.random()))
c.layer(name="path", blend="screen", opacity=.6)                                  # light cone under the sun
c.draw(polygon([(SUN[0] - 50, HZ), (SUN[0] + 50, HZ), (SUN[0] + 230, H), (SUN[0] - 210, H)]),
       fill=gradient([(0, "#ffb98e"), (.6, "#7a3f55"), (1, "#2a1530")]).within((0, HZ / H, 1, 1)), feather=45)
c.layer(name="pool", blend="screen")
c.draw(ellipse(SUN[0] + 20, 590, 165, 120), fill="#f0957a", feather=95)
c.draw(ellipse(SUN[0] + 15, 575, 95, 70), fill="#ffc29a", feather=60, opacity=.8)
c.layer(name="glitter")                                                           # broken sun reflection
for i in range(320):
    t = (i / 240) ** 1.5
    y = HZ + 2 + t * (H - HZ - 2)
    half = 40 + t * 200
    x = SUN[0] + rng.normal(0, .42) * half
    k = float(np.clip(1 - abs(x - SUN[0]) / half, 0, 1))
    w = (5 + t * 60) * rng.uniform(.4, 1) * (.4 + .6 * k)
    col = PALE if k > .6 else "#ffc79a" if k > .3 else "#c77d86"
    c.rect(x - w / 2, y, w, 1 + t * 3.5, r=2, fill=col, opacity=float(.25 + .75 * k) * (1 - .45 * t))

# ---- island, cottage and lighthouse: flat two-tone with rims toward the sun
c.layer(name="island")
rock = union(svg(f"M 770 {HZ+4} C 820 {HZ-18} 900 {HZ-30} 960 {HZ-28} C 1010 {HZ-26} 1050 {HZ-12} 1080 {HZ+4} Z"),
             svg(f"M 1050 {HZ+4} C 1075 {HZ-10} 1110 {HZ-10} 1130 {HZ+4} Z"),
             svg(f"M 740 {HZ+4} C 750 {HZ-5} 772 {HZ-6} 784 {HZ+4} Z"))
c.draw(rock, fill="#251c38")
c.draw(rock & rect(0, 0, 905, H), fill="#33264a")                                # sun-side facet
c.draw(rock - rock.move(3, 2), fill=WARM, opacity=.7)
c.layer(name="cottage")
c.rect(960, HZ - 52, 50, 50, fill="#2b2140")                                     # walls sink into the rock
c.polygon([(954, HZ - 52), (985, HZ - 72), (1016, HZ - 52)], fill="#1f1830")
c.line(954, HZ - 52, 985, HZ - 72, stroke=WARM, width=1.5, opacity=.7)
c["cottage"].below("island")                                                       # the rock hides its base
c.layer(name="cottage/window")
c.rect(972, HZ - 44, 9, 9, fill="#ffd38a")                                         # lit window: someone waits
c.cur.apply(glow(8, .6, "#ffc870"))
c.layer(name="lighthouse")
x0, x1, top, base = 896, 936, 190, HZ - 26
tower = polygon([(x0, base), (x0 + 7, top), (x1 - 7, top), (x1, base)])
c.draw(tower, fill="#3a2d52")
c.draw(tower & rect(0, 0, (x0 + x1) / 2 - 2, H), fill="#5a4470")                # lit left half
for yb in (250, 320):
    c.draw(tower & rect(0, yb, W, 22), fill="#6e2f48")
    c.draw(tower & rect(0, yb, (x0 + x1) / 2 - 2, 22), fill="#9a4658")
c.draw(tower - tower.move(2, 0), fill=WARM, opacity=.6)                          # warm left edge
c.rect(x0 - 2, top - 8, x1 - x0 + 4, 8, fill="#251c38")                           # gallery
for xr in range(int(x0), int(x1) + 1, 6):
    c.line(xr, top - 18, xr, top - 8, stroke="#251c38", width=1.2)
c.line(x0 - 2, top - 18, x1 + 2, top - 18, stroke="#251c38", width=1.5)
c.layer(name="lamp")
c.rect(x0 + 8, top - 38, x1 - x0 - 16, 22, fill="#ffe7a8")
c.polygon([(x0 + 3, top - 38), (x0 + 20, top - 58), (x1 - 3, top - 38)], fill="#1f1830")
c.cur.apply(glow(30, 1, "#ffcf7a"))
c.layer(name="gulls")
for gx, gy, gs in [(820, 168, 8), (845, 182, 5), (1010, 250, 4)]:
    c.path(f"M {gx-gs} {gy} Q {gx-gs/2} {gy-gs*.7} {gx} {gy} Q {gx+gs/2} {gy-gs*.7} {gx+gs} {gy}",
           fill=None, stroke="#2b2140", width=1.6)

# ---- the keeper: darkest shape on the brightest water, warm rim, anchored by reflection and ripples
bx, by = 420, 650                                                                  # boat centre, waterline
c.layer(name="keeper/boat")
hull = svg(f"M {bx-105} {by-26} C {bx-40} {by-18} {bx+60} {by-18} {bx+112} {by-30} "
           f"C {bx+92} {by-4} {bx+60} {by+4} {bx+5} {by+4} C {bx-55} {by+4} {bx-88} {by-4} {bx-105} {by-26} Z")
c.draw(hull, fill=INK)
c.draw(hull & svg(f"M {bx-110} {by-28} C {bx-40} {by-20} {bx+60} {by-20} {bx+115} {by-32} L {bx+115} {by-16} "
                  f"C {bx+60} {by-8} {bx-40} {by-8} {bx-110} {by-16} Z"), fill="#231a33")          # top strake
c.layer(name="keeper/body")
c.draw(svg(f"M {bx-4} {by-20} C {bx-8} {by-68} {bx+6} {by-102} {bx+30} {by-110} "
           f"L {bx+40} {by-96} C {bx+26} {by-78} {bx+28} {by-48} {bx+36} {by-20} Z"), fill=INK)   # coat
c.circle(bx + 37, by - 122, 12.5, fill=INK)                                       # head
c.ellipse(bx + 37, by - 132, 20, 4.5, fill=INK)                                   # cap brim
c.rect(bx + 27, by - 142, 20, 10, r=4, fill=INK)                                   # cap crown
c.line([(bx + 30, by - 96), (bx + 4, by - 72), (bx - 12, by - 66)], stroke=INK, width=8)   # arms to the oar
c.layer(name="keeper/oar")
c.line(bx - 8, by - 70, bx - 150, by + 20, stroke=INK, width=5)
c.draw(ellipse(bx - 152, by + 19, 14, 4).rotate(-30), fill=INK)                   # blade entering the water
rim = c["keeper/body"].duplicate("keeper/rim")                                     # rim light from the sun (left)
rim.fill("#ffb27a", mask=rim.alpha > 0)
rim.move(-3, -2)
rim.below("keeper/body")
gun = c.layer(name="keeper/gunwale")
c.draw(svg(f"M {bx-105} {by-26} C {bx-40} {by-18} {bx+60} {by-18} {bx+112} {by-30}"), fill=None,
       stroke="#ffb27a", width=2)
c.reflect("keeper", by + 2, ripple=7, dim=.75, fade=.8, breakup=.45, bands=4, seed=2)
c.reflect(["island", "cottage", "lighthouse", "lamp"], HZ + 4, ripple=3, dim=.7, fade=.7, bands=3, seed=3)

c.layer(name="ripples")                                                            # hull and oar disturb the light
for k, (rx, ry, o) in enumerate([(122, 7, .45), (150, 12, .22)]):
    c.draw(ellipse(bx + 4, by + 3, rx, ry).outline(1.6) & rect(0, by + 3, W, 40), fill=PALE, opacity=o)
c.line(bx - 95, by + 3, bx + 100, by + 3, stroke=PALE, width=1.6, opacity=.75)    # bright waterline
c.layer(name="splash")
for r, o in [(9, .7), (20, .3)]:
    c.draw(ellipse(bx - 152, by + 22, r * 1.6, r * .32).outline(1.3), fill=PALE, opacity=o)
for dx, dy, r in [(-6, -10, 1.8), (4, -16, 1.3), (12, -8, 1.5), (-16, -5, 1.1)]:
    c.circle(bx - 152 + dx, by + 20 + dy, r, fill=PALE, opacity=.85)

c.apply(vignette(.3), grain(.018, seed=1))
c.save(__file__[:-3] + ".png")
