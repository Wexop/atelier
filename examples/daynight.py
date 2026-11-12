# Day-night loop over flat hills: sun and moon ride one wheel, sky / hills / clouds follow colour keyframes.
# atelier animate examples/daynight.py --frames 120 --fps 24   (T: 0 dawn, .25 noon, .5 dusk, .75 midnight)
import math
import numpy as np
from atelier import *

W, H = 480, 300
HZ = 205                                                   # horizon (px)
c = Canvas(W, H)
P = [0, 0.25, 0.5, 0.75, 1]                                # dawn, noon, dusk, night, dawn


def phase(*cols):                                          # colour at T from one colour per phase (loops)
    return key(T, list(zip(P, cols + (cols[0],))), ease="sine")


# sky
sky_top = phase("#5a6cc8", "#2f8ff0", "#4a3a96", "#0a0f2e")
sky_mid = phase("#f4a0bc", "#7fc8f8", "#f2698c", "#1a2252")
sky_low = phase("#ffd3a0", "#d2f0ff", "#ffb35c", "#2b3570")
c.layer(name="sky").fill(gradient([(0, sky_top), (0.5, sky_mid), (HZ / H, sky_low), (1, sky_low)]))
night = key(T, [(0, 0), (0.45, 0), (0.62, 1), (0.88, 1), (1, 0)], ease="sine")
c.layer(name="stars", opacity=night).fill("#ffffff", mask=stars(density=30, size=0.9, seed=4) * top(0.62, soft=0.12))

# sun and moon: opposite ends of one wheel turning once per loop
for name, off, col, glow in (("sun", 0.0, phase("#ffb46a", "#fff2c2", "#ff7a3c", "#ff7a3c"), "#ffcf7a"),
                             ("moon", 0.5, "#eef0ff", "#9fb4ff")):
    a = math.pi - 2 * math.pi * (T + off)
    x, y = W / 2 + 205 * math.cos(a), HZ + 10 - 165 * math.sin(a)
    if y < HZ + 40:
        c.layer(name=f"{name}/halo", blend="screen", opacity=0.75).fill(glow, mask=disc(x / W, y / H, 0.06, soft=0.18))
        c.layer(name=name).circle(x, y, 22 if name == "sun" else 15, fill=col)
        c[name].apply(blur(0.8))

# clouds drift across once per loop (the wrap happens off-screen)
cloud_col = phase("#ffe0ea", "#ffffff", "#ffc0a8", "#3a4478")
for i, (y, sc, sp) in enumerate([(70, 1.0, 1), (115, 0.7, 2), (48, 0.55, 1)]):
    x = (i * 0.37 + T * sp) % 1 * (W + 220) - 110
    blob = union(ellipse(x, y, 46 * sc, 16 * sc), ellipse(x - 26 * sc, y + 4 * sc, 28 * sc, 11 * sc),
                 ellipse(x + 24 * sc, y - 8 * sc, 30 * sc, 15 * sc), ellipse(x + 44 * sc, y + 5 * sc, 22 * sc, 9 * sc))
    c.layer(name=f"cloud/{i}", opacity=0.9).draw(blob, fill=cloud_col, feather=2)

# hills, back to front, each tinted by the time of day, warm rim at dawn and dusk
hills = [(HZ - 22, 0.05, 2.1, 0.3, ("#a585c4", "#86c98f", "#b8578a", "#27315c")),
         (HZ - 4, 0.045, 3.3, 1.7, ("#7d62a6", "#58ab6e", "#82406f", "#1c2448")),
         (HZ + 30, 0.06, 2.4, 4.1, ("#523f7c", "#33855a", "#4e2a5c", "#11172f"))]
rim = key(T, [(0, 0.8), (0.12, 0), (0.4, 0), (0.5, 0.9), (0.6, 0), (0.9, 0), (1, 0.8)], ease="sine")
for i, (y, amp, f, ph, cols) in enumerate(hills):
    ridge = curve(lambda u, y=y, amp=amp, f=f, ph=ph: y / H - amp * (0.6 * np.sin(u * f * 3 + ph) + 0.4 * np.sin(u * f * 7.3 + ph * 2)))
    c.layer(name=f"hill/{i}").fill_below(ridge, phase(*cols))
    if rim > 0.01:
        c.layer(name=f"hill/{i}/rim", blend="screen", opacity=rim * 0.6).fill(
            phase("#ff9fb0", "#ffffff", "#ffa060", "#000000"), mask=ridge.below(soft=0.004) - ridge.below(soft=0.004).move(0, 0.008))

c.apply(bloom(0.75, 14, 0.35), vignette(0.2))
c.save(__file__[:-3] + ".png")
