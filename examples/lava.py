# Lava-lamp loop: warm gooey blobs rise, merge and split over a violet glow.
# atelier animate examples/lava.py --frames 96 --fps 24   (every motion uses whole cycles of T: seamless loop)
import cv2
import numpy as np
from atelier import *

W, H = 420, 420
c = Canvas(W, H)
s = c.s
yy, xx = np.mgrid[0:c.H, 0:c.W].astype(np.float32) / s + 0.5 / s     # canvas-px grid
TAU = 2 * np.pi

# background: deep violet top, hot pink low down, a breathing warm glow at the bottom
c.layer(name="bg").fill(gradient([(0, "#1b0b3a"), (0.55, "#5a1a8c"), (1, "#ff3d7f")]))
heat = wave(T, cycles=2, lo=0.7, hi=1.0)
c.layer(name="bg/heat", blend="screen").fill("#ff8a3d", mask=disc(0.5, 1.08, 0.5, soft=0.35) * heat)

# blobs: rise with whole cycles per loop (wrap happens off-screen), sway and breathe
rng = np.random.default_rng(SEED + 5)
F = np.zeros_like(xx)
for i in range(8):
    k = 1 if i < 5 else 2                                            # small blobs travel twice per loop
    ph = rng.uniform(0, 1)
    r = rng.uniform(22, 38) / k ** 0.5
    y = H + 70 - ((ph + T * k) % 1) * (H + 140)
    x = rng.uniform(0.2, 0.8) * W + 34 * np.sin(TAU * (T * k + ph * 3))
    r = r * (1 + 0.12 * np.sin(TAU * (2 * T + ph)))
    F += r * r / ((xx - x) ** 2 + ((yy - y) * 0.85) ** 2 + 1)
for cx, cy, rr, ph in ((W * 0.5, H + 30, 100, 0.0), (W * 0.45, -40, 64, 0.4)):  # wax pooled at bottom and top
    wob = 1 + 0.08 * np.sin(TAU * (T + ph) + xx / 40)
    F += (rr * wob) ** 2 / ((xx - cx) ** 2 + ((yy - cy) * 1.4) ** 2 + 1)

body = smoothstep(0.85, 1.15, F)                                    # gooey merge: metaball threshold
# glossy wax: relief from the blurred shape -> normals -> soft light from the top left + a highlight
hgt = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 12 * s)
gy, gx = np.gradient(hgt * 40 * s)
nz = 1 / np.sqrt(gx * gx + gy * gy + 1)
lam = np.clip((-gx * -0.55 - gy * -0.6) * nz + 0.58 * nz, 0, 1)        # light from upper left, toward viewer
spec = np.clip(lam, 0, 1) ** 18
col = ramp(np.clip(hgt * 1.2 - 0.1, 0, 1), [(0, "#ff2a7f"), (0.5, "#ff3f6a"), (0.85, "#ff6450"), (1, "#ff8a4c")])
col = np.clip(col * (0.78 + 0.4 * lam[..., None]) + spec[..., None] * np.array([1.0, 0.9, 0.75]) * 0.7, 0, 1)
c.layer(name="blobs/glow", blend="screen", opacity=0.55).fill(col, mask=body)
c["blobs/glow"].apply(blur(16))
c.layer(name="blobs").fill(col, mask=body)
c.layer(name="blobs/rim", blend="screen", opacity=0.55).fill("#ffd6f0", mask=np.clip(body * (1 - body) * 4, 0, 1))

c.apply(bloom(0.8, 16, 0.35), vignette(0.25))
c.save(__file__[:-3] + ".png")
