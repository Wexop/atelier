# Pixel art: 320x180, 16-colour palette, ordered dithering, exported 4x without smoothing.
import numpy as np
from atelier import *

PAL = ["#14102a", "#241a45", "#3d2a63", "#6b3a7a", "#a54d79", "#e0716a", "#f6a46b", "#ffd88a",
       "#fff4c8", "#2b2148", "#433467", "#1a2a33", "#21403a", "#2f6050", "#5a3524", "#ffe08a"]
c = Canvas(320, 180, pixel=True)
sky = c.cur
sky.fill(gradient([(0, PAL[0]), (0.25, PAL[2]), (0.45, PAL[4]), (0.6, PAL[6]), (0.68, PAL[7])]))
sky.circle(196, 112, 22, fill=PAL[8])                                    # setting sun
sky.apply(pixelate(1, PAL[:9], dither="bayer", strength=0.35))
c.layer().scatter("square", 40, region=top(0.3), size=(1, 1), color=PAL[8], seed=SEED)  # stars

far = fbm(3, 3, seed=SEED + 2).ridge_line(0.52, 0.22)
c.layer().fill_below(far, PAL[10])
near = ridged(4, 4, seed=SEED + 4).ridge_line(0.64, 0.16)
c.layer().fill_below(near, PAL[9])
land = c.layer()
land.rect(0, 120, 320, 60, fill=PAL[9])                                   # shore line base
for x, h in [(18, 34), (34, 46), (52, 30), (262, 40), (280, 52), (300, 36)]:   # pines
    for k in range(4):
        w = h * (0.45 - k * 0.08)
        land.polygon([(x - w, 124 - k * h / 5), (x + w, 124 - k * h / 5), (x, 124 - k * h / 5 - h / 2.6)], fill=PAL[11])
    land.rect(x - 1, 118, 3, 7, fill=PAL[14])
land.polygon([(226, 112), (240, 100), (254, 112)], fill=PAL[14])         # cabin
land.rect(229, 112, 22, 12, fill=PAL[14])
land.rect(236, 116, 5, 4, fill=PAL[15])                                   # lit window
land.rect(246, 100, 3, 6, fill=PAL[14])                                   # chimney + smoke
for k, (dx, dy) in enumerate([(0, 0), (2, -4), (5, -8), (9, -11), (14, -13)]):
    land.rect(247 + dx, 96 + dy, 2 + k // 2, 2, fill=PAL[3 + (k > 2)])

# lake: mirrored scene with ripples, darkened
img = c.flatten()
mir = img[60:124][::-1].copy()
ys = np.arange(len(mir))
for i in ys[::2]:                                                         # 1px ripples on every other row
    mir[i] = np.roll(mir[i], 1 if (i // 2) % 2 else -1, axis=0)
depth = np.linspace(0.75, 0.12, len(mir))[:, None, None] ** 1.3          # Fresnel: weaker reflection up close
mir[..., :3] = mir[..., :3] * depth + np.array([0.05, 0.07, 0.16]) * (1 - depth)
lake = np.zeros_like(img)
lake[124:124 + len(mir)] = mir[:180 - 124]
lake[124:, :, 3] = 1
water = c.layer()
water.rgba = lake
water.apply(pixelate(1, [PAL[i] for i in (0, 1, 2, 3, 4, 9, 10, 11)], dither="bayer", strength=0.3))  # water: cool sub-palette
rng = np.random.default_rng(SEED)
for y in range(127, 176, 3):                                              # sparse sun glints, wider near shore
    for _ in range(2):
        half = max(1, int((180 - y) / 8 * rng.uniform(0.4, 1.0)))
        x = 196 + int(rng.normal(0, 3 + (y - 124) / 6))
        water.line(x - half, y, x + half, y, stroke=PAL[7] if y < 150 else PAL[6], width=1)
water.line(234, 127, 242, 127, stroke=PAL[15], width=1)                   # window reflection
c.save(__file__[:-3] + ".png", scale=4)
