# Eroded alpine range at golden hour: wide valley lake, side light, physical sky and clouds, aerial perspective.
import numpy as np
from atelier import *
from atelier.nature import *

n = 1024
h = heightfield(n, "alpine", seed=SEED + 4, scale=3.0, warp=0.25, octaves=10)
zz, xx = np.mgrid[0:n, 0:n] / (n - 1)
rise = smoothstep(0.2, 0.7, zz)                                    # range rises with distance
sides = smoothstep(0.15, 0.48, np.abs(xx - 0.5)) * (1 - rise) * 0.3
h = h * (0.08 + 0.92 * rise) + sides + rise * 0.1
h = thermal(h, 30)
h, flow = erode(h, 2.5, seed=SEED, flow=True)
h = (h - h.min()) / (h.max() - h.min())

c = Canvas(1500, 1000)
v = c.landscape(h, flow=flow, size=22000, height=3600, water=0.07, cam=(11000, 300, 3200), fov=50,
              horizon=0.5, sun_elev=7, sun_az=-110, clouds=0.2, haze=1.6, snow=0.62, cloud_alt=(2800, 4200),
              waves=0.015, trees=0.7, aa=2, seed=SEED)
c.apply(bloom(0.9, 10, 0.15), grain(0.008, seed=SEED))
c.save(__file__[:-3] + ".png")
