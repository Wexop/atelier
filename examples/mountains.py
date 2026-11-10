# Eroded mountain range at golden hour: physical sky, raymarched clouds, aerial perspective, a lake.
import numpy as np
from atelier import *
from atelier.nature import *

n = 512
h = heightfield(n, "alpine", seed=SEED + 4, scale=2.2)
zz, xx = np.mgrid[0:n, 0:n] / (n - 1)
rise = smoothstep(0.22, 0.6, zz)                                   # range rises with distance
sides = smoothstep(0.18, 0.5, np.abs(xx - 0.5)) * (1 - rise) * 0.35  # foothills framing the lake
h = h * (0.12 + 0.88 * rise) + sides + rise * 0.12
h = thermal(h, 20)
h, flow = erode(h, 1.2, seed=SEED, flow=True)
h = (h - h.min()) / (h.max() - h.min())

c = Canvas(1500, 1000)
v = c.landscape(h, flow=flow, size=16000, height=3400, water=0.08, cam=(8000, 330, 200), fov=48,
                horizon=0.56, sun_elev=5, sun_az=-125, clouds=0.22, haze=1.4, snow=0.6, cloud_alt=(2600, 3800), waves=0.02, seed=SEED)
c.save(__file__[:-3] + ".png")
