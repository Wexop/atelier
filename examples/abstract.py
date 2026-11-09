# Abstract generative: silk ribbons from a twice domain-warped field, flowing along an S-curve.
import numpy as np
from atelier import *

c = Canvas(1600, 1000, bg="#060914")
f = domain_warp(domain_warp(fbm(1.3, 4, seed=SEED), 0.45, 1.2, seed=SEED + 1), 0.1, 2.0, seed=SEED + 2)
hue = fbm(0.8, 3, seed=SEED + 5)                                     # slow colour drift
spine = Field(lambda x, y: np.exp(-((y - 0.52 - 0.22 * np.sin(x * 5.5 + 0.6)) / 0.32) ** 2))  # S-curve band
silk = (0.5 + 0.5 * np.sin(f * np.pi * 6)) ** 2                       # ribbon folds
pal = ["#0b1e3f", "#0f5e6e", "#2fa39a", "#e8c170", "#f27059", "#fde9d0"]
col = mix(hue.ramp(pal), "#fde9d0", silk * 0.2)
c.layer().fill(shade(f, color=col, strength=0.55, light=(110, 45), ambient=0.3, spec=0.15,
                     shininess=25), mask=spine)
threads = (1 - np.abs(np.sin(f * np.pi * 16))) ** 24                # fine silk threads
c.layer(blend="screen", opacity=0.55).fill("#fff2d8", mask=threads * spine ** 2)
c.apply(bloom(0.75, 16, 0.5), curves([(0.1, 0.06), (0.6, 0.64)]), vignette(0.4), grain(0.02, seed=SEED))
c.save(__file__[:-3] + ".png")
