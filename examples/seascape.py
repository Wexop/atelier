# Open sea at sunset: physical sky, lit cumulus, FFT-like waves with Fresnel reflections, glitter and foam.
from atelier import *
from atelier.nature import *

c = Canvas(1500, 1000)
c.sea(sun_elev=7, sun_az=-18, horizon=0.5, fov=46, clouds=0.2, waves=0.11, wave_len=40, wind=-20,
      foam=0.6, cam_height=6, haze=1.3, cloud_alt=(2400, 3600), seed=SEED + 4)
c.apply(bloom(0.92, 10, 0.2), vignette(0.3), grain(0.01, seed=SEED))
c.save(__file__[:-3] + ".png")
