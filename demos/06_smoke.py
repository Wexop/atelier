import time
from atelier import *
s = Scene()
s.add(plane(), Mat("#b8b0a0", rough=0.8))
s.add(sphere(0.5).move(-1.1, 0.5, 0), Mat("#c0392b", rough=0.25))
s.add(round_box(0.9, 0.9, 0.9, 0.08).rotate(y=30).move(0, 0.45, -0.3), Mat("#2e86c1", rough=0.6))
s.add(torus(0.35, 0.12).rotate(x=70).move(1.1, 0.48, 0.1), Mat("#d4af37", rough=0.2, metal=1))
s.add(sphere(0.35).move(0.2, 0.35, 0.9), Mat("#ffffff", glass=1, rough=0.02, ior=1.5))
s.sun((-1, 1.6, 1.2), power=3)
s.camera((0, 1.5, 4.2), (0, 0.4, 0), fov=38)
c = Canvas(640, 400)
t = time.time(); r = c.render(s); print("render", round(time.time() - t, 2), "s")
c.preview("demos/out/06_smoke.jpg", 640)
