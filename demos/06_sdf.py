# Step 6 demo: photoreal still life (glass, ceramic, fruit, wood, window light, DOF).
from atelier import *

s = Scene()
s.add(plane(), Mat(wood(rings=18, seed=2).ramp(["#3a2312", "#6b4426", "#8a5a33"]), rough=0.45, proj="xz", tex_scale=0.35))
s.add(plane(-1.2, (0, 0, 1)), Mat("#cfc6b8", rough=0.9))                                   # back wall
vase = lathe([(0.20, 0), (0.30, 0.25), (0.22, 0.55), (0.11, 0.72), (0.13, 0.8)], thickness=0.03)
s.add(vase.move(-0.55, 0, -0.35), Mat("#2f5d7c", rough=0.12))                              # glazed ceramic
glass = lathe([(0.0, 0.0), (0.16, 0.01), (0.17, 0.05), (0.15, 0.3), (0.18, 0.5)], thickness=0.012)
s.add(glass.move(0.25, 0, -0.15), Mat("#e6f0ec", glass=1, rough=0.02, ior=1.5))
apple = sphere(0.15).scale((1, 0.88, 1)).sub(sphere(0.045).move(0, 0.145, 0), k=0.05)
apple = (apple | capsule((0, 0.1, 0), (0.015, 0.19, 0), 0.008)).move(0.62, 0.13, 0.22)
s.add(apple, Mat("#b3261e", rough=0.35, sss=0.4))
orange = sphere(0.14).displace(noise3(60, 2, seed=3), 0.006).move(-0.1, 0.14, 0.32)
s.add(orange, Mat("#e8801a", rough=0.55, sss=0.5))
s.add(sphere(0.07).move(0.02, 0.07, 0.48), Mat("#6e8c2a", rough=0.4, sss=0.5))            # small lime
s.area(pos=(-2.2, 2.0, 1.5), size=(1.4, 1.4), normal=(1, -0.6, -0.6), color="#fff1dc", power=26, n=(3, 2))
s.sky("#8aa0b8", "#c9c2b6", "#3a2c22", power=0.35)
s.camera((0.35, 0.75, 2.1), (0.05, 0.22, -0.1), fov=30)

c = Canvas(1200, 800)
r = c.render(s, aa=2, exposure=0.3)
c.apply(depth_of_field(r.depth, focus=r.focus_at(700, 470, c.s), aperture=5), bloom(0.85, 8, 0.3),
        vignette(0.3), grain(0.015))
c.save("demos/out/06_sdf.png")
c.preview("demos/out/06_sdf_prev.jpg")
