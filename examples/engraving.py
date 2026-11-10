# Copper-plate engraving of a 3D scene: hatches that follow the surfaces in perspective, ink contours.
from atelier import *
from atelier.stylize import render_hatch, outlines

s = Scene()
s.add(plane(0.0), Mat("#b0a898"))                                                         # ground
s.add(box(3.2, 0.12, 2.0).move(0, 0.06, -0.4) | box(2.8, 0.12, 1.6).move(0, 0.18, -0.4), Mat("#d8d0c0"))  # steps
for x in (-1.1, -0.37, 0.37, 1.1):                                                       # colonnade
    shaft = cylinder(0.14, 1.9).move(x, 1.19, -0.9)
    capital = round_box(0.42, 0.1, 0.42, 0.02).move(x, 2.19, -0.9)
    s.add(shaft | capital | round_box(0.4, 0.1, 0.4, 0.02).move(x, 0.29, -0.9), Mat("#ddd5c4"))
s.add(box(2.7, 0.18, 0.5).move(0, 2.33, -0.9), Mat("#d8d0c0"))                           # architrave
s.add(cylinder(0.14, 0.7).rotate(z=90).move(0.7, 0.38, 0.65), Mat("#d8d0c0"))            # fallen drum
s.add(sphere(0.32).move(-0.75, 0.56, 0.45), Mat("#d8d0c0"))                              # stone sphere
s.add(box(0.5, 0.2, 0.5).move(-0.75, 0.34, 0.45), Mat("#d8d0c0"))                        # its plinth
for x, z, h in [(-3.3, -3.0, 2.8), (-2.7, -3.8, 2.2), (2.4, -3.2, 2.5), (2.9, -3.9, 2.0)]:                  # cypresses
    s.add(ellipsoid(0.24, h / 2, 0.24).displace(noise3(5, 2, seed=int(x * 10)), 0.04).move(x, h / 2, z), Mat("#4a4a36"))
s.add(sphere(1).scale((9, 1.6, 3)).move(-4, -0.4, -14) | sphere(1).scale((8, 2.2, 3)).move(6, -0.6, -16),
      Mat("#a8a090"))                                                                      # distant hills
s.sun((-1.4, 1.2, 0.9), power=3.0, soft=0.02)
s.sky("#9ab", "#dde", "#887", power=0.25)
s.camera((1.6, 1.5, 5.0), (0, 1.05, -0.4), fov=44)

PAPER, INK = "#f1e9d6", "#2a1d14"
c = Canvas(1100, 850, bg=PAPER)
sky_tone = gradient(["#9a9a9a", "#e8e8e8"]) + smoothstep(0.45, 0.6, clouds(1.6, 0.45, seed=SEED + 2).scale(2.5, 0.7))
c.hatch(None, angle=0, spacing=4.5, width=1.0, color=INK, tone=sky_tone.clip(), wobble=0.25)  # engraved sky
L = c.layer()
r = L.render(s, alpha=True)
L.clear()
L.fill(PAPER, mask=r.alpha)
L.apply(render_hatch(r, s, spacing=4.5, width=1.15, layers=3, color=INK, lo=0.0, wobble=0.2),
        outlines(r, width=2.0, inner=0.9, color=INK, crease=0.5, depth=0.05, rough=0.5))
c.layer(blend="overlay", opacity=0.5).fill(paper())                                   # paper tooth
c.layer()
c.draw(rect(0, 0, 1100, 850) - rect(45, 45, 1010, 690), fill=PAPER)                    # plate margin
c.rect(45, 45, 1010, 690, fill=None, stroke=INK, width=2.5)
c.rect(36, 36, 1028, 708, fill=None, stroke=INK, width=1)
c.text("RUINS OF A DORIC TEMPLE", 550, 785, size=30, fill=INK, font="serif", align="center", spacing=0.18)
c.text("Plate IV  -  drawn from a 3D model", 550, 818, size=15, fill=INK, font="serif", italic=True, align="center")
c.save(__file__[:-3] + ".png")
