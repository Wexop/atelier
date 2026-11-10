# Ligne-claire comic panel from a 3D scene: toon bands, ink outlines, halftone shadows, speech balloon.
from atelier import *
from atelier.stylize import toon, outlines, shadow_dots

s = Scene()
rock = sphere(1.0).scale((1.9, 0.55, 1.4)).displace(noise3(1.6, 2, seed=SEED + 1), 0.22)
rock = rock | sphere(0.8).scale((1.2, 0.6, 1.0)).move(1.5, -0.1, 0.6).displace(noise3(2, 2, seed=SEED + 2), 0.18)
isle = s.add(rock.move(0, -0.05, 0), Mat("#b9a58a", rough=0.9))                          # islet
sea = s.add(plane(0.0), Mat("#4f9bc9", rough=0.6))                                        # sea
tower = lathe([(0.42, 0.0), (0.36, 1.0), (0.3, 2.0), (0.27, 2.5)])
s.add(tower.move(-0.2, 0.3, -0.1), Mat("#f3eee2", rough=0.7))
for y0 in (0.55, 1.45):                                                                   # red bands
    s.add((tower.scale(1.015) & box(2, 0.42, 2).move(0, y0 + 0.21, 0)).move(-0.2, 0.3, -0.1), Mat("#d2412f"))
s.add(cylinder(0.42, 0.07).move(-0.2, 2.84, -0.1), Mat("#2f3a44"))                      # gallery
s.add(cylinder(0.2, 0.42).move(-0.2, 3.08, -0.1), Mat("#f6d55c", rough=0.4))             # lantern
s.add(cone(0.3, 0.32).move(-0.2, 3.29, -0.1), Mat("#d2412f"))                            # cap
house = box(0.9, 0.55, 0.7).move(0, 0.27, 0)
roof = extrude([(-0.55, 0), (0.55, 0), (0, 0.38)], 0.8).move(0, 0.55, 0)
s.add(house.rotate(y=-25).move(1.05, 0.3, 0.45), Mat("#efe3c8", rough=0.8))
s.add(roof.rotate(y=-25).move(1.05, 0.3, 0.45), Mat("#3f6f9a", rough=0.7))
s.add(box(0.16, 0.3, 0.02).rotate(y=-25).move(0.98, 0.45, 0.82), Mat("#6a4a32"))        # door
hull = ellipsoid(0.5, 0.2, 0.2).onion(0.02) - box(2, 1, 1).move(0, 0.5, 0)            # rowboat
s.add(hull.rotate(y=8).move(-2.3, 0.08, 1.4), Mat("#b8462f"))
s.add(extrude([(0, 0), (0.32, 0), (0, 0.6)], 0.02).rotate(y=8).move(-2.35, 0.1, 1.4), Mat("#f4f0e4"))  # sail
s.add(cylinder(0.012, 0.7).rotate(y=30).move(-2.35, 0.4, 1.4), Mat("#5a3f2a"))
s.sun((-1.2, 1.4, 1.0), power=3.2, soft=0.01)
s.sky("#8cc4e8", "#e9f2f5", "#4f9bc9", power=0.6)
s.camera((3.2, 1.9, 5.6), (0.25, 1.35, 0), fov=40)

c = Canvas(1200, 800)
c.fill(gradient(["#7cbde6", "#d6eef7"]))                                                 # flat comic sky
def cloud(x, y, k):                                                                      # puffy inked cloud
    puffs = [(-60, 0, 34), (-25, -22, 40), (20, -30, 46), (62, -8, 36), (20, 4, 38), (-30, 6, 30)]
    return union(*[circle(x + dx * k, y + dy * k, rr * k) for dx, dy, rr in puffs]) & rect(0, 0, 1200, y + 12 * k)
for x, y, k in [(860, 120, 1.25), (1110, 250, 0.8), (560, 300, 0.7)]:
    c.draw(cloud(x, y, k), fill="#ffffff", stroke="#16130f", width=2.5)
L = c.layer()
r = L.render(s, alpha=True)
L.apply(toon(r, bands=3, shadow=0.5, colors={sea: "#3f8fc4", isle: "#c9a878"}), shadow_dots(r, size=5, lo=0.3, color="#22303f", amount=0.8),
        outlines(r, width=2.6, inner=1.0, crease=0.6))
c.layer()
for i, (x, y) in enumerate([(120, 700), (1060, 720), (1040, 560), (180, 560), (760, 765)]):
    c.stroke([(x, y), (x + 22, y - 8), (x + 44, y), (x + 66, y - 8)], "ink", 4, "#16130f", seed=i)  # waves
for x, y in [(640, 120), (700, 95), (760, 140)]:                                          # gulls
    c.stroke([(x - 16, y - 6), (x - 7, y - 8), (x, y), (x + 7, y - 8), (x + 16, y - 6)], "ink", 3, "#16130f")
balloon = ellipse(300, 150, 230, 92) | polygon([(330, 225), (420, 300), (390, 222)])
c.draw(balloon, fill="white", stroke="#16130f", width=3)
c.text("THE LAMP IS LIT...", 300, 132, size=34, fill="#16130f", font="sans", weight="bold", align="center",
       valign="middle")
c.text("SOMEONE IS HOME!", 300, 176, size=34, fill="#16130f", font="sans", weight="bold", align="center",
       valign="middle")
c.rect(6, 6, 1188, 788, fill=None, stroke="#16130f", width=8)                           # panel border
c.save(__file__[:-3] + ".png")
