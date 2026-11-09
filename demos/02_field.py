# Step 2 demo: a night landscape from fields only (close to the target API example).
from atelier import *

c = Canvas(1600, 1000)
c.fill(gradient([(0, "#070b1f"), (0.5, "#2b2350"), (0.72, "#8e4162"), (0.85, "#f4a259")]))
cloud = clouds(2.5, 0.45, seed=4).scale(3, 0.7) * band(0.40, 0.56, 0.06)
c.layer(blend="screen").fill("white", mask=stars(70, seed=3) * top(0.65, soft=0.25) * (1 - cloud))
moon = c.layer(blend="screen")
moon.fill("#ffd9a8", mask=disc(0.72, 0.22, 0.2, soft=0.3) * 0.18)        # halo
moon.fill("#fff6e0", mask=disc(0.72, 0.22, 0.04))
c.layer(blend="multiply", opacity=0.3).fill("#a08cb0", mask=disc(0.712, 0.212, 0.012, soft=.01) + disc(0.73, 0.235, 0.007, soft=.006))
# thin stratus streaks lit from below
c.layer(blend="screen", opacity=0.5).fill(gradient(["#3a2d5c", "#e08a7a"]).scale(1, 0.3), mask=cloud)

ranges = [  # baseline, amplitude, colours (atmospheric perspective: far = lighter, bluer), seed
    (0.60, 0.17, ["#5d4f86", "#7a6aa0"], 1),
    (0.70, 0.13, ["#352c55", "#4a3d6b"], 2),
    (0.82, 0.10, ["#15122a", "#241e40"], 5),
]
for y, amp, cols, sd in ranges:
    ridge = ridged(2.2, 6, seed=sd).ridge_line(y=y, amp=amp)
    relief = ridged(2.2, 6, seed=sd).scale(1, 1.6) + ridged(9, 4, seed=sd + 10).scale(0.5, 3) * 0.12
    rock = shade(relief, color=fbm(4, seed=sd).ramp(cols), strength=1.2, light=(20, 35), ambient=0.25)
    c.layer().fill_below(ridge, rock)
    c.layer(blend="screen", opacity=0.45).fill("#9a6f9e", mask=ridge.below() * (1 - Y.remap(y - 0.1, y + 0.12)))  # valley haze
c.layer().fill("#07060d", mask=fbm(3, seed=9).ridge_line(y=0.94, amp=0.05).below())
c.save("demos/out/02_field.png")
c.preview("demos/out/02_field_prev.jpg")
