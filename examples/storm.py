# Night storm over a valley: glowing campfire with drifting smoke in the foreground, mist between the ridges,
# a distant lightning strike under a rain curtain.
import numpy as np
from atelier import *

W, H = 1400, 900
FIRE = (430, 742)
FOCUS = "fire"
c = Canvas(W, H)

# sky: storm cell darker on the right, faint moonlit gap on the left
c.layer(name="sky").fill(gradient([(0, "#0f1424"), (0.45, "#222b42"), (0.7, "#3a4560")]))
c.layer(name="clouds", opacity=0.6).fill(
    "#3c4662", mask=smoothstep(0.42, 0.75, domain_warp(fbm(2.5, seed=SEED), 0.15, seed=SEED + 1)) * top(0.62, soft=0.15))
c.layer(name="cell").fill("#151a2b", mask=smoothstep(0.45, 0.7, fbm(3, seed=SEED + 2)) * right(0.55, soft=0.25) * top(0.45, soft=0.2))

# distant strike, lighting the cloud base around it
c.layer(name="flash", blend="screen").fill("#6f86c8", mask=disc(0.79, 0.3, 0.18, soft=0.9))
c.layer(name="bolt", opacity=0.85).lightning(1120, 200, 1065, 540, branches=4, width=1.6, glow=0.8, seed=SEED + 3)
c["bolt"].apply(blur(0.8))                                             # far away: softer than the fire

# rain curtain under the cell
c.layer(name="rain").rain(900, angle=10, length=(10, 26), width=0.8, opacity=0.4, depth=2,
                          region=zone(0.6, 0.15, 1, 0.75, soft=0.06), seed=SEED + 4)

# ridges back to front, mist between them
ridges = [(560, 0.12, "#323c5a"), (625, 0.09, "#1c2335")]
for i, (y, amp, col) in enumerate(ridges):
    c.layer(name=f"ridge/{i}").fill_below(fbm(3 + i, seed=SEED + 10 + i).ridge_line(y / H, amp), col)
    c.layer(name=f"mist/{i}").mist(y - 90, y + 30, color="#8e9bb8", density=0.4 - i * 0.1, seed=SEED + 20 + i)

# pines on the middle ridge, hazed into the dark
c.layer(name="pines")
for x, h in [(820, 120), (870, 90), (905, 140), (960, 100), (1250, 130), (1300, 95), (90, 110), (140, 150)]:
    c.tree(x, 640 + (x % 37), h, kind="pine", season="winter", seed=x)
c["pines"].apply(gradient_map(["#0a0e18", "#1c2438"]))                 # night silhouettes

# foreground clearing
ground = curve(lambda x: 0.835 - 0.04 * np.sin(x * 5.5 + 0.6) - 0.03 * x)
c.layer(name="ground").fill_below(ground, gradient([(0, "#0e121d"), (1, "#030408")]))
c.layer(name="glow", blend="screen", opacity=0.75).fill(
    "#c4561c", mask=disc(FIRE[0] / W, (FIRE[1] + 15) / H, 0.09, soft=1) * ground.below())

# figures warming up, rim-lit by the fire
for i, (x, y, h, flip) in enumerate([(335, 756, 150, False), (535, 760, 140, True)]):
    side = -1 if flip else 1                                           # direction of the fire
    c.layer(name=f"people/{i}/rim")
    c.figure(x + 3.5 * side, y - 1, h, pose="sit", view="side", style="silhouette", flip=flip, colors={"body": "#ffb36a"}, seed=i)
    c.layer(name=f"people/{i}")
    c.figure(x, y, h, pose="sit", view="side", style="silhouette", flip=flip, colors={"body": "#05060a"}, seed=i)

# the fire: a glow behind the logs, smoke drifting with the storm wind
c.layer(name="smoke").smoke(FIRE[0] + 4, FIRE[1] - 75, 420, width=40, drift=0.35, spread=4, curl=1.4, color="#d8c6b4",
                            shadow="#4a4f60", density=0.9, light=(270, 30), seed=SEED + 5)
c.layer(name="fire/halo", blend="screen").fill("#ff7a2a", mask=disc(FIRE[0] / W, (FIRE[1] - 30) / H, 0.05, soft=0.06))
c.layer(name="fire", blend="screen").fill("#ffd47a", mask=disc(FIRE[0] / W, (FIRE[1] - 14) / H, 0.018, soft=0.06))
c.layer(name="fire/sparks").scatter("soft", 25, region=zone((FIRE[0] - 50) / W, (FIRE[1] - 160) / H, (FIRE[0] + 60) / W,
                                    (FIRE[1] - 20) / H), size=(2, 5), color=["#ffd36e", "#ff9a40"], seed=SEED)
c.layer(name="logs")
c.line(FIRE[0] - 40, FIRE[1] + 4, FIRE[0] + 36, FIRE[1] - 6, stroke="#2a160c", width=10)
c.line(FIRE[0] - 34, FIRE[1] - 6, FIRE[0] + 42, FIRE[1] + 4, stroke="#3a1e0e", width=9)
c["logs"].apply(inner_shadow(0, -3, 3, "#ff8a3a", 0.8))                 # lit from the flames above

c.apply(curves([(0, 0), (0.25, 0.2), (0.75, 0.82), (1, 1)]), bloom(0.8, 14, 0.5), grain(0.02, seed=SEED), vignette(0.35))
c.save(__file__[:-3] + ".png")
