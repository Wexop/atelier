# 3D wooden artist's mannequin in a dynamic leap, built from figure_sdf (capsules, ellipsoids, ball joints).
from atelier import *
from atelier.figure import figure_sdf, figure_skeleton

c = Canvas(1200, 800)
s = Scene()
pose = dict(arm_l=(-50, 30, 0, 40), arm_r=(120, 25, 0, 20), leg_l=(60, 6, 20, 25), leg_r=(-35, 4, 70, 30),
            spine=(10, 4, 18), head=(-14, -10, 0), lift=1.3)
body, joints = figure_sdf("run", height=1.0, split=True, turn=-55, seed=SEED, vary=0, **pose)
sk = figure_skeleton("run", seed=SEED, vary=0, **pose)
hip_y = float(sk["joints"]["pelvis"][1]) / sk["heads"]                                     # metres (height=1)
grain_w = wood(rings=26, angle=82).ramp(["#a86c38", "#d8a66e", "#b97c44", "#e6bd88"])
s.add(body, Mat(grain_w, proj="tri", tex_scale=.45, rough=.38))
s.add(joints, Mat("#7a4e2a", rough=.3))
s.add(round_box(.5, .06, .5, .02).move(.0, .03, 0), Mat("#5a3a24", rough=.45))           # stand base
s.add(cylinder(.014, hip_y).move(0, hip_y / 2, 0), Mat("#3a2a20", rough=.3, metal=.6))  # support rod
s.add(plane(0), Mat("#d9d0c3", rough=.85))
s.add(plane(-1.2, (0, 0, 1)), Mat("#e9e2d6", rough=.95))
s.camera((-.3, .74, 2.35), (0.02, .56, 0), fov=33)
s.area((-1.5, 2.0, 1.5), size=(1.2, 1.2), normal=(1, -1.2, -1), color="#fff0dc", power=14)
s.point((1.2, 1.4, -.6), "#b8cdf0", power=2.5)
s.sky("#c9d4e2", "#efe8dc", "#8c8276", power=.3)
r = c.render(s, aa=1, look="punchy")
c.apply(vignette(.3), grain(.012, seed=SEED))
c.save(__file__[:-3] + ".png")
