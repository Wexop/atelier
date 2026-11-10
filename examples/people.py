# Flat-illustration park scene: posed characters, a dog, a bench, and a perspective crowd in the distance.
from atelier import *
from atelier.figure import *

W, H = 1600, 1000
HZ = 520                                                     # horizon (eye level)
c = Canvas(W, H, bg=gradient(["#9ccbe0", "#d9ecee", "#f5eedc"], angle=90))
c.circle(1270, 170, 70, fill="#fff6dc")
for i, (x, y, s) in enumerate([(260, 140, 1), (760, 95, .7), (1120, 250, .8)]):      # flat clouds
    c.draw(union(ellipse(x, y, 90 * s, 28 * s), ellipse(x - 40 * s, y - 18 * s, 50 * s, 36 * s),
                 ellipse(x + 35 * s, y - 24 * s, 58 * s, 40 * s)), fill="#ffffff", opacity=.85)
# distant city + hills
for i, (x, w, h) in enumerate([(40, 70, 120), (115, 55, 170), (175, 90, 95), (1180, 60, 150), (1245, 80, 210),
                               (1330, 50, 130), (1385, 95, 175), (1485, 70, 110)]):
    c.rect(x, HZ - h, w, h + 5, fill="#b9cdd6")
c.fill("#9cc28f", mask=svg(f"M0 {HZ - 30} C 300 {HZ - 80} 600 {HZ - 10} 900 {HZ - 45} S 1400 {HZ - 70} 1600 {HZ - 35} V {HZ + 20} H 0 Z"))
c.rect(0, HZ, W, H - HZ, fill=gradient(["#8fbf72", "#6fa957"], angle=90))
# path toward the vanishing point
c.polygon([(760, HZ), (840, HZ), (1250, H), (380, H)], fill=gradient(["#e8dcc0", "#d8c7a0"], angle=90))
c.scatter("grass", 260, region=rect(0, HZ + 120, W, H - HZ - 120) - polygon([(760, HZ), (840, HZ), (1290, H), (340, H)]),
          size=(8, 18), seed=SEED + 2)

def tree(x, y, s, col):
    c.draw(rect(x - 9 * s, y - 120 * s, 18 * s, 125 * s, 6 * s), fill="#6b4a33")
    c.draw(union(circle(x, y - 175 * s, 75 * s), circle(x - 55 * s, y - 130 * s, 55 * s),
                 circle(x + 55 * s, y - 135 * s, 58 * s)), fill=col)
    c.draw(circle(x + 22 * s, y - 200 * s, 30 * s), fill="#ffffff", opacity=.12)

for x, y, s, col in [(120, HZ + 40, .7, "#4f8a46"), (1480, HZ + 45, .75, "#3f7a3c"), (560, HZ + 12, .35, "#4f8a46"),
                     (1050, HZ + 18, .4, "#3f7a3c"), (1290, HZ + 25, .5, "#4f8a46")]:
    tree(x, y, s, col)
# distant crowd along the path, fading into the haze
c.crowd(26, (650, HZ + 18, 960, HZ + 120), horizon=HZ - 6, height=70, seed=SEED + 5, dogs=.08,
        tint=("#cfe2e6", .45), spacing=.45)

def shadow(x, y, w):
    c.ellipse(x, y, w, w * .16, fill="#2c4a2a", opacity=.22)

# bench with a reader
c.rect(150, 800, 300, 18, 6, fill="#8a5a3b"); c.rect(150, 760, 300, 14, 6, fill="#8a5a3b")
c.rect(170, 815, 14, 90, fill="#4a3a30"); c.rect(416, 815, 14, 90, fill="#4a3a30")
shadow(300, 910, 150)
c.figure(320, 910, 300, pose="sit", view=-80, seed=11, outfit="long pants", hair="bun", skin="tan",
         colors={"top": "#e9c46a", "bottom": "#264653", "hair": "#2b1d14"}, head=(8, -20, 4))
# a couple walking a dog toward us
shadow(700, 905, 80); shadow(820, 915, 80)
c.figure(690, 905, 340, pose="walk", view=-55, phase=.22, seed=21, outfit="long pants", hair="short",
         skin="brown", colors={"top": "#e76f51", "bottom": "#2b2d42", "hair": "#1d1a1a"})
walker = dict(pose="walk", view=-55, phase=.72, seed=22)
c.figure(815, 915, 320, outfit="tee skirt", hair="long", skin="light",
         colors={"top": "#f4f1ea", "bottom": "#2a9d8f", "hair": "#8c3b1e"}, **walker)
hand = max(figure_joints(815, 915, 320, **walker).values(), key=lambda q: q[0] - abs(q[1] - 780))
dog = dict(kind="dog", pose="walk", phase=.3, seed=4)
collar = figure_joints(990, 935, 85, **dog)["neck_top"]
c.spline([hand, ((hand[0] + collar[0]) / 2, max(hand[1], collar[1]) + 25), collar], stroke="#3a2a22", width=2)
shadow(990, 935, 60)
c.figure(990, 935, 85, colors={"body": "#c8955a"}, **dog)
# kid jumping with a balloon, parent waving
shadow(1150, 925, 50)
kid = dict(pose="cheer", body="child", seed=31, arm_l=(10, 140, 0, 20))
c.figure(1150, 925, 220, outfit="tee shorts", hair="curly", skin="dark",
         colors={"top": "#f3a712", "bottom": "#3d405b"}, **kid)
kh = figure_joints(1150, 925, 220, **kid)["hand_l"]
c.spline([kh, (kh[0] + 12, kh[1] - 60), (1226, 525)], stroke="#555", width=1.5)
c.ellipse(1225, 485, 34, 40, fill="#e63946"); c.circle(1213, 472, 8, fill="#ffffff", opacity=.35)
shadow(1370, 935, 80)
c.figure(1370, 935, 330, pose="wave", view=-25, seed=32, outfit="long pants", hair="ponytail", skin="olive",
         colors={"top": "#2e86ab", "bottom": "#6b705c", "hair": "#3b2618"})
# mid-ground: jogger and someone on the phone
shadow(500, 690, 40)
c.figure(500, 690, 175, pose="run", phase=.3, seed=41, outfit="tank shorts", hair="buzz", skin="medium",
         colors={"top": "#d62828", "bottom": "#22223b"})
shadow(1060, 700, 40)
c.figure(1060, 700, 170, pose="phone", view="3q", seed=42, outfit="coat pants", body="elder",
         colors={"top": "#6d597a", "bottom": "#455a64"}, hair="grey")
c.apply(grain(.015, 1.2, seed=SEED))
c.save(__file__[:-3] + ".png")
