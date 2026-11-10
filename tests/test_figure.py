import numpy as np
import pytest

from atelier import Canvas, Scene
from atelier.figure import (figure_skeleton, animal_skeleton, figure_shape, figure_parts, figure_sdf, POSES,
                            ANIMAL_POSES)


def test_proportions_and_grounding():
    s = figure_skeleton("stand", vary=0)
    J = s["joints"]
    assert abs(s["heads"] - 7.45) < .2
    assert J["knee_l"][1] < J["hip_l"][1] < J["shoulder_l"][1] < J["head"][1]
    assert J["shoulder_l"][0] > 0 > J["shoulder_r"][0]  # figure's left = +x
    child = figure_skeleton("stand", "child", vary=0)
    assert 4.5 < child["heads"] < 5.5
    for pose in POSES:
        sk = figure_skeleton(pose, seed=1)
        ys = [p[4][1] for p in sk["parts"] if p[0] == "cap"]
        assert min(ys) > -1e-6 and all(np.isfinite(ys))


def test_determinism_and_variation():
    a = figure_shape(100, 300, 200, pose="walk", seed=3).bounds
    b = figure_shape(100, 300, 200, pose="walk", seed=3).bounds
    c = figure_shape(100, 300, 200, pose="walk", seed=4).bounds
    assert a == b and a != c


@pytest.mark.parametrize("pose", list(POSES))
def test_pose_shapes_valid(pose):
    sh = figure_shape(200, 400, 300, pose=pose, seed=0)
    x0, y0, x1, y1 = sh.bounds
    assert y1 <= 401 and y0 < 400 - 150 and 20 < x1 - x0 < 400  # feet on the line, plausible size
    assert y1 - y0 <= (300 * 1.3 if pose not in ("cheer", "jump", "reach", "wave") else 300 * 1.7)


def test_styles_and_animals_render():
    c = Canvas(400, 200)
    for i, st in enumerate(("flat", "line", "silhouette", "ink")):
        c.figure(50 + i * 90, 180, 150, pose="wave", style=st, seed=i)
    assert c.flatten()[..., 3].sum() > 2000
    for kind in ("cat", "dog", "horse", "bird"):
        for pose in (("stand", "walk", "sit") if kind != "bird" else ("stand", "fly")):
            if kind == "horse" and pose == "sit":
                pose = "gallop"
            sh = figure_shape(100, 150, 60, kind=kind, pose=pose, seed=1)
            assert sh.bounds[3] <= 151 and sh.bounds[2] - sh.bounds[0] > 20
    parts = figure_parts(100, 300, 200, pose="stand", seed=2, outfit="tee skirt", hair="long")
    roles = {r for _, r, _, _ in parts}
    assert {"skin", "top", "bottom", "hair", "shoes"} <= roles


def test_crowd_sorted():
    c = Canvas(300, 200)
    pl = c.crowd(12, (0, 100, 300, 200), horizon=60, height=80, seed=1)
    assert len(pl) == 12 and [p[1] for p in pl] == sorted(p[1] for p in pl)
    assert pl[0][2] < pl[-1][2]  # far figures are smaller


def test_sdf_figure():
    sd = figure_sdf("walk", height=1.8, seed=1)
    p = np.random.default_rng(0).uniform(-1.5, 2, (2000, 3)).astype(np.float32)
    d = sd(p)
    assert np.isfinite(d).all()
    assert sd(np.array([[0, 1.0, 0]], np.float32))[0] < 0  # pelvis is inside
    assert sd(np.array([[0, 2.5, 0]], np.float32))[0] > 0.3  # above the head
    b, j = figure_sdf("sit", split=True)
    assert np.isfinite(j(p)).all()
    dog = figure_sdf("stand", kind="dog", height=.6)
    assert np.isfinite(dog(p)).all()
    s = Scene()
    s.add(sd)
