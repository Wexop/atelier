"""Figures: parametric human and animal skeletons with named poses, rendered as vector silhouettes,
flat illustration characters, brush gesture drawings, 3D SDF mannequins, and perspective crowds.

Skeletons are built in 3D (units = head heights, y up, figure faces +z, its left side = +x),
then projected (2D) or turned into SDF primitives (3D). Everything random takes `seed`.

    c.figure(x, y, height, pose="walk", style="flat", seed=3)       # feet at (x, y), canvas px
    figure_shape(x, y, height, pose="wave") -> Shape                 # silhouette (fill/gradient/stroke)
    figure_sdf(pose="run", height=1.75) -> SDF                       # Scene.add(..., Mat(...))
    c.crowd(60, (0, 400, 800, 600), horizon=300, height=120, seed=1)
"""
from __future__ import annotations

import math
import numpy as np
import cv2
import skia

from .core import Layer, color as _color, F32
from .vector import Shape
from . import sdf as _sdf

# ------------------------------------------------------------------ small math
_DOWN = np.array([0.0, -1.0, 0.0])
_UP = np.array([0.0, 1.0, 0.0])
_FWD = np.array([0.0, 0.0, 1.0])


def _rx(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_MIR = np.diag([-1.0, 1.0, 1.0])

# ------------------------------------------------------------------ bodies (units: head heights)
_ADULT = dict(ankle=.3, shin=1.75, thigh=1.95, spine=2.15, neck=.34, uarm=1.42, farm=1.2, hand=.72, foot=1.0,
              sw=.82, hw=.42, chest=(.78, .7, .48), waist=(.6, .55, .42), pelvis=(.74, .5, .46),
              head=(.37, .5, .45), r_uarm=(.22, .15), r_farm=(.15, .105), r_thigh=(.32, .2),
              r_shin=(.21, .12), r_neck=.17, r_hand=.115, r_foot=.13, calf=.23)
_CHILD = dict(ankle=.2, shin=1.0, thigh=1.1, spine=1.45, neck=.2, uarm=.9, farm=.75, hand=.45, foot=.7,
              sw=.6, hw=.33, chest=(.6, .55, .42), waist=(.55, .45, .4), pelvis=(.56, .42, .4),
              head=(.37, .5, .45), r_uarm=(.17, .13), r_farm=(.125, .095), r_thigh=(.27, .17),
              r_shin=(.17, .105), r_neck=.13, r_hand=.095, r_foot=.105, calf=.18)
_TODDLER = dict(ankle=.15, shin=.7, thigh=.75, spine=1.15, neck=.1, uarm=.65, farm=.55, hand=.35, foot=.6,
                sw=.55, hw=.32, chest=(.6, .55, .46), waist=(.6, .5, .46), pelvis=(.56, .42, .42),
                head=(.37, .5, .45), r_uarm=(.16, .13), r_farm=(.12, .1), r_thigh=(.26, .18),
                r_shin=(.17, .11), r_neck=.12, r_hand=.09, r_foot=.1, calf=.17)


def _lerpb(a, b, t):
    return {k: (tuple(x + (y - x) * t for x, y in zip(a[k], b[k])) if isinstance(a[k], tuple)
                else a[k] + (b[k] - a[k]) * t) for k in a}


BODIES = {"adult": _ADULT, "teen": _lerpb(_ADULT, _CHILD, .22), "child": _CHILD, "toddler": _TODDLER,
          "elder": _ADULT}
BUILDS = {"slim": (.88, .9, 1.0), "average": (1, 1, 1), "broad": (1.12, 1.0, 1.0), "heavy": (1.12, 1.4, 1.1),
          "curvy": (.95, 1.1, 1.18)}  # (shoulders/limbs, waist, hips)
_WIDTH_KEYS = ("r_uarm", "r_farm", "r_thigh", "r_shin", "r_neck", "r_hand", "calf")


def _body(body="adult", build="average", rng=None, vary=1.0):
    B = dict(BODIES[body])
    ks, kw, kh = BUILDS[build]
    j = (lambda s: 1 + rng.normal(0, s) * vary) if rng is not None else (lambda s: 1.0)
    B["sw"] *= ks * j(.04)
    B["chest"] = tuple(v * f for v, f in zip(B["chest"], (ks * j(.05), 1, kw ** .5)))
    B["waist"] = (B["waist"][0] * kw * j(.07), B["waist"][1], B["waist"][2] * kw * j(.07))
    B["pelvis"] = (B["pelvis"][0] * kh * j(.05), B["pelvis"][1], B["pelvis"][2] * kh)
    B["hw"] *= kh
    lw = ks * j(.06) * (1.08 if build == "heavy" else 1)
    for k in _WIDTH_KEYS:
        B[k] = tuple(v * lw for v in B[k]) if isinstance(B[k], tuple) else B[k] * lw
    if build == "heavy":
        B["r_thigh"] = (B["r_thigh"][0] * 1.12, B["r_thigh"][1])
    lj = j(.025)
    for k in ("shin", "thigh", "uarm", "farm"):
        B[k] *= lj
    B["spine"] *= j(.025)
    return B


# ------------------------------------------------------------------ poses
def _walk(p):
    s, c = math.sin(2 * math.pi * p), math.cos(2 * math.pi * p)
    out = {"spine": (4, 0, 7 * s), "pelvis": (-5 * s, 0), "head": (2, -4 * s, 0)}
    for side, sg in (("l", 1), ("r", -1)):
        ss, cc = s * sg, c * sg
        fwd, knee = 24 * ss, 5 + 55 * max(0.0, cc) ** 1.3 + 8 * max(0.0, -ss)
        wp = -15 * max(ss, 0) + 24 * max(-ss, 0) - 8 * max(cc, 0)
        out["leg_" + side] = (fwd, 3, knee, fwd - knee + wp)
        out["arm_" + side] = (-20 * ss, 6, 0, 14 + 18 * max(0.0, -ss))
    return out


def _run(p):
    s, c = math.sin(2 * math.pi * p), math.cos(2 * math.pi * p)
    out = {"spine": (13, 0, 12 * s), "pelvis": (-8 * s, 0), "head": (-8, -6 * s, 0), "lift": .25 * abs(s)}
    for side, sg in (("l", 1), ("r", -1)):
        ss, cc = s * sg, c * sg
        fwd, knee = 38 * ss + 12, 28 + 82 * max(0.0, cc)
        wp = 30 * max(-ss, 0) + 15 * max(cc, 0) - 8 * max(ss, 0)
        out["leg_" + side] = (fwd, 4, knee, fwd - knee + wp)
        out["arm_" + side] = (-48 * ss + 8, 12, -10, 88 - 10 * ss)
    return out


_STAND = {"spine": (3, 0, 0), "arm_l": (2, 7, 0, 10), "arm_r": (2, 7, 0, 10), "leg_l": (0, 3, 2, -2),
          "leg_r": (0, 3, 2, -2)}
POSES = {
    "stand": {},
    "idle": {"pelvis": (0, -5), "spine": (2, 5, 0), "head": (0, 0, 4), "leg_l": (2, 5, 0, -2),
             "leg_r": (8, 1, 16, -8), "arm_l": (0, 9, 0, 14), "arm_r": (6, 6, -10, 22)},
    "walk": lambda p: _walk(.25 if p is None else p),
    "run": lambda p: _run(.25 if p is None else p),
    "sit": {"spine": (6, 0, 0), "head": (-3, 0, 0), "leg_l": (88, 8, 88, 0), "leg_r": (88, 8, 88, 0),
            "arm_l": (28, 10, -15, 58), "arm_r": (28, 10, -15, 58)},
    "wave": {"spine": (2, -3, 0), "head": (0, -8, 6), "arm_r": (10, 100, 90, 75)},
    "point": {"head": (0, -30, 0), "spine": (2, 0, -10), "arm_r": (12, 82, 0, 4)},
    "cheer": {"head": (-12, 0, 0), "spine": (-4, 0, 0), "arm_l": (8, 155, 0, 18), "arm_r": (8, 155, 0, 18)},
    "jump": {"head": (-10, 0, 0), "spine": (-6, 0, 0), "arm_l": (10, 145, 0, 22), "arm_r": (10, 145, 0, 22),
             "leg_l": (30, 9, 85, 30), "leg_r": (30, 9, 85, 30), "lift": .9},
    "dance": {"pelvis": (12, 9), "spine": (2, -11, 14), "head": (0, 10, 12), "arm_l": (20, 135, 0, 40),
              "arm_r": (35, 65, 85, 100), "leg_l": (4, 7, 2, -4), "leg_r": (35, 18, 60, 20)},
    "think": {"head": (12, 12, 8), "spine": (4, 0, 0), "arm_r": (40, 4, -28, 150), "arm_l": (28, 0, -75, 98),
              "leg_r": (6, 2, 12, -6)},
    "hips": {"spine": (1, 0, 0), "arm_l": (-8, 42, -90, 95), "arm_r": (-8, 42, -90, 95),
             "leg_l": (0, 7, 0, -2), "leg_r": (0, 7, 0, -2)},
    "phone": {"head": (6, 0, 12), "arm_r": (22, 18, -35, 150), "arm_l": (4, 6, 0, 15),
              "leg_r": (8, 1, 14, -8)},
    "reach": {"head": (-20, 0, 0), "arm_r": (170, 8, 0, 5), "arm_l": (4, 8, 0, 12), "leg_r": (0, 3, 0, 15)},
}
_DEFAULT_VIEW = {"walk": "side", "run": "side", "sit": "3q", "jump": "3q"}
VIEWS = {"front": 0.0, "3q": 35.0, "side": 90.0, "back": 180.0, "3q_back": 145.0}


def _pose_angles(pose, phase=None, overrides=None, rng=None, vary=1.0, body="adult"):
    P = POSES[pose] if isinstance(pose, str) else pose
    P = P(phase) if callable(P) else dict(P)
    A = dict(_STAND)
    A.update(P)
    if body == "elder":
        A["spine"] = (A.get("spine", (0, 0, 0))[0] + 14,) + tuple(A.get("spine", (0, 0, 0))[1:])
        A["head"] = tuple(np.add(A.get("head", (0, 0, 0)), (-10, 0, 0)))
        for s in ("leg_l", "leg_r"):
            L = A[s]
            A[s] = (L[0] + 6, L[1], L[2] + 10, L[3] - 4)
    if overrides:
        A.update(overrides)
    if rng is not None and vary:
        for k, v in list(A.items()):
            if k != "lift" and isinstance(v, tuple):
                A[k] = tuple(float(x) + rng.normal(0, 3.0 * vary) for x in v)
    return A


# ------------------------------------------------------------------ human skeleton
def _seg_parts(parts, name, role, pts, radii, group):
    for i in range(len(pts) - 1):
        parts.append(("cap", name, role, group, pts[i], pts[i + 1], radii[i], radii[i + 1]))


def figure_skeleton(pose="stand", body="adult", build="average", phase=None, seed=None, vary=1.0, _heads=True,
                    **angles):
    """3D human skeleton (head-height units, feet on y=0, facing +z, figure's left = +x).
    pose: name in POSES (stand idle walk run sit wave point cheer jump dance think hips phone reach)
    or a dict of angles; phase (0-1) for walk/run cycles. angles override per joint (degrees):
    spine=(bend, side, twist) head=(nod, turn, tilt) pelvis=(turn, tilt)
    arm_l/arm_r=(forward, out, roll, elbow) leg_l/leg_r=(forward, out, knee, foot) lift=heads.
    -> dict(joints={name: xyz}, parts=[...], heads=standing height, head=(centre, R), body=B)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    B = _body(body, build, rng, vary)
    A = _pose_angles(pose, phase, angles, rng, vary, body)
    g = lambda k, n: (tuple(A.get(k, ())) + (0.0,) * n)[:n]
    bend, side, twist = g("spine", 3)
    nod, hturn, htilt = g("head", 3)
    pturn, ptilt = g("pelvis", 2)
    J, parts = {}, []
    hip_y = B["ankle"] + B["shin"] + B["thigh"]
    P0 = np.array([0.0, hip_y + .05, 0.0])
    Rp = _ry(pturn) @ _rz(ptilt)
    Rw = Rp @ _ry(twist / 2) @ _rz(-side / 2) @ _rx(bend / 2)
    Rc = Rp @ _ry(twist) @ _rz(-side) @ _rx(bend)
    W = P0 + Rw @ np.array([0, .45 * B["spine"], 0])
    N = W + Rc @ np.array([0, .55 * B["spine"], 0])
    J.update(pelvis=P0, waist=W, neck=N)
    chest_c = N + Rc @ np.array([0, -B["chest"][1] * .92, -.03])
    hip_c = P0 + Rp @ np.array([0, -.1, 0])
    parts.append(("hull", "hips", "bottom", "torso", [(W + Rw @ np.array([0, -.1, 0]), Rw,
                  np.array(B["waist"]) * .97), (hip_c, Rp, np.array(B["pelvis"]))]))
    sh_r = np.array([B["r_uarm"][0] * 1.05, B["r_uarm"][0] * .9, B["r_uarm"][0] * 1.1])
    sh_c = [N + Rc @ np.array([sx * (B["sw"] - .06), -.2, -.03]) for sx in (1, -1)]
    parts.append(("hull", "torso", "top", "torso", [(chest_c, Rc, np.array(B["chest"])),
                  (W, Rw, np.array(B["waist"]))] + [(c, Rc, sh_r) for c in sh_c]))
    # head + neck
    Rh = Rc @ _ry(hturn) @ _rx(nod) @ _rz(-htilt)
    T = N + Rc @ _rx(nod * .5) @ np.array([0, B["neck"], .02])
    Hc = T + Rh @ np.array([0, .44, .05])
    J.update(neck_top=T, head=Hc)
    parts.append(("cap", "neck", "skin", "neck", N + Rc @ np.array([0, -.15, -.02]), T, B["r_neck"],
                  B["r_neck"] * .95))
    parts.append(("ell", "head", "skin", "head", Hc, Rh, np.array(B["head"])))
    # arms
    for sd, M in (("l", np.eye(3)), ("r", _MIR)):
        fwd, out, rot, elb = g("arm_" + sd, 4)
        S = N + Rc @ (M @ np.array([B["sw"], -.2, -.03]))
        Ru = _rx(-fwd) @ _rz(out) @ _ry(rot)
        Rf = Ru @ _rx(-elb)
        Ruw, Rfw = Rc @ M @ Ru @ M, Rc @ M @ Rf @ M
        E = S + Ruw @ (_DOWN * B["uarm"])
        Wr = E + Rfw @ (_DOWN * B["farm"])
        Ht = Wr + Rfw @ (_rx(-12) @ _DOWN * B["hand"] * .78)
        J.update({"shoulder_" + sd: S, "elbow_" + sd: E, "wrist_" + sd: Wr, "hand_" + sd: Ht})
        ru, rf = B["r_uarm"], B["r_farm"]
        parts.append(("cap", "uarm_" + sd, "sleeve", "arm_" + sd, S, E, ru[0], ru[1]))
        mid = E + (Wr - E) * .3
        parts.append(("cap", "farm_" + sd, "cuff", "arm_" + sd, E, mid, rf[0] * 1.02, rf[0] * 1.05))
        parts.append(("cap", "farm2_" + sd, "cuff", "arm_" + sd, mid, Wr, rf[0] * 1.05, rf[1]))
        parts.append(("cap", "hand_" + sd, "skin", "arm_" + sd, Wr + (Ht - Wr) * .25, Ht, B["r_hand"],
                      B["r_hand"] * .82))
    # legs
    for sd, M in (("l", np.eye(3)), ("r", _MIR)):
        fwd, out, knee, foot = g("leg_" + sd, 4)
        H = P0 + Rp @ (M @ np.array([B["hw"], -.05, 0]))
        Rt = _rx(-fwd) @ _rz(out)
        Rs = Rt @ _rx(knee)
        Rft = Rs @ _rx(foot)
        Rtw, Rsw, Rfw = (Rp @ M @ R @ M for R in (Rt, Rs, Rft))
        K = H + Rtw @ (_DOWN * B["thigh"])
        An = K + Rsw @ (_DOWN * B["shin"])
        fdir, fdown = Rfw @ _FWD, Rfw @ _DOWN
        heel = An + fdir * -.1 + fdown * .14
        toe = An + fdir * (B["foot"] * .78) + fdown * .17
        J.update({"hip_" + sd: H, "knee_" + sd: K, "ankle_" + sd: An, "toe_" + sd: toe, "heel_" + sd: heel})
        rt, rs = B["r_thigh"], B["r_shin"]
        tm = H + (K - H) * .45
        parts.append(("cap", "thigh_" + sd, "shorts", "leg_" + sd, H, tm, rt[0], (rt[0] + rt[1]) / 2 * 1.04))
        parts.append(("cap", "thigh2_" + sd, "pants", "leg_" + sd, tm, K, (rt[0] + rt[1]) / 2 * 1.04, rt[1]))
        cm = K + (An - K) * .3
        parts.append(("cap", "shin_" + sd, "pants", "leg_" + sd, K, cm, rt[1] * .98, B["calf"]))
        parts.append(("cap", "shin2_" + sd, "pants", "leg_" + sd, cm, An, B["calf"], rs[1]))
        parts.append(("cap", "foot_" + sd, "shoes", "leg_" + sd, heel, toe, B["r_foot"], B["r_foot"] * .85))
    sk = dict(joints=J, parts=parts, head=(Hc, Rh), body=B, kind="human", angles=A)
    _ground(sk, A.get("lift", 0.0))
    if _heads:
        sk["heads"] = _standing_heads(body)
    return sk


_STAND_CACHE: dict = {}


def _standing_heads(body):
    if body not in _STAND_CACHE:
        _STAND_CACHE[body] = figure_skeleton("stand", body, vary=0, _heads=False)["top"]
    return _STAND_CACHE[body]


def _part_pts(p):
    """Sample extreme points of a part in 3D (for bounds/grounding)."""
    if p[0] == "cap":
        a, b, ra, rb = p[4:8]
        return [a + _DOWN * ra, b + _DOWN * rb, a + _UP * ra, b + _UP * rb]
    els = [p[4:7]] if p[0] == "ell" else p[4]
    out = []
    for c, R, r in els:
        ext = np.sqrt(((R * np.asarray(r)[None, :]) ** 2).sum(1))  # world half-extent per axis
        out += [c - ext * _UP, c + ext * _UP]
    return out


def _move_parts(sk, d):
    sk["joints"] = {k: v + d for k, v in sk["joints"].items()}
    new = []
    for p in sk["parts"]:
        if p[0] == "cap":
            new.append(p[:4] + (p[4] + d, p[5] + d) + p[6:])
        elif p[0] == "ell":
            new.append(p[:4] + (p[4] + d,) + p[5:])
        else:
            new.append(p[:4] + ([(c + d, R, r) for c, R, r in p[4]],))
    sk["parts"] = new
    hc, rh = sk["head"]
    sk["head"] = (hc + d, rh)


def _ground(sk, lift=0.0):
    ys = [q[1] for p in sk["parts"] for q in _part_pts(p)]
    _move_parts(sk, np.array([0.0, lift - min(ys), 0.0]))
    sk["top"] = max(ys) - min(ys)


# ------------------------------------------------------------------ colours
SKIN = {"porcelain": "#f7e1d3", "light": "#f1c9a9", "fair": "#e8b48e", "medium": "#d29a6c", "olive": "#c08a5b",
        "tan": "#a86e45", "brown": "#8a5534", "dark": "#68391f", "deep": "#4a2915"}
HAIR = {"black": "#1d1a1a", "darkbrown": "#3b2618", "brown": "#6a4126", "auburn": "#8c3b1e", "blonde": "#d9b26a",
        "ginger": "#c0602a", "grey": "#a9a6a2", "white": "#e9e5dc"}
_TOPS = ["#e4572e", "#2e86ab", "#f3a712", "#4c956c", "#a23b72", "#3d405b", "#f2cc8f", "#81b29a", "#e07a5f",
         "#5f0f40", "#f4f1ea", "#264653", "#e9c46a", "#d62828", "#6d597a"]
_BOTTOMS = ["#2b2d42", "#3d405b", "#264653", "#5c4d3c", "#1f3b57", "#6b705c", "#22223b", "#8d6e63", "#455a64"]
_SHOES = ["#1b1b1b", "#3e2723", "#f5f5f5", "#5d4037", "#263238", "#7b2d26"]
_FACE_CUT = (.355, .47, .42)
HAIRSTYLES = {  # (front ellipsoids [(local centre, radii)], back ellipsoids, face-cut centre y)
    "short": ([((0, .1, -.05), (.405, .45, .47))], [], -.2),
    "buzz": ([((0, .07, -.03), (.385, .46, .46))], [], -.17),
    "bob": ([((0, .03, -.04), (.45, .56, .51))], [], -.24),
    "long": ([((0, .05, -.04), (.43, .54, .5))], [((0, -.55, -.16), (.43, .74, .27))], -.24),
    "bun": ([((0, .1, -.05), (.4, .45, .47)), ((0, .44, -.26), (.19, .17, .19))], [], -.2),
    "ponytail": ([((0, .1, -.05), (.4, .45, .47))], [((0, -.25, -.5), (.12, .45, .13))], -.2),
    "curly": ([((0, .14, -.05), (.58, .58, .6))], [], -.22),
    "bald": ([], [], 0),
}
_HAIR_PICK = ["short", "short", "bob", "long", "bun", "ponytail", "curly", "buzz", "bald"]


def _col3(c):
    return np.asarray(_color(c), np.float64)[:3]


def _shade(c, k):
    return tuple(float(v) for v in np.clip(_col3(c) * k, 0, 1))


def _outfit(outfit, colors, rng, skin, hair):
    """-> (top, bottom, hairstyle, colour dict)."""
    words = (outfit or "").split()
    tops = [w for w in words if w in ("tee", "long", "tank", "coat")]
    bots = [w for w in words if w in ("pants", "shorts", "skirt", "dress")]
    top = tops[0] if tops else str(rng.choice(["tee", "tee", "long", "long", "coat", "tank"]))
    bot = bots[0] if bots else str(rng.choice(["pants", "pants", "pants", "shorts", "skirt", "dress"]))
    hs = hair if hair in HAIRSTYLES else str(rng.choice(_HAIR_PICK))
    C = {"skin": SKIN.get(skin, skin) if skin is not None else str(rng.choice(list(SKIN.values()))),
         "hair": str(rng.choice(list(HAIR.values())[:7])),
         "top": str(rng.choice(_TOPS)), "bottom": str(rng.choice(_BOTTOMS)), "shoes": str(rng.choice(_SHOES))}
    if hair is not None and hair not in HAIRSTYLES:
        C["hair"] = HAIR.get(hair, hair)
    C.update(colors or {})
    return top, bot, hs, C


# ------------------------------------------------------------------ 2D projection
class _View:
    def __init__(self, x, y, k, yaw, flip):
        self.x, self.y, self.k = x, y, k
        self.V = _ry(yaw)
        self.f = -1.0 if flip else 1.0
        self.P = np.array([self.V[0] * self.f * k, [0, -k, 0]])

    def pt(self, p):
        return np.asarray(p, np.float64) @ self.P.T + np.array([self.x, self.y])

    def depth(self, p):
        return float(np.asarray(p) @ self.V[2])

    def ell2(self, c, R, r):
        """Projected ellipsoid -> (centre2, axes U (2x2), radii (2,))."""
        S = self.P @ (R * np.square(r)[None, :]) @ R.T @ self.P.T
        w, U = np.linalg.eigh(S)
        return self.pt(c), U, np.sqrt(np.maximum(w, 1e-12))

    def ell_path(self, c, R, r):
        c2, U, rr = self.ell2(c, R, np.asarray(r, np.float64))
        p = skia.Path()
        p.addOval(skia.Rect.MakeLTRB(-rr[0], -rr[1], rr[0], rr[1]))
        m = skia.Matrix()
        m.setAll(U[0, 0], U[0, 1], c2[0], U[1, 0], U[1, 1], c2[1], 0, 0, 1)
        p.transform(m)
        return p

    def ell_pts(self, c, R, r, n=64):
        c2, U, rr = self.ell2(c, R, np.asarray(r, np.float64))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return c2 + (U @ (rr[:, None] * np.stack([np.cos(t), np.sin(t)]))).T


def _cap2d(a, ra, b, rb, n=14):
    """Closed polygon = convex hull of two circles (tapered capsule outline)."""
    d = b - a
    L = float(np.hypot(*d))
    if L <= abs(ra - rb) + 1e-6:
        c, r = (a, ra) if ra >= rb else (b, rb)
        t = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        return c + r * np.stack([np.cos(t), np.sin(t)], 1)
    th = math.atan2(d[1], d[0])
    phi = math.acos(max(-1.0, min(1.0, (ra - rb) / L)))
    tb = np.linspace(th - phi, th + phi, n)
    ta = np.linspace(th + phi, th + 2 * np.pi - phi, n)
    return np.concatenate([b + rb * np.stack([np.cos(tb), np.sin(tb)], 1),
                           a + ra * np.stack([np.cos(ta), np.sin(ta)], 1)])


def _poly_path(P):
    p = skia.Path()
    p.addPoly([skia.Point(float(x), float(y)) for x, y in P], True)
    return p


def _union_paths(paths):
    paths = [p for p in paths if p is not None]
    if len(paths) == 1:
        return paths[0]
    ob = skia.OpBuilder()
    for p in paths:
        ob.add(p, skia.PathOp.kUnion_PathOp)
    out = ob.resolve()
    return out if out is not None else paths[0]


def _part_path(v, p):
    if p[0] == "cap":
        a, b, ra, rb = p[4:8]
        return _poly_path(_cap2d(v.pt(a), ra * v.k, v.pt(b), rb * v.k))
    if p[0] == "ell":
        return v.ell_path(*p[4:7])
    if p[0] == "poly":
        return _poly_path(v.pt(np.asarray(p[4])))
    pts = np.concatenate([v.ell_pts(c, R, r) for c, R, r in p[4]]).astype(np.float32)
    return _poly_path(cv2.convexHull(pts)[:, 0])


def _part_depth(v, p):
    if p[0] == "cap":
        return max(v.depth(p[4]), v.depth(p[5]))
    if p[0] == "ell":
        return v.depth(p[4])
    if p[0] == "poly":
        return float(np.mean([v.depth(q) for q in p[4]]))
    return float(np.mean([v.depth(c) for c, _, _ in p[4]]))


def _resolve_view(view, pose, kind="human"):
    if view is None:
        view = "side" if kind != "human" else _DEFAULT_VIEW.get(pose if isinstance(pose, str) else "", "front")
    return VIEWS[view] if isinstance(view, str) else float(view)


def _skirt(sk, bot):
    J, B = sk["joints"], sk["body"]
    hips = next(p for p in sk["parts"] if p[1] == "hips")
    hc, Rp, pr = hips[4][1]
    km = (J["knee_l"] + J["knee_r"]) / 2
    t = {"skirt": .8, "dress": .92, "coat": .82}[bot]
    bc = hc + (km - hc) * t
    dx = abs(J["knee_l"][0] - J["knee_r"][0]) / 2 + B["r_thigh"][1] * 1.7
    dz = abs(J["knee_l"][2] - J["knee_r"][2]) / 2 + B["r_thigh"][1] * 1.7
    rb = np.array([max(pr[0] * 1.15, dx), .03, max(pr[2] * 1.05, dz)])
    return ("hull", "skirt", "top" if bot in ("dress", "coat") else "bottom", "skirt",
            [(hc, Rp, pr * np.array([1.04, 1, 1.06])), (bc, Rp, rb)])


def _human_items(sk, v, top="tee", bot="pants", hair="short"):
    """Projected drawable parts -> (list of (group, role, skia.Path, depth) back to front, torso depth)."""
    role_map = {"sleeve": "top" if top in ("tee", "long", "coat") else "skin",
                "cuff": "top" if top in ("long", "coat") else "skin",
                "shorts": "bottom" if bot in ("pants", "shorts") else "skin",
                "pants": "bottom" if bot == "pants" else "skin"}
    parts = list(sk["parts"])
    if bot == "dress" or top == "coat":
        parts = [p[:2] + ("top",) + p[3:] if p[1] == "hips" else p for p in parts]
    if bot in ("skirt", "dress") or top == "coat":
        parts.append(_skirt(sk, bot if bot in ("skirt", "dress") else "coat"))
    groups = {}
    for p in parts:
        groups.setdefault(p[3], []).append(p)
    Hc, Rh = sk["head"]
    hr = np.array(sk["body"]["head"])
    torso_d = _part_depth(v, next(p for p in parts if p[1] == "torso")) - .3
    out, leg_d = [], []
    for gname, ps in groups.items():
        ds = [_part_depth(v, p) for p in ps]
        d = max(ds) if gname.startswith("arm") else float(np.mean(ds))
        if gname == "torso":
            d = torso_d
        elif gname == "neck":
            d = torso_d + .01
        elif gname == "head":
            d = max(d, torso_d) + .3
        elif gname.startswith("arm"):
            d += .05
        if gname.startswith("leg"):
            th = ps[0]
            if v.depth(th[5]) - v.depth(th[4]) < .3:  # thigh not pointing at the viewer: tucked under the hips
                d = torso_d - .05 + .01 * math.tanh(d)
            leg_d.append(d)
        out.append([gname, ps, d])
    for o in out:
        if o[0] == "skirt":
            o[2] = max(leg_d + [torso_d]) + .01
    out.sort(key=lambda o: o[2])
    items = []
    fwd_d = float((Rh @ _FWD) @ v.V[2])
    style = HAIRSTYLES.get(hair, HAIRSTYLES["short"])
    ears = [Hc + Rh @ np.array([sd * hr[0] * .97, -.03, -.03]) for sd in (1, -1)]
    for gname, ps, d in out:
        if gname == "head":
            for ec in ears:
                if v.depth(ec) < v.depth(Hc):
                    items.append((gname, "skin", v.ell_path(ec, Rh, (.05, .1, .08)), d))
        for p in ps:
            items.append((gname, role_map.get(p[2], p[2]), _part_path(v, p), d))
        if gname == "head":
            for ec in ears:
                if v.depth(ec) >= v.depth(Hc):
                    items.append((gname, "skin", v.ell_path(ec, Rh, (.05, .1, .08)), d))
            items.append((gname, "skin", v.ell_path(Hc + Rh @ np.array([0, -.08, .43]), Rh, (.055, .085, .07)), d))
            if style[0]:
                hp = _union_paths([v.ell_path(Hc + Rh @ np.array(c, np.float64), Rh, r) for c, r in style[0]])
                if fwd_d > -.45:
                    cut = v.ell_path(Hc + Rh @ np.array([0, style[2], .12]), Rh, _FACE_CUT)
                    hp = skia.Op(hp, cut, skia.PathOp.kDifference_PathOp) or hp
                items.append((gname, "hair", hp, d))
    sk_i = next((i for i, it in enumerate(items) if it[0] == "skirt"), None)
    if sk_i is not None:  # shirt / dress bodice over the skirt top edge
        t_it = [it for it in items if it[0] == "torso" and it[1] == "top"]
        if t_it:
            plain = bot == "dress" or top == "coat"
            items.insert(sk_i + 1, ("torso2" if plain else "torso3", "top", t_it[-1][2], items[sk_i][3]))
    if style[1]:  # long hair / ponytail: behind the head (and behind the body unless seen from the back)
        hb = _union_paths([v.ell_path(Hc + Rh @ np.array(c, np.float64), Rh, r) for c, r in style[1]])
        dz = float(np.mean([v.depth(Hc + Rh @ np.array(c, np.float64)) for c, _ in style[1]]))
        hd = torso_d - .02 if dz < v.depth(Hc) else max(o[2] for o in out) + .01
        idx = next((i for i, it in enumerate(items) if it[3] > hd), len(items))
        items.insert(idx, ("hairback", "hair", hb, hd))
    return items, torso_d


def _face(sk, v):
    """Visible face features as (kind, points2D, size_px)."""
    Hc, Rh = sk["head"]
    hr = np.array(sk["body"]["head"])
    vd = v.V[2]
    out = []

    def vis(loc):
        n = Rh @ (np.asarray(loc, np.float64) / hr ** 2)
        return float(n @ vd) / np.linalg.norm(n)

    def P(loc):
        return v.pt(Hc + Rh @ np.asarray(loc, np.float64))

    for sx in (1, -1):
        e = (sx * .14, .0, .415)
        if vis(e) > .2:
            out.append(("eye", P(e), .046 * v.k * (0.6 + 0.4 * vis(e))))
            out.append(("brow", np.array([P((sx * .07, .12, .44)), P((sx * .2, .13, .38))]), .028 * v.k))
        c = (sx * .21, -.17, .37)
        if vis(c) > .25:
            out.append(("cheek", P(c), .065 * v.k))
    if vis((0, -.05, .45)) > .45:
        out.append(("nose", np.array([P((.0, -.02, .47)), P((.035, -.13, .48)), P((-.01, -.14, .47))]), .022 * v.k))
    if vis((0, -.27, .4)) > .25:
        out.append(("mouth", np.array([P((-.09, -.25, .39)), P((0, -.29, .41)), P((.09, -.25, .39))]), .03 * v.k))
    return out


# ------------------------------------------------------------------ public 2D API
def _build2d(x, y, height, pose, kind, body, build, view, flip, seed, phase, outfit, hair, skin, colors, vary,
             angles):
    rng = np.random.default_rng(seed)
    if kind == "human":
        sk = figure_skeleton(pose, body, build or "average", phase, seed, vary, **angles)
        top, bot, hs, C = _outfit(outfit, colors, rng, skin, hair)
    else:
        sk = animal_skeleton(kind, pose, phase, seed, vary, **angles)
        top = bot = hs = None
        C = _animal_colors(kind, colors, rng)
    if "_tint" in C:
        tc, t = C.pop("_tint")
        tc = _col3(tc)
        C = {r: tuple(float(q) for q in _col3(c) * (1 - t) + tc * t) for r, c in C.items()}
    v = _View(x, y, height / sk["heads"], _resolve_view(view, pose, kind), flip)
    if kind == "human":
        items, td = _human_items(sk, v, top, bot, hs)
    else:
        items, td = _animal_items(sk, v)
    return items, td, sk, v, C


def figure_parts(x, y, height, pose="stand", kind="human", body="adult", build=None, view=None, flip=False,
                 seed=0, phase=None, outfit=None, hair=None, skin=None, colors=None, vary=1.0, **angles):
    """Drawable parts of a figure, back to front: [(group, role, Shape, colour)].
    Roles: skin top bottom shoes hair (human) / body belly dark mane (animals)."""
    items, td, sk, v, C = _build2d(x, y, height, pose, kind, body, build, view, flip, seed, phase, outfit, hair,
                                   skin, colors, vary, angles)
    return [(g, r, Shape(p), C.get(r, C.get("body", "#888"))) for g, r, p, d in items]


def figure_joints(x, y, height, pose="stand", kind="human", body="adult", build=None, view=None, flip=False,
                  seed=0, phase=None, vary=1.0, **angles):
    """2D canvas positions of the skeleton joints {name: (x, y)} for the same arguments as c.figure
    (attach props: leashes, balloons, bags). Human joints: head neck pelvis shoulder_l elbow_l wrist_l
    hand_l hip_l knee_l ankle_l toe_l (+ _r) ...; animals: shoulder hip neck_top head muzzle tail2 lf0-4..."""
    sk, v = _build2d(x, y, height, pose, kind, body, build, view, flip, seed, phase, None, None, None, None,
                     vary, angles)[2:4]
    return {k: tuple(float(q) for q in v.pt(p)) for k, p in sk["joints"].items()}


def figure_shape(x, y, height, pose="stand", kind="human", body="adult", build=None, view=None, flip=False,
                 seed=0, phase=None, outfit=None, hair=None, vary=1.0, **angles):
    """Silhouette of a figure as one Shape (feet at (x, y), standing height in px). Works with
    fills, gradients, strokes, booleans and as a mask. kind: human | cat | dog | horse | bird."""
    items = _build2d(x, y, height, pose, kind, body, build, view, flip, seed, phase, outfit, hair, None, None,
                     vary, angles)[0]
    return Shape(_union_paths([it[2] for it in items]))


def _merge_items(items):
    """Merge consecutive items with the same group and role (one outline per limb, no seams)."""
    out = []
    for g, r, p, d in items:
        if out and out[-1][0] == g and out[-1][1] == r:
            out[-1][2].append(p)
        else:
            out.append([g, r, [p], d])
    return [(g, r, _union_paths(ps), d) for g, r, ps, d in out]


_BRUSH_OF = {"stick": "round", "ink": "ink", "chalk": "chalk", "pencil": "round", "brush": "bristle"}


def _chains(sk):
    J = sk["joints"]
    if sk["kind"] == "human":
        c = [["pelvis", "waist", "neck", "neck_top"], ["shoulder_l", "neck", "shoulder_r"], ["hip_l", "pelvis", "hip_r"]]
        for s in "lr":
            c.append([f"shoulder_{s}", f"elbow_{s}", f"wrist_{s}", f"hand_{s}"])
            c.append([f"hip_{s}", f"knee_{s}", f"ankle_{s}", f"toe_{s}"])
    else:
        c = sk["chains"]
    return [[J[n] for n in ch if n in J] for ch in c]


def figure(self, x, y, height, pose="stand", style="flat", kind="human", body="adult", build=None, view=None,
           flip=False, seed=0, phase=None, outfit=None, hair=None, skin=None, colors=None, fill=None,
           outline=None, width=None, face=True, shade=.86, opacity=1.0, vary=1.0, **angles):
    """Draw a figure with its feet at (x, y); height = standing height in px (animals: shoulder height).
    style: flat (illustration: clothes, hair, face) | line (colouring-book line art) | silhouette |
    stick | ink | chalk | pencil | brush (gesture strokes; fill = colour, width = stroke px).
    kind: human | cat | dog | horse | bird. pose: see POSES / ANIMAL_POSES; phase (0-1) for walk/run cycles.
    body: adult teen child toddler elder; build: slim average broad heavy curvy.
    view: front | 3q | side | back | 3q_back | yaw degrees (default per pose); flip mirrors.
    outfit: 'tee|long|tank|coat pants|shorts|skirt|dress'; hair: style (short buzz bob long bun ponytail
    curly bald) or colour; skin: SKIN name or colour; colors: {skin hair top bottom shoes | body belly dark}
    override. outline: None = auto darker tone, False, or a colour. Angles override joints
    (spine=(bend, side, twist), arm_r=(forward, out, roll, elbow), leg_l=(forward, out, knee, foot) ...).
    Returns the silhouette Shape."""
    items, td, sk, v, C = _build2d(x, y, height, pose, kind, body, build, view, flip, seed, phase, outfit, hair,
                                   skin, colors, vary, angles)
    k = v.k
    sil = None
    if style in _BRUSH_OF:
        col = fill if fill is not None else "#1c1a1f"
        w = width if width is not None else max(1.0, .11 * k)
        for i, ch in enumerate(_chains(sk)):
            if len(ch) >= 2:
                self.stroke(v.pt(np.array(ch)), brush=_BRUSH_OF[style], size=w, color=col, seed=seed + i,
                            opacity=opacity)
        if kind == "human" or "head" in sk:
            Hc, Rh = sk["head"]
            pts = v.ell_pts(Hc, Rh, np.array(sk["body"]["head"]) * .96, 40)
            self.stroke(np.concatenate([pts, pts[:3]]), brush=_BRUSH_OF[style], size=w * .8, color=col,
                        seed=seed + 99, opacity=opacity)
        return Shape(_union_paths([it[2] for it in items]))
    if style == "silhouette":
        sil = Shape(_union_paths([it[2] for it in items]))
        st = {} if outline in (None, False) else {"stroke": outline, "width": width or max(.6, .03 * k)}
        self.draw(sil, fill=fill if fill is not None else "#1f1d24", opacity=opacity, **st)
        return sil
    line_art = style == "line"
    if line_art:
        C = {r: (fill if fill is not None else "#fbf8f2") for r in
             ("skin", "hair", "top", "bottom", "shoes", "body", "belly", "dark", "mane", "beak")}
        if outline in (None, False):
            outline = "#1c1a1f"
    elif style != "flat":
        raise ValueError("style: flat | line | silhouette | stick | ink | chalk | pencil | brush")
    lw = width if width is not None else max(.5, (.035 if line_art else .022) * k)
    side_view = abs(math.sin(math.radians(_resolve_view(view, pose, kind)))) > .3
    for g, r, p, d in _merge_items(items):
        col = C.get(r, C.get("body", "#888"))
        if not line_art and side_view and d < td - .2 and g not in ("hairback",):
            col = _shade(col, shade)
        ol = None if outline is False else (outline if outline is not None else _shade(col, .72))
        if g == "torso2":
            ol = None
        if ol is not None and k < 3:
            ol = None
        self.draw(Shape(p), fill=col, stroke=ol, width=lw, opacity=opacity)
    if face and k >= 5:
        _draw_face(self, sk, v, C, line_art)
    return Shape(_union_paths([it[2] for it in items]))


def _draw_face(L, sk, v, C, line_art=False):
    if sk["kind"] != "human":
        for kind, P, r in _animal_face(sk, v):
            L.draw(Shape(_ell(P, r, r)), fill="#15110f")
        return
    skin = C.get("skin", "#e0b090")
    dark = "#1c1a1f" if line_art else "#2a1d18"
    for kind, P, r in _face(sk, v):
        if kind == "eye":
            L.draw(Shape(_ell(P, r * .85, r * 1.1)), fill=dark)
        elif kind == "brow" and not line_art:
            L.draw(Shape(_open(P)), fill=None, stroke=_shade(C.get("hair", dark), .8), width=r, cap="round")
        elif kind == "cheek" and not line_art:
            L.draw(Shape(_ell(P, r, r * .8)), fill="#e8707a", opacity=.22)
        elif kind == "nose":
            L.draw(Shape(_open(P)), fill=None, stroke=dark if line_art else _shade(skin, .72), width=r, cap="round")
        elif kind == "mouth":
            L.draw(Shape(_open(P, True)), fill=None, stroke=dark if line_art else "#8a3b30", width=r, cap="round")


def _ell(P, rx, ry):
    p = skia.Path()
    p.addOval(skia.Rect.MakeLTRB(P[0] - rx, P[1] - ry, P[0] + rx, P[1] + ry))
    return p


def _open(P, curve=False):
    p = skia.Path()
    p.moveTo(*map(float, P[0]))
    if curve and len(P) == 3:
        p.quadTo(*map(float, P[1]), *map(float, P[2]))
    else:
        for q in P[1:]:
            p.lineTo(*map(float, q))
    return p


# ------------------------------------------------------------------ animals (units: withers height ~ 1)
_SPECIES = {
    "horse": dict(L=.98, w=.13, chest=(.2, .29, .3), belly=(.2, .25, .36), hind=(.21, .29, .28),
                  front=(.3, .33, .27, .09), hind_leg=(.33, .34, .3, .09), r_front=(.1, .065, .045, .05),
                  r_hind=(.13, .075, .045, .05), neck=.52, neck_up=52, r_neck=(.16, .1), skull=(.095, .12, .15),
                  muzzle=.38, r_muzzle=(.095, .068), head_down=62, ear=(.1, .03), tail=(.75, .05), coat=
                  ["#8b4a2b", "#5a3a28", "#c79a6a", "#2b2522", "#e8e1d6", "#a0612f"], dark="#211a16"),
    "dog": dict(L=.78, w=.1, chest=(.17, .24, .22), belly=(.15, .17, .3), hind=(.15, .19, .19),
                front=(.28, .3, .12, .09), hind_leg=(.3, .3, .2, .09), r_front=(.075, .05, .045, .05),
                r_hind=(.1, .055, .045, .05), neck=.3, neck_up=48, r_neck=(.11, .085), skull=(.105, .11, .12),
                muzzle=.2, r_muzzle=(.07, .055), head_down=12, ear=(.16, .05), tail=(.42, .035),
                coat=["#c8955a", "#3a2c24", "#e9d8b8", "#8a5a34", "#5b5f66", "#f2ede4"], dark="#221a15"),
    "cat": dict(L=.92, w=.09, chest=(.15, .21, .2), belly=(.15, .19, .32), hind=(.17, .23, .22),
                front=(.3, .32, .13, .1), hind_leg=(.36, .36, .26, .1), r_front=(.065, .045, .04, .045),
                r_hind=(.1, .05, .04, .045), neck=.18, neck_up=50, r_neck=(.1, .09), skull=(.16, .15, .15),
                muzzle=.08, r_muzzle=(.06, .05), head_down=5, ear=(.12, .055), tail=(.95, .045),
                coat=["#d98a3d", "#5c5c66", "#2a2826", "#e8e2d6", "#9c7a5a"], dark="#1d1917"),
}
_ANIMAL_STAND = {
    "horse": {"front": (-22, 0, 0, 28), "hind": (28, -32, 2, 28), "tail": (-40, -82), "head": (0, 0)},
    "dog": {"front": (-28, 2, 8, 75), "hind": (30, -38, 6, 80), "tail": (35, 10), "head": (0, 0)},
    "cat": {"front": (-26, 2, 10, 80), "hind": (32, -42, 8, 85), "tail": (25, 60), "head": (0, 0)},
}


def _quad_gait(sp, p, gait):
    A0 = _ANIMAL_STAND[sp]
    out = {}
    if gait == "walk":
        off, amp, lift = {"lf": .25, "rf": .75, "lh": 0.0, "rh": .5}, 20, 0.0
    elif gait == "trot":
        off, amp, lift = {"lf": 0.0, "rh": 0.0, "rf": .5, "lh": .5}, 26, 0.0
    else:  # run / gallop
        off, amp, lift = {"lf": 0.0, "rf": .1, "lh": .5, "rh": .6}, 38, .12
    for leg, o in off.items():
        s, c = math.sin(2 * math.pi * (p + o)), math.cos(2 * math.pi * (p + o))
        a = list(A0["front" if leg[1] == "f" else "hind"])
        if leg[1] == "f":
            a[0] += amp * s
            a[1] += amp * .6 * s
            a[2] += -75 * max(0.0, c) + amp * .3 * s
            a[3] += -40 * max(0.0, c)
        else:
            a[0] += amp * s
            a[1] += -25 * max(0.0, c) + amp * .5 * s
            a[2] += 40 * max(0.0, c) + amp * .4 * s
            a[3] += 30 * max(0.0, c)
        out[leg] = tuple(a)
    if gait == "run":
        out["pitch"] = 6 * math.sin(2 * math.pi * p)
        out["lift"] = lift * max(0.0, math.sin(2 * math.pi * (p + .25)))
        out["tail"] = (A0["tail"][0] * .3 - 10, 0) if sp != "cat" else (-5, 15)
        out["head"] = (-10, 0)
    return out


def _animal_pose(sp, pose, p):
    A0 = _ANIMAL_STAND[sp]
    A = {"front": A0["front"], "hind": A0["hind"], "tail": A0["tail"], "head": A0["head"], "pitch": 0.0,
         "lift": 0.0}
    if pose in ("walk", "trot", "run", "gallop"):
        A.update(_quad_gait(sp, .0 if p is None else p, "run" if pose == "gallop" else pose))
    elif pose == "sit":
        A.update(pitch=42, front=(-42, -42, -42, 48), hind=(25, -158, 48, 48), head=(-12, 0),
                 tail=(-30, -20) if sp != "cat" else (-35, 175))
    elif pose == "lie":
        A.update(pitch=0, front=(-30, 95, 90, 90), hind=(70, -115, 90, 90), head=(-5, 0),
                 tail=(-20, -10))
    elif pose == "rear":
        A.update(pitch=48, front=(40, -60, -80, -40), hind=(-10, -40, -20, 30), head=(-30, 0),
                 tail=(-40, -40))
    elif pose == "graze":
        A.update(head=(75, 0), front=(-14, 4, 4, 28))
    elif pose == "stretch":
        A.update(pitch=-18, front=(55, 75, 85, 85), hind=(10, -20, 0, 75), tail=(60, 20), head=(-10, 0))
    elif pose != "stand":
        raise ValueError(f"unknown animal pose '{pose}'. Available: {', '.join(ANIMAL_POSES)}")
    for k in ("lf", "rf"):
        A.setdefault(k, A["front"])
    for k in ("lh", "rh"):
        A.setdefault(k, A["hind"])
    return A


ANIMAL_POSES = ("stand", "walk", "trot", "run", "gallop", "sit", "lie", "rear", "graze", "stretch")
_ASTAND: dict = {}


def animal_skeleton(kind="dog", pose="stand", phase=None, seed=None, vary=1.0, _heads=True, **angles):
    """3D quadruped skeleton (units ~ withers height, facing +z). kind: cat | dog | horse | bird.
    Angles override: pitch (body, + = front up), lift, head=(down, turn), tail=(base, tip) degrees
    up from backward, lf/rf/lh/rh = 4 segment angles (degrees from straight down, + = forward)."""
    if kind == "bird":
        return _bird_skeleton(pose, phase, seed, vary, _heads, **angles)
    S = _SPECIES[kind]
    rng = np.random.default_rng(seed) if seed is not None else None
    A = _animal_pose(kind, pose, phase)
    A.update(angles)
    if rng is not None and vary:
        for k in ("lf", "rf", "lh", "rh", "tail", "head"):
            A[k] = tuple(float(x) + rng.normal(0, 3 * vary) for x in A[k])
    J, parts = {}, []
    Rb = _rx(-A["pitch"])
    root = np.array([0.0, 1.0, 0.0])
    Sj = root + Rb @ np.array([0, 0, S["L"] / 2])
    Hj = root + Rb @ np.array([0, 0, -S["L"] / 2])
    J.update(shoulder=Sj, hip=Hj, root=root)
    hull = [(Sj + Rb @ np.array([0, .1, .03]), Rb, np.array(S["chest"])),
            (root + Rb @ np.array([0, .04, 0]), Rb, np.array(S["belly"])),
            (Hj + Rb @ np.array([0, .1, -.03]), Rb, np.array(S["hind"]))]
    parts.append(("hull", "body", "body", "body", hull))
    # legs
    chains = []
    for leg in ("lf", "rf", "lh", "rh"):
        front = leg[1] == "f"
        sx = 1 if leg[0] == "l" else -1
        base = (Sj if front else Hj) + Rb @ np.array([sx * S["w"], 0, 0])
        lens, rads = (S["front"], S["r_front"]) if front else (S["hind_leg"], S["r_hind"])
        pts = [base]
        for ang, ln in zip(A[leg], lens):
            pts.append(pts[-1] + Rb @ _rx(-ang) @ _DOWN * ln)
        names = [f"{leg}{i}" for i in range(5)]
        J.update(dict(zip(names, pts)))
        chains.append(names)
        r = list(rads) + [rads[-1] * .9]
        for i in range(4):
            role = "dark" if (kind == "horse" and i == 3) else "body"
            parts.append(("cap", f"{leg}{i}", role, "leg_" + leg, pts[i], pts[i + 1], r[i], r[i + 1]))
    # neck + head
    hd, ht = A["head"][0], A["head"][1] if len(A["head"]) > 1 else 0.0
    N0 = Sj + Rb @ np.array([0, .14, .02])
    ndir = Rb @ _rx(-S["neck_up"] + hd * .4) @ _FWD
    NE = N0 + ndir * S["neck"]
    Rh = Rb @ _ry(ht) @ _rx(S["head_down"] + hd)
    sc = NE + Rh @ np.array([0, .02, S["skull"][2] * .4])
    J.update(neck=N0, neck_top=NE, head=sc)
    parts.append(("cap", "neck", "body", "head", N0, NE, S["r_neck"][0], S["r_neck"][1]))
    if kind == "horse":
        mane = [N0 + Rb @ np.array([0, .06, -.07]), NE + Rh @ np.array([0, .09, -.06])]
        parts.append(("cap", "mane", "dark", "head", mane[0], mane[1], .04, .025))
    parts.append(("ell", "skull", "body", "head", sc, Rh, np.array(S["skull"])))
    mz = sc + Rh @ np.array([0, -S["skull"][1] * .3, S["skull"][2] * .5])
    tip = mz + Rh @ np.array([0, -.02, S["muzzle"]])
    parts.append(("cap", "muzzle", "body", "head", mz, tip, S["r_muzzle"][0], S["r_muzzle"][1]))
    parts.append(("ell", "nose", "dark", "head", tip + Rh @ np.array([0, .01, S["r_muzzle"][1] * .75]), Rh,
                  np.array([.025, .02, .02]) * (1.6 if kind == "horse" else 1)))
    J["muzzle"] = tip
    el, ew = S["ear"]
    for sx in (1, -1):
        eb = sc + Rh @ np.array([sx * S["skull"][0] * .55, S["skull"][1] * .75, -.01])
        if kind == "dog":
            parts.append(("ell", "ear", "ear", "head", eb + Rh @ np.array([sx * .05, -.06, -.03]),
                          Rh @ _rz(sx * 12), np.array([.03, el / 1.7, .05])))
        else:
            et = eb + Rh @ np.array([sx * .02, el, -.03 if kind == "horse" else .0])
            parts.append(("cap", "ear", "body", "head", eb, et, ew, .006))
    # tail
    tl, tr = S["tail"]
    T0 = Hj + Rb @ np.array([0, .14, -.13])
    a1, a2 = A["tail"][0], A["tail"][1]
    T1 = T0 + Rb @ (np.array([0, math.sin(math.radians(a1)), -math.cos(math.radians(a1))]) * tl * .5)
    T2 = T1 + Rb @ (np.array([0, math.sin(math.radians(a2)), -math.cos(math.radians(a2))]) * tl * .5)
    trole = "dark" if kind == "horse" else "body"
    hz = kind == "horse"
    parts.append(("cap", "tail", trole, "tail", T0, T1, tr * 1.1, tr * (1.4 if hz else .9)))
    parts.append(("cap", "tail2", trole, "tail", T1, T2, tr * (1.4 if hz else .9), tr * (1.9 if hz else .7)))
    J.update(tail0=T0, tail1=T1, tail2=T2)
    chains += [["hip", "root", "shoulder", "neck", "neck_top", "head", "muzzle"], ["hip", "tail0", "tail1", "tail2"]]
    sk = dict(joints=J, parts=parts, head=(sc, Rh), body=dict(head=S["skull"]), kind=kind, chains=chains,
              species=S)
    _ground(sk, A.get("lift", 0.0))
    if _heads:
        if kind not in _ASTAND:
            st = animal_skeleton(kind, "stand", vary=0, _heads=False)
            _ASTAND[kind] = float(st["joints"]["shoulder"][1] + .1 + S["chest"][1])
        sk["heads"] = _ASTAND[kind]
    return sk


def _bird_skeleton(pose="stand", phase=None, seed=None, vary=1.0, _heads=True, **angles):
    """Small songbird / pigeon: stand | walk | peck | fly (phase flaps the wings)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    jit = (lambda: rng.normal(0, 3 * vary)) if rng is not None else (lambda: 0.0)
    pitch = {"stand": 25, "walk": 18, "peck": -30, "fly": 0}.get(pose, 25) + angles.get("pitch", 0) + jit()
    Rb = _rx(-pitch)
    root = np.array([0.0, .55, 0.0])
    J, parts = {"root": root}, []
    parts.append(("hull", "body", "body", "body", [(root + Rb @ np.array([0, 0, .06]), Rb, np.array([.17, .19, .24])),
                                                    (root + Rb @ np.array([0, -.02, -.22]), Rb,
                                                     np.array([.1, .08, .18]))]))
    hc = root + Rb @ np.array([0, .2, .2]) if pose != "peck" else root + Rb @ np.array([0, -.02, .34])
    Rh = _rx(-angles.get("head", (0,))[0] if "head" in angles else 0)
    parts.append(("ell", "skull", "body", "head", hc, Rh, np.array([.1, .1, .11])))
    bk = hc + Rh @ np.array([0, -.015, .09])
    parts.append(("cap", "beak", "dark", "head", bk, bk + Rh @ np.array([0, -.03, .12]), .035, .006))
    tb = root + Rb @ np.array([0, .0, -.32])
    parts.append(("cap", "tail", "dark", "tail", tb, tb + Rb @ np.array([0, -.04, -.22]), .05, .06))
    if pose == "fly":
        ph = 0.25 if phase is None else phase
        fa = 55 * math.sin(2 * math.pi * ph)
        for sx in (1, -1):
            wb = root + np.array([sx * .1, .08, .05])
            wt = wb + np.array([sx * .45 * math.cos(math.radians(fa)), .45 * math.sin(math.radians(fa)), -.08])
            parts.append(("cap", "wing", "dark", "wing_" + ("l" if sx > 0 else "r"), wb, wt, .09, .03))
    else:
        parts.append(("ell", "wing", "dark", "body", root + Rb @ np.array([.12, .03, -.05]), Rb,
                      np.array([.04, .12, .24])))
        for sx in (1, -1):
            hp = root + np.array([sx * .06, -.08, 0])
            st = {"walk": 15 * sx, "peck": 5}.get(pose, 0)
            kn = hp + _rx(-st) @ _DOWN * .1
            ft = kn + _rx(-st - 10) @ _DOWN * .2
            J.update({"hip_" + str(sx): hp, "knee_" + str(sx): kn, "foot_" + str(sx): ft})
            parts.append(("cap", "leg", "dark", "legs", kn, ft, .018, .014))
            parts.append(("cap", "toe", "dark", "legs", ft, ft + np.array([0, 0, .12]), .014, .01))
    J.update(head=hc, beak=bk)
    sk = dict(joints=J, parts=parts, head=(hc, Rh), body=dict(head=(.1, .1, .11)), kind="bird",
              chains=[["root", "head", "beak"]], species={"coat": ["#6b7b8c", "#8a6a4f", "#3a3f47", "#c0563b"],
                                                             "dark": "#2a2522"})
    _ground(sk, .6 if pose == "fly" else 0.0)
    sk["heads"] = 1.0
    return sk


def _animal_colors(kind, colors, rng):
    S = _SPECIES.get(kind) or {"coat": ["#6b7b8c", "#8a6a4f", "#3a3f47"], "dark": "#2a2522"}
    C = {"body": str(rng.choice(S["coat"])), "dark": S["dark"]}
    C["ear"] = _shade(C["body"], .72)
    if kind == "bird":
        C["dark"] = _shade(C["body"], .6)
    C.update(colors or {})
    return C


def _animal_items(sk, v):
    groups = {}
    for p in sk["parts"]:
        groups.setdefault(p[3], []).append(p)
    body = next(p for p in sk["parts"] if p[1] == "body")
    td = _part_depth(v, body) - .05
    out = []
    for g, ps in groups.items():
        d = float(np.mean([_part_depth(v, p) for p in ps]))
        if g == "body":
            d = td
        elif g == "head":
            d = max(d, td + .02)
        elif g == "tail":
            d = td - .01
        out.append((g, ps, d))
    out.sort(key=lambda o: o[2])
    items = [(g, p[2], _part_path(v, p), d) for g, ps, d in out for p in ps]
    return items, td


def _animal_face(sk, v):
    hc, Rh = sk["head"]
    r = np.array(sk["body"]["head"])
    out = []
    for sx in (1, -1):
        loc = np.array([sx * r[0] * .62, r[1] * .25, r[2] * .62])
        n = Rh @ (loc / r ** 2)
        vis = float(n @ v.V[2]) / np.linalg.norm(n)
        if vis > .1:
            out.append(("eye", v.pt(hc + Rh @ loc), max(.6, .022 * v.k * (1.3 if sk["kind"] == "cat" else 1))))
    return out


# ------------------------------------------------------------------ 3D (SDF)
def _round_cone(a, b, r1, r2):
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    ba = b - a
    l2 = float(ba @ ba) or 1e-9
    rr = float(r1 - r2)
    a2 = l2 - rr * rr
    il2 = 1.0 / l2

    def f(p):
        pa = p - a
        y = pa @ ba
        z = y - l2
        q = pa * F32(l2) - y[:, None] * ba
        x2 = (q * q).sum(1)
        y2, z2 = y * y * l2, z * z * l2
        k = math.copysign(1, rr) * rr * rr * x2
        d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
        d2 = np.sqrt(x2 + y2) * il2 - r1
        d1 = np.sqrt(x2 + z2) * il2 - r2
        return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3)).astype(F32)
    return f, (a + b) / 2, math.sqrt(l2) / 2 + max(r1, r2)


def _oell(c, R, r):
    c, R, r = np.asarray(c, F32), np.asarray(R, F32), np.asarray(r, F32)

    def f(p):
        q = (p - c) @ R
        k0 = np.sqrt(((q / r) ** 2).sum(1))
        k1 = np.sqrt(((q / (r * r)) ** 2).sum(1))
        return (k0 * (k0 - 1) / np.maximum(k1, 1e-9)).astype(F32)
    return f, c, float(r.max())


def _ball(c, r):
    c, r = np.asarray(c, F32), F32(r)
    return (lambda p: (np.sqrt(((p - c) ** 2).sum(1)) - r).astype(F32)), c, float(r)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _fast_union(prims, k=0.0):
    """Union of (f, centre, radius) primitives; each is evaluated only near its bounding sphere."""
    k = F32(k)
    cs = np.array([c for _, c, _ in prims], F32)
    rs = np.array([r for _, _, r in prims], F32)

    def f(p):
        d = np.full(len(p), 1e9, F32)
        for (fn, c, r) in prims:
            lb = np.sqrt(((p - c) ** 2).sum(1)) - r
            m = lb < d + k
            if not m.any():
                continue
            allm = m.all()
            dd = fn(p) if allm else fn(p[m])
            if allm:
                d = _smin(d, dd, k) if k > 0 else np.minimum(d, dd)
            else:
                d[m] = _smin(d[m], dd, k) if k > 0 else np.minimum(d[m], dd)
        return d
    lo, hi = (cs - rs[:, None]).min(0), (cs + rs[:, None]).max(0)
    ctr = (lo + hi) / 2
    return _sdf.SDF(f, (ctr.astype(F32), float(np.linalg.norm(hi - lo) / 2) + float(k)))


def figure_sdf(pose="stand", kind="human", height=1.75, body="adult", build=None, seed=0, phase=None,
               style="mannequin", split=False, at=(0.0, 0.0, 0.0), turn=0.0, vary=1.0, **angles):
    """3D figure as an SDF (metres, feet on y=at[1], facing +z then rotated by `turn` degrees about y).
    style: mannequin (wooden doll: segments + ball joints) | smooth (clay, everything blended) | blocky
    (capsule limbs, no joint balls). split=True -> (segments, joints) to give joints a 2nd material.
    kind: human | cat | dog | horse | bird (height = withers height for animals)."""
    if kind == "human":
        sk = figure_skeleton(pose, body, build or "average", phase, seed, vary, **angles)
    else:
        sk = animal_skeleton(kind, pose, phase, seed, vary, **angles)
    k = height / sk["heads"]
    Rt = _ry(turn)
    off = np.asarray(at, np.float64)
    T = lambda q: (Rt @ np.asarray(q, np.float64)) * k + off
    shrink = .86 if style == "mannequin" else 1.0
    segs, joints, torso = [], [], []
    for p in sk["parts"]:
        if p[0] == "cap":
            if p[1] in ("hand_l", "hand_r") and style == "mannequin":
                a, b = T(p[4]), T(p[5])
                segs.append(_round_cone(a, b, p[6] * k, p[7] * k * .9))
            else:
                segs.append(_round_cone(T(p[4]), T(p[5]), p[6] * k * shrink, p[7] * k * shrink))
        elif p[0] == "ell":
            segs.append(_oell(T(p[4]), Rt @ p[5], np.asarray(p[6]) * k))
        else:
            torso.append(_fast_union([_oell(T(c), Rt @ R, np.asarray(r) * k) for c, R, r in p[4]], .12 * k))
    if kind == "human" and style == "mannequin":
        J, B = sk["joints"], sk["body"]
        for nm, r in (("shoulder", B["r_uarm"][0] * .9), ("elbow", B["r_farm"][0] * 1.08),
                      ("wrist", B["r_farm"][1] * 1.1), ("hip", B["r_thigh"][0] * .72),
                      ("knee", B["r_shin"][0] * 1.08), ("ankle", B["r_shin"][1] * 1.15)):
            for sd in "lr":
                joints.append(_ball(T(J[f"{nm}_{sd}"]), r * k))
        joints.append(_ball(T((J["neck"] + J["neck_top"]) / 2), B["r_neck"] * 1.05 * k))
    tor = [(t.f, t.bounds[0], t.bounds[1]) for t in torso]
    if style == "smooth":
        return _fast_union(segs + tor + joints, .06 * k)
    body_sdf = _fast_union(segs + tor, .015 * k if style == "mannequin" else .03 * k)
    if not joints:
        return (body_sdf, None) if split else body_sdf
    jt = _fast_union(joints)
    return (body_sdf, jt) if split else body_sdf | jt


# ------------------------------------------------------------------ crowds
def crowd(self, n=40, box=None, horizon=None, height=None, poses=("walk", "walk", "walk", "stand", "idle", "phone"),
          style="flat", seed=0, kids=.12, elders=.08, dogs=0.0, tint=None, colors=None, spacing=.35, **kw):
    """Scatter n varied figures in perspective. box=(x0, y0, x1, y1) = region of feet (y0 far, y1 near);
    horizon = eye-level y (figure size -> 0 there; default y0 - (y1 - y0)); height = adult height at y1.
    tint=(colour, amount) fades distant figures toward colour (amount at y0, 0 at y1).
    colors = palette list for clothes (or None). kw -> c.figure. Drawn back to front; returns placements
    [(x, y, h)]."""
    W, H = self.w, self.h
    x0, y0, x1, y1 = box if box is not None else (0, H * .6, W, H)
    hz = horizon if horizon is not None else y0 - (y1 - y0)
    hmax = height if height is not None else (y1 - hz) * .45
    rng = np.random.default_rng(seed)
    placed = []
    for i in range(n * 30):
        if len(placed) >= n:
            break
        y = y0 + (y1 - y0) * rng.random() ** .85
        h = hmax * (y - hz) / max(y1 - hz, 1e-6)
        x = x0 + (x1 - x0) * rng.random()
        if any(abs(x - px) < h * spacing and abs(y - py) < h * .05 + 2 for px, py, ph in placed):
            continue
        placed.append((x, y, h))
    placed.sort(key=lambda t: t[1])
    out = []
    for i, (x, y, h) in enumerate(placed):
        r = np.random.default_rng((seed, i))
        u = r.random()
        if u < dogs:
            kind, body, hh = str(r.choice(["dog", "dog", "cat"])), "adult", h * .3
            pose = str(r.choice(["walk", "stand", "trot"])) if kind == "dog" else "walk"
            view = "side"
        else:
            kind = "human"
            body = "child" if u < dogs + kids else ("elder" if u < dogs + kids + elders else
                                                    str(r.choice(["adult", "adult", "adult", "teen"])))
            hh = h * {"child": .66, "elder": .95, "teen": .93}.get(body, 1.0) * (1 + r.normal(0, .04))
            pose = str(r.choice(list(poses)))
            view = str(r.choice(["side", "3q", "side", "3q_back"])) if pose in ("walk", "run") else \
                str(r.choice(["front", "3q", "back", "3q_back", "3q"]))
        cl = None
        if colors is not None and kind == "human":
            cl = {"top": str(r.choice(colors)), "bottom": str(r.choice(colors))}
        if tint is not None:
            t = float(tint[1]) * (1 - (y - y0) / max(y1 - y0, 1e-6))
            cl = dict(cl or {}, _tint=(tint[0], t))
        kw2 = dict(kw)
        if kind == "human":
            kw2.setdefault("build", str(r.choice(["average", "average", "slim", "heavy", "broad", "curvy"])))
        else:
            kw2.pop("build", None)
        self.figure(x, y, hh, pose=pose, style=style, kind=kind, body=body, view=view,
                    flip=bool(r.random() < .5), seed=int(r.integers(1 << 30)), phase=float(r.random()),
                    colors=cl, **kw2)
        out.append((x, y, hh))
    return out


Layer.figure = figure
Layer.crowd = crowd

__all__ = ["figure_skeleton", "animal_skeleton", "figure_shape", "figure_joints", "figure_parts", "figure_sdf", "POSES",
           "ANIMAL_POSES", "SKIN", "HAIR", "HAIRSTYLES", "BODIES", "VIEWS"]
