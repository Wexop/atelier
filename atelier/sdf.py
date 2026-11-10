"""Light 3D engine: vectorised sphere tracing of signed distance fields (numpy, CPU).

s = Scene()
s.add(sphere(0.5).move(0, 0.5, 0), Mat("#c33", rough=0.3))
s.add(plane(), Mat(wood().ramp("wood"), proj="xz", tex_scale=0.5))
s.sun((1, 2, 1)); s.camera((0, 1.2, 4), (0, 0.5, 0), fov=35)
r = c.render(s, aa=2)          # paints the current layer; r.depth / r.normal / r.id / r.mask(id)
Units are arbitrary (think metres); y is up. Colours in sRGB, lighting in linear.
"""
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

from .core import F32, Layer, color as _col, to_linear, to_srgb, premul, CONFIG
from .field import Field, Ctx
from .fx import tonemap_linear

# ------------------------------------------------------------------ small vector helpers
def _v(x):
    return np.asarray(x, F32)


def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _len(v):
    return np.sqrt((v * v).sum(-1))


def _dot(a, b):
    return (a * b).sum(-1)


def _rot(rx=0.0, ry=0.0, rz=0.0):
    """Rotation matrix from Euler degrees (applied x, then y, then z)."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    X = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Z = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return (Z @ Y @ X).astype(F32)


# ------------------------------------------------------------------ 3D noise
_G3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1], [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], F32)


def _hash3(ix, iy, iz, seed):
    h = (ix.view(np.uint32) * np.uint32(0x8DA6B343) ^ iy.view(np.uint32) * np.uint32(0xD8163841)
         ^ iz.view(np.uint32) * np.uint32(0xCB1AB31F) ^ np.uint32((seed * 0x27D4EB2F + 0x165667B1) & 0xFFFFFFFF))
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h


def perlin3(p, seed=0):
    """3D gradient noise ~[-1, 1] at points p (N, 3)."""
    f = np.floor(p)
    i = f.astype(np.int32)
    r = (p - f).astype(F32)
    u = r * r * r * (r * (r * 6 - 15) + 10)
    out = 0
    acc = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                g = _G3[_hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed) & np.uint32(15)]
                acc[(dx, dy, dz)] = (g[:, 0] * (r[:, 0] - dx) + g[:, 1] * (r[:, 1] - dy) + g[:, 2] * (r[:, 2] - dz))
    lerp = lambda a, b, t: a + (b - a) * t
    x0 = lerp(acc[0, 0, 0], acc[1, 0, 0], u[:, 0]); x1 = lerp(acc[0, 1, 0], acc[1, 1, 0], u[:, 0])
    x2 = lerp(acc[0, 0, 1], acc[1, 0, 1], u[:, 0]); x3 = lerp(acc[0, 1, 1], acc[1, 1, 1], u[:, 0])
    y0 = lerp(x0, x1, u[:, 1]); y1 = lerp(x2, x3, u[:, 1])
    return lerp(y0, y1, u[:, 2]) * 1.1


_TLS = threading.local()      # .fw = per-point footprint while a material is evaluated (texture AA)


def noise3(scale=1.0, octaves=4, seed=0, gain=0.5, lacunarity=2.0):
    """3D fbm, returns f(p (N,3), fw=None) -> (N,) in ~[0,1]. For displace() and solid textures.
    scale = float or (sx, sy, sz) (anisotropic). fw = footprint per point (world units): octaves finer
    than ~2 per footprint fade to their mean (band-limited, no aliasing). The path tracer sets fw
    automatically while it evaluates materials (so f(p, n) colour callables are anti-aliased too)."""
    sv = np.asarray(scale, F32) * np.ones(3, F32) if np.ndim(scale) else F32(scale)
    smax = float(np.max(sv))
    def f(p, fw=None):
        p = np.asarray(p, F32)
        if fw is None:
            fw = getattr(_TLS, "fw", None)
            if fw is not None and len(fw) != len(p):
                fw = None
        q = p * sv
        tot, amp, norm, freq = np.zeros(len(p), F32), 1.0, 0.0, smax
        for o in range(octaves):
            if fw is None:
                tot = tot + amp * perlin3(q + F32(o * 13.17), seed + o)
            else:
                w = np.clip(2 - 4 * fw * F32(freq), 0, 1).astype(F32)
                m = w > 0
                if m.all():
                    tot = tot + amp * w * perlin3(q + F32(o * 13.17), seed + o)
                elif m.any():
                    tot[m] += amp * w[m] * perlin3(q[m] + F32(o * 13.17), seed + o)
            norm += amp
            amp *= gain
            freq *= lacunarity
            q = q * F32(lacunarity)
        return np.clip(0.5 + 0.6 * tot / norm, 0, 1)
    return f


# ------------------------------------------------------------------ SDF nodes
class SDF:
    """Signed distance function node. f(p: (N,3)) -> (N,). bounds = (centre, radius) or None."""

    def __init__(self, f, bounds=None, lip=1.0, xf=None):
        self.f, self.bounds, self.lip = f, bounds, lip
        self.xf = xf        # world -> object-space mapping (None = identity), for Mat(local=True)

    def __call__(self, p):
        return self.f(np.asarray(p, F32))

    def _x(self, g):
        """Object-space chain: world point -> g -> this node's local mapping."""
        xf = self.xf
        return g if xf is None else (lambda p: xf(g(p)))

    def local(self, p):
        """Map world points to object space (follows .move/.rotate/.scale/...)."""
        p = np.asarray(p, F32)
        return p if self.xf is None else self.xf(p)

    # transforms
    def move(self, x=0.0, y=0.0, z=0.0):
        t = _v((x, y, z))
        b = None if self.bounds is None else (self.bounds[0] + t, self.bounds[1])
        g = lambda p: p - t
        return SDF(lambda p: self.f(g(p)), b, self.lip, self._x(g))

    def rotate(self, x=0.0, y=0.0, z=0.0):
        """Rotate by Euler angles in degrees."""
        R = _rot(x, y, z)
        b = None if self.bounds is None else (R @ self.bounds[0], self.bounds[1])
        g = lambda p: p @ R
        return SDF(lambda p: self.f(g(p)), b, self.lip, self._x(g))

    def scale(self, k):
        """Uniform scale (k float) or non-uniform (kx, ky, kz) — the latter is approximate."""
        if np.ndim(k):
            kv = _v(k)
            m = float(kv.min())
            b = None if self.bounds is None else (self.bounds[0] * kv, self.bounds[1] * float(kv.max()))
            return SDF(lambda p: self.f(p / kv) * m, b, self.lip, self._x(lambda p: p / kv))
        k = float(k)
        b = None if self.bounds is None else (self.bounds[0] * k, self.bounds[1] * k)
        return SDF(lambda p: self.f(p / k) * k, b, self.lip, self._x(lambda p: p / F32(k)))

    def mirror(self, axis="x"):
        i = "xyz".index(axis)
        def g(p):
            q = p.copy()
            q[:, i] = np.abs(q[:, i])
            return q
        b = None if self.bounds is None else (_v([0 if j == i else self.bounds[0][j] for j in range(3)]),
                                              self.bounds[1] + abs(float(self.bounds[0][i])))
        return SDF(lambda p: self.f(g(p)), b, self.lip, self._x(g))

    def twist(self, k):
        """Twist around y: k radians per unit height."""
        def g(p):
            a = p[:, 1] * F32(k)
            c, s = np.cos(a), np.sin(a)
            return np.stack([c * p[:, 0] - s * p[:, 2], p[:, 1], s * p[:, 0] + c * p[:, 2]], 1)
        b = None if self.bounds is None else (self.bounds[0], self.bounds[1] * 1.2)
        return SDF(lambda p: self.f(g(p)), b, self.lip * 0.6, self._x(g))

    def bend(self, k):
        """Bend in the xy plane (k = curvature)."""
        def g(p):
            a = p[:, 0] * F32(k)
            c, s = np.cos(a), np.sin(a)
            return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)
        b = None if self.bounds is None else (self.bounds[0], self.bounds[1] * 1.3)
        return SDF(lambda p: self.f(g(p)), b, self.lip * 0.6, self._x(g))

    def repeat(self, period, count=None):
        """Infinite (or ±count) repetition with spacing period (float or (px, py, pz); 0 = no repeat)."""
        per = _v(period) * np.ones(3, F32)
        cnt = None if count is None else _v(count) * np.ones(3, F32)
        def g(p):
            safe = np.where(per > 0, per, 1)
            idx = np.round(p / safe)
            if cnt is not None:
                idx = np.clip(idx, -cnt, cnt)
            return np.where(per > 0, p - safe * idx, p)
        b = None
        if cnt is not None and self.bounds is not None:
            b = (self.bounds[0], self.bounds[1] + float(np.linalg.norm(per * cnt)))
        return SDF(lambda p: self.f(g(p)), b, self.lip, self._x(g))

    def onion(self, thickness):
        """Hollow shell of given thickness."""
        return SDF(lambda p: np.abs(self.f(p)) - F32(thickness / 2), self.bounds, self.lip, self.xf)

    def round(self, r):
        b = None if self.bounds is None else (self.bounds[0], self.bounds[1] + r)
        return SDF(lambda p: self.f(p) - F32(r), b, self.lip, self.xf)

    def displace(self, fn, amount=0.05):
        """Add amount*(fn(p)-0.5) to the surface; fn = noise3(...) or any p -> [0,1]."""
        b = None if self.bounds is None else (self.bounds[0], self.bounds[1] + abs(amount))
        return SDF(lambda p: self.f(p) + F32(amount) * (fn(p) - 0.5), b, self.lip * 0.7, self.xf)

    def elongate(self, x=0.0, y=0.0, z=0.0):
        h = _v((x, y, z)) / 2
        g = lambda p: p - np.clip(p, -h, h)
        b = None if self.bounds is None else (self.bounds[0], self.bounds[1] + float(np.linalg.norm(h)))
        return SDF(lambda p: self.f(g(p)), b, self.lip, self._x(g))

    # booleans
    def union(self, o, k=0.0):
        return _boolean(self, o, k, "u")

    def inter(self, o, k=0.0):
        return _boolean(self, o, k, "i")

    def sub(self, o, k=0.0):
        return _boolean(self, o, k, "s")

    __or__ = lambda a, b: a.union(b)
    __and__ = lambda a, b: a.inter(b)
    __sub__ = lambda a, b: a.sub(b)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _merge_bounds(A, B, extra=0.0):
    if A is None or B is None:
        return None
    (ca, ra), (cb, rb) = A, B
    d = float(np.linalg.norm(cb - ca))
    if d + rb <= ra:
        return ca, ra + extra
    if d + ra <= rb:
        return cb, rb + extra
    r = (d + ra + rb) / 2
    c = ca + (cb - ca) * ((r - ra) / max(d, 1e-9))
    return c.astype(F32), r + extra


def _boolean(a, b, k, kind):
    k = float(k)
    if kind == "u":
        f = (lambda p: np.minimum(a.f(p), b.f(p))) if k <= 0 else (lambda p: _smin(a.f(p), b.f(p), F32(k)))
        bounds = _merge_bounds(a.bounds, b.bounds, k)
    elif kind == "i":
        f = (lambda p: np.maximum(a.f(p), b.f(p))) if k <= 0 else (lambda p: -_smin(-a.f(p), -b.f(p), F32(k)))
        bounds = a.bounds if a.bounds is not None else b.bounds
    else:
        f = (lambda p: np.maximum(a.f(p), -b.f(p))) if k <= 0 else (lambda p: -_smin(-a.f(p), b.f(p), F32(k)))
        bounds = a.bounds
    return SDF(f, bounds, min(a.lip, b.lip), a.xf)


# ------------------------------------------------------------------ primitives (centred at origin)
def sphere(r=0.5):
    return SDF(lambda p: _len(p) - F32(r), (_v((0, 0, 0)), r))


def box(w=1.0, h=1.0, d=1.0):
    """Box with full sizes w (x), h (y), d (z)."""
    b = _v((w, h, d)) / 2
    def f(p):
        q = np.abs(p) - b
        return _len(np.maximum(q, 0)) + np.minimum(q.max(1), 0)
    return SDF(f, (_v((0, 0, 0)), float(np.linalg.norm(b))))


def round_box(w=1.0, h=1.0, d=1.0, r=0.05):
    return box(w - 2 * r, h - 2 * r, d - 2 * r).round(r)


def cylinder(r=0.5, h=1.0):
    """Capped cylinder along y, height h."""
    def f(p):
        dx = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - F32(r)
        dy = np.abs(p[:, 1]) - F32(h / 2)
        return np.minimum(np.maximum(dx, dy), 0) + np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    return SDF(f, (_v((0, 0, 0)), math.hypot(r, h / 2)))


def capsule(a=(0, -0.5, 0), b=(0, 0.5, 0), r=0.2):
    A, B = _v(a), _v(b)
    ba = B - A
    def f(p):
        pa = p - A
        t = np.clip(pa @ ba / float(ba @ ba), 0, 1)
        return _len(pa - t[:, None] * ba) - F32(r)
    return SDF(f, ((A + B) / 2, float(np.linalg.norm(ba)) / 2 + r))


def torus(R=0.5, r=0.15):
    """Torus in the xz plane."""
    def f(p):
        qx = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - F32(R)
        return np.sqrt(qx * qx + p[:, 1] ** 2) - F32(r)
    return SDF(f, (_v((0, 0, 0)), R + r))


def cone(r1=0.5, h=1.0, r2=0.0):
    """Capped cone along y (base radius r1 at -h/2, top radius r2 at +h/2)."""
    hh = h / 2
    k1 = np.array([r2, hh], F32)
    k2 = np.array([r2 - r1, 2 * hh], F32)
    k2d = float(k2 @ k2)
    def f(p):
        qx = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
        qy = p[:, 1]
        cax = qx - np.minimum(qx, np.where(qy < 0, F32(r1), F32(r2)))
        cay = np.abs(qy) - F32(hh)
        t = np.clip(((k1[0] - qx) * k2[0] + (k1[1] - qy) * k2[1]) / k2d, 0, 1)
        cbx = qx - k1[0] + k2[0] * t
        cby = qy - k1[1] + k2[1] * t
        s = np.where((cbx < 0) & (cay < 0), -1.0, 1.0).astype(F32)
        return s * np.sqrt(np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby))
    return SDF(f, (_v((0, 0, 0)), math.hypot(max(r1, r2), hh)))


def plane(y=0.0, normal=(0, 1, 0)):
    n = _norm(_v(normal))
    return SDF(lambda p: p @ n - F32(y), None)


def ellipsoid(rx=0.5, ry=0.3, rz=0.4):
    r = _v((rx, ry, rz))
    def f(p):
        k0 = _len(p / r)
        k1 = _len(p / (r * r))
        return k0 * (k0 - 1) / np.maximum(k1, 1e-9)
    return SDF(f, (_v((0, 0, 0)), float(r.max())), 0.9)


def _seg_dist2(px, py, V, closed):
    """Squared distance from points to a 2D polyline/polygon + sign (inside = -1) if closed.
    Vectorised over segments."""
    A = V if closed else V[:-1]
    B = np.roll(V, -1, 0) if closed else V[1:]
    E = (B - A).astype(F32)
    ee = np.maximum((E * E).sum(1), 1e-12).astype(F32)
    N = px.shape[0]
    d = np.empty(N, F32)
    s = np.ones(N, F32)
    step = max(1, 2_000_000 // max(len(A), 1))
    for a in range(0, N, step):
        x, y = px[a:a + step, None], py[a:a + step, None]
        wx, wy = x - A[None, :, 0], y - A[None, :, 1]
        t = np.clip((wx * E[:, 0] + wy * E[:, 1]) / ee, 0, 1)
        bx, by = wx - E[:, 0] * t, wy - E[:, 1] * t
        d[a:a + step] = (bx * bx + by * by).min(1)
        if closed:
            c1, c2 = y >= A[None, :, 1], y < B[None, :, 1]
            c3 = E[:, 0] * wy > E[:, 1] * wx
            flips = ((c1 & c2 & c3) | (~c1 & ~c2 & ~c3)).sum(1)
            s[a:a + step] = np.where(flips % 2 == 1, -1.0, 1.0)
    return d, s


class _Grid2D:
    """Pre-computed 2D distance field over a box, sampled bicubically; exact-ish lower bound outside."""

    def __init__(self, fn, x0, x1, y0, y1, res=1536):
        sc = res / max(x1 - x0, y1 - y0)
        self.nx, self.ny = max(8, int((x1 - x0) * sc) + 1), max(8, int((y1 - y0) * sc) + 1)
        self.x0, self.y0, self.sc = x0, y0, sc
        self.x1, self.y1 = x0 + (self.nx - 1) / sc, y0 + (self.ny - 1) / sc
        gx, gy = np.meshgrid(np.linspace(self.x0, self.x1, self.nx, dtype=F32),
                             np.linspace(self.y0, self.y1, self.ny, dtype=F32))
        self.tex = fn(gx.ravel(), gy.ravel()).reshape(self.ny, self.nx).astype(F32)

    def __call__(self, x, y):
        cx, cy = np.clip(x, self.x0, self.x1), np.clip(y, self.y0, self.y1)
        n = x.size
        cols = 1024 if n > 1024 else max(n, 1)
        rows = -(-n // cols)
        pad = rows * cols - n
        mx = np.pad((cx - self.x0) * self.sc, (0, pad)).reshape(rows, cols).astype(F32)
        my = np.pad((cy - self.y0) * self.sc, (0, pad)).reshape(rows, cols).astype(F32)
        d = cv2.remap(self.tex, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).ravel()[:n]
        return d + np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def _dense(points, smooth):
    P = np.asarray(points, np.float64)
    if smooth and len(P) > 2:
        from .brush import _catmull
        P, _ = _catmull(P, 16)
    return P.astype(F32)


def lathe(profile, thickness=None, smooth=True):
    """Solid of revolution around y from a profile [(radius, y), ...] (bottom to top).
    thickness=None: solid (profile closed through the axis); else a shell of that thickness
    (glasses, bowls, vases). smooth=True passes a spline through the profile points."""
    V = _dense(profile, smooth)
    if thickness is None:
        V = np.vstack([[0, V[0, 1]], V, [0, V[-1, 1]]]).astype(F32)
    rmax = float(np.abs(V[:, 0]).max())
    y0, y1 = float(V[:, 1].min()), float(V[:, 1].max())
    def exact(qx, qy):
        if thickness is None:
            d2, s = _seg_dist2(qx, qy, V, True)
            return s * np.sqrt(d2)
        d2, _ = _seg_dist2(qx, qy, V, False)
        return np.sqrt(d2) - F32(thickness / 2)
    m = 0.15 * max(rmax, y1 - y0) + (thickness or 0)
    grid = _Grid2D(exact, 0.0, rmax + m, y0 - m, y1 + m)
    def f(p):
        return grid(np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2), p[:, 1])
    return SDF(f, (_v((0, (y0 + y1) / 2, 0)), math.hypot(rmax, (y1 - y0) / 2) + (thickness or 0)))


def extrude(points, depth=0.2, smooth=False):
    """Extrude a 2D polygon [(x, y), ...] (xy plane) along z by depth (centred)."""
    V = _dense(points, smooth)
    ext = float(np.abs(V).max())
    def exact(x, y):
        d2, s = _seg_dist2(x, y, V, True)
        return s * np.sqrt(d2)
    lo, hi = V.min(0), V.max(0)
    m = 0.15 * float((hi - lo).max())
    grid = _Grid2D(exact, float(lo[0]) - m, float(hi[0]) + m, float(lo[1]) - m, float(hi[1]) + m)
    def f(p):
        dx = grid(p[:, 0], p[:, 1])
        dz = np.abs(p[:, 2]) - F32(depth / 2)
        return np.minimum(np.maximum(dx, dz), 0) + np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dz, 0) ** 2)
    return SDF(f, (_v((float(V[:, 0].mean()), float(V[:, 1].mean()), 0)), math.hypot(ext * 2, depth / 2)))


# ------------------------------------------------------------------ materials
def _proj_eval(c, p, n, proj, ts, fw=None):
    """Sample a 2D Field on 3D points (triplanar or planar projection). fw (world footprint per
    point) buckets points by pixel size so the Field's own octave cut-off band-limits it."""
    ts = F32(ts)
    def samp(u, v, px):
        out = np.asarray(c.fn(u * ts, v * ts, Ctx(1.0, px, 1.0)), F32)
        if out.ndim == 2:
            return out[:, :3] if out.shape[1] >= 3 else np.repeat(out[:, :1], 3, 1)
        return np.stack([np.broadcast_to(out, u.shape)] * 3, 1)
    def proj_all(p, n, px):
        if proj == "tri":
            w = np.abs(n) ** 4
            w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
            return (samp(p[:, 2], -p[:, 1], px) * w[:, :1] + samp(p[:, 0], p[:, 2], px) * w[:, 1:2]
                    + samp(p[:, 0], -p[:, 1], px) * w[:, 2:3])
        i, j = {"xz": (0, 2), "xy": (0, 1), "zy": (2, 1)}[proj]
        return samp(p[:, i], (-p[:, j] if j == 1 else p[:, j]), px)
    if fw is None:
        return proj_all(p, n, 1e-3)
    px = np.maximum(np.asarray(fw, F32) * ts, 1e-4)
    lev = np.floor(np.log2(px) * 2).astype(np.int32)       # half-octave buckets
    out = np.empty((len(p), 3), F32)
    for L in np.unique(lev):
        m = lev == L
        out[m] = proj_all(p[m], n[m], float(2 ** (L / 2)))
    return out


def _src_eval(c, p, n, proj, ts, fw=None, chans=3):
    """Raw (unclipped) values of a colour/scalar source: Field | f(p, n) | constant."""
    if isinstance(c, Field):
        v = _proj_eval(c, p, n, proj, ts, fw)
        return v if chans == 3 else v[:, 0]
    if callable(c):
        v = np.asarray(c(p, n), F32)
        if chans == 3:
            return v[:, :3] if v.ndim == 2 else np.stack([np.broadcast_to(v, (len(p),))] * 3, 1)
        return v[:, 0] if v.ndim == 2 else np.broadcast_to(v, (len(p),)).astype(F32)
    if chans == 3:
        return np.broadcast_to(_col(c)[:3], (len(p), 3)).astype(F32)
    return np.full(len(p), float(c), F32)


_TETRA = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], F32)


class Mat:
    """Material. color: colour | 2D Field (projected) | f(p, n) -> (N,3) sRGB.
    rough 0..1 | Field | f(p, n) -> (N,) (roughness map), metal 0..1, glass 0..1 (+ior; colour = tint
    after 0.1 units of travel in path tracing), sss 0..1 (wrap lighting, sss_color tint), emit colour (+power).
    proj: 'tri' (triplanar) | 'xz' | 'xy' | 'zy'; tex_scale = repeats/unit.
    local=True: textures/bump/wear in object space (follow .move/.rotate/.scale of the object).
    bump: micro-normal strength (~0.1-1) of noise3(bump_scale) or bump_fn(p)->[0,1].
    sheen 0..1 (+sheen_color): cloth/velvet rim. wear: imperfection list [smudges(), scratches(), dust(), edge_wear()]."""

    def __init__(self, color="#cccccc", rough=0.5, metal=0.0, glass=0.0, ior=1.45, sss=0.0, sss_color=None,
                 emit=None, power=1.0, proj="tri", tex_scale=1.0, spec=0.5, bump=0.0, bump_scale=60.0,
                 bump_fn=None, local=False, sheen=0.0, sheen_color=None, wear=None):
        self.color, self.rough, self.metal, self.glass, self.ior = color, rough, metal, glass, ior
        self.sss, self.sss_color, self.emit, self.power = sss, sss_color, emit, power
        self.proj, self.tex_scale, self.spec = proj, tex_scale, spec
        self.bump, self.bump_scale, self.local = bump, bump_scale, local
        self.bump_fn = bump_fn if bump_fn is not None else noise3(bump_scale, 3, seed=11)
        self.sheen, self.sheen_color = sheen, sheen_color
        self.wear = [] if wear is None else (list(wear) if isinstance(wear, (list, tuple)) else [wear])

    def albedo(self, p, n):
        return to_linear(np.clip(_src_eval(self.color, p, n, self.proj, self.tex_scale), 0, 1))

    def eval(self, p, n, sdf=None, fw=None, bump=True):
        """Per-point material: dict(alb (N,3) linear, rough (N,), n (bumped normal)). fw = world footprint
        per point (band-limits textures). sdf = the object (object-space textures, edge_wear)."""
        p = np.asarray(p, F32)
        N = len(p)
        xf = sdf.xf if (self.local and sdf is not None) else None
        pt = p if xf is None else xf(p)
        nt = n
        if xf is not None:
            nt = _norm(xf(p + n * F32(1e-3)) - pt)
        fwl = None if fw is None else np.asarray(fw, F32)
        if fwl is not None and xf is not None:      # footprint in object units (scale-aware)
            fwl = fwl * _len(xf(p + _v((1e-3, 0, 0))) - pt) / F32(1e-3)
        _TLS.fw = fwl
        try:
            alb = np.clip(_src_eval(self.color, pt, nt, self.proj, self.tex_scale, fwl), 0, 1)
            rough = _src_eval(self.rough, pt, nt, self.proj, self.tex_scale, fwl, 1)
            nb = n
            if self.bump and bump:
                e = F32(0.05 / self.bump_scale)
                g = 0
                for k in _TETRA:
                    q = p + k * e
                    g = g + k * np.asarray(self.bump_fn(q if xf is None else xf(q)), F32)[:, None]
                g = g / (4 * e) / F32(self.bump_scale)
                g = g - n * (g * n).sum(1, keepdims=True)
                nb = _norm(n - F32(self.bump) * g)
            ctx = dict(p=pt, pw=p, n=n, nl=nt, alb=alb, rough=np.array(rough, F32), sdf=sdf,
                       fw=(np.zeros(N, F32) if fwl is None else fwl))
            for w in self.wear:
                w(ctx)
        finally:
            _TLS.fw = None
        return dict(alb=to_linear(np.clip(ctx["alb"], 0, 1)), rough=np.clip(ctx["rough"], 0, 1).astype(F32), n=nb)


# ------------------------------------------------------------------ imperfections (Mat(wear=[...]))
def _fade(fw, freq):
    """1 while a feature of frequency freq is resolved, -> 0 when it gets smaller than the footprint."""
    return np.clip(1.5 - fw * F32(freq), 0, 1)


def smudges(amount=0.5, scale=5.0, seed=0):
    """Greasy smudges / fingerprints: blotchy roughness increase and slight dulling (glass, metal, glaze)."""
    blot, fine = noise3(scale, 3, seed=seed + 101), noise3(scale * 14, 2, seed=seed + 102)
    def f(c):
        m = np.clip((blot(c["p"]) - 0.47) * 4, 0, 1) * (0.55 + 0.45 * fine(c["p"]))
        c["rough"] = c["rough"] + F32(amount * 0.45) * m * (1 - c["rough"])
        c["alb"] = c["alb"] * (1 - F32(0.06 * amount) * m)[:, None]
    return f


def scratches(amount=0.5, density=40.0, seed=0):
    """Fine random scratches (short thin streaks in 3 directions): brighter, rougher lines."""
    rng = np.random.default_rng(seed + 7)
    Rs = [_rot(*rng.uniform(0, 360, 3)) for _ in range(3)]
    gate = noise3(density / 10, 2, seed=seed + 103)
    k = F32(density)
    def f(c):
        p, fw = c["p"], c["fw"]
        acc = np.zeros(len(p), F32)
        for i, R in enumerate(Rs):
            q = (p @ R) * np.array([k / 12, k / 12, k], F32)
            v = np.abs(perlin3(q, seed + 200 + i))
            acc = np.maximum(acc, np.clip(1 - v / 0.05, 0, 1))
        g = np.clip((gate(p) - 0.5) * 5, 0, 1)
        fd = _fade(fw, density * 2)
        s = (acc * fd + 0.05 * (1 - fd)) * g
        c["rough"] = c["rough"] + F32(amount * 0.5) * s * (1 - c["rough"])
        c["alb"] = np.clip(c["alb"] * (1 + F32(0.25 * amount) * s)[:, None], 0, 1)
    return f


def dust(amount=0.3, color="#c9c2b6", seed=0):
    """Dust: a faint film plus specks, mostly on up-facing surfaces (world up)."""
    film, specks = noise3(4, 3, seed=seed + 104), noise3(350, 1, seed=seed + 105)
    dc = _col(color)[:3].astype(F32)
    def f(c):
        p = c["p"]
        up = np.clip(c["n"][:, 1], 0, 1) ** 2
        fd = _fade(c["fw"], 350)
        sp = np.clip((specks(p) - 0.7) * 6, 0, 1) * fd + 0.04 * (1 - fd)
        cov = np.clip(F32(amount) * (0.25 + 0.75 * up) * (0.35 * film(p) + sp), 0, 1)
        c["alb"] = c["alb"] * (1 - cov[:, None]) + dc * cov[:, None]
        c["rough"] = c["rough"] + (1 - c["rough"]) * cov
    return f


def edge_wear(amount=0.5, width=0.01, color=None, seed=0):
    """Worn convex edges (needs the object SDF): lighter/other colour (color=None -> lightened albedo)
    and changed roughness, broken up by noise. width = how far the wear reaches (world units)."""
    br = noise3(0.3 / width, 3, seed=seed + 106)
    wc = None if color is None else _col(color)[:3].astype(F32)
    def f(c):
        if c["sdf"] is None:
            return
        d_in = np.asarray(c["sdf"](c["pw"] - c["n"] * F32(width)), F32)
        edge = np.clip(1 + d_in / F32(width), 0, 1)
        m = np.clip((edge * (0.6 + amount) + (br(c["p"]) - 0.5) * 1.2 - 0.35) * 3, 0, 1) * F32(min(1, amount * 1.5))
        tgt = np.clip(c["alb"] * 1.6 + 0.06, 0, 1) if wc is None else np.broadcast_to(wc, c["alb"].shape)
        c["alb"] = c["alb"] * (1 - m[:, None]) + tgt * m[:, None]
        c["rough"] = c["rough"] * (1 - m) + np.clip(c["rough"] * 0.7 + 0.15, 0, 1) * m
    return f


# ------------------------------------------------------------------ render result
class Render:
    """Output of a render: rgb (sRGB, H×W×3), linear, alpha, depth (inf = sky), normal, id (-1 = sky)."""

    def __init__(self, rgb, linear, alpha, depth, normal, ids):
        self.rgb, self.linear, self.alpha, self.depth, self.normal, self.id = rgb, linear, alpha, depth, normal, ids

    def mask(self, obj_id, soft=True):
        m = (self.id == obj_id).astype(F32)
        return cv2.GaussianBlur(m, (0, 0), 0.7) if soft else m

    def focus_at(self, x, y, s=1.0):
        """Depth at canvas point (x, y) — use as depth_of_field(focus=...)."""
        H, W = self.depth.shape
        xi, yi = min(W - 1, max(0, int(x * s))), min(H - 1, max(0, int(y * s)))
        d = self.depth[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else 1e3

    @property
    def rgba(self):
        return np.concatenate([self.rgb, self.alpha[..., None]], -1)


# ------------------------------------------------------------------ scene
class Scene:
    def __init__(self):
        self.objects: list = []      # (sdf, mat)
        self.lights: list = []
        self.cam = dict(pos=_v((0, 1, 4)), target=_v((0, 0.5, 0)), fov=40.0, up=_v((0, 1, 0)))
        self.sky_cfg = dict(zenith=_v(to_linear(_col("#5d86c8")[:3])), horizon=_v(to_linear(_col("#d6e2ee")[:3])),
                            ground=_v(to_linear(_col("#4a4440")[:3])), power=1.0)
        self.fog_cfg = None
        self.steps, self.max_dist, self.shadow_steps = 160, 60.0, 48
        self.bg = None

    # building
    def add(self, sdf, mat=None, **kw):
        """Add an object; returns its id (for r.mask(id))."""
        if mat is None:
            mat = Mat(**kw) if kw else Mat()
        elif not isinstance(mat, Mat):
            mat = Mat(mat, **kw)
        self.objects.append((sdf, mat))
        return len(self.objects) - 1

    def camera(self, pos=(0, 1, 4), target=(0, 0.5, 0), fov=40.0, up=(0, 1, 0)):
        """fov = vertical field of view in degrees."""
        self.cam = dict(pos=_v(pos), target=_v(target), fov=float(fov), up=_v(up))
        return self

    def sun(self, dir=(-1.0, 2.0, 1.0), color="#fff3e0", power=3.0, soft=0.04):
        """Directional light coming FROM direction `dir`; soft = penumbra (0 = hard)."""
        self.lights.append(dict(kind="sun", dir=_norm(_v(dir)), col=to_linear(_col(color)[:3]) * F32(power),
                                soft=max(soft, 1e-3)))
        return self

    def point(self, pos=(1, 2, 1), color="#ffffff", power=5.0, radius=0.05):
        self.lights.append(dict(kind="point", pos=_v(pos), col=to_linear(_col(color)[:3]) * F32(power),
                                soft=max(radius, 1e-3)))
        return self

    def area(self, pos=(0, 2, 2), size=(1.0, 1.0), normal=(0, -1, -1), color="#ffffff", power=6.0, n=(3, 2),
             pattern=None, visible=False):
        """Rectangular area light approximated by n[0]*n[1] soft point lights (render); sampled over its
        surface in pathtrace, where it also shows in reflections. pattern (pathtrace): None | "window"
        (panes + mullions) | "softbox" | f(u, v) -> [0,1]; visible: also seen by camera rays."""
        nv = _norm(_v(normal))
        t1 = _norm(np.cross(nv, _v((0, 1, 0)) if abs(nv[1]) < 0.9 else _v((1, 0, 0))))
        t2 = np.cross(nv, t1)
        k = n[0] * n[1]
        pts = []
        for i in range(n[0]):
            for j in range(n[1]):
                u = ((i + 0.5) / n[0] - 0.5) * size[0]
                v = ((j + 0.5) / n[1] - 0.5) * size[1]
                pts.append(_v(pos) + t1 * u + t2 * v)
        self.lights.append(dict(kind="area", pos=_v(pos), pts=pts, col=to_linear(_col(color)[:3]) * F32(power / k),
                                soft=max(size) * 0.5, rad=to_linear(_col(color)[:3]) * F32(power), t1=t1.astype(F32),
                                t2=t2.astype(F32), nv=nv, size=(float(size[0]), float(size[1])), pattern=pattern,
                                visible=visible))
        return self

    def studio(self, key=(-2.4, 1.9, 1.2), target=(0, 0.25, 0), size=(1.2, 1.6), power=45.0, color="#ffeedd",
               window=True, fill=0.2, bg="#2a2622"):
        """Photo-studio set: a big key panel at `key` aimed at `target` ("window" panes or a softbox),
        a dim cool fill panel opposite (fill = relative power), a dark gradient environment. The panels
        light the scene and show up in glass/metal reflections (pathtrace)."""
        k, t = _v(key), _v(target)
        self.area(k, size, t - k, color, power, pattern="window" if window else "softbox")
        if fill > 0:
            fp = _v((2 * t[0] - k[0], t[1] + (k[1] - t[1]) * 0.6, k[2]))   # mirrored across the target
            self.area(fp, (size[0] * 1.6, size[1]), t - fp, "#dfe8ff", power * fill, pattern="softbox")
        b = np.asarray(_col(bg)[:3], F32)
        self.sky(tuple(np.clip(b * 0.8, 0, 1)), tuple(np.clip(b * 1.25, 0, 1)), tuple(np.clip(b * 0.5, 0, 1)),
                 power=1.0)
        return self

    def sky(self, zenith="#5d86c8", horizon="#d6e2ee", ground="#4a4440", power=1.0):
        self.sky_cfg = dict(zenith=to_linear(_col(zenith)[:3]), horizon=to_linear(_col(horizon)[:3]),
                            ground=to_linear(_col(ground)[:3]), power=float(power))
        return self

    def fog(self, color="#c8d0da", density=0.05):
        self.fog_cfg = (to_linear(_col(color)[:3]), float(density))
        return self

    # distance evaluation
    def _dist(self, p, exclude=(), skip_glass=False, want_id=False):
        best = np.full(len(p), np.inf, F32)
        ids = np.full(len(p), -1, np.int32) if want_id else None
        for k, (sdf, mat) in enumerate(self.objects):
            if k in exclude or (skip_glass and (mat.glass > 0.5 or mat.emit is not None)):  # no shadows from glass/lamps
                continue
            if sdf.bounds is not None:
                c, r = sdf.bounds
                lb = _len(p - c) - F32(r)
                m = lb < best
                if not m.any():
                    continue
                if m.all():
                    d = sdf.f(p)
                else:
                    d = np.where(m, np.inf, lb).astype(F32)
                    d[m] = sdf.f(p[m])
            else:
                d = sdf.f(p)
            if want_id:
                ids = np.where(d < best, k, ids)
            best = np.minimum(best, d)
        return (best, ids) if want_id else best

    def _lip(self):
        return min([o[0].lip for o in self.objects] + [1.0])

    def _march(self, o, d, pixang, exclude=(), tmin=1e-3, tmax=None):
        N = len(o)
        tmax = tmax or self.max_dist
        t = np.full(N, tmin, F32)
        hit = np.zeros(N, bool)
        idx = np.arange(N)
        step = F32(0.95 * self._lip())
        for _ in range(self.steps):
            if idx.size == 0:
                break
            ti = t[idx]
            dist = self._dist(o[idx] + d[idx] * ti[:, None], exclude)
            eps = np.maximum(F32(1e-4), ti * F32(pixang * 0.5))
            h = dist < eps
            hit[idx[h]] = True
            t[idx] = ti + dist * step
            keep = ~h & (t[idx] < tmax)
            idx = idx[keep]
        return t, hit

    def _normal(self, p, eps):
        e = eps[:, None]
        k = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], F32)
        n = 0
        for kk in k:
            n = n + kk * self._dist(p + kk * e)[:, None]
        return _norm(n)

    def _shadow(self, p, L, tmax, soft):
        """Soft shadow factor in [0,1] (IQ), glass objects don't cast shadows."""
        N = len(p)
        res = np.ones(N, F32)
        t = np.full(N, 0.02, F32)
        k = (1.0 / np.broadcast_to(np.asarray(soft, F32), (N,))).astype(F32)
        idx = np.arange(N)
        tm = np.broadcast_to(np.asarray(tmax, F32), (N,))
        for _ in range(self.shadow_steps):
            if idx.size == 0:
                break
            ti = t[idx]
            h = self._dist(p[idx] + L[idx] * ti[:, None], skip_glass=True)
            res[idx] = np.minimum(res[idx], k[idx] * h / ti)
            t[idx] = ti + np.clip(h, 0.01, 0.25)
            keep = (res[idx] > 0.002) & (t[idx] < tm[idx])
            idx = idx[keep]
        r = np.clip(res, 0, 1)
        return r * r * (3 - 2 * r)

    def _ao(self, p, n):
        occ = np.zeros(len(p), F32)
        sca = 1.0
        for i in range(5):
            h = F32(0.01 + 0.06 * i)
            d = self._dist(p + n * h)
            occ += (h - d) * F32(sca)
            sca *= 0.85
        return np.clip(1 - 2.5 * occ, 0, 1)

    def _env(self, d):
        """Sky gradient radiance (linear) for directions d, without the sun disc."""
        c = self.sky_cfg
        y = d[:, 1:2]
        up = np.clip(y, 0, 1) ** 0.5
        col = np.where(y >= 0, c["horizon"] * (1 - up) + c["zenith"] * up,
                       c["horizon"] * np.exp(np.minimum(y, 0) * 12) + c["ground"] * (1 - np.exp(np.minimum(y, 0) * 12)))
        return (col * F32(c["power"])).astype(F32)

    def _sky(self, d):
        col = self._env(d)
        for L in self.lights:
            if L["kind"] == "sun":
                cs = _dot(d, L["dir"])
                disk = np.clip((cs - 0.99985) / 0.0001, 0, 1) * 40 + np.clip(cs, 0, 1) ** 200 * 0.6
                col = col + L["col"] * disk[:, None]
        return col.astype(F32)

    def _irr(self, n):
        """Diffuse sky irradiance for normals n."""
        c = self.sky_cfg
        y = n[:, 1:2]
        a = 0.5 + 0.5 * y
        top = c["zenith"] * 0.6 + c["horizon"] * 0.4
        return ((top * a + c["ground"] * (1 - a) * 0.8 + c["horizon"] * 0.15) * F32(c["power"])).astype(F32)

    def _materials(self, p, n, ids):
        N = len(p)
        alb = np.zeros((N, 3), F32)
        par = {k: np.zeros(N, F32) for k in ("rough", "metal", "glass", "ior", "sss", "spec")}
        emit = np.zeros((N, 3), F32)
        sssc = np.zeros((N, 3), F32)
        for k in np.unique(ids):
            if k < 0:
                continue
            m = ids == k
            mat = self.objects[k][1]
            ev = mat.eval(p[m], n[m], self.objects[k][0])
            a = ev["alb"]
            alb[m] = a
            n[m] = ev["n"]
            for key in par:
                par[key][m] = ev["rough"] if key == "rough" else getattr(mat, key)
            if mat.emit is not None:
                emit[m] = to_linear(_col(mat.emit)[:3]) * F32(mat.power)
            sssc[m] = to_linear(_col(mat.sss_color)[:3]) if mat.sss_color is not None else np.sqrt(a)
        return alb, par, emit, sssc

    @staticmethod
    def _brdf(n, V, Ld, rad, sh, diff_c, F0, a2, kk, ndv, wrap, sssc):
        ndl_raw = _dot(n, Ld)[:, None]
        ndl = np.clip(ndl_raw, 0, 1)
        Hh = _norm(Ld + V)
        ndh = np.clip(_dot(n, Hh), 0, 1)[:, None]
        vdh = np.clip(_dot(V, Hh), 0, 1)[:, None]
        D = a2 / (math.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
        G = (ndl / (ndl * (1 - kk) + kk)) * (ndv / (ndv * (1 - kk) + kk))
        Fs = F0 + (1 - F0) * (1 - vdh) ** 5
        spec = D * G * Fs / (4 * ndv * np.maximum(ndl, 1e-3)) * ndl
        dif = np.clip((ndl_raw + wrap) / (1 + wrap), 0, 1)
        back = wrap * np.clip(-ndl_raw, 0, 1) * 0.35       # light bleeding through (sss)
        return rad * sh * (diff_c * dif + spec) + rad * diff_c * sssc * back * (0.5 + 0.5 * sh)

    def _shade(self, p, v, t, pixang, depth, exclude=()):
        """Shade hit points p seen along directions v. Returns linear rgb, normals, ids."""
        N = len(p)
        eps = np.maximum(F32(2e-4), t * F32(pixang * 0.5))
        n = self._normal(p, eps)
        _, ids = self._dist(p, exclude, want_id=True)
        alb, par, emit, sssc = self._materials(p, n, ids)
        rough = np.clip(par["rough"], 0.03, 1)[:, None]
        metal = par["metal"][:, None]
        glass = par["glass"][:, None]
        V = -v
        ndv = np.clip(_dot(n, V), 1e-3, 1)[:, None]
        F0 = 0.04 * (1 - metal) + alb * metal
        diff_c = alb * (1 - metal) * (1 - glass)
        pp = p + n * (eps[:, None] * 4)
        col = np.zeros((N, 3), F32)
        a2 = (rough ** 2) ** 2
        kk = (rough + 1) ** 2 / 8
        wrap = par["sss"][:, None]
        for L in self.lights:
            # one (soft) shadow per light; area lights shade from each sample point
            if L["kind"] == "sun":
                samples = [(np.broadcast_to(L["dir"], (N, 3)), np.broadcast_to(L["col"], (N, 3)))]
                Lc, tmax, soft = samples[0][0], np.full(N, 30.0, F32), L["soft"]
            else:
                dv = L["pos"] - p
                dist = _len(dv)
                Lc, tmax = dv / dist[:, None], dist
                soft = L["soft"] / np.maximum(dist, 0.1) if L["kind"] == "area" else L["soft"]
                samples = [(Lc, L["col"] * F32(len(L.get("pts", [0]))) / (dist[:, None] ** 2))]
            sh = np.ones(N, F32)
            if depth == 0:   # bounces skip shadow marching (cheap, barely visible)
                lit = _dot(n, Lc) > -0.3
                sh = np.zeros(N, F32)
                if lit.any():
                    sv = soft[lit] if np.ndim(soft) else soft
                    sh[lit] = self._shadow(pp[lit], np.ascontiguousarray(Lc[lit]), tmax[lit], sv)
            sh = sh[:, None]
            a2l = a2
            if L["kind"] == "area":   # widen the highlight by the light's apparent size
                al = np.clip(rough ** 2 + (L["soft"] / np.maximum(dist, 0.1))[:, None] * 0.5, 0, 1)
                a2l = al * al
            for Ld, rad in samples:
                col += self._brdf(n, V, Ld, rad, sh, diff_c, F0, a2l, kk, ndv, wrap, sssc)
        ao = self._ao(p, n)[:, None] if depth == 0 else np.ones((N, 1), F32)
        col += self._irr(n) * diff_c * ao
        # reflections
        R = _norm(v - 2 * _dot(v, n)[:, None] * n)
        Fr = F0 + (np.maximum(1 - rough, F0) - F0) * (1 - ndv) ** 5
        env = self._sky(R) * (1 - rough ** 2) + self._irr(n) * rough ** 2
        sharp = ((rough[:, 0] < 0.45) | (glass[:, 0] > 0)) & (depth == 0)
        if sharp.any():
            rc = self._trace(pp[sharp], R[sharp], pixang * 2, depth + 1)[0]
            blend = np.clip((0.45 - rough[sharp]) / 0.25, 0, 1)
            env[sharp] = rc * blend + env[sharp] * (1 - blend)
        col += env * Fr * ao * (1 - glass) + env * Fr * glass
        # transmission (one bounce): refract at entry, continue without this object
        gm = glass[:, 0] > 0
        if gm.any():
            ior = par["ior"][gm][:, None]
            vi, ni = v[gm], n[gm]
            cosi = np.clip(-_dot(vi, ni), 0, 1)[:, None]
            eta = 1 / ior
            k = 1 - eta * eta * (1 - cosi * cosi)
            T = _norm(eta * vi + (eta * cosi - np.sqrt(np.maximum(k, 0))) * ni)
            T = np.where(k > 0, T, vi)
            excl = tuple(set(exclude) | set(np.unique(ids[gm]).tolist()))
            start = p[gm] - ni * (eps[gm][:, None] * 6)
            if depth == 0:
                tc = self._trace(start, T, pixang, depth + 1, excl)[0]
            else:
                tc = self._sky(T)
            Fg = (F0[gm] + (1 - F0[gm]) * (1 - cosi) ** 5)
            tint = np.clip(alb[gm], 1e-3, 1) ** (1 / np.maximum(cosi, 0.12))   # longer path at grazing angles
            g = glass[gm]
            col[gm] = col[gm] + g * tc * tint * (1 - Fg)        # col already holds reflections + highlights
        col += emit
        return col, n, ids

    def _trace(self, o, d, pixang, depth=0, exclude=()):
        t, hit = self._march(o, d, pixang, exclude)
        col = self._sky(d)
        N = len(o)
        nrm = np.zeros((N, 3), F32)
        ids = np.full(N, -1, np.int32)
        if hit.any():
            ph = o[hit] + d[hit] * t[hit][:, None]
            c, nh, ih = self._shade(ph, d[hit], t[hit], pixang, depth, exclude)
            if self.fog_cfg is not None:
                fc, dens = self.fog_cfg
                f = np.exp(-dens * t[hit])[:, None]
                c = c * f + fc * (1 - f)
            col[hit] = c
            nrm[hit] = nh
            ids[hit] = ih
        t = np.where(hit, t, np.inf).astype(F32)
        return col, t, nrm, ids

    def rays(self, W, H):
        cam = self.cam
        fwd = _norm(cam["target"] - cam["pos"])
        right = _norm(np.cross(fwd, cam["up"]))
        up = np.cross(right, fwd)
        th = math.tan(math.radians(cam["fov"]) / 2)
        ys, xs = np.mgrid[0:H, 0:W].astype(F32)
        u = ((xs + 0.5) / W * 2 - 1) * th * W / H
        v = (1 - (ys + 0.5) / H * 2) * th
        d = fwd + u[..., None] * right + v[..., None] * up
        d = _norm(d.reshape(-1, 3)).astype(F32)
        o = np.broadcast_to(cam["pos"], d.shape).astype(F32)
        return o, d, 2 * th / H

    def render(self, w=640, h=400, aa=1, exposure=0.0, tonemap="agx", look=None, threads=None):
        """Render to a Render (rgb sRGB, linear, alpha, depth, normal, id) at w x h (x aa supersampling)."""
        if not self.lights:
            self.sun()
        W, H = int(w * aa), int(h * aa)
        o, d, pixang = self.rays(W, H)
        # z-depth along the camera axis (for DOF) = t * cos(angle to forward)
        fwd = _norm(self.cam["target"] - self.cam["pos"])
        N = len(d)
        chunk = 32768
        parts = [(i, min(N, i + chunk)) for i in range(0, N, chunk)]
        workers = threads or CONFIG.get("threads") or os.cpu_count() or 2
        col = np.zeros((N, 3), F32)
        dep = np.zeros(N, F32)
        nrm = np.zeros((N, 3), F32)
        ids = np.zeros(N, np.int32)
        def job(ab):
            a, b = ab
            c_, t_, n_, i_ = self._trace(o[a:b], d[a:b], pixang)
            col[a:b], dep[a:b], nrm[a:b], ids[a:b] = c_, t_ * _dot(d[a:b], fwd), n_, i_
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(job, parts))
        col = col.reshape(H, W, 3)
        dep = dep.reshape(H, W)
        nrm = nrm.reshape(H, W, 3)
        ids = ids.reshape(H, W)
        alpha = (ids >= 0).astype(F32)
        if aa > 1:
            col = cv2.resize(col, (w, h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
            nrm = _norm(cv2.resize(nrm, (w, h), interpolation=cv2.INTER_AREA).reshape(-1, 3)).reshape(h, w, 3)
            fin = np.isfinite(dep)
            dd = np.where(fin, dep, 0)
            cnt = cv2.resize(fin.astype(F32), (w, h), interpolation=cv2.INTER_AREA)
            dep = np.where(cnt > 0.5, cv2.resize(dd, (w, h), interpolation=cv2.INTER_AREA) / np.maximum(cnt, 1e-6),
                           np.inf).astype(F32)
            ids = ids[aa // 2::aa, aa // 2::aa][:h, :w]
        rgb = tonemap_linear(col, tonemap, exposure, look)
        return Render(rgb.astype(F32), col, alpha, dep, nrm, ids)


# ------------------------------------------------------------------ canvas integration
def _layer_render(self, scene, aa=1, exposure=0.0, tonemap="agx", look=None, alpha=False, threads=None):
    """Render a Scene into this layer at the canvas raster size; returns the Render
    (depth/normal/id at raster size, for depth_of_field and masks)."""
    if CONFIG.get("preview"):      # quick look: no supersampling
        aa = 1
    r = scene.render(self.W, self.H, aa, exposure, tonemap, look, threads)
    a = r.alpha if alpha else np.ones_like(r.alpha)
    self.rgba = np.concatenate([r.rgb, a[..., None]], -1)
    return r


Layer.render = _layer_render

__all__ = ["SDF", "sphere", "box", "round_box", "cylinder", "capsule", "torus", "cone", "plane", "ellipsoid",
           "lathe", "extrude", "Mat", "Scene", "Render", "noise3", "perlin3",
           "smudges", "scratches", "dust", "edge_wear"]

# path tracer (atelier/pt.py): installs Layer.pathtrace, exports pathtrace / denoise
try:
    from .pt import pathtrace, denoise  # noqa: E402,F401
    __all__ += ["pathtrace", "denoise"]
except ImportError:  # pt.py not present
    pass
