"""Procedural fields: lazy, vectorised functions of normalised coordinates.

A Field is evaluated only when painted, at the layer's resolution, so the same
script renders identically at preview size and at full size.
Coordinates: x, y in [0, 1] across the canvas (y down). Noises are isotropic
(they scale x by the aspect ratio internally) and return values ~[0, 1].
Fields support + - * / ** abs, comparisons (-> 0/1), numpy ufuncs (np.sin(f)),
and methods: ramp, warp, rotate, scale, move, clip, remap, ridge_line, shade.
"""
from __future__ import annotations

import inspect
import math
import cv2
import numpy as np

from .core import Layer, color, coords, F32

TAU = 2 * math.pi


class Ctx:
    """Evaluation context: ar = W/H aspect ratio, px = one raster pixel in normalised-height
    units, s = render scale (raster px per canvas px)."""
    __slots__ = ("ar", "px", "s")

    def __init__(self, ar=1.0, px=1 / 512, s=1.0):
        self.ar, self.px, self.s = ar, px, s


# ------------------------------------------------------------------ Field
def _ev(v, x, y, k):
    return v.fn(x, y, k) if isinstance(v, Field) else v


def _match(a, b):
    """Broadcast a scalar field (h,w) against a colour field (h,w,C)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.ndim != b.ndim:
        if a.ndim < b.ndim:
            a = a[..., None]
        else:
            b = b[..., None]
    return a, b


class Field:
    """Lazy field. fn(x, y) or fn(x, y, ctx) -> array; ctx has .ar (aspect) and .px (pixel size)."""

    def __init__(self, fn):
        if isinstance(fn, Field):
            fn = fn.fn
        elif not callable(fn):
            v = fn if isinstance(fn, (int, float)) else color(fn)[:3]
            fn = (lambda x, y, k, v=v: np.broadcast_to(np.asarray(v, F32), x.shape + np.shape(v)))
        else:
            try:
                n = len(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                n = 3
            if n == 2:
                f2 = fn
                fn = lambda x, y, k: f2(x, y)
        self.fn = fn

    # evaluation
    def render(self, H, W, box=None, s=1.0):
        x, y = coords(H, W, box)
        out = np.asarray(self.fn(x, y, Ctx(W / H, 1 / H, s)), F32)
        if out.ndim == 0 or out.shape[:2] != x.shape:
            out = np.broadcast_to(out, x.shape + out.shape[2:] if out.ndim > 2 else x.shape)
        return out

    def __call__(self, x, y, ar=1.0, px=1 / 512):
        return self.fn(np.asarray(x, F32), np.asarray(y, F32), Ctx(ar, px))

    def eval(self, w=512, h=512, s=1.0):
        """Evaluate to an array (h, w[, C]) — handy for inspection or numpy work."""
        return self.render(h, w, None, s)

    # arithmetic
    def _bin(self, o, op, rev=False):
        def fn(x, y, k):
            a, b = _match(self.fn(x, y, k), _ev(o, x, y, k))
            return op(b, a) if rev else op(a, b)
        return Field(fn)

    __add__ = lambda s, o: s._bin(o, np.add)
    __radd__ = lambda s, o: s._bin(o, np.add, True)
    __sub__ = lambda s, o: s._bin(o, np.subtract)
    __rsub__ = lambda s, o: s._bin(o, np.subtract, True)
    __mul__ = lambda s, o: s._bin(o, np.multiply)
    __rmul__ = lambda s, o: s._bin(o, np.multiply, True)
    __truediv__ = lambda s, o: s._bin(o, np.divide)
    __rtruediv__ = lambda s, o: s._bin(o, np.divide, True)
    __pow__ = lambda s, o: s._bin(o, lambda a, b: np.power(np.maximum(a, 0), b))
    __gt__ = lambda s, o: s._bin(o, lambda a, b: (a > b).astype(F32))
    __lt__ = lambda s, o: s._bin(o, lambda a, b: (a < b).astype(F32))
    __ge__ = __gt__
    __le__ = __lt__
    __neg__ = lambda s: Field(lambda x, y, k: -s.fn(x, y, k))
    __abs__ = lambda s: Field(lambda x, y, k: np.abs(s.fn(x, y, k)))
    __array_priority__ = 100

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if method != "__call__":
            return NotImplemented
        return Field(lambda x, y, k: ufunc(*[_ev(i, x, y, k) for i in inputs], **kw))

    def map(self, f):
        """Apply any array function to the evaluated values."""
        return Field(lambda x, y, k: f(self.fn(x, y, k)))

    def clip(self, lo=0.0, hi=1.0):
        return self.map(lambda v: np.clip(v, lo, hi))

    def remap(self, lo, hi, to=(0.0, 1.0), clamp=True):
        """Linearly map [lo, hi] -> to (default [0,1])."""
        a, b = to
        def f(v):
            t = (v - lo) / (hi - lo)
            return a + (b - a) * (np.clip(t, 0, 1) if clamp else t)
        return self.map(f)

    def ramp(self, stops="sunset", space="oklab"):
        return ramp(self, stops, space)

    def shade(self, **kw):
        return shade(self, **kw)

    # domain transforms
    def at(self, fx, fy):
        """Evaluate at transformed coords: fx(x, y, k), fy(x, y, k) -> new x, y."""
        return Field(lambda x, y, k: self.fn(fx(x, y, k), fy(x, y, k), k))

    def move(self, dx=0.0, dy=0.0):
        return Field(lambda x, y, k: self.fn(x - dx, y - dy, k))

    def scale(self, sx, sy=None, center=(0.5, 0.5)):
        sy = sx if sy is None else sy
        cx, cy = center
        return Field(lambda x, y, k: self.fn(cx + (x - cx) / sx, cy + (y - cy) / sy, k))

    def rotate(self, deg, center=(0.5, 0.5)):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        cx, cy = center
        def fn(x, y, k):
            u, v = (x - cx) * k.ar, y - cy
            return self.fn(cx + (c * u + s * v) / k.ar, cy - s * u + c * v, k)
        return Field(fn)

    def within(self, x0, y0=None, x1=None, y1=None):
        """Re-map this field so its [0,1] space spans a box in canvas px (x0, y0, x1, y1),
        or the bounds of a Shape: gradient([...]).within(shape)."""
        if y0 is None:
            x0, y0, x1, y1 = x0.bounds if hasattr(x0, "bounds") else x0
        bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        def fn(x, y, k):
            ch = 1 / (k.px * k.s)          # canvas height in canvas px
            cw = k.ar * ch
            return self.fn((x * cw - x0) / bw, (y * ch - y0) / bh, Ctx(bw / bh, 1 / (k.s * bh), k.s))
        return Field(fn)

    def warp(self, strength=0.1, scale=2.0, seed=0, octaves=3):
        return domain_warp(self, strength, scale, seed, octaves)

    def ridge_line(self, y=0.6, amp=0.15, row=0.5):
        """Silhouette curve: y_line(x) = y + amp*(0.5 - f(x, row)). Use with fill_below()."""
        return Curve(lambda x, k: y + amp * (0.5 - self.fn(x, np.full_like(x, row), k)))


X = Field(lambda x, y, k: x)
Y = Field(lambda x, y, k: y)


def radial(cx=0.5, cy=0.5):
    """Distance to (cx, cy) in canvas-height units (aspect-correct)."""
    return Field(lambda x, y, k: np.hypot((x - cx) * k.ar, y - cy))


def angle(cx=0.5, cy=0.5):
    """Angle around (cx, cy) in [0, 1) (0 = right, clockwise on screen)."""
    return Field(lambda x, y, k: (np.arctan2(y - cy, (x - cx) * k.ar) / TAU) % 1.0)


def grid(w, h):
    """Normalised pixel-centre coordinate arrays x, y of shape (h, w)."""
    return coords(h, w)


# ------------------------------------------------------------------ helpers (arrays or fields)
def _lift(f, *args):
    if any(isinstance(a, Field) for a in args):
        return Field(lambda x, y, k: f(*[_ev(a, x, y, k) for a in args]))
    return f(*args)


def smoothstep(e0, e1, v):
    def f(a, b, t):
        t = np.clip((t - a) / (b - a), 0, 1)
        return t * t * (3 - 2 * t)
    return _lift(f, e0, e1, v)


def mix(a, b, t):
    """Linear blend a*(1-t) + b*t; a, b may be colours, arrays or fields."""
    a = Field(a) if isinstance(a, (str, tuple, list)) else a
    b = Field(b) if isinstance(b, (str, tuple, list)) else b
    def f(a, b, t):
        a, t2 = _match(a, t)
        b, t3 = _match(b, t)
        return a * (1 - t3) + b * t3
    return _lift(f, a, b, t)


def aastep(f, edge=0.5):
    """Anti-aliased step (f > edge) with ~1px transition, slope-aware."""
    f = Field(f)
    def fn(x, y, k):
        v, gx, gy = _grad(f, x, y, k)
        v = v - edge
        w = np.maximum(np.hypot(gx, gy) * k.px, 1e-6)
        return np.clip(v / w + 0.5, 0, 1)
    return Field(fn)


# ------------------------------------------------------------------ hashing & base noises
_M1, _M2, _M3 = np.uint32(0x8DA6B343), np.uint32(0xD8163841), np.uint32(0x9E3779B1)
_G = np.array([[np.cos(a), np.sin(a)] for a in (np.arange(16) * TAU / 16 + 0.13)], F32)


def _hash(ix, iy, seed):
    h = ix.view(np.uint32) * _M1 ^ iy.view(np.uint32) * _M2 ^ np.uint32((seed * 0x27D4EB2F + 0x165667B1) & 0xFFFFFFFF)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h


def _floor(v):
    f = np.floor(v)
    return f.astype(np.int32), (v - f).astype(F32)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin(x, y, seed, period=None):
    xi, xf = _floor(x)
    yi, yf = _floor(y)
    u, v = _fade(xf), _fade(yf)
    wrap = (lambda a: a % period) if period else (lambda a: a)
    def g(dx, dy):
        h = _hash(wrap(xi + dx), wrap(yi + dy), seed) & np.uint32(15)
        gg = _G[h]
        return gg[..., 0] * (xf - dx) + gg[..., 1] * (yf - dy)
    a = g(0, 0) + u * (g(1, 0) - g(0, 0))
    b = g(0, 1) + u * (g(1, 1) - g(0, 1))
    return (a + v * (b - a)) * 1.6  # std ~0.35


def _value(x, y, seed, period=None):
    xi, xf = _floor(x)
    yi, yf = _floor(y)
    u, v = _fade(xf), _fade(yf)
    wrap = (lambda a: a % period) if period else (lambda a: a)
    r = lambda dx, dy: (_hash(wrap(xi + dx), wrap(yi + dy), seed) >> np.uint32(8)).astype(F32) / F32(1 << 24)
    a = r(0, 0) + u * (r(1, 0) - r(0, 0))
    b = r(0, 1) + u * (r(1, 1) - r(0, 1))
    return ((a + v * (b - a)) * 2 - 1) * 0.78  # std ~0.35


_F2, _G2 = 0.5 * (math.sqrt(3) - 1), (3 - math.sqrt(3)) / 6


def _simplex(x, y, seed):
    s = (x + y) * _F2
    i, _ = _floor(x + s)
    j, _ = _floor(y + s)
    t = (i + j).astype(F32) * _G2
    x0, y0 = x - (i - t), y - (j - t)
    i1 = (x0 > y0).astype(np.int32)
    j1 = 1 - i1
    pts = [(x0, y0, 0, 0), (x0 - i1 + _G2, y0 - j1 + _G2, i1, j1),
           (x0 - 1 + 2 * _G2, y0 - 1 + 2 * _G2, 1, 1)]
    n = 0
    for px, py, di, dj in pts:
        tt = np.maximum(0.5 - px * px - py * py, 0)
        gg = _G[_hash(i + di, j + dj, seed) & np.uint32(15)]
        n = n + tt ** 4 * (gg[..., 0] * px + gg[..., 1] * py)
    return n * 64.0  # std ~0.35


def _worley(x, y, seed, jitter=1.0):
    xi, xf = _floor(x)
    yi, yf = _floor(y)
    f1 = np.full(x.shape, 9.0, F32)
    f2 = f1.copy()
    cid = np.zeros(x.shape, np.uint32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            h = _hash(xi + dx, yi + dy, seed)
            px = dx + 0.5 + jitter * (((h & np.uint32(0xFFFF)).astype(F32) / 65535) - 0.5)
            py = dy + 0.5 + jitter * (((h >> np.uint32(16)).astype(F32) / 65535) - 0.5)
            d = (px - xf) ** 2 + (py - yf) ** 2
            m = d < f1
            f2 = np.where(m, f1, np.minimum(f2, d))
            f1 = np.where(m, d, f1)
            cid = np.where(m, h, cid)
    return np.sqrt(f1), np.sqrt(f2), cid


# Fast path: perlin/value noise are pre-computed on a tileable lattice (period _P cells,
# _T px per cell) and sampled with bicubic cv2.remap -> ~10x faster than direct hashing.
_P, _T = 32, 24
_TILES: dict = {}


def _tile(base, seed):
    key = (base, seed)
    if key not in _TILES:
        g = np.arange(_P * _T, dtype=F32) / _T
        x, y = np.meshgrid(g, g)
        _TILES[key] = (_perlin if base == "perlin" else _value)(x, y, seed, period=_P).astype(F32)
    return _TILES[key]


def _noise2(base, u, v, seed):
    """Base noise in ~[-1,1] at arbitrary coords u, v (lattice units)."""
    if base == "simplex":
        return _simplex(u, v, seed)
    u, v = np.asarray(u, F32), np.asarray(v, F32)
    shp = u.shape
    if u.ndim != 2 or max(shp) > 30000:
        n = u.size
        cols = 2048 if n > 2048 else max(n, 1)
        rows = -(-n // cols)
        pad = rows * cols - n
        u = np.pad(u.ravel(), (0, pad)).reshape(rows, cols)
        v = np.pad(v.ravel(), (0, pad)).reshape(rows, cols)
    # BORDER_WRAP only wraps one period: reduce coords into the tile explicitly.
    mu = (u - np.floor(u * F32(1 / _P)) * F32(_P)) * F32(_T)
    mv = (v - np.floor(v * F32(1 / _P)) * F32(_P)) * F32(_T)
    out = cv2.remap(_tile(base, seed), mu, mv, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_WRAP)
    if out.shape != shp:
        out = out.ravel()[:int(np.prod(shp))].reshape(shp)
    return out


def _noise_field(base, scale, seed):
    return Field(lambda x, y, k: 0.5 + 0.5 * _noise2(base, x * k.ar * scale, y * scale, seed))


def perlin(scale=4.0, seed=0):
    """Gradient noise in ~[0,1]; scale = features per canvas height."""
    return _noise_field("perlin", scale, seed)


def simplex(scale=4.0, seed=0):
    return _noise_field("simplex", scale, seed)


def value(scale=4.0, seed=0):
    return _noise_field("value", scale, seed)


def _octaves(x, y, k, scale, octaves, lacunarity, gain, seed, base, kind):
    if base not in ("perlin", "simplex", "value"):
        raise ValueError("base: perlin | simplex | value")
    u0, v0 = x * k.ar * scale, y * scale
    total, amp, norm, freq = np.zeros(np.shape(x), F32), 1.0, 0.0, 1.0
    weight = 1.0
    for i in range(int(octaves)):
        a = 0.5 + i * 1.7  # rotate each octave to hide lattice alignment
        c, s = math.cos(a), math.sin(a)
        if scale * freq * k.px > 0.7:  # sub-pixel octave: only aliasing, skip
            norm += amp
            if kind != "fbm":
                total = total + amp * (0.45 if kind == "ridged" else 0.4)
            amp *= gain
            freq *= lacunarity
            continue
        u = (c * u0 - s * v0) * freq + i * 17.31
        v = (s * u0 + c * v0) * freq + i * 9.73
        n = _noise2(base, u, v, seed)
        if kind == "ridged":
            n = (1 - np.abs(n)) ** 2
            n = n * weight
            weight = np.clip(n * 2, 0, 1)
        elif kind == "turb":
            n = np.abs(n)
        total = total + amp * n
        norm += amp
        amp *= gain
        freq *= lacunarity
    t = total / norm
    if kind == "fbm":
        return 0.5 + 0.5 * np.tanh(2.2 * t)      # soft saturation: no flat plateaus
    if kind == "ridged":
        return np.clip((t - 0.02) * 1.15, 0, 1)
    return np.clip((t - 0.05) * 1.6, 0, 1)


def fbm(scale=4.0, octaves=5, lacunarity=2.0, gain=0.5, seed=0, base="perlin"):
    """Fractal sum of noise, ~[0,1]. base: perlin|simplex|value."""
    return Field(lambda x, y, k: _octaves(x, y, k, scale, octaves, lacunarity, gain, seed, base, "fbm"))


def ridged(scale=3.0, octaves=6, lacunarity=2.0, gain=0.5, seed=0, base="perlin"):
    """Ridged multifractal (sharp crests, mountains), ~[0,1]."""
    return Field(lambda x, y, k: _octaves(x, y, k, scale, octaves, lacunarity, gain, seed, base, "ridged"))


def turbulence(scale=4.0, octaves=5, lacunarity=2.0, gain=0.5, seed=0, base="perlin"):
    """Sum of |noise| (billowy, creases), ~[0,1]."""
    return Field(lambda x, y, k: _octaves(x, y, k, scale, octaves, lacunarity, gain, seed, base, "turb"))


def worley(scale=6.0, mode="F1", seed=0, jitter=1.0):
    """Cellular noise. mode: F1 | F2 | F2-F1 (cell edges) | id (random [0,1) per cell)."""
    def fn(x, y, k):
        f1, f2, cid = _worley(x * k.ar * scale, y * scale, seed, jitter)
        if mode == "F1":
            return np.clip(f1, 0, 1)
        if mode == "F2":
            return np.clip(f2, 0, 1.2)
        if mode == "F2-F1":
            return np.clip(f2 - f1, 0, 1)
        if mode == "id":
            return (cid >> np.uint32(8)).astype(F32) / F32(1 << 24)
        raise ValueError("worley mode: F1 | F2 | F2-F1 | id")
    return Field(fn)


voronoi = worley


def domain_warp(f, strength=0.1, scale=2.0, seed=0, octaves=3):
    """Displace f's lookup coords by an fbm vector field (strength in canvas-height units)."""
    f = Field(f)
    def fn(x, y, k):
        wx = _octaves(x, y, k, scale, octaves, 2.0, 0.5, seed + 7, "perlin", "fbm") - 0.5
        wy = _octaves(x, y, k, scale, octaves, 2.0, 0.5, seed + 13, "perlin", "fbm") - 0.5
        return f.fn(x + 2 * strength * wx / k.ar, y + 2 * strength * wy, k)
    return Field(fn)


# ------------------------------------------------------------------ regions / curves
def _edge(d, k, soft):
    """Coverage from signed distance d (>0 inside), d in canvas-height units."""
    w = max(soft, k.px)
    return np.clip(d / w + 0.5, 0, 1)


def top(f=0.5, soft=0.0):
    """Mask of y < f (fraction of height)."""
    return Field(lambda x, y, k: _edge(f - y, k, soft))


def bottom(f=0.5, soft=0.0):
    return Field(lambda x, y, k: _edge(y - f, k, soft))


def left(f=0.5, soft=0.0):
    return Field(lambda x, y, k: _edge((f - x) * k.ar, k, soft))


def right(f=0.5, soft=0.0):
    return Field(lambda x, y, k: _edge((x - f) * k.ar, k, soft))


def band(y0, y1, soft=0.0):
    return Field(lambda x, y, k: _edge(np.minimum(y - y0, y1 - y), k, soft))


def zone(x0, y0, x1, y1, soft=0.0):
    """Rectangle mask, normalised coords."""
    return Field(lambda x, y, k: _edge(np.minimum(np.minimum((x - x0) * k.ar, (x1 - x) * k.ar),
                                                  np.minimum(y - y0, y1 - y)), k, soft))


def disc(cx=0.5, cy=0.5, r=0.25, soft=0.0):
    """Disc mask; r in canvas-height units."""
    return Field(lambda x, y, k: _edge(r - np.hypot((x - cx) * k.ar, y - cy), k, soft))


class Curve:
    """y = f(x) in normalised coords. .below()/.above() -> masks; callable on x arrays."""

    def __init__(self, fn):
        try:
            n = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            n = 2
        self.fn = fn if n == 2 else (lambda x, k, f=fn: f(x))

    def __call__(self, x, ar=1.0):
        return self.fn(np.asarray(x, F32), Ctx(ar))

    def _row(self, x, k):
        if x.ndim == 2 and x.shape[0] > 1 and x[0, 0] == x[-1, 0]:
            r = self.fn(x[0], k)
            slope = np.gradient(r, x[0]) / k.ar if x.shape[1] > 1 else 0 * r
            return np.broadcast_to(r, x.shape), np.broadcast_to(np.sqrt(1 + slope ** 2), x.shape)
        return self.fn(x, k), 1.0

    def below(self, soft=0.0):
        def fn(x, y, k):
            c, sl = self._row(x, k)
            return _edge((y - c) / sl, k, soft)
        return Field(fn)

    def above(self, soft=0.0):
        return 1 - self.below(soft)

    def __add__(self, o):
        return Curve(lambda x, k: self.fn(x, k) + (o.fn(x, k) if isinstance(o, Curve) else o))


def curve(fn):
    """Curve from y = fn(x) (normalised)."""
    return Curve(fn)


def _fill_below(self, curve, src, soft=0.0):
    """Fill the region under a Curve (e.g. a ridge_line silhouette)."""
    return self.fill(src, mask=curve.below(soft))


def _fill_above(self, curve, src, soft=0.0):
    return self.fill(src, mask=curve.above(soft))


Layer.fill_below = _fill_below
Layer.fill_above = _fill_above


# ------------------------------------------------------------------ ready-made patterns (scalar [0,1])
def wood(scale=1.0, rings=14, seed=0, grain=0.35, angle=0.0):
    """Flat-sawn wood: long grain along x, ring arcs, dark late-wood lines. [0,1] (0 = dark).
    Colour it: wood().ramp('wood'). angle rotates the grain."""
    warp = fbm(1.2 * scale, 4, seed=seed).scale(6, 1)
    fine = fbm(60 * scale, 3, seed=seed + 1).scale(25, 1)
    cy = 0.5 + 0.35 * math.sin(seed * 2.1)
    def fn(x, y, k):
        u = (x - 0.5) * k.ar
        d = np.hypot(y - cy, u * 0.12 / scale) + 0.05 * (warp.fn(x, y, k) - 0.5) / scale
        r = (d * rings * scale) % 1.0
        tone = 1 - 0.75 * smoothstep(0.55, 0.95, r) - 0.25 * (1 - smoothstep(0.0, 0.08, r))
        return np.clip(tone * (1 - grain) + grain * fine.fn(x, y, k), 0, 1)
    f = Field(fn)
    return f.rotate(angle) if angle else f


def marble(scale=2.0, veins=5.0, seed=0, sharp=1.0):
    """Marble: thin dark veins with soft halos, [0,1] (1 = vein). marble().ramp('marble')."""
    w = fbm(scale, 4, seed=seed)
    t = turbulence(scale * 2, 5, seed=seed + 3)
    w2 = fbm(scale * 1.5, 4, seed=seed + 5)
    def fn(x, y, k):
        tt = t.fn(x, y, k)
        a = np.abs(np.sin((x * k.ar + y) * veins + w.fn(x, y, k) * 6 + tt * 1.2))
        b = np.abs(np.sin((x * k.ar - y * 0.6) * veins * 1.7 + w2.fn(x, y, k) * 7 + tt * 2))
        thin = (1 - a) ** (12 * sharp) * 0.75
        halo = (1 - a) ** 3 * 0.3
        sec = (1 - b) ** (25 * sharp) * 0.4
        cloud = (w2.fn(x * 1.7, y * 1.7, k) - 0.3) * 0.25
        return np.clip(np.maximum(thin, sec) + halo + cloud, 0, 1)
    return Field(fn)


def clouds(scale=3.0, cover=0.5, sharp=4.0, seed=0):
    """Cumulus-ish density [0,1]; cover = fraction of sky covered."""
    f = domain_warp(fbm(scale, 6, seed=seed), 0.15, scale * 0.5, seed + 3)
    return f.map(lambda v: np.clip((v - (1 - cover)) * sharp + 0.5, 0, 1) ** 1.2)


def terrain_height(scale=2.0, seed=0, warp=0.08):
    """Eroded-looking height map [0,1]."""
    r = domain_warp(ridged(scale, 7, seed=seed), warp, scale * 0.6, seed + 5)
    b = fbm(scale * 0.5, 3, seed=seed + 9)
    return r * 0.7 + b * 0.3


def water_ripples(scale=8.0, seed=0, n=6):
    """Interfering circular ripples [0,1]."""
    rng = np.random.default_rng(seed)
    cs = rng.random((n, 2))
    ph = rng.random(n) * TAU
    def fn(x, y, k):
        acc = 0
        for (cx, cy), p in zip(cs, ph):
            d = np.hypot((x - cx) * k.ar, y - cy)
            acc = acc + np.sin(d * scale * TAU + p) / (1 + d * 4)
        return 0.5 + 0.5 * np.tanh(acc * 0.6)
    return Field(fn)


def fabric(scale=60.0, seed=0, fuzz=0.2):
    """Plain-weave cloth height [0,1]; scale = threads per canvas height. Use with shade()."""
    fib = fbm(scale * 3, 2, seed=seed + 1)
    slub = fbm(scale * 0.15, 3, seed=seed)
    def fn(x, y, k):
        u, v = x * k.ar * scale, y * scale
        fu, fv = u - np.floor(u), v - np.floor(v)
        over = ((np.floor(u) + np.floor(v)) % 2) == 0
        pu, pv = np.sin(np.pi * fu), np.sin(np.pi * fv)
        warp_ = np.sqrt(pu) * (0.55 + 0.45 * pv)   # vertical thread on top
        weft = np.sqrt(pv) * (0.55 + 0.45 * pu)    # horizontal thread on top
        h = np.where(over, warp_, weft)
        return np.clip(h * (1 - fuzz) + fuzz * fib.fn(x, y, k) + (slub.fn(x, y, k) - 0.5) * 0.15, 0, 1)
    return Field(fn)


def paper(scale=1.0, seed=0):
    """Paper tooth + fibres, ~0.5 +- small: use as overlay / multiply texture."""
    a = fbm(180 * scale, 3, seed=seed)
    b = fbm(12 * scale, 4, seed=seed + 1)
    fib = turbulence(60 * scale, 2, seed=seed + 2).scale(1, 6)
    return 0.5 + (a - 0.5) * 0.5 + (b - 0.5) * 0.25 - (fib - 0.5) * 0.15


def stars(density=60.0, size=1.2, seed=0, prob=0.6):
    """Star field [0,1]: density = cells per canvas height, size in px."""
    def fn(x, y, k):
        f1, _, cid = _worley(x * k.ar * density, y * density, seed, 0.8)
        r = (cid >> np.uint32(8)).astype(F32) / F32(1 << 24)
        on = r < prob
        b = (r / prob) ** 3 * 0.9 + 0.1
        rad = size * k.px * density * (0.5 + b)
        return np.where(on, np.exp(-(f1 / rad) ** 2) * b, 0).astype(F32)
    return Field(fn)


# ------------------------------------------------------------------ colour
def rgb2hsv(c):
    c = np.asarray(c, F32)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c[..., :3].max(-1), c[..., :3].min(-1)
    d = mx - mn
    dz = np.where(d == 0, 1, d)
    h = np.where(mx == r, (g - b) / dz % 6, np.where(mx == g, (b - r) / dz + 2, (r - g) / dz + 4)) / 6
    h = np.where(d == 0, 0, h)
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return np.stack([h % 1, s, mx], -1).astype(F32)


def hsv2rgb(c):
    c = np.asarray(c, F32)
    h, s, v = c[..., 0] % 1, c[..., 1], c[..., 2]
    k = lambda n: (n + h * 6) % 6
    f = lambda n: v - v * s * np.clip(np.minimum(k(n), 4 - k(n)), 0, 1)
    return np.stack([f(5), f(3), f(1)], -1).astype(F32)


def rgb2oklab(c):
    from .core import to_linear
    l = to_linear(np.asarray(c, F32)[..., :3])
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], F32)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], F32)
    return (np.cbrt(l @ M1.T) @ M2.T).astype(F32)


def oklab2rgb(c):
    from .core import to_srgb
    M2i = np.array([[1, 0.3963377774, 0.2158037573],
                    [1, -0.1055613458, -0.0638541728],
                    [1, -0.0894841775, -1.2914855480]], F32)
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]], F32)
    return np.clip(to_srgb((np.asarray(c, F32) @ M2i.T) ** 3 @ M1i.T), 0, 1)


PALETTES = {
    "sunset": ["#0b1026", "#3b2a5a", "#8e4162", "#e07a5f", "#f4a259", "#f9dc9c"],
    "night": ["#05060f", "#0b1026", "#1b1b2f", "#2e2e4d", "#4a4e8c", "#8f9bd6"],
    "ocean": ["#03045e", "#023e8a", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"],
    "forest": ["#13200f", "#2d4a2b", "#4f6f3a", "#8a9a5b", "#c2b97f", "#e8e1c4"],
    "desert": ["#3d2c2e", "#7a4b3a", "#b86f52", "#d9a066", "#eacb8a", "#f5ead0"],
    "earth": ["#2b2118", "#5c4033", "#8b6b4a", "#b89b72", "#d8c8a8", "#f2ead8"],
    "fire": ["#000000", "#3d0000", "#8b0000", "#e25822", "#ffb000", "#fff5c0"],
    "ice": ["#0a1931", "#185adb", "#6fa8dc", "#b9d7ea", "#e8f4fa", "#ffffff"],
    "pastel": ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff", "#ffc6ff"],
    "neon": ["#0d0221", "#2a0a4a", "#7b2cbf", "#ff2a6d", "#05d9e8", "#d1f7ff"],
    "vapor": ["#ff71ce", "#b967ff", "#01cdfe", "#05ffa1", "#fffb96"],
    "viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "wood": ["#3b2414", "#5e3a1e", "#8a5a2b", "#b07d43", "#c99a5b"],
    "marble": ["#f4f1ec", "#e6e2da", "#b9b4ab", "#6d6a66", "#2a2a2e"],
    "mono": ["#000000", "#ffffff"],
    "retro16": ["#1a1c2c", "#5d275d", "#b13e53", "#ef7d57", "#ffcd75", "#a7f070", "#38b764", "#257179",
                "#29366f", "#3b5dc9", "#41a6f6", "#73eff7", "#f4f4f4", "#94b0c2", "#566c86", "#333c57"],
    "green4": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
}


def palette(name=None):
    """palette('sunset') -> list of hex colours; palette() -> available names."""
    if name is None:
        return list(PALETTES)
    if name not in PALETTES:
        raise ValueError(f"unknown palette '{name}'. Available: {', '.join(PALETTES)}")
    return list(PALETTES[name])


def cosine_palette(a, b, c, d, n=8):
    """IQ cosine palette a + b*cos(2pi(c*t + d)); returns n colours usable as ramp stops."""
    t = np.linspace(0, 1, n)[:, None]
    return np.clip(np.array(a) + np.array(b) * np.cos(TAU * (np.array(c) * t + np.array(d))), 0, 1).astype(F32)


def _stops(stops):
    """-> (positions (n,), colours (n,4))."""
    if isinstance(stops, str):
        stops = palette(stops)
    if isinstance(stops, np.ndarray) and stops.ndim == 2:
        stops = list(stops)
    stops = list(stops)
    if all(isinstance(s, tuple) and len(s) == 2 and not isinstance(s[1], (int, float)) for s in stops):
        pos = np.array([p for p, _ in stops], F32)
        cols = np.array([color(c) for _, c in stops], F32)
    else:
        pos = np.linspace(0, 1, len(stops)).astype(F32)
        cols = np.array([color(c) for c in stops], F32)
    o = np.argsort(pos, kind="stable")
    return pos[o], cols[o]


def make_lut(stops, n=1024, space="oklab"):
    """Colour lookup table (n, 4) from stops; interpolation in oklab or srgb."""
    pos, cols = _stops(stops)
    t = np.linspace(0, 1, n, dtype=F32)
    rgb = cols[:, :3]
    if space == "oklab":
        rgb = rgb2oklab(rgb)
    out = np.stack([np.interp(t, pos, rgb[:, i]) for i in range(3)] +
                   [np.interp(t, pos, cols[:, 3])], -1).astype(F32)
    if space == "oklab":
        out[:, :3] = oklab2rgb(out[:, :3])
    return out


def _apply_lut(v, table):
    idx = np.clip(np.asarray(v, F32) * (len(table) - 1) + 0.5, 0, len(table) - 1).astype(np.int32)
    out = table[idx]
    return out if table[:, 3].min() < 1 else out[..., :3]


def ramp(f, stops="sunset", space="oklab"):
    """Map a scalar field/array [0,1] onto colours. stops: palette name, list of colours,
    list of (pos, colour), or (n,3) array. Returns a colour field (or array if f is an array)."""
    table = make_lut(stops, space=space)
    if isinstance(f, Field):
        return f.map(lambda v: _apply_lut(v, table))
    return _apply_lut(f, table)


def gradient(stops, angle=90.0, kind="linear", center=(0.5, 0.5), radius=0.5, space="oklab"):
    """Colour gradient field. linear: angle in degrees (0 = left->right, 90 = top->bottom);
    radial: from center out to radius (canvas-height units); conic: around center."""
    table = make_lut(stops, space=space)
    cx, cy = center
    def fn(x, y, k):
        if kind == "linear":
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            ext = abs(c) * k.ar + abs(s)
            t = ((x - 0.5) * k.ar * c + (y - 0.5) * s) / ext + 0.5
        elif kind == "radial":
            t = np.hypot((x - cx) * k.ar, y - cy) / radius
        elif kind == "conic":
            t = (np.arctan2(y - cy, (x - cx) * k.ar) / TAU - angle / 360) % 1.0
        else:
            raise ValueError("gradient kind: linear | radial | conic")
        return _apply_lut(t, table)
    return Field(fn)


def _is_grid(x, y):
    return (x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1 and x[0, 0] == x[-1, 0]
            and y[0, 0] == y[0, -1] and x[0, 1] > x[0, 0])


def _grad(h, x, y, k):
    """Value and gradient (d/dy-units) of field h; exact central differences on pixel grids."""
    if _is_grid(x, y):
        # pad by one pixel so the gradient at the box edge is centred too
        dx, dy = x[0, 1] - x[0, 0], y[1, 0] - y[0, 0]
        xp = np.pad(x, 1, mode="reflect", reflect_type="odd")
        yp = np.pad(y, 1, mode="reflect", reflect_type="odd")
        hp = h.fn(xp, yp, k)
        h0 = hp[1:-1, 1:-1]
        gx = (hp[1:-1, 2:] - hp[1:-1, :-2]) / (2 * dx * k.ar)
        gy = (hp[2:, 1:-1] - hp[:-2, 1:-1]) / (2 * dy)
        return h0, gx, gy
    e = k.px
    h0 = h.fn(x, y, k)
    gx = (h.fn(x + e / k.ar, y, k) - h.fn(x - e / k.ar, y, k)) / (2 * e)
    gy = (h.fn(x, y + e, k) - h.fn(x, y - e, k)) / (2 * e)
    return h0, gx, gy


def shade(height, light=(135, 40), strength=1.0, color=None, ambient=0.25, spec=0.0, shininess=40):
    """2.5D lighting of a height field. light = (azimuth deg, 0 = from right, 90 = from top;
    elevation deg). Without color: grey shading with flat = 0.5 (use on an overlay/soft_light
    layer). With color (colour/field): lit colour. strength scales the bumps."""
    h = Field(height)
    col = None if color is None else (color if isinstance(color, Field) else Field(color))
    az, el = math.radians(light[0]), math.radians(light[1])
    L = np.array([np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), np.sin(el)], F32)
    Hv = L + np.array([0, 0, 1], F32)
    Hv /= np.linalg.norm(Hv)
    def fn(x, y, k):
        h0, gx, gy = _grad(h, x, y, k)
        gx, gy = gx * (strength * 0.05), gy * (strength * 0.05)
        inv = 1 / np.sqrt(gx * gx + gy * gy + 1)
        nx, ny, nz = -gx * inv, -gy * inv, inv
        lam = np.maximum(nx * L[0] + ny * L[1] + nz * L[2], 0)
        sp = spec * np.maximum(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0) ** shininess if spec else 0
        if col is None:
            return np.clip(0.5 * lam / L[2] + sp, 0, 1)
        c = col.fn(x, y, k)[..., :3]
        r = lam / L[2]                                  # 1 on flat ground
        r = np.where(r > 1, 1 + (r - 1) * 0.3, r)       # compress highlights (low sun)
        return np.clip(c * (ambient + (1 - ambient) * r)[..., None] + np.asarray(sp)[..., None], 0, 1)
    return Field(fn)


__all__ = ["Field", "Ctx", "X", "Y", "radial", "angle", "grid", "smoothstep", "mix", "aastep",
           "perlin", "simplex", "value", "fbm", "ridged", "turbulence", "worley", "voronoi",
           "domain_warp", "top", "bottom", "left", "right", "band", "zone", "disc", "Curve", "curve",
           "wood", "marble", "clouds", "terrain_height", "water_ripples", "fabric", "paper", "stars",
           "rgb2hsv", "hsv2rgb", "rgb2oklab", "oklab2rgb", "PALETTES", "palette", "cosine_palette",
           "make_lut", "ramp", "gradient", "shade"]
