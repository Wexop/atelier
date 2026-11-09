"""Brush engine: strokes along Catmull-Rom splines, painterly tips, scatter, hatching.

c.stroke(points, brush="round", size=12, color="black", pressure=None, ...)
  brushes: round, round_soft, airbrush, ink, chalk, bristle, oil, watercolor (see BRUSHES)
c.scatter("star"|"dot"|"soft"|"grass"|"leaf"|"petal"|"square"|"person"|Shape, n, region=mask, ...)
c.hatch(region, angle=45, spacing=8, width=1.2, tone=None, cross=False)
km_mix(a, b, t): subtractive (pigment-like) colour mixing.
Sizes are in canvas px; everything random takes a seed.
"""
from __future__ import annotations

import math
import numpy as np
import cv2
import skia

from .core import color as _core_color
from .core import F32, Layer, Canvas, color, to_linear, to_srgb, resolve, resolve_mask, premul, unpremul, \
    _composite, coords
from .field import Field, Ctx, fbm, PALETTES, rgb2hsv, hsv2rgb, smoothstep
from .vector import Shape

BRUSHES = {
    "round":      dict(tip="dab", soft=0.0, spacing=0.08, taper=(0.0, 0.0)),
    "round_hard": dict(tip="dab", soft=0.0, spacing=0.08, taper=(0.0, 0.0)),
    "round_soft": dict(tip="dab", soft=0.5, spacing=0.1, taper=(0.0, 0.0)),
    "airbrush":   dict(tip="dab", soft=1.0, spacing=0.1, flow=0.08, taper=(0.0, 0.0)),
    "ink":        dict(tip="dab", soft=0.0, spacing=0.05, taper=(0.12, 0.3), speed=0.8),
    "chalk":      dict(tip="chalk", soft=0.0, spacing=0.1, taper=(0.05, 0.1), tooth=2.5),
    "bristle":    dict(tip="bristle", n=14, spacing=0.1, taper=(0.05, 0.15), var=0.08, gaps=0.25),
    "oil":        dict(tip="bristle", n=18, spacing=0.1, taper=(0.12, 0.22), var=0.035, gaps=0.12,
                       impasto=1.0, pickup=0.35),
    "watercolor": dict(tip="watercolor", soft=0.0, spacing=0.1, taper=(0.05, 0.1), edge=1.0, granulation=0.5,
                       bleed=0.3),
}


def brushes():
    """Available brush names."""
    return list(BRUSHES)


# ------------------------------------------------------------------ colour mixing
def _rgb(v):
    return color(v)[:3] if isinstance(v, (str, tuple, list)) else np.asarray(v, F32)[..., :3]


def _is_src(v):
    """True for paint sources that vary in space (Field, callable, image array)."""
    return isinstance(v, Field) or (callable(v) and not isinstance(v, np.ndarray)) or \
        (isinstance(v, np.ndarray) and v.ndim >= 2)


def km_mix(a, b, t=0.5):
    """Kubelka-Munk-ish subtractive mix of sRGB colours/arrays a, b (t = amount of b).
    Blue + yellow -> green, mixtures darken like pigments."""
    a, b = _rgb(a), _rgb(b)
    ra = np.clip(to_linear(a), 1e-4, 1)
    rb = np.clip(to_linear(b), 1e-4, 1)
    ks = lambda r: (1 - r) ** 2 / (2 * r)
    t = np.asarray(t, F32)
    if t.ndim and t.ndim < np.ndim(ra):
        t = t[..., None]
    k = ks(ra) * (1 - t) + ks(rb) * t
    r = 1 + k - np.sqrt(k * k + 2 * k)
    # KM alone over-darkens light mixes: blend 25% of the linear mix back
    lin = ra * (1 - t) + rb * t
    return np.clip(to_srgb(r * 0.75 + lin * 0.25), 0, 1).astype(F32)


# ------------------------------------------------------------------ path sampling
def _catmull(P, sub=12):
    """Dense polyline through points + parameter u (point index space)."""
    n = len(P)
    if n == 1:
        return P.copy(), np.zeros(1, F32)
    if n == 2:
        t = np.linspace(0, 1, sub + 1)[:, None]
        return P[0] + (P[1] - P[0]) * t, np.linspace(0, 1, sub + 1).astype(F32)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0, 1, sub, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out, us = [], []
    for i in range(n - 1):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        out.append(seg)
        us.append(i + t[:, 0])
    out.append(P[-1:])
    us.append([n - 1])
    return np.vstack(out), np.concatenate(us).astype(F32)


def _resample(points, step):
    """Uniform samples along a spline through points. Returns xy (m,2), u (point-index param),
    t (arc fraction 0..1), tangent (m,2)."""
    P = np.asarray(points, np.float64).reshape(-1, 2)
    dense, u = _catmull(P)
    seg = np.hypot(*np.diff(dense, axis=0).T) if len(dense) > 1 else np.zeros(0)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    m = max(2, int(L / max(step, 0.3)) + 1) if L > 0 else 1
    d = np.linspace(0, L, m)
    xy = np.stack([np.interp(d, cum, dense[:, 0]), np.interp(d, cum, dense[:, 1])], 1)
    uu = np.interp(d, cum, u)
    tan = np.gradient(xy, axis=0) if m > 1 else np.array([[1.0, 0.0]])
    tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    return xy, uu, (d / L if L > 0 else np.zeros(m)), tan, L


def _points_of(points):
    if isinstance(points, Shape):
        return points.points(spacing=4)
    return np.asarray(points, np.float64).reshape(-1, 2)


def _pressure(pressure, uu, t, n):
    if pressure is None:
        return np.ones_like(t)
    if callable(pressure):
        return np.asarray(pressure(t), np.float64) * np.ones_like(t)
    p = np.asarray(pressure, np.float64)
    if p.ndim == 0:
        return np.full_like(t, float(p))
    return np.interp(uu, np.linspace(0, n - 1, len(p)), p)


def _taper(t, taper):
    a, b = taper
    w = np.ones_like(t)
    if a > 0:
        w *= np.clip(t / a, 0, 1) ** 0.6
    if b > 0:
        w *= np.clip((1 - t) / b, 0, 1) ** 0.6
    return np.maximum(w, 0.05)


# ------------------------------------------------------------------ rasterisation helpers
def _box(xy, r, pad, W, H):
    x0 = int(math.floor((xy[:, 0] - r).min() - pad)); y0 = int(math.floor((xy[:, 1] - r).min() - pad))
    x1 = int(math.ceil((xy[:, 0] + r).max() + pad)); y1 = int(math.ceil((xy[:, 1] + r).max() + pad))
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def _surface(box, kind="a8"):
    w, h = box[2] - box[0], box[3] - box[1]
    info = skia.ImageInfo.MakeA8(w, h) if kind == "a8" else skia.ImageInfo.MakeN32Premul(w, h)
    s = skia.Surface.MakeRaster(info)
    c = s.getCanvas()
    c.translate(-box[0], -box[1])
    return s, c


def _read(surf, kind="a8"):
    img = surf.makeImageSnapshot()
    if kind == "a8":
        a = img.toarray(colorType=skia.kAlpha_8_ColorType)
        return (a[..., 0] if a.ndim == 3 else a).astype(F32) * F32(1 / 255)
    a = img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    return a.astype(F32) * F32(1 / 255)


def _dabs(xy, r, box, aa=True, flow=1.0):
    """Coverage of circular dabs (raster px)."""
    surf, cv = _surface(box)
    p = skia.Paint(AntiAlias=aa, Color=skia.Color4f(1, 1, 1, float(flow)))
    for (x, y), rr in zip(xy, r):
        cv.drawCircle(float(x), float(y), float(max(rr, 0.35)), p)
    return _read(surf)


def _field_at(f, xy_canvas, layer):
    """Sample a paint source at canvas-px points -> (n,3)."""
    c = layer.canvas
    if isinstance(f, Field) or (callable(f) and not isinstance(f, np.ndarray)):
        f = f if isinstance(f, Field) else Field(f)
        x = (xy_canvas[:, 0] / c.w).astype(F32)[None]
        y = (xy_canvas[:, 1] / c.h).astype(F32)[None]
        v = np.asarray(f.fn(x, y, Ctx(c.W / c.H, 1 / c.H, c.s)), F32)[0]
        if v.ndim == 1:
            v = np.stack([v] * 3, -1)
        return v[:, :3]
    if isinstance(f, np.ndarray) and f.ndim >= 2:
        img = resolve(f, c.H, c.W)
        xi = np.clip((xy_canvas[:, 0] * c.s).astype(int), 0, c.W - 1)
        yi = np.clip((xy_canvas[:, 1] * c.s).astype(int), 0, c.H - 1)
        return img[yi, xi, :3]
    return np.broadcast_to(color(f)[:3], (len(xy_canvas), 3)).copy()


def _under(layer, box):
    """Straight rgb + alpha currently in the layer under box."""
    x0, y0, x1, y1 = box
    pm = layer.px[y0:y1, x0:x1]
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-4, pm[..., :3] / np.maximum(a, 1e-4), 1.0)
    return rgb, a[..., 0]


# ------------------------------------------------------------------ tips
def _paint_rgba(layer, rgba_pm, box, opacity):
    """Composite a premultiplied RGBA patch onto the layer."""
    x0, y0, x1, y1 = box
    patch = layer.px[y0:y1, x0:x1]
    _composite(patch, rgba_pm, "normal", opacity)
    layer.touch()


def _tip_dab(layer, xy, r, P, col, box, opacity, flow, seed):
    soft = P.get("soft", 0.0)
    cov = _dabs(xy, r * (1 - 0.35 * soft), box, layer.aa, P.get("flow", flow))
    if soft > 0 and layer.aa:
        cov = cv2.GaussianBlur(cov, (0, 0), max(0.5, float(np.median(r)) * soft * 0.6))
    layer.paint(np.clip(cov, 0, 1) * opacity, col, box)


def _tip_chalk(layer, xy, r, P, col, box, opacity, flow, seed):
    cov = _dabs(xy, r, box, layer.aa)
    cov = cv2.GaussianBlur(cov, (0, 0), 0.7)
    c = layer.canvas
    tooth = fbm(c.h / P.get("tooth", 2.5), 3, seed=seed + 11)
    tex = tooth.render(c.H, c.W, box, c.s)
    pr = cv2.GaussianBlur(cov, (0, 0), max(1.0, float(np.median(r)) * 0.4))
    k = np.clip((tex - 1 + pr * 1.15) * 3.5, 0, 1)
    layer.paint(cov * k * opacity, col, box)


def _bristle_paths(xy, r, tan, P, rng, s):
    """Polylines for each bristle (raster px) with per-bristle offset, width, gaps."""
    n = P.get("n", 14)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    offs = ((np.arange(n) + 0.5) / n * 2 - 1 + rng.uniform(-0.5, 0.5, n) / n) * 0.95  # stratified
    widths = rng.uniform(0.7, 1.3, n)
    gaps = P.get("gaps", 0.2)
    m = len(xy)
    lines = xy[None] + nrm[None] * (offs[:, None] * r[None])[..., None]          # (n, m, 2)
    # paint runs out near the end / random gaps on outer bristles (smoothed noise, window 3)
    z = rng.standard_normal((n, m + 2))
    noise = (z[:, :-2] + z[:, 1:-1] + z[:, 2:]) / 1.732
    dry = (np.linspace(0, 1, m) ** 2)[None] * gaps * 2 + np.abs(offs)[:, None] * gaps
    on = noise > (dry - 0.5) * 2.5 - 1.2 + gaps * 1.5
    return [(lines[j], on[j], widths[j]) for j in range(n)], offs


def _poly(pts):
    p = skia.Path()
    p.addPoly([skia.Point(float(x), float(y)) for x, y in pts], False)
    return p


def _tip_bristle(layer, xy, r, P, col, box, opacity, flow, seed, tan=None, t=None):
    rng = np.random.default_rng(seed)
    base = np.asarray(col, F32)[:3] if isinstance(col, np.ndarray) else color(col)[:3]
    # bristles only need a coarse polyline (the spline is already smooth)
    k = max(1, len(xy) // 24)
    sel = np.unique(np.r_[np.arange(0, len(xy), k), len(xy) - 1])
    xs, rs, ts, tns = xy[sel], r[sel], t[sel], tan[sel]
    paths, offs = _bristle_paths(xs, rs, tns, P, rng, layer.s)
    n = len(paths)
    rb = np.maximum(rs * 2 / n * 1.9, 0.6)
    surf, cv = _surface(box, "rgba")
    var = P.get("var", 0.08)
    body = P.get("body", 0.85)
    if body > 0:  # paint film under the bristle streaks (one polygon following the width)
        nrm = np.stack([-tns[:, 1], tns[:, 0]], 1) * (rs * 0.82)[:, None]
        outline = np.vstack([xs + nrm, (xs - nrm)[::-1]])
        bp = skia.Paint(AntiAlias=layer.aa, Color=skia.Color4f(float(base[0]), float(base[1]), float(base[2]),
                                                               float(body * 0.35)))
        poly = _poly(outline)
        poly.close()
        cv.drawPath(poly, bp)
    hsv = rgb2hsv(base[None])[0] + rng.normal(0, 1, (n, 3)) * np.array([var * 0.15, var * 0.6, var], F32)
    cols = np.clip(hsv2rgb(hsv), 0, 1)
    paint = skia.Paint(AntiAlias=layer.aa, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap,
                       StrokeJoin=skia.Paint.kRound_Join)
    for j, (line, on, wj) in enumerate(paths):
        cj = cols[j]
        paint.setColor(skia.Color4f(float(cj[0]), float(cj[1]), float(cj[2]), 1.0))
        # split into runs where the bristle touches the canvas
        idx = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a < 2:
                continue
            w = float(np.mean(rb[a:b]) * wj)
            paint.setStrokeWidth(w)
            cv.drawPath(_poly(line[a:b]), paint)
    pm = _read(surf, "rgba")
    cov = pm[..., 3]
    rgb = pm[..., :3] / np.maximum(cov[..., None], 1e-4)
    pickup = P.get("pickup", 0.0)
    tmap = None
    if pickup > 0 or P.get("impasto", 0) > 0:  # smooth: dab steps would show up in the relief
        tmap = cv2.GaussianBlur(_param_map(xs, rs, ts, box, layer.aa), (0, 0), max(1.0, float(np.median(rs)) * 0.6))
    if pickup > 0:
        under, ua = _under(layer, box)
        # brush picks up wet paint it passes over; more as it runs dry (along t)
        amt = (pickup * (0.4 + 0.6 * tmap) * ua)
        rgb = km_mix(rgb, under, amt)
    out = np.concatenate([rgb * cov[..., None], cov[..., None]], -1)
    _paint_rgba(layer, out, box, opacity)
    imp = P.get("impasto", 0.0)
    if imp > 0:
        bo = body * 0.35
        ridge = np.clip((cov - bo) / (1 - bo), 0, 1)       # where bristles laid paint over the film
        if layer.height is None:
            layer.height = np.zeros((layer.H, layer.W), F32)
        x0, y0, x1, y1 = box
        h = layer.height[y0:y1, x0:x1]
        ridge = cv2.GaussianBlur(ridge, (0, 0), max(0.6, 0.5 * layer.s))
        soft = cv2.GaussianBlur(cov, (0, 0), max(1.0, float(np.median(r)) * 0.25))
        thick = imp * soft * (1.0 + 0.4 * ridge) * float(np.median(r)) / layer.s * 0.04
        thick = thick * (1.2 - 0.6 * tmap)
        layer.height[y0:y1, x0:x1] = h * (1 - soft * 0.8) + thick
        layer.touch()


def _param_map(xy, r, t, box, aa):
    """Per-pixel arc-length fraction of the stroke (later dabs overwrite earlier ones)."""
    surf, cv = _surface(box, "rgba")
    p = skia.Paint(AntiAlias=False)
    step = max(1, len(xy) // 256)
    for i in range(0, len(xy), step):
        v = float(t[i])
        p.setColor(skia.Color4f(v, v, v, 1))
        cv.drawCircle(float(xy[i, 0]), float(xy[i, 1]), float(max(r[i], 0.5) * 1.05), p)
    return _read(surf, "rgba")[..., 0]


def _tip_watercolor(layer, xy, r, P, col, box, opacity, flow, seed):
    c = layer.canvas
    cov = _dabs(xy, r, box, True)
    rm = float(np.median(r))
    wet = cv2.GaussianBlur(cov, (0, 0), max(1.0, rm * 0.35))
    nf = fbm(c.h / max(rm / c.s * 3, 4), 4, seed=seed + 3)
    noise = nf.render(c.H, c.W, box, c.s)
    # irregular wet edge
    wet = np.clip((wet - 0.5 + (noise - 0.5) * 0.7 * P.get("bleed", 0.3) * 2) * 3 + 0.5, 0, 1)
    wet = cv2.GaussianBlur(wet, (0, 0), 0.8)
    edge = np.clip(wet - cv2.GaussianBlur(wet, (0, 0), max(1.0, rm * 0.22)), 0, 1)
    gran = fbm(c.h / 4.0, 2, seed=seed + 5).render(c.H, c.W, box, c.s)          # pigment grains ~4px
    pool = fbm(c.h / max(rm / c.s * 2.5, 6), 3, seed=seed + 9).render(c.H, c.W, box, c.s)  # pooling
    g = P.get("granulation", 0.5)
    dens = wet * (0.55 + (pool - 0.5) * 0.8 + g * (gran - 0.5) * 0.3) + edge * P.get("edge", 1.0) * 2.0
    dens = np.clip(dens, 0, 2) * opacity
    _glaze(layer, dens, col, box)


def _glaze(layer, dens, col, box):
    """Transparent pigment layer: result = (existing paint over white) * T**density."""
    x0, y0, x1, y1 = box
    T = np.clip(resolve(col, layer.H, layer.W, box, layer.s)[..., :3], 0.02, 1)
    lin = to_linear(T)
    pm = layer.px[y0:y1, x0:x1]
    a = pm[..., 3:4]
    base = to_linear(pm[..., :3] + (1 - a))                # paint over white paper
    new = to_srgb(base * lin ** dens[..., None])
    # smallest alpha that reproduces `new` exactly when composited over white
    na = np.maximum(np.maximum(a, 1 - new.min(-1, keepdims=True)), np.clip(dens, 0, 1)[..., None] * 0.0)
    # store premultiplied so that over-white compositing gives `new`
    pm[..., :3] = new - (1 - na)
    pm[..., 3:4] = na
    np.clip(pm, 0, 1, out=pm)
    layer.touch()


# ------------------------------------------------------------------ public: stroke
def stroke(self, points, brush="round", size=12.0, color="black", pressure=None, opacity=1.0,
           jitter=0.0, spacing=None, taper=None, seed=0, **kw):
    """Paint a stroke along a smooth spline through `points` [(x, y), ...] (or a Shape).
    size = diameter in canvas px; pressure: list (per point), function t->p, or scalar;
    jitter: dab scatter (fraction of size); taper=(start, end) fractions; extra kw override
    brush params (soft, flow, n, gaps, pickup, impasto, edge, granulation, bleed, tooth, speed)."""
    if isinstance(brush, str) and brush not in BRUSHES:
        raise ValueError(f"unknown brush '{brush}'. Available: {', '.join(BRUSHES)}")
    P = dict(BRUSHES[brush]) if isinstance(brush, str) else dict(brush)
    P.update(kw)
    s = self.s
    pts = _points_of(points)
    step = max(0.3, (spacing or P.get("spacing", 0.1)) * size * s)
    xy, uu, t, tan, L = _resample(pts * s, step)
    pr = _pressure(pressure, uu, t, len(pts))
    r = size * s / 2 * pr * _taper(t, taper or P.get("taper", (0, 0)))
    if P.get("speed"):
        sp = np.gradient(uu) if len(uu) > 1 else np.ones(1)
        sp = sp / max(np.median(sp), 1e-9)
        r = r * np.clip(1.25 - P["speed"] * (sp - 1) * 0.6, 0.35, 1.4)
    rng = np.random.default_rng(seed)
    if jitter:
        xy = xy + rng.normal(0, jitter * size * s * 0.5, xy.shape)
    pad = r.max() * 0.6 + 4
    box = _box(xy, r, pad, self.W, self.H)
    if box[0] >= box[2] or box[1] >= box[3]:
        return self
    tip = P["tip"]
    col = color
    if tip == "bristle" and _is_src(col):  # bristles need one base colour: sample mid-stroke
        col = _field_at(col, pts[len(pts) // 2:len(pts) // 2 + 1], self)[0]
    if tip == "dab":
        _tip_dab(self, xy, r, P, col, box, opacity, 1.0, seed)
    elif tip == "chalk":
        _tip_chalk(self, xy, r, P, col, box, opacity, 1.0, seed)
    elif tip == "bristle":
        _tip_bristle(self, xy, r, P, col, box, opacity, 1.0, seed, tan=tan, t=t)
    elif tip == "watercolor":
        _tip_watercolor(self, xy, r, P, col, box, opacity, 1.0, seed)
    else:
        raise ValueError(f"unknown tip '{tip}'")
    return self


def impasto(layer, strength=1.0, light=(135, 45)):
    """Shade a layer's paint height map (written by the oil brush) -> lit RGBA premultiplied."""
    h = layer.height
    gy, gx = np.gradient(h)
    gx, gy = gx * layer.s * strength * 0.6, gy * layer.s * strength * 0.6
    az, el = math.radians(light[0]), math.radians(light[1])
    L = np.array([math.cos(az) * math.cos(el), -math.sin(az) * math.cos(el), math.sin(el)], F32)
    inv = 1 / np.sqrt(gx * gx + gy * gy + 1)
    lam = (-gx * L[0] - gy * L[1] + L[2]) * inv
    shade = np.clip(lam / L[2], 0.3, 1.6)
    Hv = (L + np.array([0, 0, 1], F32))
    Hv /= np.linalg.norm(Hv)
    spec = np.clip((-gx * Hv[0] - gy * Hv[1] + Hv[2]) * inv, 0, 1) ** 40 * 0.25 * strength
    pm = layer.px.copy()
    a = pm[..., 3:4]
    pm[..., :3] = np.clip(pm[..., :3] * shade[..., None] + spec[..., None] * a, 0, None)
    pm[..., :3] = np.minimum(pm[..., :3], a)
    return pm


# hook impasto into flattening
_orig_flatten = Canvas.flatten


def _flatten(self):
    if not any(getattr(L, "height", None) is not None for L in self.layers):
        return _orig_flatten(self)
    key = tuple((id(L), id(L.px), L.v, L.blend, L.opacity, id(L._mask), getattr(L, "relief", 1.0))
                for L in self.layers)
    if self.__dict__.get("_fkey") == key:
        return self._flat.copy()
    dst = np.zeros((self.H, self.W, 4), F32)
    for L in self.layers:
        if L.opacity > 0:
            src = impasto(L, getattr(L, "relief", 1.0)) if L.height is not None else L.px
            _composite(dst, src, L.blend, L.opacity, L._mask)
    self._fkey, self._flat = key, unpremul(dst)
    return self._flat.copy()


Canvas.flatten = _flatten


# ------------------------------------------------------------------ scatter
def _tpl(kind):
    p = skia.Path()
    if kind in ("dot", "soft", "circle"):
        p.addCircle(0, 0, 0.5)
    elif kind == "square":
        p.addRect(-0.5, -0.5, 0.5, 0.5)
    elif kind in ("star", "sparkle"):
        pts = []
        for i in range(8):
            a = i * math.pi / 4 - math.pi / 2
            rr = 0.5 if i % 2 == 0 else 0.09
            pts.append((rr * math.cos(a), rr * math.sin(a)))
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
        p.close()
        p.addCircle(0, 0, 0.12)
    elif kind == "grass":
        p.moveTo(-0.05, 0)
        p.quadTo(-0.02, -0.55, 0.14, -1.0)
        p.quadTo(0.03, -0.5, 0.05, 0)
        p.close()
    elif kind == "leaf":
        p.moveTo(0, -0.5)
        p.quadTo(0.32, -0.05, 0, 0.5)
        p.quadTo(-0.32, -0.05, 0, -0.5)
        p.close()
    elif kind == "petal":
        p.addOval(skia.Rect.MakeLTRB(-0.18, -0.5, 0.18, 0.0))
    elif kind == "person":
        p.addCircle(0, -0.86, 0.11)
        p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-0.17, -0.72, 0.17, 0.0), 0.1, 0.1))
    else:
        raise ValueError("scatter kind: dot soft star grass leaf petal square person, a Shape, or a brush name")
    return p


_SCATTER_DEF = {
    "star": dict(size=(1.0, 4.0), color="#fff8ec", rot=(0, 0)),
    "dot": dict(size=(2.0, 6.0), color="black", rot=(0, 0)),
    "soft": dict(size=(4.0, 14.0), color="white", rot=(0, 0)),
    "grass": dict(size=(10.0, 30.0), color=["#3f6b2a", "#557d32", "#2f5521", "#6d8f3a"], rot=(-14, 14)),
    "leaf": dict(size=(6.0, 14.0), color=["#2f5d2a", "#46783a", "#6c9441", "#8aa64a"], rot=(0, 360)),
    "petal": dict(size=(5.0, 10.0), color=["#f6c1cf", "#f2a3b8", "#fde2e8"], rot=(0, 360)),
    "square": dict(size=(3.0, 8.0), color="black", rot=(0, 90)),
    "person": dict(size=(9.0, 15.0), color=["#1c1c24", "#2a2633", "#3a3040"], rot=(-3, 3)),
}


def scatter(self, what="dot", n=200, region=None, size=None, color=None, color_jitter=0.0, rot=None,
            seed=0, opacity=1.0, size_range=None, sort=True):
    """Scatter n instances over a region mask (Field/Shape/array; None = whole canvas).
    what: dot | soft | star | grass | leaf | petal | square | person | a Shape (unit-sized,
    centred on 0,0) | a brush name (soft dab). size=(min, max) px; color: colour, list,
    palette name, Field/array (sampled at each position); color_jitter: HSV variation;
    rot=(min, max) degrees. Instances are drawn back-to-front by y."""
    c = self.canvas
    rng = np.random.default_rng(seed)
    key = what if isinstance(what, str) else "shape"
    if key in BRUSHES:
        key = "soft"
    d = _SCATTER_DEF.get(key, dict(size=(4.0, 10.0), color="black", rot=(0, 0)))
    size = size_range or size or d["size"]
    color = d["color"] if color is None else color
    rot = d["rot"] if rot is None else rot
    # positions from the region density
    gw = min(512, c.W)
    gh = max(1, round(gw * c.H / c.W))
    if region is None:
        dens = np.ones((gh, gw), F32)
    else:
        dens = resolve_mask(region, gh, gw, None, c.s * gw / c.W)
        dens = np.broadcast_to(dens, (gh, gw))
    p = dens.ravel().astype(np.float64)
    if p.sum() <= 0:
        return self
    idx = rng.choice(p.size, size=n, p=p / p.sum())
    xy = np.stack([(idx % gw + rng.random(n)) / gw * c.w, (idx // gw + rng.random(n)) / gh * c.h], 1)
    if sort:
        o = np.argsort(xy[:, 1], kind="stable")
        xy = xy[o]
    sz = rng.uniform(size[0], size[1], n) if isinstance(size, (tuple, list)) else np.full(n, float(size))
    angs = rng.uniform(rot[0], rot[1], n) if isinstance(rot, (tuple, list)) else np.full(n, float(rot))
    # colours
    if isinstance(color, str) and color in PALETTES:
        color = PALETTES[color]
    if isinstance(color, (list, tuple)) and len(color) and isinstance(color[0], (str, tuple, list)):
        pal = np.array([_col(q)[:3] for q in color], F32)
        cols = pal[rng.integers(0, len(pal), n)]
    else:
        cols = _field_at(color, xy, self)
    if color_jitter:
        hsv = rgb2hsv(cols)
        hsv += rng.normal(0, 1, hsv.shape) * np.array([color_jitter * 0.12, color_jitter * 0.5, color_jitter], F32)
        cols = np.clip(hsv2rgb(hsv), 0, 1)
    tpl = what.path if isinstance(what, Shape) else _tpl(key)
    soft = key == "soft"
    s = c.s
    surf, cv = _surface((0, 0, c.W, c.H), "rgba")
    paint = skia.Paint(AntiAlias=self.aa)
    for (x, y), z, a, col in zip(xy, sz, angs, cols):
        paint.setColor(skia.Color4f(float(col[0]), float(col[1]), float(col[2]), 1.0))
        if soft:
            paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, float(z * s * 0.22)))
        cv.save()
        cv.translate(float(x * s), float(y * s))
        cv.rotate(float(a))
        cv.scale(float(z * s), float(z * s))
        cv.drawPath(tpl, paint)
        cv.restore()
    pm = _read(surf, "rgba")
    _composite(self.px, pm, "normal", opacity)
    self.touch()
    return self


def _col(q):
    return _core_color(q)


# ------------------------------------------------------------------ hatch
def hatch(self, region=None, angle=45.0, spacing=8.0, width=1.2, color="black", tone=None, cross=False,
          wobble=0.0, seed=0, opacity=1.0):
    """Parallel hatching inside region. tone (Field/array, 0 = dark) modulates line width
    (engraving). cross=True adds a second set at angle+90 (or pass angle=[a1, a2, ...])."""
    c = self.canvas
    angles = list(angle) if isinstance(angle, (list, tuple)) else [angle] + ([angle + 90] if cross else [])
    W, H, s = c.W, c.H, c.s
    m = resolve_mask(region, H, W, None, s) if region is not None else np.ones((H, W), F32)
    m = np.broadcast_to(m, (H, W))
    from .core import _bbox
    bb = _bbox(m > 0)
    if bb is None:
        return self
    y0, y1, x0, x1 = bb
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32)
    darkness = None
    if tone is not None:
        darkness = 1 - resolve_mask(tone, H, W, (x0, y0, x1, y1), s)
    wob = fbm(c.h / 40, 3, seed=seed).render(H, W, (x0, y0, x1, y1), s) - 0.5 if wobble else 0
    cov = np.zeros((y1 - y0, x1 - x0), F32)
    for k, a in enumerate(angles):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        v = (-xx * sa + yy * ca) / (spacing * s) + wob * wobble
        d = np.abs(v - np.floor(v) - 0.5) * spacing * s
        hw = width * s / 2
        if darkness is not None:
            # each extra set only appears in darker tones
            thr = k / max(len(angles), 1) * 0.6
            hw = hw * np.clip((darkness - thr) / (1 - thr), 0, 1) * 1.8
        cov = np.maximum(cov, np.clip(hw - d + 0.5, 0, 1))
    self.paint(cov * m[y0:y1, x0:x1] * opacity, color, (x0, y0, x1, y1))
    return self


Layer.stroke = stroke
Layer.scatter = scatter
Layer.hatch = hatch

__all__ = ["BRUSHES", "brushes", "km_mix", "impasto"]


# ------------------------------------------------------------------ painterly rendering
def paint_like(self, ref, brush="oil", sizes=(48, 24, 12), spacing=0.6, tol=0.08, length=8,
               angle=0.0, angle_jitter=12.0, pressure=(0.75, 1.0), color_jitter=0.04, seed=0, **kw):
    """Repaint a reference (Field / array / image path / Canvas) with brush strokes, coarse to
    fine (Hertzmann-style). Strokes follow the reference's iso-lines; where it is flat they go at
    `angle` degrees (number or Field of degrees). Finer passes only touch areas that still differ
    from the reference by more than `tol`. sizes in canvas px. Returns number of strokes."""
    c = self.canvas
    if isinstance(ref, Canvas):
        ref = ref.flatten()
    if isinstance(ref, str):
        from .core import load
        ref = load(ref)
    R = resolve(ref, c.H, c.W, None, c.s)[..., :3] * np.ones((c.H, c.W, 1), F32)
    rng = np.random.default_rng(seed)
    ang = None
    if callable(angle):  # Field or f(x, y) of degrees
        ang = np.asarray((angle if isinstance(angle, Field) else Field(angle)).render(c.H, c.W, None, c.s), F32)
    count = 0
    for pi, size in enumerate(sizes):
        sig = max(0.8, size * c.s * 0.35)
        B = cv2.GaussianBlur(R, (0, 0), sig)
        lum = cv2.GaussianBlur(B, (0, 0), sig * 2.5) @ np.array([0.2126, 0.7152, 0.0722], F32)  # big forms only
        gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        cur = self.rgba
        cur_rgb = cur[..., :3] * cur[..., 3:4] + (1 - cur[..., 3:4])
        err = np.abs(cur_rgb - B).mean(-1)
        if pi > 0:
            err = cv2.GaussianBlur(err, (0, 0), sig)
        step = size * spacing
        gxs = np.arange(-step / 2, c.w + step, step)    # overshoot so borders get covered
        gys = np.arange(-step / 2, c.h + step, step)
        P = np.stack(np.meshgrid(gxs, gys), -1).reshape(-1, 2)
        P = P + rng.uniform(-0.5, 0.5, P.shape) * step
        P = P[rng.permutation(len(P))]
        H, W = c.H, c.W
        flat = 0.02 * (1 if c.s >= 1 else c.s)
        def direction(x, y, prev=None):
            xi, yi = min(W - 1, max(0, int(x * c.s))), min(H - 1, max(0, int(y * c.s)))
            if mag[yi, xi] > flat:
                d = np.array([-gy[yi, xi], gx[yi, xi]]) / mag[yi, xi]
            else:
                a = math.radians(ang[yi, xi] if ang is not None else angle)
                d = np.array([math.cos(a), math.sin(a)])
            if prev is not None and d @ prev < 0:
                d = -d
            return d
        for (x, y) in P:
            xi, yi = min(W - 1, max(0, int(x * c.s))), min(H - 1, max(0, int(y * c.s)))
            if pi > 0 and err[yi, xi] < tol:
                continue
            col = B[yi, xi].copy()
            pts = [(x, y)]
            d = direction(x, y)
            if angle_jitter:
                a = math.radians(rng.normal(0, angle_jitter))
                d = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
            sz = size * rng.uniform(0.8, 1.2)
            px, py = x, y
            for k in range(length):
                px, py = px + d[0] * sz * 0.6, py + d[1] * sz * 0.6
                qi, qj = min(W - 1, max(0, int(px * c.s))), min(H - 1, max(0, int(py * c.s)))
                if np.abs(B[qj, qi] - col).mean() > tol * 2.5 or not (-sz < px < c.w + sz and -sz < py < c.h + sz):
                    break
                pts.append((px, py))
                d = direction(px, py, d)
            if len(pts) < 2:
                pts = [(x - d[0] * sz * 0.4, y - d[1] * sz * 0.4), (x + d[0] * sz * 0.4, y + d[1] * sz * 0.4)]
            if color_jitter:
                hsv = rgb2hsv(col[None])[0] + rng.normal(0, color_jitter, 3) * np.array([0.08, 0.6, 1.0], F32)
                col = np.clip(hsv2rgb(hsv[None])[0], 0, 1)
            pr = rng.uniform(*pressure) if isinstance(pressure, tuple) else pressure
            self.stroke(pts, brush, sz, tuple(float(v) for v in col), pressure=pr,
                        seed=int(rng.integers(1 << 30)), **kw)
            count += 1
    return count


Layer.paint_like = paint_like
