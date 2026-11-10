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
    t = np.linspace(0, 1, sub, endpoint=False)[None, :, None]
    t2, t3 = t * t, t * t * t
    p0, p1, p2, p3 = (Q[i:i + n - 1, None] for i in range(4))           # all segments at once
    seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    us = (np.arange(n - 1)[:, None] + t[0, :, 0][None]).ravel()
    return np.vstack([seg.reshape(-1, 2), P[-1:]]), np.append(us, n - 1).astype(F32)


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


_KINDS = {"a8": lambda w, h: skia.ImageInfo.MakeA8(w, h),
          "rgba": lambda w, h: skia.ImageInfo.MakeN32Premul(w, h),
          "f16": lambda w, h: skia.ImageInfo.Make(w, h, skia.ColorType.kRGBA_F16_ColorType,
                                                  skia.AlphaType.kPremul_AlphaType)}


def _surface(box, kind="a8"):
    w, h = box[2] - box[0], box[3] - box[1]
    s = skia.Surface.MakeRaster(_KINDS[kind](w, h))
    c = s.getCanvas()
    c.translate(-box[0], -box[1])
    return s, c


def _read(surf, kind="a8"):
    img = surf.makeImageSnapshot()
    if kind == "a8":
        a = img.toarray(colorType=skia.kAlpha_8_ColorType)
        return (a[..., 0] if a.ndim == 3 else a).astype(F32) * F32(1 / 255)
    if kind == "f16":
        return img.toarray(colorType=skia.kRGBA_F32_ColorType, alphaType=skia.kPremul_AlphaType)
    a = img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    return a.astype(F32) * F32(1 / 255)


def _path(pts, verbs, weights=None):
    """skia.Path from numpy points (k,2) + verb codes (0 move 1 line 3 conic 5 close) in one call
    (deserialises SkPath's binary layout: far faster than per-point Python calls)."""
    pts = np.ascontiguousarray(pts, np.float32).reshape(-1, 2)
    w = np.zeros(0, np.float32) if weights is None else np.ascontiguousarray(weights, np.float32)
    verbs = np.ascontiguousarray(verbs, np.uint8)
    blob = np.array([5, len(pts), len(w), len(verbs)], np.uint32).tobytes() + pts.tobytes() + w.tobytes() + \
        verbs.tobytes()
    blob += b"\0" * (-len(blob) % 4)
    p = skia.Path()
    if len(verbs) and not p.readFromMemory(blob):
        raise RuntimeError("path deserialisation failed")
    return p


_CIRCLE_VERBS = np.array([0, 3, 3, 3, 3, 5], np.uint8)
_CIRCLE_K = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], np.float32)


def _circles(xy, r):
    """One path holding a circle per (x, y, r) (4 conics each)."""
    pts = np.asarray(xy, np.float32)[:, None] + _CIRCLE_K[None] * np.asarray(r, np.float32)[:, None, None]
    n = len(pts)
    return _path(pts, np.tile(_CIRCLE_VERBS, n), np.full(4 * n, math.sqrt(0.5), np.float32))


def _polygon(pts):
    n = len(pts)
    v = np.ones(n + 1, np.uint8)
    v[0], v[-1] = 0, 5
    return _path(pts, v)


def _ribbons(L, R, seg, n=1, S=1):
    """Ribbon polygons: one per run of True seg[j, i] (left edge L forward, right edge R back).
    L, R: (S*n, m, 2) for S strokes of n bristles; seg: (S*n, m-1), segment i joins points i, i+1.
    Returns points, verbs and per-stroke cut indices into both (for slicing one path per stroke)."""
    d = np.diff(np.pad(seg.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    js, a = np.nonzero(d == 1)
    e = np.nonzero(d == -1)[1]                    # last point index of each run
    npt = e - a + 1
    cnt = 2 * npt
    K, total = len(a), int(cnt.sum())
    rid = np.repeat(np.arange(K), cnt)
    i = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    fwd = i < npt[rid]
    col = np.where(fwd, a[rid] + i, e[rid] - (i - npt[rid]))
    pts = np.where(fwd[:, None], L[js[rid], col], R[js[rid], col])
    verbs = np.ones(total + K, np.uint8)
    vs = np.cumsum(cnt) - cnt + np.arange(K)
    verbs[vs] = 0
    verbs[vs + cnt] = 5
    st = js // n
    pcut = np.concatenate([[0], np.cumsum(np.bincount(st, cnt, S))]).astype(np.int64)
    vcut = np.concatenate([[0], np.cumsum(np.bincount(st, cnt + 1, S))]).astype(np.int64)
    return pts, verbs, pcut, vcut


def _argb(rgb, a=1.0):
    """Packed skia colours from (n,3) floats."""
    c = np.clip(np.round(np.asarray(rgb, np.float64) * 255), 0, 255).astype(np.int64)
    return ((int(round(a * 255)) << 24) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).tolist()


def _dabs(xy, r, box, aa=True, flow=1.0):
    """Coverage of circular dabs (raster px)."""
    surf, cv = _surface(box)
    p = skia.Paint(AntiAlias=aa, Color=skia.Color4f(1, 1, 1, float(flow)))
    r = np.maximum(r, 0.35)
    if flow >= 1:   # overlaps do not accumulate: one path
        cv.drawPath(_circles(xy, r), p)
    else:
        for (x, y), rr in zip(xy.tolist(), r.tolist()):
            cv.drawCircle(x, y, rr, p)
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


def _bristle_draw(layer, xs, rs, tns, base, rng, P, opacity):
    """Paint S bristle strokes (in order) on shared skia surfaces, then pick up wet paint, composite
    and lay impasto once for the batch. xs (S,M,2) coarse centre line in raster px, rs (S,M) radius,
    tns (S,M,2) unit tangents, base (S,3) colours; random tip data from rng."""
    S, M = rs.shape
    pad = float(rs.max()) * 0.6 + 4
    box = (max(0, int(math.floor(float((xs[..., 0] - rs).min()) - pad))),
           max(0, int(math.floor(float((xs[..., 1] - rs).min()) - pad))),
           min(layer.W, int(math.ceil(float((xs[..., 0] + rs).max()) + pad))),
           min(layer.H, int(math.ceil(float((xs[..., 1] + rs).max()) + pad))))
    if box[0] >= box[2] or box[1] >= box[3]:
        return
    x0, y0, x1, y1 = box
    aa = layer.aa
    n, var, body, gaps = P.get("n", 14), P.get("var", 0.08), P.get("body", 0.85), P.get("gaps", 0.2)
    pickup, imp = P.get("pickup", 0.0), P.get("impasto", 0.0)
    # --- bristle geometry for all strokes at once
    nrm = np.stack([-tns[..., 1], tns[..., 0]], -1)                                     # (S, M, 2)
    offs = ((np.arange(n) + 0.5) / n * 2 - 1 + rng.uniform(-0.5, 0.5, (S, n)) / n) * 0.95  # stratified
    widths = rng.uniform(0.7, 1.3, (S, n))
    lines = xs[:, None] + nrm[:, None] * (offs[..., None] * rs[:, None])[..., None]    # (S, n, M, 2)
    # paint runs out near the end / random gaps on outer bristles (smoothed noise, window 3)
    z = rng.standard_normal((S, n, M + 2))
    noise = (z[..., :-2] + z[..., 1:-1] + z[..., 2:]) / 1.732
    dry = (np.linspace(0, 1, M) ** 2)[None, None] * gaps * 2 + np.abs(offs)[..., None] * gaps
    on = noise > (dry - 0.5) * 2.5 - 1.2 + gaps * 1.5
    hsv = rgb2hsv(base)[:, None] + rng.normal(0, 1, (S, n, 3)) * np.array([var * 0.15, var * 0.6, var], F32)
    cols = np.clip(hsv2rgb(hsv), 0, 1)
    rb = np.maximum(rs * 2 / n * 1.9, 0.6)
    hw = (rb[:, None] * widths[..., None] * 0.5)[..., None]                            # (S, n, M, 1)
    seg = (on[..., :-1] & on[..., 1:]).reshape(S * n, M - 1)
    rp, rv, pcut, vcut = _ribbons((lines + nrm[:, None] * hw).reshape(S * n, M, 2),
                                  (lines - nrm[:, None] * hw).reshape(S * n, M, 2), seg, n, S)
    nb = nrm * (rs * 0.82)[..., None]
    outline = np.concatenate([xs + nb, (xs - nb)[:, ::-1]], 1)                          # (S, 2M, 2)
    pverbs = np.ones(2 * M + 1, np.uint8)
    pverbs[0], pverbs[-1] = 0, 5
    im = M // 2
    gA = (xs[:, im] - nrm[:, im] * (rs[:, im] * 0.95)[:, None]).tolist()
    gB = (xs[:, im] + nrm[:, im] * (rs[:, im] * 0.95)[:, None]).tolist()
    pos = np.clip((offs / 0.95 + 1) / 2, 1e-3, 1 - 1e-3)
    pos = np.maximum.accumulate(pos + np.arange(n) * 1e-6, axis=1).tolist()
    argb = np.array(_argb(cols.reshape(-1, 3))).reshape(S, n)
    argb = np.concatenate([argb[:, :1], argb, argb[:, -1:]], 1).tolist()
    bcol = np.clip(base, 0, 1).tolist()
    ends = np.stack([xs[:, 0], xs[:, -1]], 1).tolist()
    # --- surfaces
    surf, cv = _surface(box, "rgba")
    tcv = hcv = None
    if pickup > 0 and imp <= 0:      # with impasto the t map rides in the height surface (green channel)
        tsurf, tcv = _surface(box, "rgba")
        nw = nrm * (rs * 1.05)[..., None]
        tout = np.concatenate([xs + nw, (xs - nw)[:, ::-1]], 1)
    rmed = np.median(rs, 1)
    if imp > 0:
        hsurf, hcv = _surface(box, "f16")
        vmax = float(rmed.max()) * 1.7 * max(imp, 1.0)       # height values stored as fractions of vmax
        v = imp * rmed / vmax
        hcols = [np.array(_argb(np.stack([np.stack([v * 1.2 * g, 0 * v, 0 * v], 1),
                                          np.stack([v * 0.6 * g, 1 + 0 * v, 0 * v], 1)], 1).reshape(-1, 3),
                                0.8)).reshape(S, 2).tolist() for g in (1.0, 1.4)]
    bp, gp = skia.Paint(AntiAlias=aa), skia.Paint(AntiAlias=aa)
    tp = skia.Paint(AntiAlias=False)
    hp = skia.Paint(AntiAlias=False)        # height is blurred afterwards
    lin = skia.GradientShader.MakeLinear
    for i in range(S):
        body_path = _path(outline[i], pverbs) if body > 0 else None
        if body_path is not None:     # paint film under the bristle streaks (follows the width)
            b = bcol[i]
            bp.setColor(skia.Color4f(b[0], b[1], b[2], body * 0.35))
            cv.drawPath(body_path, bp)
        bpath = None
        if vcut[i + 1] > vcut[i]:
            bpath = _path(rp[pcut[i]:pcut[i + 1]], rv[vcut[i]:vcut[i + 1]])
            # per-bristle colour: a gradient across the stroke with one stop per bristle
            gp.setShader(lin([skia.Point(*gA[i]), skia.Point(*gB[i])], argb[i], [0.0] + pos[i] + [1.0]))
            cv.drawPath(bpath, gp)
        p0, p1 = skia.Point(*ends[i][0]), skia.Point(*ends[i][1])
        if tcv is not None:          # arc-length parameter map (later strokes overwrite earlier ones)
            tp.setShader(lin([p0, p1], [0xFF000000, 0xFFFFFFFF]))
            tcv.drawPath(_path(tout[i], pverbs), tp)
        if hcv is not None:          # paint thickness: loaded at the start, ridges where bristles lay paint
            for path, hc in ((body_path, hcols[0][i]), (bpath, hcols[1][i])):
                if path is not None:
                    hp.setShader(lin([p0, p1], hc))
                    hcv.drawPath(path, hp)
    pm = _read(surf, "rgba")
    cov = pm[..., 3]
    if not cov.any():
        return
    rgb = pm[..., :3] / np.maximum(cov[..., None], 1e-4)
    rm = float(np.median(rmed))
    hpm = _read(hsurf, "f16") if imp > 0 else None
    if pickup > 0:   # brush picks up wet paint it passes over; more as it runs dry (along t)
        tmap = hpm[..., 1] / np.maximum(hpm[..., 3], 1e-3) if imp > 0 else _read(tsurf, "rgba")[..., 0]
        tmap = cv2.GaussianBlur(tmap, (0, 0), max(1.0, rm * 0.6))
        under, ua = _under(layer, box)
        rgb = km_mix(rgb, under, pickup * (0.4 + 0.6 * tmap) * ua)
    out = np.concatenate([rgb * cov[..., None], cov[..., None]], -1)
    _paint_rgba(layer, out, box, opacity)
    if imp > 0:
        hpm = cv2.GaussianBlur(np.ascontiguousarray(hpm[..., [0, 3]]), (0, 0), max(0.6, 0.5 * layer.s))
        if layer.height is None:
            layer.height = np.zeros((layer.H, layer.W), F32)
        h = layer.height[y0:y1, x0:x1]
        h *= 1 - hpm[..., 1]
        h += hpm[..., 0] * F32(vmax / layer.s * 0.04 / 0.8)
        layer.touch()


def _coarse(pts, npts, M, sub=8):
    """Vectorised Catmull-Rom + uniform arc-length resampling of S polylines (padded (S, N, 2) with
    npts >= 2 valid points each) to M points. Returns xy (S, M, 2), unit tangents (S, M, 2)."""
    S, N = pts.shape[:2]
    rows = np.arange(S)
    last, prev = pts[rows, npts - 1], pts[rows, npts - 2]
    P = np.where((np.arange(N)[None] < npts[:, None])[..., None], pts, last[:, None])   # pad with last
    E = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, last[:, None]], 1)                 # (S, N+2, 2)
    E[rows, npts + 1] = 2 * last - prev                                                 # end tangent
    t = np.linspace(0, 1, sub, endpoint=False)[None, None, :, None]
    a0, a1, a2, a3 = (E[:, k:k + N - 1, None] for k in range(4))
    seg = 0.5 * ((2 * a1) + (-a0 + a2) * t + (2 * a0 - 5 * a1 + 4 * a2 - a3) * t * t +
                 (-a0 + 3 * a1 - 3 * a2 + a3) * t * t * t)                             # (S, N-1, sub, 2)
    seg = np.where((np.arange(N - 1)[None] < npts[:, None] - 1)[..., None, None], seg, last[:, None, None])
    dense = np.concatenate([seg.reshape(S, -1, 2), last[:, None]], 1)
    sl = np.hypot(*np.diff(dense, axis=1).transpose(2, 0, 1))
    cum = np.concatenate([np.zeros((S, 1)), np.cumsum(sl, 1)], 1)                      # (S, D)
    d = cum[:, -1:] * np.linspace(0, 1, M)[None]                                        # (S, M)
    j = np.clip((cum[:, None, :] <= d[..., None]).sum(-1) - 1, 0, cum.shape[1] - 2)
    c0, c1 = np.take_along_axis(cum, j, 1), np.take_along_axis(cum, j + 1, 1)
    f = ((d - c0) / np.maximum(c1 - c0, 1e-9))[..., None]
    x0 = np.take_along_axis(dense, j[..., None].repeat(2, 2), 1)
    x1 = np.take_along_axis(dense, (j + 1)[..., None].repeat(2, 2), 1)
    xy = x0 + (x1 - x0) * f
    tan = np.gradient(xy, axis=1)
    nt = np.hypot(tan[..., 0], tan[..., 1])[..., None]
    tan = np.where(nt > 1e-9, tan / np.maximum(nt, 1e-9), np.array([1.0, 0.0]))
    return xy, tan


def _dab_batch(layer, items, opacity):
    """Hard round dab strokes (flow 1) drawn in order on one RGBA surface. items: (xy, r, rgb, box)."""
    x0 = min(it[3][0] for it in items); y0 = min(it[3][1] for it in items)
    x1 = max(it[3][2] for it in items); y1 = max(it[3][3] for it in items)
    box = (x0, y0, x1, y1)
    surf, cv = _surface(box, "rgba")
    p = skia.Paint(AntiAlias=layer.aa)
    for xy, r, rgb, _ in items:
        p.setColor(skia.Color4f(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0))
        cv.drawPath(_circles(xy, np.maximum(r, 0.35)), p)
    _paint_rgba(layer, _read(surf, "rgba"), box, opacity)


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
def _params(brush, kw):
    if isinstance(brush, str) and brush not in BRUSHES:
        raise ValueError(f"unknown brush '{brush}'. Available: {', '.join(BRUSHES)}")
    P = dict(BRUSHES[brush]) if isinstance(brush, str) else dict(brush)
    P.update(kw)
    return P


def _geom(layer, points, P, size, pressure, spacing, taper, jitter, seed):
    """Resampled stroke in raster px: pts (canvas), xy, t, tan, r, box (None if off-canvas)."""
    s = layer.s
    pts = _points_of(points)
    step = max(0.3, (spacing or P.get("spacing", 0.1)) * size * s)
    xy, uu, t, tan, L = _resample(pts * s, step)
    pr = _pressure(pressure, uu, t, len(pts))
    r = size * s / 2 * pr * _taper(t, taper or P.get("taper", (0, 0)))
    if P.get("speed"):
        sp = np.gradient(uu) if len(uu) > 1 else np.ones(1)
        sp = sp / max(np.median(sp), 1e-9)
        r = r * np.clip(1.25 - P["speed"] * (sp - 1) * 0.6, 0.35, 1.4)
    if jitter:
        xy = xy + np.random.default_rng(seed).normal(0, jitter * size * s * 0.5, xy.shape)
    box = _box(xy, r, r.max() * 0.6 + 4, layer.W, layer.H)
    if box[0] >= box[2] or box[1] >= box[3]:
        box = None
    return pts, xy, t, tan, r, box


def stroke(self, points, brush="round", size=12.0, color="black", pressure=None, opacity=1.0,
           jitter=0.0, spacing=None, taper=None, seed=0, **kw):
    """Paint a stroke along a smooth spline through `points` [(x, y), ...] (or a Shape).
    size = diameter in canvas px; pressure: list (per point), function t->p, or scalar;
    jitter: dab scatter (fraction of size); taper=(start, end) fractions; extra kw override
    brush params (soft, flow, n, gaps, pickup, impasto, edge, granulation, bleed, tooth, speed)."""
    P = _params(brush, kw)
    pts, xy, t, tan, r, box = _geom(self, points, P, size, pressure, spacing, taper, jitter, seed)
    if box is None:
        return self
    tip = P["tip"]
    col = color
    if tip == "bristle":
        if _is_src(col):  # bristles need one base colour: sample mid-stroke
            col = _field_at(col, pts[len(pts) // 2:len(pts) // 2 + 1], self)[0]
        k = max(1, len(xy) // 24)       # bristles only need a coarse polyline (the spline is already smooth)
        sel = np.unique(np.r_[np.arange(0, len(xy), k), len(xy) - 1])
        if len(sel) < 2:
            sel = np.array([0, 0])
        _bristle_draw(self, xy[sel][None], r[sel][None], tan[sel][None], _rgb(col)[None].astype(np.float64),
                      np.random.default_rng(seed), P, opacity)
    elif tip == "dab":
        _tip_dab(self, xy, r, P, col, box, opacity, 1.0, seed)
    elif tip == "chalk":
        _tip_chalk(self, xy, r, P, col, box, opacity, 1.0, seed)
    elif tip == "watercolor":
        _tip_watercolor(self, xy, r, P, col, box, opacity, 1.0, seed)
    else:
        raise ValueError(f"unknown tip '{tip}'")
    return self


def _stroke_batch(layer, pts, npts, sizes, cols, prs, seed, brush, kw):
    """Paint S strokes in order: pts (S, N, 2) canvas px padded, npts (S,) valid points (>= 2), sizes,
    cols (S, 3), prs (S,) pressures. Bristle and hard round tips share one surface per batch (geometry
    vectorised across strokes; wet pickup / impasto computed once for the batch)."""
    kw = dict(kw)
    opacity, jitter = kw.pop("opacity", 1.0), kw.pop("jitter", 0.0)
    spacing, taper = kw.pop("spacing", None), kw.pop("taper", None)
    P = _params(brush, kw)
    tip = P["tip"]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1 << 30, size=len(pts))
    if tip == "bristle" and not jitter and not P.get("speed"):
        s = layer.s
        xy, tan = _coarse(pts * s, npts, 20)
        r = (sizes * s / 2 * prs)[:, None] * _taper(np.linspace(0, 1, 20), taper or P.get("taper", (0, 0)))[None]
        _bristle_draw(layer, xy, r, tan, np.asarray(cols, np.float64)[:, :3], rng, P, opacity)
        return
    if tip == "dab" and P.get("soft", 0) == 0 and P.get("flow", 1.0) >= 1:
        items = []
        for i in range(len(pts)):
            _, xy, t, tan, r, box = _geom(layer, pts[i, :npts[i]], P, sizes[i], prs[i], spacing, taper, jitter,
                                          int(seeds[i]))
            if box is not None:
                items.append((xy, r, np.asarray(cols[i], F32)[:3], box))
        if items:
            _dab_batch(layer, items, opacity)
        return
    for i in range(len(pts)):
        layer.stroke(pts[i, :npts[i]], brush, float(sizes[i]), tuple(float(v) for v in cols[i][:3]),
                     pressure=float(prs[i]), opacity=opacity, jitter=jitter, spacing=spacing, taper=taper,
                     seed=int(seeds[i]), **kw)


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
        cov = np.maximum(cov, np.clip(hw - d + 0.5, 0, 1) * np.clip(2 * hw, 0, 1))   # zero width -> no line
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
    H, W, s = c.H, c.W, c.s
    if callable(angle):  # Field or f(x, y) of degrees
        ang = np.asarray((angle if isinstance(angle, Field) else Field(angle)).render(H, W, None, s), F32)
    else:
        ang = np.full((H, W), float(angle), F32)
    ang = np.radians(ang)
    count = 0
    for pi, size in enumerate(sizes):
        sig = max(0.8, size * s * 0.35)
        B = cv2.GaussianBlur(R, (0, 0), sig)
        lum = cv2.GaussianBlur(B, (0, 0), sig * 2.5) @ np.array([0.2126, 0.7152, 0.0722], F32)  # big forms only
        gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        flat = 0.02 * (1 if s >= 1 else s)
        edge = mag > flat     # direction field: along iso-lines, `angle` where the reference is flat
        DX = np.where(edge, -gy / np.maximum(mag, 1e-9), np.cos(ang)).astype(F32)
        DY = np.where(edge, gx / np.maximum(mag, 1e-9), np.sin(ang)).astype(F32)
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
        ix = lambda x: np.clip((x * s).astype(np.int64), 0, W - 1)
        iy = lambda y: np.clip((y * s).astype(np.int64), 0, H - 1)
        xi, yi = ix(P[:, 0]), iy(P[:, 1])
        if pi > 0:
            keep = err[yi, xi] >= tol
            P, xi, yi = P[keep], xi[keep], yi[keep]
        N = len(P)
        if N == 0:
            continue
        col = B[yi, xi]
        d = np.stack([DX[yi, xi], DY[yi, xi]], 1).astype(np.float64)
        if angle_jitter:
            a = np.radians(rng.normal(0, angle_jitter, N))
            ca, sa = np.cos(a), np.sin(a)
            d = np.stack([d[:, 0] * ca - d[:, 1] * sa, d[:, 0] * sa + d[:, 1] * ca], 1)
        d0 = d.copy()
        sz = size * rng.uniform(0.8, 1.2, N)
        # all strokes walk the direction field in lockstep (each stops at a colour change / off canvas)
        pts = np.zeros((N, length + 1, 2))
        pts[:, 0] = P
        npts = np.ones(N, np.int64)
        alive = np.ones(N, bool)
        p = P.astype(np.float64).copy()
        for k in range(length):
            p = np.where(alive[:, None], p + d * (sz * 0.6)[:, None], p)
            qi, qj = ix(p[:, 0]), iy(p[:, 1])
            ok = (np.abs(B[qj, qi] - col).mean(-1) <= tol * 2.5) & (-sz < p[:, 0]) & (p[:, 0] < c.w + sz) &                 (-sz < p[:, 1]) & (p[:, 1] < c.h + sz)
            alive &= ok
            if not alive.any():
                break
            pts[alive, k + 1] = p[alive]
            npts += alive
            nd = np.stack([DX[qj, qi], DY[qj, qi]], 1)
            nd = np.where(((nd * d).sum(1) < 0)[:, None], -nd, nd)
            d = np.where(alive[:, None], nd, d)
        if color_jitter:
            hsv = rgb2hsv(col) + rng.normal(0, color_jitter, (N, 3)) * np.array([0.08, 0.6, 1.0], F32)
            col = np.clip(hsv2rgb(hsv), 0, 1)
        prs = rng.uniform(*pressure, N) if isinstance(pressure, tuple) else np.full(N, float(pressure))
        short = npts < 2      # no room to walk: a short dash along the start direction
        h = d0[short] * (sz[short] * 0.4)[:, None]
        pts[short, 0], pts[short, 1] = P[short] - h, P[short] + h
        npts[short] = 2
        _stroke_batch(self, pts, npts, sz, col, prs, int(rng.integers(1 << 30)), brush, kw)
        count += N
    return count


Layer.paint_like = paint_like
