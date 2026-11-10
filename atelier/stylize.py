"""Stylisation: 3D renders -> illustration, and painterly / graphic image filters.

Render-aware (r = c.render(scene)); each returns an effect for c.apply(...) / L.apply(...):
  toon(r, bands=3)            cel shading: flat object colours quantised by the render's light
  outlines(r, width=2)        ink lines from object-id / depth / normal discontinuities
  render_hatch(r, scene)      engraving hatches following the surfaces (scene gives the camera)
  shadow_dots(r, size=6)      comic halftone dots in the shadows
  shading(r)                  (H, W) light amount in [0, 1] (drives hatch(tone=...), masks, ...)
Image filters f(img, s) like atelier.fx:
  kuwahara xdog edge_smooth watercolorize pencil linocut stained_glass mosaic
Sizes are in canvas px; everything random takes a seed.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from .core import F32, color as _col, to_linear, to_srgb
from .field import fbm, paper as _paper, rgb2hsv, hsv2rgb

LUMA = np.array([0.2126, 0.7152, 0.0722], F32)


# ------------------------------------------------------------------ helpers
def _lum(rgb):
    return rgb[..., :3] @ LUMA


def _blur(a, sigma):
    if sigma <= 0.3:
        return a
    return cv2.GaussianBlur(np.ascontiguousarray(a), (0, 0), float(sigma)).reshape(a.shape)


def _fit(a, h, w, nearest=False):
    """Resize a render buffer to the image raster if needed."""
    a = np.asarray(a)
    if a.shape[:2] == (h, w):
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if a.dtype.kind in "iu":
        return cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(a.dtype)
    return cv2.resize(np.ascontiguousarray(a, F32), (w, h), interpolation=interp).reshape((h, w) + a.shape[2:])


def _flat(img, bg=(1.0, 1.0, 1.0)):
    """Straight RGBA -> RGB over a background colour."""
    a = img[..., 3:4]
    return img[..., :3] * a + np.asarray(bg, F32) * (1 - a)


def _out(img, rgb, cov=None):
    """Replace colour (optionally only through coverage cov), keep/extend alpha."""
    out = img.copy()
    if cov is None:
        out[..., :3] = rgb
        out[..., 3] = 1.0
        return out
    c = cov[..., None]
    out[..., :3] = img[..., :3] * (1 - c) + rgb * c
    out[..., 3] = np.maximum(img[..., 3], cov)
    return out


def _noise(h, w, s, scale_px, octaves=3, seed=0):
    """fbm raster ~[0,1] with features about scale_px canvas px wide."""
    return fbm(max(h / s / max(scale_px, 0.5), 0.5), octaves, seed=seed).render(h, w, None, s).astype(F32)


def _ink(col):
    return _col(col)[:3].astype(F32)


def _line_cov(edge, half):
    """Anti-aliased line coverage of half-width `half` px (scalar or map) around boolean edge pixels."""
    if not edge.any():
        return np.zeros(edge.shape, F32)
    d = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(half + 0.5 - d, 0, 1).astype(F32)


# ------------------------------------------------------------------ render-aware: light
def _shading(r, h, w, light=None):
    """(perceptual shading 0..1 (1 = fully lit, can exceed 1 on highlights), object mask, ids, lit-ratio)."""
    ids = _fit(r.id, h, w, True)
    obj = ids >= 0
    if light is not None:
        n = _fit(r.normal, h, w)
        L = np.asarray(light, F32)
        L = L / max(np.linalg.norm(L), 1e-6)
        sh = np.clip(n @ L, 0, 1)
        return np.where(obj, sh, 1.0).astype(F32), obj, ids, sh ** 2.2
    lin = _fit(r.linear, h, w)
    l = _lum(lin)
    ratio = np.ones((h, w), F32)
    for k in np.unique(ids[obj]):
        m = ids == k
        ref = float(np.percentile(l[m], 92))
        ratio[m] = l[m] / max(ref, 1e-5)
    return np.clip(ratio, 0, 4) ** (1 / 2.2), obj, ids, ratio


def shading(r, light=None):
    """Light amount per pixel (H, W) in [0, 1] from a Render: each object's luminance relative to its own
    brightest part (cast shadows and AO included), or Lambert n.light if light=(x, y, z) is given.
    Sky = 1. Use as tone for hatch(tone=...) or as a mask."""
    h, w = r.id.shape
    return np.clip(_shading(r, h, w, light)[0], 0, 1)


def _albedo(r, h, w, ids, obj, ratio, colors):
    """Flat colour per object (linear): colors={id: colour} or estimated from its well-lit pixels."""
    lin = _fit(r.linear, h, w)
    base = np.zeros((h, w, 3), F32)
    for k in np.unique(ids[obj]):
        m = ids == k
        if colors and k in colors:
            c = to_linear(_col(colors[k])[:3])
        else:
            rk = ratio[m]
            sel = rk > np.percentile(rk, 60)
            px = lin[m][sel] if sel.sum() > 4 else lin[m]
            c = np.median(px, 0) / max(float(np.median(rk[sel] if sel.sum() > 4 else rk)), 1e-4)
        base[m] = np.clip(c, 0, 1)
    return base


def toon(r, bands=3, colors=None, light=None, shadow=0.42, tint="#5a6a9a", spec=1.3, sky=None, soft=0.6):
    """Cel shading from a Render: each object gets a flat colour (colors={id: colour} or estimated)
    shaded in `bands` steps of the render's own lighting (shadows included; light=(x,y,z) -> Lambert).
    shadow = darkest band level, tint = colour the shadows lean to, spec = lit ratio above which a
    white highlight band appears (None = off), sky = colour for background pixels (None = keep)."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        sh, obj, ids, ratio = _shading(r, h, w, light)
        base = _albedo(r, h, w, ids, obj, ratio, colors)
        q = np.floor(np.clip(sh, 0, 0.999) * bands) / max(bands - 1, 1)
        q = _blur(q.astype(F32), soft * s)                      # anti-aliased band edges
        lev = shadow + (1 - shadow) * q
        col = base * lev[..., None]
        if tint is not None:                                  # cool, slightly saturated shadows
            tl = to_linear(_col(tint)[:3])
            tl = tl / max(float(tl @ LUMA), 1e-4)
            col = col * (1 + (tl - 1) * ((1 - q) * 0.35)[..., None])
        rgb = to_srgb(np.clip(col, 0, 1))
        if spec is not None:
            hi = _blur((ratio > spec).astype(F32) * obj, 0.5 * s)
            rgb = rgb * (1 - hi[..., None] * 0.85) + hi[..., None] * 0.85
        m = _blur(obj.astype(F32), 0.35 * s)
        bg = img[..., :3] if sky is None else np.broadcast_to(_ink(sky), img[..., :3].shape)
        out = img.copy()
        out[..., :3] = bg * (1 - m[..., None]) + rgb * m[..., None]
        out[..., 3] = 1.0 if sky is not None else np.maximum(img[..., 3], m)
        return out
    return f


# ------------------------------------------------------------------ render-aware: lines
def _edges(r, h, w, depth=0.03, crease=0.5):
    """Boolean edge maps: sky silhouette, object boundary, depth fold, crease (on the nearer pixel)."""
    ids = _fit(r.id, h, w, True)
    d = _fit(r.depth, h, w, True).astype(np.float64)
    d = np.where(np.isfinite(d), d, 1e6)
    n = _fit(r.normal, h, w, True)
    sky = np.zeros((h, w), bool)
    obj = np.zeros((h, w), bool)
    fold = np.zeros((h, w), bool)
    cre = np.zeros((h, w), bool)
    for ax in (0, 1):
        sl_a = (slice(None), slice(1, None)) if ax == 1 else (slice(1, None), slice(None))
        sl_b = (slice(None), slice(None, -1)) if ax == 1 else (slice(None, -1), slice(None))
        ia, ib, da, db = ids[sl_a], ids[sl_b], d[sl_a], d[sl_b]
        diff = ia != ib
        a_near = da < db
        s_edge = diff & ((ia < 0) | (ib < 0))
        o_edge = diff & ~s_edge
        for E, msk in ((sky, s_edge), (obj, o_edge)):
            E[sl_a] |= msk & a_near
            E[sl_b] |= msk & ~a_near
        same = ~diff & (ia >= 0)
        rel = np.abs(da - db) / np.maximum(np.minimum(da, db), 1e-6)
        # depth folds: a jump much bigger than the local slope (second difference)
        if depth:
            dd = np.zeros((h, w))
            if ax == 1:
                dd[:, 1:-1] = np.abs(d[:, 2:] + d[:, :-2] - 2 * d[:, 1:-1]) / d[:, 1:-1]
            else:
                dd[1:-1] = np.abs(d[2:] + d[:-2] - 2 * d[1:-1]) / d[1:-1]
            fold |= (dd > depth) & (ids >= 0)
        if crease:
            nd = 1 - (n[sl_a] * n[sl_b]).sum(-1)
            c = same & (nd > crease) & (rel < 0.2)
            cre[sl_a] |= c & a_near
            cre[sl_b] |= c & ~a_near
    fold &= ~sky & ~obj
    cre &= ~sky & ~obj & ~fold
    return sky, obj, fold, cre


def outlines(r, width=2.0, inner=0.8, color="#16130f", depth=0.03, crease=0.5, rough=0.0, seed=0):
    """Ink lines from a Render: silhouettes against the sky 1.3x `width`, object boundaries `width`,
    depth folds 0.75x, creases (normal angle, 1 - cos > crease) `inner` (canvas px). depth = relative
    depth jump for folds (0 = off). rough > 0 wobbles the lines by that many px (hand-inked look)."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        sky, obj, fold, cre = _edges(r, h, w, depth, crease)
        cov = np.zeros((h, w), F32)
        for E, k in ((sky, 1.3 * width), (obj, width), (fold, 0.75 * width), (cre, inner)):
            if k > 0:
                cov = np.maximum(cov, _line_cov(E, k * s / 2 - 0.5))
        if rough:
            yy, xx = np.mgrid[0:h, 0:w].astype(F32)
            nx = (_noise(h, w, s, 30, 3, seed) - 0.5) * 2 * rough * s
            ny = (_noise(h, w, s, 30, 3, seed + 1) - 0.5) * 2 * rough * s
            cov = cv2.remap(cov, (xx + nx).astype(F32), (yy + ny).astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return _out(img, _ink(color), _blur(cov, 0.35 * s))
    return f


# ------------------------------------------------------------------ render-aware: hatching / dots
def _world(r, scene, h, w):
    """World positions (h, w, 3) of visible surfaces + camera basis (fwd, right)."""
    cam = scene.cam
    pos, tgt, up = (np.asarray(cam[k], np.float64) for k in ("pos", "target", "up"))
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-9)
    H, W = r.depth.shape
    o, d, _ = scene.rays(W, H)
    d = d.reshape(H, W, 3).astype(np.float64)
    dep = np.where(np.isfinite(r.depth), r.depth, 0.0)
    t = dep / np.maximum(d @ fwd, 1e-6)
    p = pos + d * t[..., None]
    return _fit(p.astype(F32), h, w), fwd, right


def _phase_cov(v, obj, ids, hw, spacing_px, wob=0.0):
    """Coverage of lines at integer values of a world-space phase v, kept about spacing_px apart on screen
    by octaves (mip-mapped hatching: where lines crowd with distance every other one fades out)."""
    gy, gx = np.gradient(v)
    g = np.hypot(gx, gy)
    for k in np.unique(ids[obj]):                       # tame gradient spikes at object borders
        m = ids == k
        lo_, hi_ = np.percentile(g[m], (2, 90))
        g[m] = np.clip(g[m], lo_, hi_)
    g = np.maximum(g, 1e-9)
    L = np.log2(1 / (spacing_px * g))
    k0 = np.floor(L)
    t = (L - k0).astype(F32)
    v = v + wob * spacing_px * g                        # wobble in units of one line period
    out = 0
    for kk, wt in ((k0, 1.0), (k0 + 1, t)):
        sc = np.exp2(kk)
        vv = v * sc
        fr = g * sc                                     # cycles per px
        dist = np.abs(vv - np.floor(vv) - 0.5) / fr
        c = np.clip(hw - dist + 0.5, 0, 1) * np.clip(2 * hw, 0, 1)
        out = np.maximum(out, c * wt)
    return (out * obj).astype(F32)


def render_hatch(r, scene=None, spacing=5.0, width=1.0, layers=3, color="#1b1712", light=None, angle=45.0,
                 lo=0.12, gamma=1.0, value=0.6, wobble=0.25, seed=0):
    """Engraving hatches on a Render, darker = more sets and thicker lines. With scene= (the Scene that made r)
    lines follow the surfaces in 3D (rings around cylinders, receding lines on the ground: real perspective);
    without it they are straight screen lines at angle, angle+90, +45, -45. spacing/width in canvas px,
    lo = darkness where the first set starts, gamma > 1 = lighter, value = how much darker object colours
    darken the tone (0 = light only)."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        sh, obj, ids, ratio = _shading(r, h, w, light)
        sh = np.clip(sh, 0, 1)
        if value:
            alb = to_srgb(_albedo(r, h, w, ids, obj, ratio, None)) @ LUMA
            alb = alb / max(float(alb.max()), 1e-3)            # lightest object = white paper
            sh = sh * np.clip(alb, 0, 1) ** value
        dark = (1 - sh ** (1 / gamma)) * obj
        wob = (_noise(h, w, s, 40, 3, seed) - 0.5) * wobble if wobble else 0
        cov = np.zeros((h, w), F32)
        sp = spacing * s
        if scene is not None:
            p, fwd, right = _world(r, scene, h, w)
            n = _fit(r.normal, h, w)
            Y = np.array([0, 1, 0], F32)
            Fh = np.array([fwd[0], 0, fwd[2]], F32)
            Fh /= max(np.linalg.norm(Fh), 1e-6)
            Rh = np.array([right[0], 0, right[2]], F32)
            Rh /= max(np.linalg.norm(Rh), 1e-6)
            nz = lambda v: v / np.linalg.norm(v)
            axes = [(Y, Fh), (Rh, Fh), (nz(Y + Rh), nz(Fh + Rh)), (nz(Y - Rh), nz(Fh - Rh))]
        else:
            yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        for k in range(layers):
            thr = lo + k * (1 - lo) / layers * 0.8
            hw = width * s / 2 * np.clip((dark - thr) / (1 - thr) * 1.8, 0, 1) * 1.4
            if not (hw > 0.05).any():
                continue
            if scene is not None:
                A, B = axes[k % 4]
                wgt = np.clip((np.abs(n @ A) - 0.55) / 0.25, 0, 1)
                c = 0
                for ax, wa in ((A, 1 - wgt), (B, wgt)):
                    if (wa > 0).any():
                        c = c + _phase_cov(p @ ax, obj, ids, hw, sp, wob) * wa
            else:
                a = math.radians(angle + (0, 90, 45, -45)[k % 4])
                vv = (-xx * math.sin(a) + yy * math.cos(a)) / sp + wob
                fr = 1 / sp
                c = np.clip(hw - np.abs(vv - np.floor(vv) - 0.5) / fr + 0.5, 0, 1) * np.clip(2 * hw, 0, 1) * obj
            cov = np.maximum(cov, c)
        return _out(img, _ink(color), cov)
    return f


def shadow_dots(r, size=6.0, angle=45.0, color="#1b1712", lo=0.35, amount=0.9, light=None):
    """Comic halftone dots in the shadows of a Render: dot area grows with darkness beyond `lo`.
    size = screen cell in canvas px."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        sh, obj, ids, _ = _shading(r, h, w, light)
        dark = np.clip((1 - np.clip(sh, 0, 1) - lo) / (1 - lo), 0, 1) * amount * obj
        cell = size * s
        y, x = np.mgrid[0:h, 0:w].astype(F32)
        c, sn = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        u, v = (x * c + y * sn) / cell, (-x * sn + y * c) / cell
        du, dv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
        d = np.sqrt(du * du + dv * dv) * cell
        rad = np.sqrt(_blur(dark.astype(F32), cell * 0.25)) * cell * 0.62
        cov = np.clip(rad - d + 0.5, 0, 1) * (rad > 0.25) * obj
        return _out(img, _ink(color), cov.astype(F32))
    return f


# ------------------------------------------------------------------ painterly filters
def kuwahara(radius=6.0, sharp=8.0, aniso=1.0, sectors=8):
    """Anisotropic Kuwahara (oil-paint look): each pixel takes the mean of the most uniform of `sectors`
    sectors of an ellipse stretched along the local edge direction. radius in canvas px, sharp = how
    strongly the most uniform sector wins, aniso: 0 = round, 1 = stretched along structures."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3], F32)
        R = max(1.5, radius * s)
        L = _lum(rgb)
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        st = max(1.0, R * 0.35)
        jxx, jyy, jxy = _blur(gx * gx, st), _blur(gy * gy, st), _blur(gx * gy, st)
        tr = jxx + jyy
        dis = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy * jxy, 0))
        A = np.where(tr > 1e-9, dis / np.maximum(tr, 1e-9), 0) * aniso         # anisotropy 0..1
        phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2                 # along the edge
        ca, sa = np.cos(phi).astype(F32), np.sin(phi).astype(F32)
        ea, eb = (R * (1 + A)).astype(F32), (R / (1 + A)).astype(F32)
        ns = int(np.clip(round(R / 1.6), 2, 5))
        g = np.linspace(-1, 1, 2 * ns + 1)
        offs = [(u, v) for u in g for v in g if u * u + v * v <= 1.0001]
        th = 2 * np.pi * np.arange(sectors) / sectors
        src = np.concatenate([rgb, L[..., None]], -1)
        out = np.empty_like(rgb)
        yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        q = float(sharp)
        for y0 in range(0, h, 192):                         # row bands keep memory bounded
            y1 = min(h, y0 + 192)
            sl = slice(y0, y1)
            M = np.zeros((sectors, y1 - y0, w, 3), F32)
            S1 = np.zeros((sectors, y1 - y0, w), F32)
            S2 = np.zeros((sectors, y1 - y0, w), F32)
            Wt = np.zeros(sectors, F32)
            for u, v in offs:
                dx = ca[sl] * ea[sl] * u - sa[sl] * eb[sl] * v
                dy = sa[sl] * ea[sl] * u + ca[sl] * eb[sl] * v
                smp = cv2.remap(src, (xx[sl] + dx).astype(F32), (yy[sl] + dy).astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                gw = math.exp(-2.0 * (u * u + v * v))
                if u == 0 and v == 0:
                    wk = np.full(sectors, 1.0 / sectors)
                else:
                    wk = np.maximum(0, np.cos(math.atan2(v, u) - th)) ** 4
                for k in np.flatnonzero(wk > 1e-3):
                    wv = F32(gw * wk[k])
                    M[k] += smp[..., :3] * wv
                    l = smp[..., 3]
                    S1[k] += l * wv
                    S2[k] += l * l * wv
                    Wt[k] += wv
            Wt = np.maximum(Wt, 1e-6)[:, None, None]
            mean = M / Wt[..., None]
            sd = np.sqrt(np.maximum(S2 / Wt - (S1 / Wt) ** 2, 0)) + F32(2e-3)
            a = (sd.min(0)[None] / sd) ** q
            out[sl] = (a[..., None] * mean).sum(0) / a.sum(0)[..., None]
        res = img.copy()
        res[..., :3] = np.clip(out, 0, 1)
        return res
    return f


def edge_smooth(radius=4.0, tone=0.1, passes=2):
    """Edge-preserving smoothing (bilateral): flattens texture, keeps contours. radius in canvas px,
    tone = colour difference still smoothed across."""
    def f(img, s=1.0):
        rgb = np.ascontiguousarray(img[..., :3], F32)
        for _ in range(passes):
            rgb = cv2.bilateralFilter(rgb, -1, float(tone), float(max(1.0, radius * s)))
        out = img.copy()
        out[..., :3] = np.clip(rgb, 0, 1)
        return out
    return f


def _levels(L, lo=2, hi=98):
    """Stretch luminance so its lo..hi percentiles span 0..1 (filters behave the same on dark images)."""
    a, b = np.percentile(L[::4, ::4], (lo, hi))
    return np.clip((L - a) / max(b - a, 1e-3), 0, 1).astype(F32)


def xdog(sigma=1.0, k=1.6, p=20.0, eps=0.9, phi=12.0, fill=0.0, ink="#16130f", paper=None, amount=1.0,
         smooth=2.0):
    """XDoG line art (Winnemoeller): sharpened difference of Gaussians, soft-thresholded, relative to the
    local brightness (works on dark and light images alike). sigma = line scale (canvas px); eps < 1:
    lower = fewer, bolder lines; phi = edge sharpness; fill > 0 also inks tones darker than fill (0..1).
    paper=None draws the ink over the image (ink over painting), else on a paper colour. smooth = edge-
    preserving denoise radius (canvas px) so grain and texture do not turn into specks."""
    def f(img, s=1.0):
        L = _levels(_lum(_flat(img)))
        if smooth:
            L = cv2.bilateralFilter(np.ascontiguousarray(L, F32), -1, 0.12, max(1.0, smooth * s))
        g1 = _blur(L, sigma * s)
        g2 = _blur(L, sigma * k * s)
        R = 1 + (1 + p) * (g1 - g2) / np.maximum(g2, 0.3)        # 1 on flats of any brightness
        T = np.where(R >= eps, 1.0, 1 + np.tanh(phi * (R - eps)))
        cov = np.clip(1 - T, 0, 1).astype(F32)
        if fill:
            cov = np.maximum(cov, np.clip((fill - _blur(L, 1.5 * sigma * s)) / 0.04 + 0.5, 0, 1))
        cov = cov * amount
        if paper is not None:
            base = np.broadcast_to(_ink(paper), img[..., :3].shape)
            out = img.copy()
            out[..., :3] = base * (1 - cov[..., None]) + _ink(ink) * cov[..., None]
            out[..., 3] = 1.0
            return out
        return _out(img, _ink(ink), cov)
    return f


def watercolorize(bleed=2.0, edge=1.0, granulation=0.4, paper="#f6f1e5", wash=0.7, simplify=5.0, ink=0.0,
                  ink_color="#2a2118", seed=0):
    """Turn an image into a watercolour: simplified washes (mean shift), wet bleeding (noisy displacement +
    blur, `bleed` canvas px), pigment pooling at wash borders (edge), granulation in paper tooth, lighter
    transparent colour (wash < 1) over `paper`. ink > 0 adds pen lines (XDoG of the source) on top:
    watercolour over ink."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        pap = _ink(paper)
        rgb = np.clip(_flat(img, pap), 0, 1)
        u8 = (rgb * 255 + 0.5).astype(np.uint8)
        ms = cv2.pyrMeanShiftFiltering(u8, max(2.0, simplify * s), 30).astype(F32) / 255
        if bleed:
            yy, xx = np.mgrid[0:h, 0:w].astype(F32)
            nx = (_noise(h, w, s, bleed * 6, 3, seed) - 0.5) * 2 * bleed * s
            ny = (_noise(h, w, s, bleed * 6, 3, seed + 1) - 0.5) * 2 * bleed * s
            ms = cv2.remap(ms, (xx + nx).astype(F32), (yy + ny).astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ms = _blur(ms, bleed * s * 0.3)
        pl = to_linear(pap)
        D = -np.log(np.clip(to_linear(ms) / pl, 1e-3, 1)) * wash                    # pigment density
        Dm = D.mean(-1)
        rim = np.clip(Dm - _blur(Dm, (bleed + 3) * s), 0, None) / (Dm + 0.05)      # pigment pools at borders
        tooth = _paper(1.0, seed=seed + 2).render(h, w, None, s).astype(F32)
        gran = _noise(h, w, s, 2.5, 2, seed + 3)
        pool = _noise(h, w, s, 60, 3, seed + 4)
        mod = 1 + edge * np.clip(rim, 0, 1.2) * 1.6 + granulation * ((gran - 0.5) * 0.5 + (0.5 - tooth) * 0.6) \
            + (pool - 0.5) * 0.35
        out = to_srgb(pl * np.exp(-D * np.clip(mod, 0.2, 3)[..., None]))
        out = out * (1 + (tooth[..., None] - 0.5) * 0.12)
        if ink:
            lines = xdog(1.0, eps=0.86, ink=ink_color, paper="#ffffff")(img, s)[..., 0]
            cov = np.clip((1 - lines) * ink, 0, 1)[..., None]
            out = out * (1 - cov) + _ink(ink_color) * cov
        return _out(img, np.clip(out, 0, 1))
    return f


def _strokes_tex(h, w, s, angle, length, seed):
    """Directional noise (pencil strokes) ~[0,1]."""
    rng = np.random.default_rng(seed)
    n = rng.random((h, w), dtype=np.float32)
    L = max(3, int(length * s))
    k = np.zeros((L, L), F32)
    c = (L - 1) / 2
    ca, sa = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
    for t in np.linspace(-c, c, L * 2):
        k[int(round(c + t * sa)), int(round(c + t * ca))] = 1
    k /= k.sum()
    m = cv2.filter2D(n, -1, k, borderType=cv2.BORDER_REFLECT)
    m = (m - m.mean()) / max(float(m.std()), 1e-6)
    return np.clip(0.5 + m * 0.22, 0, 1)


def pencil(color="#2b2a2e", paper="#f7f4ec", hatch=0.7, angle=35.0, line=2.5, colored=0.0, seed=0):
    """Pencil sketch: contour lines by colour dodge (line = blur radius in canvas px), tone by directional
    strokes (hatch amount, crossing strokes in the darks), paper grain. colored > 0 tints with the image."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        rgb = _flat(img)
        L = _lum(rgb)
        B = _blur(1 - L, line * s)
        sketch = np.clip(L / np.maximum(1 - B, 1e-3), 0, 1) ** 3
        dark = np.clip(1 - _blur(L, 1.2 * s), 0, 1)
        t1 = _strokes_tex(h, w, s, angle, 14, seed)
        t2 = _strokes_tex(h, w, s, angle + 75, 14, seed + 1)
        tone = np.clip(dark * 1.3 - 0.12 + (t1 - 0.5) * 0.9, 0, 1) * hatch
        tone = np.maximum(tone, np.clip(dark * 1.3 - 0.55 + (t2 - 0.5) * 0.9, 0, 1) * hatch)
        grain = _paper(1.0, seed=seed + 2).render(h, w, None, s).astype(F32)
        cov = np.clip((1 - sketch) + tone * (1 - sketch * 0.3), 0, 1)
        cov = np.clip(cov * (1 + (0.5 - grain) * 0.9), 0, 1)
        lead = np.broadcast_to(_ink(color), rgb.shape)
        if colored:
            lead = lead * (1 - colored) + np.clip(rgb * 0.85, 0, 1) * colored
        out = _ink(paper) * (1 - cov[..., None]) + lead * cov[..., None]
        out = out * (1 + (grain[..., None] - 0.5) * 0.08)
        return _out(img, np.clip(out, 0, 1))
    return f


def linocut(lines=7.0, angle=-20.0, bend=1.0, contrast=1.2, gamma=0.75, ink="#17130f", paper="#f2ebdc", detail=1.0,
            seed=0):
    """Linocut / woodcut: solid blacks, white highlights, carved line screen in the mid-tones whose
    lines bend with the forms (bend), wobbly gouge edges and paper specks in the ink. lines = line
    period in canvas px, detail = strength of carved contour edges, gamma < 1 = lighter."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        L = _levels(_lum(_flat(img))) ** gamma
        L = np.clip((cv2.bilateralFilter(np.ascontiguousarray(L, F32), -1, 0.12, max(1.0, 2 * s)) - 0.5) *
                    contrast + 0.5, 0, 1)
        per = max(2.0, lines * s)
        yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        a = math.radians(angle)
        big = _blur(L, per * 2.5)
        wob = _noise(h, w, s, lines * 12, 3, seed) - 0.5
        v = (-xx * math.sin(a) + yy * math.cos(a)) / per + big * bend * 3 + wob * 0.8
        tri = np.abs(v - np.floor(v) - 0.5) * 2                      # 0 on the line, 1 between
        jag = (_noise(h, w, s, 4.0, 2, seed + 1) - 0.5) * 0.05
        screen = 0.3 + 0.5 * tri
        ink_cov = np.clip((screen - L + jag) * per * 0.9 + 0.5, 0, 1)
        if detail:                                                   # carved outlines along contours
            g1, g2 = _blur(L, 0.8 * s), _blur(L, 2.4 * s)
            dog = np.clip((g2 - g1) * 14 * detail - 0.15, 0, 1)
            ink_cov = np.maximum(ink_cov, dog)
            ink_cov *= 1 - np.clip((g1 - g2) * 10 * detail - 0.2, 0, 1) * (L < 0.3)   # white cut lines in darks
        speck = _noise(h, w, s, 1.5, 2, seed + 2)
        ink_cov = ink_cov * (1 - np.clip((speck - 0.86) * 14, 0, 1) * 0.7)
        out = _ink(paper) * (1 - ink_cov[..., None]) + _ink(ink) * ink_cov[..., None]
        return _out(img, out)
    return f


# ------------------------------------------------------------------ segmentation filters
def _cells(h, w, s, cell, rgb, edges, seed):
    """Voronoi labels (h, w) with seeds denser along image edges; returns labels, distance to seed."""
    rng = np.random.default_rng(seed)
    n = max(4, int(h * w / (cell * s) ** 2))
    L = _lum(rgb)
    g = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
    g = _blur(g, cell * s * 0.3)
    dens = 1 + edges * g / max(float(g.mean()), 1e-6) * 0.5
    sub = max(1, int(cell * s / 4))
    dsm = dens[::sub, ::sub]
    p = dsm.ravel().astype(np.float64)
    idx = rng.choice(p.size, size=n, replace=False, p=p / p.sum()) if n < p.size else np.arange(p.size)
    ys = np.clip((idx // dsm.shape[1]) * sub + rng.integers(0, sub, len(idx)), 0, h - 1)
    xs = np.clip((idx % dsm.shape[1]) * sub + rng.integers(0, sub, len(idx)), 0, w - 1)
    src = np.full((h, w), 255, np.uint8)
    src[ys, xs] = 0
    dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(h, w), dist


def _boundary(lab):
    e = np.zeros(lab.shape, bool)
    dx = lab[:, 1:] != lab[:, :-1]
    dy = lab[1:] != lab[:-1]
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:] |= dy
    e[:-1] |= dy
    return e


def _cell_mean(lab, rgb):
    n = int(lab.max()) + 1
    cnt = np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
    m = np.stack([np.bincount(lab.ravel(), rgb[..., c].ravel(), n) / cnt for c in range(3)], -1)
    return m.astype(F32)


def stained_glass(cell=26.0, lead=3.0, lead_color="#1c1a19", glow=0.35, saturation=1.35, edges=2.0, seed=0):
    """Stained glass: Voronoi panes (cell = mean size in canvas px, denser along image edges), each a
    saturated mean colour with light glowing through (glow) and mottled glass, joined by lead lines."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        rgb = _flat(img)
        lab, dist = _cells(h, w, s, cell, rgb, edges, seed)
        m = _cell_mean(lab, rgb)
        hsv = rgb2hsv(m)
        hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0, 1)
        hsv[:, 2] = np.clip(hsv[:, 2], 0, 1) ** 0.7
        pane = np.clip(hsv2rgb(hsv), 0, 1)[lab]
        r = np.clip(dist / (cell * s * 0.6), 0, 1)
        light = 1 + glow * (0.6 - r)                                     # brighter in the middle of a pane
        mott = _noise(h, w, s, cell * 0.35, 3, seed + 1)
        pane = np.clip(pane * (light * (1 + (mott - 0.5) * 0.35))[..., None], 0, 1)
        cov = _blur(_line_cov(_boundary(lab), lead * s / 2 - 0.5), 0.4 * s)
        out = pane * (1 - cov[..., None]) + _ink(lead_color) * cov[..., None]
        return _out(img, out)
    return f


def mosaic(tile=10.0, grout=1.2, grout_color="#d9d2c3", jitter=0.3, vary=0.06, seed=0):
    """Mosaic: rows of square tesserae (tile canvas px, staggered, hand-laid wobble = jitter) in the mean
    colour of the image under each, with brightness variation, bevelled edges and grout gaps."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        rgb = _flat(img)
        T = max(2.0, tile * s)
        yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        xw = xx + (_noise(h, w, s, tile * 6, 2, seed) - 0.5) * 2 * jitter * T
        yw = yy + (_noise(h, w, s, tile * 6, 2, seed + 1) - 0.5) * 2 * jitter * T
        row = np.floor(yw / T)
        xs = xw + (row % 2) * T * 0.5
        col = np.floor(xs / T)
        _, lab = np.unique((row * 100003 + col).astype(np.int64), return_inverse=True)
        lab = lab.reshape(h, w)
        m = _cell_mean(lab, rgb)
        rng = np.random.default_rng(seed)
        m = np.clip(m * (1 + rng.normal(0, vary, (len(m), 1))), 0, 1).astype(F32)
        fx, fy = xs / T - col, yw / T - row
        d = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)) * T          # px to tile border
        bevel = 1 - 0.18 * np.clip(1 - d / (T * 0.22), 0, 1) + 0.06 * (0.5 - fy)
        out = m[lab] * bevel[..., None]
        g = np.clip(grout * s / 2 + 0.5 - d, 0, 1)
        out = out * (1 - g[..., None]) + _ink(grout_color) * g[..., None]
        return _out(img, np.clip(out, 0, 1))
    return f


__all__ = ["toon", "outlines", "render_hatch", "shadow_dots", "shading", "kuwahara", "edge_smooth", "xdog",
           "watercolorize", "pencil", "linocut", "stained_glass", "mosaic"]
