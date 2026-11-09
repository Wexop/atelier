"""Effects and grading. Each function returns an effect f(img, s=1) -> img working on straight
RGBA float32 (s = render scale; radii are in canvas px). Use with c.apply(...) (whole image)
or layer.apply(...) (one layer); both accept mask=. Effects also work on plain arrays:
blur(4)(arr).
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from PIL import Image

from .core import F32, rgba, premul, unpremul, color as _col, to_linear, to_srgb, resolve, resolve_mask, Layer, Canvas
from .field import rgb2oklab, oklab2rgb, make_lut, PALETTES, _apply_lut

LUMA = np.array([0.2126, 0.7152, 0.0722], F32)


def _lum(rgb):
    return rgb[..., :3] @ LUMA


def _gblur(a, sigma):
    """Gaussian blur (sigma in raster px); large sigmas go through a downsampled pyramid."""
    if sigma <= 0.3:
        return a
    h, w = a.shape[:2]
    if sigma > 8:
        f = sigma / 4
        sw, sh = max(1, round(w / f)), max(1, round(h / f))
        small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma * sw / w)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return out.reshape(a.shape)
    return cv2.GaussianBlur(a, (0, 0), sigma).reshape(a.shape)


def _spatial(img, fn):
    """Run a spatial op on premultiplied RGBA so transparent edges don't bleed dark."""
    pm = fn(premul(img))
    return unpremul(np.clip(pm, 0, None))


def _color(img, fn):
    out = img.copy()
    out[..., :3] = fn(img[..., :3])
    return out


def _arr(src, img, s):
    """Resolve a Field / array / colour to the image raster size."""
    h, w = img.shape[:2]
    return resolve(src, h, w, None, s)


# ------------------------------------------------------------------ blur & sharpen
def blur(radius=4.0):
    """Gaussian blur; radius ~ sigma in canvas px."""
    return lambda img, s=1.0: _spatial(img, lambda pm: _gblur(pm, radius * s))


def motion_blur(length=20.0, angle=0.0):
    """Linear motion blur; angle in degrees (0 = horizontal)."""
    def f(img, s=1.0):
        L = max(1, int(round(length * s)))
        k = np.zeros((L * 2 + 1, L * 2 + 1), F32)
        c = L
        dx, dy = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
        for t in np.linspace(-L / 2, L / 2, L * 2 + 1):
            k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
        k /= k.sum()
        return _spatial(img, lambda pm: cv2.filter2D(pm, -1, k, borderType=cv2.BORDER_REFLECT))
    return f


def sharpen(amount=0.6, radius=1.0):
    """Unsharp mask."""
    def f(img, s=1.0):
        b = _gblur(img[..., :3], radius * s)
        return _color(img, lambda c: np.clip(c + (c - b) * amount, 0, 1))
    return f


# ------------------------------------------------------------------ light
def bloom(threshold=0.75, radius=12.0, strength=0.8, tint=None):
    """Glow around bright areas (computed in linear light, 3 radii)."""
    def f(img, s=1.0):
        lin = to_linear(img[..., :3])
        l = lin @ LUMA
        t = to_linear(np.float32(threshold))
        knee = np.clip((l - t) / max(1 - t, 1e-3), 0, 1) ** 2
        bright = lin * knee[..., None]
        if tint is not None:
            bright = bright * to_linear(_col(tint)[:3])
        acc = sum(_gblur(bright, radius * s * m) * wgt for m, wgt in ((0.5, 0.2), (1, 0.4), (3, 0.4)))
        return _color(img, lambda c: np.clip(to_srgb(lin + acc * strength), 0, 1))
    return f


def glow(radius=10.0, strength=1.0, color=None):
    """Halo around a layer's content (neon text, sprites). color=None uses the content colour."""
    def f(img, s=1.0):
        pm = premul(img)
        halo = _gblur(pm, radius * s)
        if color is not None:
            halo = halo[..., 3:4] * np.append(_col(color)[:3], 1)[None, None]
        halo = np.clip(halo * strength, 0, 1)
        out = pm + halo * (1 - pm[..., 3:4])
        return unpremul(np.clip(out, 0, 1))
    return f


def drop_shadow(dx=6.0, dy=8.0, radius=8.0, color="#000000", opacity=0.5):
    """Shadow behind a layer's content."""
    def f(img, s=1.0):
        a = img[..., 3]
        M = np.float32([[1, 0, dx * s], [0, 1, dy * s]])
        sh = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
        sh = _gblur(sh, radius * s) * opacity
        shadow = np.concatenate([np.broadcast_to(_col(color)[:3], img.shape[:2] + (3,)), sh[..., None]], -1)
        pm = premul(img)
        out = pm + premul(shadow) * (1 - pm[..., 3:4])
        return unpremul(out)
    return f


def inner_shadow(dx=4.0, dy=4.0, radius=6.0, color="#000000", opacity=0.6):
    """Shadow inside a layer's content (engraved / inset look)."""
    def f(img, s=1.0):
        a = img[..., 3]
        M = np.float32([[1, 0, dx * s], [0, 1, dy * s]])
        inv = cv2.warpAffine(1 - a, M, (a.shape[1], a.shape[0]), borderValue=1)
        k = (_gblur(inv, radius * s) * a * opacity)[..., None]
        return _color(img, lambda c: c * (1 - k) + _col(color)[:3] * k)
    return f


def vignette(amount=0.35, softness=0.6, roundness=1.0):
    """Darken edges. roundness 1 = ellipse following the frame."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(F32)
        u, v = (x / w - 0.5) * 2, (y / h - 0.5) * 2
        ar = w / h
        u = u * (ar ** (1 - roundness))
        d = np.sqrt(u * u + v * v) / math.sqrt(2) * 1.25
        k = np.clip((d - (1 - softness)) / max(softness, 1e-3), 0, 1)
        k = 1 - amount * (k * k * (3 - 2 * k))
        return _color(img, lambda c: c * k[..., None])
    return f


def grain(amount=0.04, size=1.0, seed=0, color=False):
    """Film grain, strongest in mid-tones; size in canvas px; deterministic by seed."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        g = max(size * s, 1.0)
        gh, gw = max(1, round(h / g)), max(1, round(w / g))
        rng = np.random.default_rng(seed)
        n = rng.standard_normal((gh, gw, 3 if color else 1)).astype(F32)
        if (gh, gw) != (h, w):
            n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC).reshape(h, w, -1)
        l = _lum(img)[..., None]
        wgt = 0.12 + 3.2 * l * (1 - l)
        return _color(img, lambda c: np.clip(c + n * amount * wgt, 0, 1))
    return f


def chromatic_aberration(amount=2.0):
    """Radial RGB split; amount = px offset at the corners (R out, B in)."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(F32)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        r = math.hypot(cx, cy)
        out = img.copy()
        for ch, sign in ((0, 1), (2, -1)):
            k = 1 - sign * amount * s / r
            mx, my = cx + (x - cx) * k, cy + (y - cy) * k
            out[..., ch] = cv2.remap(img[..., ch], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out
    return f


def lens_distortion(k=0.08, zoom=None):
    """Barrel (k>0) / pincushion (k<0) distortion; zoom auto-crops borders."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(F32)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        nx, ny = (x - cx) / cx, (y - cy) / cx
        r2 = nx * nx + ny * ny
        z = zoom if zoom is not None else 1 / (1 + max(k, 0) * (1 + (h / w) ** 2))
        fct = (1 + k * r2) * z
        mx, my = cx + nx * fct * cx, cy + ny * fct * cx
        pm = premul(img)
        out = cv2.remap(pm, mx.astype(F32), my.astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return unpremul(out)
    return f


# ------------------------------------------------------------------ tone & colour
def exposure(stops=0.5):
    return lambda img, s=1.0: _color(img, lambda c: np.clip(to_srgb(to_linear(c) * 2 ** stops), 0, 1))


def contrast(amount=0.2, pivot=0.5):
    """S-free linear contrast around pivot (amount -1..1)."""
    return lambda img, s=1.0: _color(img, lambda c: np.clip((c - pivot) * (1 + amount) + pivot, 0, 1))


def levels(black=0.0, white=1.0, gamma=1.0, out_black=0.0, out_white=1.0):
    def fn(c):
        t = np.clip((c - black) / max(white - black, 1e-6), 0, 1) ** (1 / gamma)
        return out_black + t * (out_white - out_black)
    return lambda img, s=1.0: _color(img, fn)


def curves(points, channel="rgb"):
    """Tone curve through (in, out) points in [0,1] (monotone cubic). channel: rgb | r | g | b."""
    pts = sorted(points)
    if pts[0][0] > 0:
        pts = [(0.0, pts[0][1] if len(pts) == 1 else 0.0)] + pts
    if pts[-1][0] < 1:
        pts = pts + [(1.0, 1.0)]
    from scipy.interpolate import PchipInterpolator  # lazy: slow import
    xs, ys = zip(*pts)
    table = np.clip(PchipInterpolator(xs, ys)(np.linspace(0, 1, 1024)), 0, 1).astype(F32)
    idx = {"rgb": [0, 1, 2], "r": [0], "g": [1], "b": [2]}[channel]
    def fn(c):
        c = c.copy()
        for i in idx:
            c[..., i] = table[np.clip((c[..., i] * 1023 + 0.5).astype(np.int32), 0, 1023)]
        return c
    return lambda img, s=1.0: _color(img, fn)


def lut(table):
    """Per-channel 1D LUT: (n,) or (n,3) array over [0,1], or a function c -> c."""
    if callable(table):
        return lambda img, s=1.0: _color(img, lambda c: np.clip(table(c), 0, 1))
    t = np.asarray(table, F32)
    if t.ndim == 1:
        t = np.repeat(t[:, None], 3, 1)
    n = len(t)
    def fn(c):
        i = np.clip((c * (n - 1) + 0.5).astype(np.int32), 0, n - 1)
        return np.stack([t[i[..., k], k] for k in range(3)], -1)
    return lambda img, s=1.0: _color(img, fn)


def hue_sat(hue=0.0, sat=1.0, light=0.0):
    """Hue rotation (degrees), saturation multiplier, lightness offset — in OKLab (perceptual)."""
    ch, sh = math.cos(math.radians(hue)), math.sin(math.radians(hue))
    def fn(c):
        lab = rgb2oklab(c)
        a, b = lab[..., 1], lab[..., 2]
        a2, b2 = (a * ch - b * sh) * sat, (a * sh + b * ch) * sat
        return oklab2rgb(np.stack([lab[..., 0] + light, a2, b2], -1))
    return lambda img, s=1.0: _color(img, fn)


def saturation(amount=1.2):
    return hue_sat(sat=amount)


def color_balance(shadows=(0, 0, 0), mids=(0, 0, 0), highlights=(0, 0, 0)):
    """Add RGB offsets (-1..1) weighted by tonal range, e.g. teal shadows / warm highlights:
    color_balance(shadows=(-.03, 0, .05), highlights=(.05, .02, -.04))."""
    S, M, Hh = (np.asarray(v, F32) for v in (shadows, mids, highlights))
    def fn(c):
        l = _lum(c)[..., None]
        ws, wh = (1 - l) ** 2, l ** 2
        wm = np.clip(1 - ws - wh, 0, 1)
        return np.clip(c + S * ws + M * wm + Hh * wh, 0, 1)
    return lambda img, s=1.0: _color(img, fn)


def gradient_map(stops="sunset", amount=1.0):
    """Map luminance onto a palette (duotone / tritone looks)."""
    table = make_lut(stops)
    def fn(c):
        m = _apply_lut(_lum(c), table)[..., :3]
        return c * (1 - amount) + m * amount
    return lambda img, s=1.0: _color(img, fn)


def invert():
    return lambda img, s=1.0: _color(img, lambda c: 1 - c)


_AGX = np.array([[0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                 [0.0784335999999992, 0.878468636469772, 0.0784336],
                 [0.0792237451477643, 0.0791661274605434, 0.879142973793104]], F32)
_AGX_INV = np.array([[1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                     [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                     [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]], F32)


def tonemap_linear(lin, kind="agx", exposure=0.0, look=None):
    """Linear HDR rgb -> display sRGB [0,1]. kind: agx | filmic (ACES fit) | reinhard | clip.
    look (agx only): None (base, neutral) | 'punchy' (more contrast and saturation)."""
    x = np.maximum(lin, 0) * F32(2 ** exposure)
    if kind == "agx":
        v = x @ _AGX
        v = np.clip(np.log2(np.maximum(v, 1e-10)), -12.47393, 4.026069)
        v = (v + 12.47393) / (4.026069 + 12.47393)
        v2 = v * v
        v4 = v2 * v2
        v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232
        if look == "punchy":
            v = np.maximum(v, 0) ** 1.35
            l = (v @ LUMA)[..., None]
            v = l + 1.4 * (v - l)
        return np.clip(v @ _AGX_INV, 0, 1).astype(F32)
    if kind == "filmic":
        y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
        return to_srgb(np.clip(y, 0, 1))
    if kind == "reinhard":
        return to_srgb(x / (1 + x))
    return to_srgb(np.clip(x, 0, 1))


def tone_map(kind="agx", exposure=0.0, look="punchy"):
    """Filmic tone mapping of an sRGB image (linearised first): soft highlight roll-off.
    On 2D art use exposure ~0.5-1.5; look='punchy' keeps contrast/saturation (agx)."""
    return lambda img, s=1.0: _color(img, lambda c: tonemap_linear(to_linear(c), kind, exposure, look))


# ------------------------------------------------------------------ distortion & stylise
def displace(field, strength=10.0, field_y=None):
    """Offset pixels by a field ([0,1], 0.5 = no move); strength in canvas px."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        fx = resolve_mask(field, h, w, None, s)
        if field_y is not None:
            fy = resolve_mask(field_y, h, w, None, s)
        elif hasattr(field, "move"):
            fy = resolve_mask(field.move(0.37, 0.21), h, w, None, s)
        else:
            fy = np.roll(fx, (h // 3, w // 3), (0, 1))
        y, x = np.mgrid[0:h, 0:w].astype(F32)
        mx = x + (fx - 0.5) * 2 * strength * s
        my = y + (fy - 0.5) * 2 * strength * s
        out = cv2.remap(premul(img), mx.astype(F32), my.astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return unpremul(out)
    return f


def posterize(levels=4):
    n = max(2, int(levels)) - 1
    return lambda img, s=1.0: _color(img, lambda c: np.round(c * n) / n)


_BAYER8 = (np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                     [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                     [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                     [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], F32) + 0.5) / 64 - 0.5


def _palette_arr(palette, img, colors):
    if palette is None:
        q = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").quantize(
            colors or 16, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = np.array(q.getpalette()[:3 * (colors or 16)], F32).reshape(-1, 3) / 255
        return pal
    if isinstance(palette, str):
        palette = PALETTES[palette]
    return np.array([_col(c)[:3] for c in palette], F32)


def quantize(img_rgb, pal, dither=None, strength=1.0):
    """Map rgb (h,w,3) to nearest palette colour (OKLab distance). dither: None | 'bayer' | 'fs'."""
    if dither == "fs":
        pimg = Image.new("P", (1, 1))
        flat = (np.clip(pal, 0, 1) * 255 + 0.5).astype(np.uint8).ravel().tolist()
        pimg.putpalette(flat + flat[:3] * (256 - len(pal)))
        src = Image.fromarray((np.clip(img_rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
        idx = np.asarray(src.quantize(palette=pimg, dither=Image.Dither.FLOYDSTEINBERG))
        return pal[np.clip(idx, 0, len(pal) - 1)]
    src = img_rgb
    if dither == "bayer":
        h, w = src.shape[:2]
        spread = strength * 1.0 / max(2, round(len(pal) ** (1 / 3)))
        t = np.tile(_BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
        src = np.clip(src + t[..., None] * spread, 0, 1)
    lab = rgb2oklab(src).reshape(-1, 3)
    plab = rgb2oklab(pal)
    best = np.empty(len(lab), np.int32)
    for i in range(0, len(lab), 262144):
        d = ((lab[i:i + 262144, None, :] - plab[None]) ** 2).sum(-1)
        best[i:i + 262144] = d.argmin(1)
    return pal[best].reshape(img_rgb.shape)


def pixelate(size=4.0, palette=None, colors=None, dither=None, strength=0.6):
    """Chunky pixels (size in canvas px) + optional palette ('retro16', list, or colors=N
    auto-palette) and dither ('bayer' | 'fs'). On a pixel=True canvas use size=1."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        k = max(1, int(round(size * s)))
        sw, sh = max(1, w // k), max(1, h // k)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA) if k > 1 else img
        rgb = small[..., :3]
        if palette is not None or colors:
            rgb = quantize(rgb, _palette_arr(palette, rgb, colors), dither, strength)
        a = np.where(small[..., 3:4] > 0.5, 1.0, 0.0).astype(F32) if (palette is not None or colors) else small[..., 3:4]
        out = np.concatenate([rgb, a], -1)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST) if k > 1 else out
    return f


def halftone(size=8.0, angle=45.0, ink="#111111", paper=None, cmyk=False):
    """Print-style dots; size = cell in canvas px. cmyk=True: 4 rotated screens."""
    def screen(dark, s, ang):
        h, w = dark.shape
        cell = size * s
        y, x = np.mgrid[0:h, 0:w].astype(F32)
        c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        u, v = (x * c + y * sn) / cell, (-x * sn + y * c) / cell
        du, dv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
        d = np.sqrt(du * du + dv * dv) * cell
        avg = _gblur(dark, cell * 0.45)
        r = np.sqrt(np.clip(avg, 0, 1)) * cell * 0.72
        return np.clip(r - d + 0.5, 0, 1)
    def f(img, s=1.0):
        rgb = img[..., :3]
        pap = _col(paper)[:3] if paper is not None else np.ones(3, F32)
        if not cmyk:
            cov = screen(1 - _lum(rgb), s, angle)[..., None]
            return _color(img, lambda c: pap * (1 - cov) + _col(ink)[:3] * cov)
        k = 1 - rgb.max(-1)
        cmy = (1 - rgb - k[..., None]) / np.maximum(1 - k[..., None], 1e-3)
        out = np.broadcast_to(pap, rgb.shape).copy()
        for ch, ang, col in ((0, 15, (0, .68, .94)), (1, 75, (.93, 0, .55)), (2, 0, (1, .95, 0))):
            cov = screen(cmy[..., ch] * (1 - k), s, ang)[..., None]
            out *= 1 - cov * (1 - np.array(col, F32))
        cov = screen(k, s, 45)[..., None]
        out *= 1 - cov * 0.92
        return _color(img, lambda c: out)
    return f


def depth_of_field(depth, focus, aperture=8.0, max_blur=None):
    """Depth blur. depth: (H,W) array (e.g. from the SDF renderer, same units as focus);
    aperture = blur radius in canvas px for objects far behind the focus plane."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        d = np.asarray(depth, F32)
        d = np.where(np.isfinite(d), d, F32(1e6))          # sky = infinitely far
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
        coc = aperture * s * np.abs(d - focus) / np.maximum(d, 1e-6)
        mb = (max_blur if max_blur is not None else aperture * 3) * s
        coc = np.clip(coc, 0, mb)
        pm = premul(img)
        pm[..., :3] = to_linear(np.clip(pm[..., :3], 0, None))
        levels_ = [0.0] + [mb * 2 ** -i for i in range(5, -1, -1)]
        stack = [pm] + [_gblur(pm, r) for r in levels_[1:]]
        out = np.zeros_like(pm)
        idx = np.interp(coc, levels_, np.arange(len(levels_))).astype(F32)
        lo = np.floor(idx).astype(np.int32)
        t = (idx - lo)[..., None]
        for i in range(len(levels_)):
            wgt = np.where(lo == i, 1 - t[..., 0], 0) + np.where(lo == i - 1, t[..., 0], 0)
            if wgt.any():
                out += stack[i] * wgt[..., None]
        out[..., :3] = to_srgb(out[..., :3])
        return unpremul(out)
    return f


# ------------------------------------------------------------------ masked apply
def _apply_layer(self, *fx, mask=None):
    """Apply effects to this layer; with mask=, only where the mask is (blended)."""
    img = self.rgba
    out = img
    for f in fx:
        out = f(out, self.s)
    if mask is not None:
        m = resolve_mask(mask, self.H, self.W, None, self.s)[..., None]
        out = img * (1 - m) + out * m
    self.rgba = out
    return self


def _apply_canvas(self, *fx, mask=None):
    """Merge all layers and apply effects to the whole image (final grading)."""
    return self.merge().apply(*fx, mask=mask)


Layer.apply = _apply_layer
Canvas.apply = _apply_canvas

__all__ = ["blur", "motion_blur", "sharpen", "bloom", "glow", "drop_shadow", "inner_shadow", "vignette",
           "grain", "chromatic_aberration", "lens_distortion", "exposure", "contrast", "levels", "curves",
           "lut", "hue_sat", "saturation", "color_balance", "gradient_map", "invert", "tone_map",
           "tonemap_linear", "displace", "posterize", "quantize", "pixelate", "halftone", "depth_of_field"]
