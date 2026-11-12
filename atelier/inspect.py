"""Self-critique tools: quality metrics, contact sheets, A/B compare, composition guides,
reference analysis and colour/tone transfer.

Images can be a Canvas, a Layer, a path, a PIL image or an array (H x W x 3/4, float 0-1 or uint8).
Coordinates reported to the user (crops, focal point) are in *canvas px* (raster px / canvas.s),
so they can be fed straight back to `atelier render scene.py --crop x,y,w,h`.

  m = metrics(c)                      # dict: noise, aliasing, banding, clipping, sharpness, colourfulness ...
  contact_sheet(c, "sheet.jpg")       # one image: view, value study, histogram, clip map, auto zooms, numbers
  compare("a.png", "b.png", "ab.jpg") # A | B | amplified diff | split
  guides(c, "guides.jpg")             # thirds, golden, diagonals, focal point, visual mass, horizon tilt
  ref = analyze_reference("ref.jpg")  # palette, value thumbnail, light direction, key, contrast
  c.apply(match_palette("ref.jpg", .8), transfer_tone("ref.jpg", .5))
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import F32, load, rgba, to_u8
from .field import rgb2oklab, oklab2rgb

LUMA = np.array([0.2126, 0.7152, 0.0722], F32)
_FONTS = os.path.join(os.path.dirname(__file__), "fonts")
_FCACHE: dict = {}

# thresholds used to flag problems (also printed in the sheet)
LIMITS = {"clip_highlights": 1.0, "clip_shadows": 1.0, "clip_channels": 5.0, "noise": 4.0,
          "aliasing": 35.0, "banding": 2.0}


# ---------------------------------------------------------------- input helpers
def _src(img, bg=0.5):
    """Anything image-like -> (rgb float32 H x W x 3 in [0,1], scale s raster px per canvas px).
    Transparent areas are composited over grey `bg`. Canvas input is quantised like an export
    (dithered 8-bit), so metrics see what the saved file will contain."""
    s = 1.0
    if hasattr(img, "flatten") and hasattr(img, "layers"):      # Canvas
        s = float(img.s)
        a = img.flatten()
        a = to_u8(np.clip(a, 0, 1)).astype(F32) / 255
    elif hasattr(img, "px") and hasattr(img, "canvas"):        # Layer
        s = float(img.s)
        a = to_u8(np.clip(img.rgba, 0, 1)).astype(F32) / 255
    else:
        a = load(img) if not isinstance(img, np.ndarray) else rgba(img.astype(F32) / 255 if img.dtype == np.uint8 else img)
    a = rgba(a)
    rgb = a[..., :3] * a[..., 3:4] + bg * (1 - a[..., 3:4])
    return np.ascontiguousarray(np.clip(rgb, 0, 1), F32), s


def _font(size, kind="sans"):
    key = (size, kind)
    if key not in _FCACHE:
        name = {"sans": "Inter-Regular.otf", "bold": "Inter-SemiBold.otf", "mono": "JetBrainsMono-Regular.ttf"}[kind]
        try:
            _FCACHE[key] = ImageFont.truetype(os.path.join(_FONTS, name), size)
        except OSError:
            _FCACHE[key] = ImageFont.load_default()
    return _FCACHE[key]


def _box(a, k):
    k = max(1, int(k))
    return cv2.boxFilter(np.asarray(a, F32), -1, (k, k), borderType=cv2.BORDER_REFLECT)


def _hex(rgb):
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(rgb, 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------- analysis core
def _noise_map(Y):
    """Immerkaer noise estimate on luma (8-bit levels) restricted to low-structure pixels."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], F32)
    r = np.abs(cv2.filter2D(Y, -1, k, borderType=cv2.BORDER_REFLECT))
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    flat = (g <= max(np.percentile(g, 40), 1e-3) + 1e-6).astype(F32)
    c = math.sqrt(math.pi / 2) / 6
    sigma = float(c * (r * flat).sum() / max(flat.sum(), 1))
    return sigma, r * c, flat, g


def _alias_map(Y):
    """Hard (1-pixel) steps on diagonal edges = jaggies. Y in [0,1]. Returns (hard, strong) maps."""
    H, W = Y.shape
    Yb = cv2.GaussianBlur(Y, (0, 0), 1.2)
    gx = cv2.Sobel(Yb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Yb, cv2.CV_32F, 0, 1, ksize=3)
    ax, ay = np.abs(gx), np.abs(gy)
    diag = np.minimum(ax, ay) > 0.3 * np.maximum(ax, ay)       # 17..73 deg
    hard = np.zeros((H, W), F32)
    strong = np.zeros((H, W), F32)
    for axis, dom in ((1, ax >= ay), (0, ay > ax)):
        Z = Y if axis == 1 else Y.T
        D = (dom & diag) if axis == 1 else (dom & diag).T
        d1 = Z[:, 2:-1] - Z[:, 1:-2]          # step x -> x+1
        d3 = Z[:, 3:] - Z[:, :-3]             # step x-1 -> x+2
        dl = np.abs(Z[:, 1:-2] - Z[:, :-3])
        dr = np.abs(Z[:, 3:] - Z[:, 2:-1])
        st = (np.abs(d3) > 0.2) & D[:, 1:-2] & (np.abs(d1) >= dl) & (np.abs(d1) >= dr)
        hd = st & (np.abs(d1) >= 0.88 * np.abs(d3)) & (d1 * d3 > 0)
        sm = np.zeros(Z.shape, F32)
        hm = np.zeros(Z.shape, F32)
        sm[:, 1:-2] = st
        hm[:, 1:-2] = hd
        if axis == 0:
            sm, hm = sm.T, hm.T
        strong += sm
        hard += hm
    return np.minimum(hard, 1), np.minimum(strong, 1)


def _banding_map(q):
    """Banding = wide plateaus separated by small steps (<= 20 levels) inside slow gradients.
    q = uint8-valued float RGB (0-255). Returns weighted per-pixel banding evidence [0,1]."""
    m = np.zeros(q.shape[:2], F32)
    for dy, dx in ((0, 1), (1, 0)):
        d = np.abs(q[dy:, dx:] - q[:q.shape[0] - dy, :q.shape[1] - dx]).max(-1)
        if dx:
            m[:, 1:] = np.maximum(m[:, 1:], d)
            m[:, :-1] = np.maximum(m[:, :-1], d)
        else:
            m[1:] = np.maximum(m[1:], d)
            m[:-1] = np.maximum(m[:-1], d)
    plateau = m < 0.5
    near = cv2.dilate(m, np.ones((31, 31), np.uint8))
    k61 = np.ones((61, 61), np.uint8)
    maxstep = cv2.dilate(m, k61)
    rng = (cv2.dilate(q, k61) - cv2.erode(q, k61)).max(-1)
    Y = q @ LUMA
    S = cv2.GaussianBlur(Y, (0, 0), 6)
    slope = np.hypot(cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=3)) / 8
    # staircase: several small steps (range >= ~2 steps) and no big shape edge nearby
    ok = plateau & (near >= 0.5) & (maxstep <= 20) & (rng >= 1.8 * maxstep) & (slope > 0.03)
    # how wide plateaus are: a dithered gradient has tiny plateaus, a banded one wide ones
    wide = _box(plateau.astype(F32), 9) > 0.85
    w = np.clip(near / 3, 0.34, 1)
    return (ok & wide).astype(F32) * w


def _pixel_grid(rgb):
    """Block size k if the image is nearest-upscaled pixel art (k x k identical blocks), else 1."""
    H, W = rgb.shape[:2]
    for k in (8, 6, 5, 4, 3, 2):
        h, w = H - H % k, W - W % k
        if h < 4 * k or w < 4 * k:
            continue
        a = rgb[:h, :w]
        rep = np.repeat(np.repeat(a[::k, ::k], k, 0), k, 1)
        if float((np.abs(rep - a).max(-1) > 3 / 255).mean()) < 0.002:
            return k
    return 1


def _analyze(img, maps=False):
    rgb, s = _src(img)
    H, W = rgb.shape[:2]
    grid = _pixel_grid(rgb)
    pixel_art = bool(getattr(img, "pixel", False)) or grid > 1
    q = np.round(rgb * 255).astype(F32)
    Y8 = q @ LUMA
    Y = Y8 / 255
    lab = rgb2oklab(rgb)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    mn, mx = rgb.min(-1), rgb.max(-1)
    blown = mn >= 254.5 / 255
    crushed = mx <= 0.5 / 255
    chan = ((mx >= 254.5 / 255) | (mn <= 0.5 / 255)) & ~blown & ~crushed & (C > 0.08)
    sigma, nmap, flat, grad = _noise_map(Y8)
    hard, strong = _alias_map(Y)
    band = _banding_map(q)
    rg = q[..., 0] - q[..., 1]
    yb = 0.5 * (q[..., 0] + q[..., 1]) - q[..., 2]
    colorful = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    dx = np.abs(np.diff(Y8, axis=1))[:-1]
    dy = np.abs(np.diff(Y8, axis=0))[:, :-1]
    gmag = np.hypot(dx, dy)
    bins = np.array([0, .2, .4, .6, .8, 1.01], F32)
    vals = np.histogram(np.clip(L, 0, 1), bins)[0] / L.size * 100
    nstrong = float(strong.sum())
    m = {
        "size": (W, H), "scale": round(s, 4),
        "mean": round(float(L.mean() * 100), 1),
        "contrast": round(float(L.std() * 100), 1),
        "range": (round(float(np.percentile(L, 2) * 100), 1), round(float(np.percentile(L, 98) * 100), 1)),
        "clip_shadows": round(float(crushed.mean() * 100), 2),
        "clip_highlights": round(float(blown.mean() * 100), 2),
        "clip_channels": round(float(chan.mean() * 100), 2),
        "saturation": round(float(C.mean() * 100), 1),
        "colorfulness": round(float(colorful), 1),
        "noise": round(sigma, 2),
        "aliasing": round(float(hard.sum() / nstrong * 100) if nstrong > 50 else 0.0, 1),
        "banding": round(float(band.mean() * 100), 2),
        "sharpness": round(float(np.percentile(gmag, 99.5)), 1),
        "values": [round(float(v), 1) for v in vals],
        "pixel_grid": grid, "pixel_art": pixel_art,
    }
    m["flags"] = _flags(m)
    if not maps:
        return m
    mp = {"pixel_art": pixel_art, "rgb": rgb, "s": s, "L": L, "C": C, "blown": blown, "crushed": crushed, "chan": chan,
          "noise": nmap * flat, "flat": flat, "hard": hard, "strong": strong, "band": band, "grad": grad, "Y": Y}
    return m, mp


def _flags(m):
    f = []
    if m["clip_highlights"] > LIMITS["clip_highlights"]:
        f.append(f"highlights clipped {m['clip_highlights']}% (pure white areas lose detail)")
    if m["clip_shadows"] > LIMITS["clip_shadows"]:
        f.append(f"shadows crushed {m['clip_shadows']}% (pure black)")
    if m["clip_channels"] > LIMITS["clip_channels"]:
        f.append(f"{m['clip_channels']}% saturated colours hit 0/255 in a channel (hue shifts, flat colour)")
    if m["noise"] > LIMITS["noise"]:
        f.append(f"noise/grain {m['noise']} levels in flat areas")
    if m["aliasing"] > LIMITS["aliasing"] and not m.get("pixel_art"):
        f.append(f"aliasing: {m['aliasing']}% of diagonal edges are hard 1-px steps (jaggies)")
    if m["banding"] > LIMITS["banding"]:
        f.append(f"banding in gradients over {m['banding']}% of the image")
    if m["contrast"] < 12:
        f.append(f"low contrast (value std {m['contrast']})")
    if m["range"][1] - m["range"][0] < 45:
        f.append(f"narrow value range {m['range'][0]}-{m['range'][1]} (no real darks or lights)")
    v = m["values"]
    if v[0] + v[1] < 3:
        f.append(f"no darks: only {v[0] + v[1]:.1f}% of pixels in the 2 darkest value bands")
    if v[3] + v[4] < 3:
        f.append(f"no lights: only {v[3] + v[4]:.1f}% of pixels in the 2 lightest value bands")
    if max(m["values"]) > 60:
        f.append("one value band covers >60% of the image (weak value structure)")
    return f


def metrics(img) -> dict:
    """Image-quality numbers (all floats, image in canvas or file px):
    mean/contrast/range  OKLab lightness x100 (mean, std, 2-98 percentile)
    clip_shadows/clip_highlights  % pure black / pure white;  clip_channels  % colours with a channel at 0/255
    saturation  mean OKLab chroma x100 (0 grey, 10 muted, 20+ vivid);  colorfulness  Hasler-Susstrunk (<15 dull, >60 vivid)
    noise  sigma in 8-bit levels in flat areas (<1 clean, >4 grainy);  aliasing  % hard 1-px steps on diagonal edges (AA'd ~10-30, jaggy >50)
    banding  % of image with stepped gradients;  sharpness  99.5th pct gradient in levels/px
    values  % of pixels in 5 lightness bands dark->light;  flags  list of human-readable problems."""
    return _analyze(img)


# ---------------------------------------------------------------- composition
def saliency(img, size=128):
    """Saliency-like map [0,1] (H x W of the image): spectral residual + lightness contrast + chroma."""
    rgb = img if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3 and img.dtype == F32 else _src(img)[0]
    H, W = rgb.shape[:2]
    k = size / max(H, W)
    sw, sh = max(8, round(W * k)), max(8, round(H * k))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    lab = rgb2oklab(small)
    L = lab[..., 0]
    F = np.fft.fft2(L)
    logA = np.log(np.abs(F) + 1e-6)
    res = logA - cv2.blur(logA.astype(F32), (3, 3))
    sr = np.abs(np.fft.ifft2(np.exp(res + 1j * np.angle(F)))) ** 2
    sr = cv2.GaussianBlur(sr.astype(F32), (0, 0), 2.5)
    blur = cv2.GaussianBlur(lab, (0, 0), max(sw, sh) / 8)
    contrast = np.sqrt(((lab - blur) ** 2).sum(-1) + 1e-12)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    nz = lambda a: (a - a.min()) / (np.ptp(a) + 1e-9)
    sal = 0.4 * nz(sr) + 0.4 * nz(cv2.GaussianBlur(contrast, (0, 0), 2)) + 0.2 * nz(chroma)
    sal = nz(cv2.GaussianBlur(sal.astype(F32), (0, 0), 1.5))
    # mild centre-free border suppression (frames rarely hold the subject)
    yy, xx = np.mgrid[0:sh, 0:sw]
    edge = np.minimum.reduce([xx, yy, sw - 1 - xx, sh - 1 - yy]) / (0.06 * max(sw, sh))
    sal = sal * np.clip(edge, 0.3, 1)
    return cv2.resize(sal.astype(F32), (W, H), interpolation=cv2.INTER_LINEAR)


def _tilt(rgb):
    """Dominant long near-horizontal line (horizon) -> (angle deg, (x0,y0,x1,y1) raster px) or None."""
    H, W = rgb.shape[:2]
    k = min(1.0, 800 / max(H, W))
    g = cv2.resize((rgb @ LUMA * 255).astype(np.uint8), (max(1, round(W * k)), max(1, round(H * k))), interpolation=cv2.INTER_AREA)
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 30, 90)
    lines = cv2.HoughLinesP(e, 1, np.pi / 720, 60, minLineLength=int(0.3 * g.shape[1]), maxLineGap=12)
    if lines is None:
        return None
    best = None
    for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4).tolist():
        ang = math.degrees(math.atan2(-(y1 - y0), x1 - x0))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        if abs(ang) > 8:
            continue
        ln = math.hypot(x1 - x0, y1 - y0)
        if best is None or ln > best[0]:
            best = (ln, ang, (x0 / k, y0 / k, x1 / k, y1 / k))
    return None if best is None else (round(best[1], 2), best[2])


def composition(img) -> dict:
    """Focal point (saliency peak), centre of visual mass, balance and horizon tilt, in canvas px."""
    rgb, s = _src(img)
    H, W = rgb.shape[:2]
    sal = saliency(rgb)
    b = cv2.GaussianBlur(sal, (0, 0), max(H, W) / 40)
    fy, fx = np.unravel_index(int(np.argmax(b)), b.shape)
    w = sal ** 2
    yy, xx = np.mgrid[0:H, 0:W]
    my, mx_ = float((w * yy).sum() / w.sum()), float((w * xx).sum() / w.sum())
    pts = [(W * a, H * c) for a in (1 / 3, 2 / 3) for c in (1 / 3, 2 / 3)]
    near = min(math.hypot(fx - px, fy - py) for px, py in pts) / math.hypot(W, H)
    t = _tilt(rgb)
    out = {
        "focal": (round(fx / s), round(fy / s)),
        "mass": (round(mx_ / s), round(my / s)),
        "balance": (round((mx_ / W - 0.5) * 2, 2), round((my / H - 0.5) * 2, 2)),
        "focal_thirds_dist": round(near, 3),
        "tilt": None if t is None else t[0],
        "_horizon": None if t is None else t[1],
        "_sal": sal, "_s": s,
    }
    return out


def _draw_guides(im, comp, W, H, k=1.0, lw=2, which="tgdfm"):
    """Draw composition guides on a PIL image showing the raster scaled by k."""
    d = ImageDraw.Draw(im, "RGBA")
    w, h = W * k, H * k
    if "t" in which:
        for f in (1 / 3, 2 / 3):
            d.line([(w * f, 0), (w * f, h)], fill=(0, 230, 255, 200), width=lw)
            d.line([(0, h * f), (w, h * f)], fill=(0, 230, 255, 200), width=lw)
    if "g" in which:
        for f in (0.382, 0.618):
            d.line([(w * f, 0), (w * f, h)], fill=(255, 210, 0, 150), width=max(1, lw - 1))
            d.line([(0, h * f), (w, h * f)], fill=(255, 210, 0, 150), width=max(1, lw - 1))
    if "d" in which:
        d.line([(0, 0), (w, h)], fill=(255, 255, 255, 90), width=max(1, lw - 1))
        d.line([(0, h), (w, 0)], fill=(255, 255, 255, 90), width=max(1, lw - 1))
    s = comp["_s"]
    if "m" in which:
        x, y = comp["mass"][0] * s * k, comp["mass"][1] * s * k
        r = max(6, 0.025 * max(w, h))
        d.rectangle([x - r, y - r, x + r, y + r], outline=(80, 255, 80, 255), width=lw + 1)
    if "f" in which:
        x, y = comp["focal"][0] * s * k, comp["focal"][1] * s * k
        r = max(8, 0.035 * max(w, h))
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 40, 200, 255), width=lw + 1)
        d.line([(x - r * 1.6, y), (x + r * 1.6, y)], fill=(255, 40, 200, 255), width=lw)
        d.line([(x, y - r * 1.6), (x, y + r * 1.6)], fill=(255, 40, 200, 255), width=lw)
    if comp.get("_horizon") is not None and comp["tilt"] is not None:
        x0, y0, x1, y1 = (v * k for v in comp["_horizon"])
        col = (255, 60, 60, 255) if 0.2 < abs(comp["tilt"]) < 5 else (120, 255, 120, 220)
        d.line([(x0, y0), (x1, y1)], fill=col, width=lw + 1)
    return im


def guides(img, path=None, width=1000):
    """Composition overlay: thirds (cyan), golden ratio (yellow), diagonals (white), focal point
    (magenta circle), centre of visual mass (green square), longest near-horizontal line
    (red if tilted 0.2-5 deg). Writes `path` if given; returns (overlay uint8 RGB, composition dict)."""
    rgb, _ = _src(img)
    H, W = rgb.shape[:2]
    comp = composition(img)
    k = width / W
    im = Image.fromarray(to_u8(cv2.resize(rgb, (width, max(1, round(H * k))), interpolation=cv2.INTER_AREA), False))
    _draw_guides(im, comp, W, H, k, lw=max(2, width // 400))
    if path:
        _save(im, path)
    return np.asarray(im), {k_: v for k_, v in comp.items() if not k_.startswith("_")}


# ---------------------------------------------------------------- palette / reference
def _kmeans(x, k=6, w=None, iters=12, seed=0):
    """Small deterministic weighted k-means (k-means++ init). x: N x D."""
    rng = np.random.default_rng(seed)
    w = np.ones(len(x), F32) if w is None else np.asarray(w, F32)
    cent = [x[rng.choice(len(x), p=w / w.sum())]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.array(cent)[None]) ** 2).sum(-1), 1) * w
        if d.sum() <= 0:
            break
        cent.append(x[rng.choice(len(x), p=d / d.sum())])
    cent = np.array(cent, F32)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - cent[None]) ** 2).sum(-1), 1)
        for j in range(len(cent)):
            sel = lab == j
            if sel.any():
                cent[j] = (x[sel] * w[sel, None]).sum(0) / w[sel].sum()
    lab = np.argmin(((x[:, None] - cent[None]) ** 2).sum(-1), 1)
    share = np.array([w[lab == j].sum() for j in range(len(cent))]) / w.sum()
    return cent, share


def _palette(rgb, k=6, seed=0, n=12000):
    flat = rgb.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    sub = flat[rng.choice(len(flat), min(n, len(flat)), replace=False)]
    lab = rgb2oklab(sub)
    lab_w = lab * np.array([1.0, 1.6, 1.6], F32)   # chroma weighs a bit more than lightness
    w = 1 + 4 * np.hypot(lab[:, 1], lab[:, 2])     # saturated accents are not swallowed by greys
    cent, share = _kmeans(lab_w, k, w, seed=seed)
    cent = cent / np.array([1.0, 1.6, 1.6], F32)
    order = np.argsort(-share)
    cols = oklab2rgb(cent[order])
    return [(_hex(c), round(float(sh), 3)) for c, sh in zip(cols, share[order]) if sh > 0.004]


def analyze_reference(src, k=6, seed=0) -> dict:
    """Study a reference image: palette [(hex, weight)] (weighted k-means in OKLab), 5-value thumbnail
    (uint8 H x W, 160 px wide), key (low/mid/high), contrast, light direction estimate
    ('from upper-left', azimuth deg: 0 = from right, 90 = from top), temperature of lights vs shadows."""
    rgb, _ = _src(src)
    H, W = rgb.shape[:2]
    lab = rgb2oklab(rgb)
    L = lab[..., 0]
    tw = 160
    th = max(1, round(H * tw / W))
    small = cv2.resize(L, (tw, th), interpolation=cv2.INTER_AREA)
    thumb = (np.clip(np.floor(small * 5), 0, 4) * (255 / 4)).astype(np.uint8)
    big = cv2.GaussianBlur(cv2.resize(L, (256, max(1, round(H * 256 / W))), interpolation=cv2.INTER_AREA), (0, 0), 12)
    gx = cv2.Sobel(big, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(big, cv2.CV_32F, 0, 1, ksize=3)
    vx, vy = float(gx.mean()), float(-gy.mean())
    mag = float(np.hypot(gx, gy).mean()) + 1e-9
    az = math.degrees(math.atan2(vy, vx)) % 360
    strength = math.hypot(vx, vy) / mag
    names = ["right", "upper-right", "top", "upper-left", "left", "lower-left", "bottom", "lower-right"]
    light = names[int(((az + 22.5) % 360) // 45)] if strength > 0.12 else "diffuse / frontal"
    lo, hi = np.percentile(L, 10), np.percentile(L, 90)
    warm = lambda m: float((lab[..., 2][m]).mean() + 0.5 * (lab[..., 1][m]).mean()) if m.any() else 0.0
    mean = float(L.mean())
    return {
        "palette": _palette(rgb, k, seed),
        "values": thumb,
        "key": "low" if mean < 0.45 else "high" if mean > 0.7 else "mid",
        "mean": round(mean * 100, 1),
        "contrast": round(float(L.std() * 100), 1),
        "light": light, "light_azimuth": round(az), "light_strength": round(strength, 2),
        "temperature": {"lights": round(warm(L >= hi) * 100, 1), "shadows": round(warm(L <= lo) * 100, 1)},
    }


def _ref_rgb(ref):
    return _src(ref)[0]


def match_palette(ref, amount=1.0):
    """Effect: histogram-match each OKLab channel (L, a, b) to the reference -> f(img, s).
    c.apply(match_palette("ref.jpg", .8)). Alpha kept."""
    rl = rgb2oklab(_ref_rgb(ref)).reshape(-1, 3)
    qs = np.linspace(0, 100, 257)
    rq = np.percentile(rl, qs, axis=0)

    def f(img, s=1.0):
        img = rgba(img)
        lab = rgb2oklab(np.clip(img[..., :3], 0, 1))
        out = lab.copy()
        for ch in range(3):
            src_q = np.percentile(lab[..., ch], qs)
            src_q = np.maximum.accumulate(src_q + np.arange(len(qs)) * 1e-7)
            out[..., ch] = np.interp(lab[..., ch], src_q, rq[:, ch])
        lab = lab + amount * (out - lab)
        res = img.copy()
        res[..., :3] = oklab2rgb(lab)
        return res
    return f


def transfer_tone(ref, amount=1.0, light=True):
    """Effect: Reinhard-style mean/std transfer in OKLab (gentler than match_palette) -> f(img, s).
    light=False transfers only colour (a, b) and keeps the lightness structure."""
    rl = rgb2oklab(_ref_rgb(ref)).reshape(-1, 3)
    rm, rs = rl.mean(0), rl.std(0) + 1e-6

    def f(img, s=1.0):
        img = rgba(img)
        lab = rgb2oklab(np.clip(img[..., :3], 0, 1))
        m, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        t = (lab - m) / sd * rs + rm
        if not light:
            t[..., 0] = lab[..., 0]
        lab = lab + amount * (t - lab)
        res = img.copy()
        res[..., :3] = oklab2rgb(lab)
        return res
    return f


# ---------------------------------------------------------------- drawing helpers
def _save(im, path):
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".jpg", ".jpeg"):
        im.convert("RGB").save(path, quality=90, subsampling=0)
    else:
        im.save(path)
    return str(path)


def _fit(a, bw, bh, nearest=False):
    """Resize array (float rgb or uint8) to fit bw x bh; returns PIL image."""
    h, w = a.shape[:2]
    k = min(bw / w, bh / h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    r = cv2.resize(a, (nw, nh), interpolation=interp)
    if r.dtype != np.uint8:
        r = to_u8(r, False)
    return Image.fromarray(r), k


BG = (24, 24, 27)
FG = (235, 235, 235)
DIM = (150, 150, 158)
RED = (255, 90, 80)
OK = (110, 220, 120)


def _text(d, xy, s, size=18, kind="sans", fill=FG):
    d.text(xy, s, font=_font(size, kind), fill=fill)
    return xy[1] + round(size * 1.3)


def _value_study(L):
    H, W = L.shape
    sm = cv2.GaussianBlur(L, (0, 0), max(1.0, max(H, W) / 300))
    v = np.clip(np.floor(sm * 5), 0, 4)
    greys = np.array([20, 75, 135, 190, 240], F32)[v.astype(int)] / 255
    return np.repeat(greys[..., None], 3, -1).astype(F32)


def _clip_view(mp):
    g = (mp["rgb"] @ LUMA)[..., None] * 0.55 + 0.1
    out = np.repeat(g, 3, -1).astype(F32)
    out[mp["chan"]] = (1.0, 0.85, 0.1)
    out[mp["blown"]] = (1.0, 0.15, 0.15)
    out[mp["crushed"]] = (0.2, 0.45, 1.0)
    return out


def _histogram(rgb, w, h, m):
    im = Image.new("RGB", (w, h), (14, 14, 16))
    d = ImageDraw.Draw(im, "RGBA")
    q = np.round(rgb * 255).astype(np.int32)
    Y = np.round(rgb @ LUMA * 255).astype(np.int32)
    ph = h - 26
    for band in range(1, 5):   # value-band separators (same bands as the value study)
        x = band * 0.2 * (w - 1)
        d.line([(x, 0), (x, ph)], fill=(60, 60, 66), width=1)
    hy = np.bincount(Y.ravel(), minlength=256)[:256].astype(F32)
    top = np.sqrt(np.percentile(hy[1:255], 99.5) + 1)
    for ch, col in ((0, (255, 80, 80, 140)), (1, (80, 230, 80, 140)), (2, (90, 140, 255, 140))):
        hc = np.sqrt(np.bincount(q[..., ch].ravel(), minlength=256)[:256].astype(F32)) / top
        pts = [(i / 255 * (w - 1), ph - min(1.0, v) * (ph - 4)) for i, v in enumerate(hc)]
        d.line(pts, fill=col, width=2)
    hv = np.sqrt(hy) / top
    for i, v in enumerate(hv):
        x = i / 255 * (w - 1)
        col = (255, 255, 255, 170)
        if i == 0 and m["clip_shadows"] > 0.05:
            col = (90, 140, 255, 255)
        if i == 255 and m["clip_highlights"] > 0.05:
            col = (255, 60, 60, 255)
        d.line([(x, ph), (x, ph - min(1.0, v) * (ph - 4))], fill=col, width=max(2, w // 256 + 1) if i in (0, 255) else max(1, w // 256 + 1))
    for i in range(5):     # value band ramp under the plot
        g = int([20, 75, 135, 190, 240][i])
        d.rectangle([i * w / 5, ph + 2, (i + 1) * w / 5, h], fill=(g, g, g))
        pct = m["values"][i]
        d.text((i * w / 5 + 6, ph + 4), f"{pct:.0f}%", font=_font(16, "mono"), fill=(0, 0, 0) if g > 120 else (255, 255, 255))
    return im


def _pick_crops(mp, n=4, size=100, focal=None):
    """Choose up to n zoom windows (raster px) where defects are most likely. Priority:
    aliasing (else strongest edges), banding, noise (if visible), brightest highlight, focal point, detail."""
    rgb, H, W = mp["rgb"], mp["rgb"].shape[0], mp["rgb"].shape[1]
    cs = int(min(size, H, W))
    taken = np.zeros((H, W), bool)
    out = []

    def pick(name, score_map=None, valmap=None, min_score=0.0, center=None):
        if len(out) >= n:
            return False
        if center is None:
            sm = _box(score_map, cs)
            sm[taken] = -1
            idx = int(np.argmax(sm))
            if sm.flat[idx] <= min_score:
                return False
            cy, cx = np.unravel_index(idx, sm.shape)
        else:
            cx, cy = (int(np.clip(v, 0, lim - 1)) for v, lim in zip(center, (W, H)))
            if taken[cy, cx]:
                return False
        x0 = int(np.clip(cx - cs // 2, 0, W - cs))
        y0 = int(np.clip(cy - cs // 2, 0, H - cs))
        v = None if valmap is None else valmap(x0, y0)
        out.append((name, (x0, y0, cs, cs), v))
        mx, my = x0 + cs // 2, y0 + cs // 2   # no other window centre closer than one window size
        e = cs * 5 // 4
        taken[max(0, my - e):my + e, max(0, mx - e):mx + e] = True
        return True

    strong, hard, flat, nm = mp["strong"], mp["hard"], mp["flat"], mp["noise"]
    win = lambda a, x0, y0: a[y0:y0 + cs, x0:x0 + cs]
    edges = np.minimum(mp["grad"], 255) / 255
    if mp.get("pixel_art") or not pick("aliasing", hard + 0.02 * strong,
                                        lambda x0, y0: win(hard, x0, y0).sum() / max(1.0, win(strong, x0, y0).sum()) * 100, 0.002):
        pick("edges", edges)
    pick("banding", mp["band"], lambda x0, y0: float(win(mp["band"], x0, y0).mean() * 100), 0.02)
    local_noise = lambda x0, y0: float(win(nm, x0, y0).sum() / max(1.0, win(flat, x0, y0).sum()))
    sm = _box(nm, cs) / np.maximum(_box(flat, cs), 0.05)
    if float(sm.max()) > 1.2:
        pick("noise", sm, local_noise)
    Lb = cv2.GaussianBlur(mp["L"], (0, 0), 3)
    Lb[taken] = -1
    cy, cx = np.unravel_index(int(np.argmax(Lb)), Lb.shape)
    pick("brightest", None, lambda x0, y0: (round(float(win(mp["L"], x0, y0).max() * 100), 1), round(float(win(mp["blown"], x0, y0).mean() * 100), 2)),
         center=(cx, cy))
    if focal is not None:
        pick("focal point", center=focal)
    pick("detail", edges)
    pick("detail", edges)
    return out


# ---------------------------------------------------------------- contact sheet
def contact_sheet(img, path="sheet.jpg", width=1280, crops=4, crop_size=100, title=None) -> dict:
    """Write ONE review image: full view (zoom windows outlined), 5-value study with thirds / focal point /
    visual mass, clipping map, luminance histogram, palette, auto zoom crops at full resolution labelled
    `--crop x,y,w,h` in canvas px, metrics with typical ranges and problem flags.
    Returns {"path", "metrics", "crops": [(name, (x,y,w,h) canvas px, value)], "composition", "palette"}."""
    m, mp = _analyze(img, maps=True)
    rgb, s = mp["rgb"], mp["s"]
    H, W = rgb.shape[:2]
    comp = composition(rgb)
    pal = _palette(rgb, 6)
    nearest = m["pixel_art"]
    M, G, lab_h = 16, 12, 30
    inner = width - 2 * M
    head = 46
    portrait = H / W > 0.9
    if portrait:
        cw = (inner - 2 * G) // 3
        boxes = [(cw, round(cw * 1.5))] * 3
    else:
        mainw = round(inner * 0.64)
        sidew = inner - mainw - G
        mainh = min(round(mainw * H / W), 680)
        boxes = [(mainw, mainh)] + [(sidew, (mainh - lab_h - G) // 2)] * 2
    main, km = _fit(rgb, *boxes[0], nearest=nearest)
    vs, kv = _fit(_value_study(mp["L"]), *boxes[1])
    _draw_guides(vs, comp, W, H, kv, lw=2, which="tfm")
    cl, _ = _fit(_clip_view(mp), *boxes[2])
    if portrait:
        toph = max(main.size[1], vs.size[1], cl.size[1]) + lab_h
    else:
        toph = max(main.size[1], vs.size[1] + cl.size[1] + lab_h + G) + lab_h
    picks = _pick_crops(mp, crops, max(16, round(crop_size)), focal=comp["focal"])
    ncr = max(1, len(picks))
    cwid = (inner - (ncr - 1) * G) // ncr
    crop_h = cwid + 58
    histw, hist_h = round(inner * 0.42), 190
    bot_h = lab_h + hist_h + 12 + 52 + 8
    flags = list(m["flags"])
    tilt = comp["tilt"]
    if tilt is not None and 0.2 < abs(tilt) < 5:
        flags.append(f"horizon tilted {tilt:+.1f} deg (level it or tilt it clearly)")
    rows_h = 11 * 27 + 10 + 24 * max(1, len(flags))
    bot_h = max(bot_h, rows_h)
    total_h = head + toph + G + lab_h + crop_h + G + bot_h + M
    sheet = Image.new("RGB", (width, total_h), BG)
    d = ImageDraw.Draw(sheet, "RGBA")
    name = title or (os.path.basename(str(img)) if isinstance(img, (str, os.PathLike)) else "image")
    hdr = f"{name}   {round(W / s)}x{round(H / s)} canvas px"
    if s != 1:
        hdr += f"   (raster {W}x{H}, scale {s:g})"
    if m["pixel_grid"] > 1:
        hdr += f"   pixel art, {m['pixel_grid']}x blocks"
    _text(d, (M, 10), hdr, 24, "bold")
    nf = len(flags)
    _text(d, (width - M - 190, 12), (f"{nf} flag" + ("s" if nf != 1 else "")) if nf else "no flags", 22, "bold", RED if nf else OK)
    y = head
    if portrait:
        cells = [(M + i * (boxes[0][0] + G), y) for i in range(3)]
    else:
        x2 = M + boxes[0][0] + G
        cells = [(M, y), (x2, y), (x2, y + lab_h + vs.size[1] + G)]
    labels = ["full view (yellow = zoom windows)", "5 values, thirds, focal (o), mass (#)", None]
    for (cx, cy), im_, lbl in zip(cells, (main, vs, cl), labels):
        if lbl:
            _text(d, (cx, cy + 2), lbl, 18, "sans", DIM)
        sheet.paste(im_, (cx, cy + lab_h))
    cx, cy = cells[2]           # clip legend in its own colours
    xx = cx
    for txt, col in (("clipped:", DIM), ("white", (255, 80, 80)), ("black", (90, 140, 255)), ("channel", (255, 215, 30))):
        d.text((xx, cy + 2), txt, font=_font(18, "sans"), fill=col)
        xx += d.textlength(txt, font=_font(18, "sans")) + 10
    for i, (nm, (x0, y0, cw_, ch_), v) in enumerate(picks):
        ox, oy = cells[0][0], cells[0][1] + lab_h
        r = [ox + x0 * km, oy + y0 * km, ox + (x0 + cw_) * km, oy + (y0 + ch_) * km]
        d.rectangle(r, outline=(0, 0, 0, 160), width=4)
        d.rectangle(r, outline=(255, 220, 0, 255), width=2)
        d.rectangle([r[0], r[1], r[0] + 20, r[1] + 22], fill=(255, 220, 0, 255))
        d.text((r[0] + 5, r[1] + 1), str(i + 1), font=_font(17, "bold"), fill=(0, 0, 0))
    y = head + toph + G
    mag = cwid / max(16, round(crop_size))
    _text(d, (M, y + 2), f"zooms: {round(crop_size)} px windows at x{mag:.1f} (nearest, real pixels)   ->   "
          "atelier render scene.py --crop x,y,w,h", 18, "sans", DIM)
    y += lab_h
    fmt = {"aliasing": "{:.0f}% hard", "banding": "{:.0f}% area", "noise": "{:.1f} lvl", "brightest": "L {:.0f}, {:.0f}% clip"}
    for i, (nm, (x0, y0, cw_, ch_), v) in enumerate(picks):
        x = M + i * (cwid + G)
        z = rgb[y0:y0 + ch_, x0:x0 + cw_]
        sheet.paste(Image.fromarray(to_u8(cv2.resize(z, (cwid, cwid), interpolation=cv2.INTER_NEAREST), False)), (x, y))
        d.rectangle([x, y, x + 22, y + 24], fill=(255, 220, 0, 255))
        d.text((x + 6, y + 2), str(i + 1), font=_font(18, "bold"), fill=(0, 0, 0))
        txt = f"{i + 1}. {nm}" + ("" if v is None else "  " + fmt[nm].format(*(v if isinstance(v, tuple) else (v,))))
        yy = _text(d, (x, y + cwid + 5), txt, 19, "bold")
        _text(d, (x, yy), f"--crop {round(x0 / s)},{round(y0 / s)},{round(cw_ / s)},{round(ch_ / s)}", 18, "mono", DIM)
    y = head + toph + G + lab_h + crop_h + G
    _text(d, (M, y + 2), "luma histogram  (bands = value study)", 18, "sans", DIM)
    sheet.paste(_histogram(rgb, histw, hist_h, m), (M, y + lab_h))
    py = y + lab_h + hist_h + 12
    sw = histw // max(1, len(pal))
    for i, (hx, sh) in enumerate(pal):
        c = tuple(int(hx[j:j + 2], 16) for j in (1, 3, 5))
        d.rectangle([M + i * sw, py, M + (i + 1) * sw - 4, py + 52], fill=c)
        tc = (0, 0, 0) if 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 140 else (255, 255, 255)
        d.text((M + i * sw + 5, py + 4), hx, font=_font(15, "mono"), fill=tc)
        d.text((M + i * sw + 5, py + 27), f"{sh * 100:.0f}%", font=_font(15, "mono"), fill=tc)
    lim = LIMITS
    v = m["values"]
    clip_bad = (m["clip_shadows"] > lim["clip_shadows"] or m["clip_highlights"] > lim["clip_highlights"]
                or m["clip_channels"] > lim["clip_channels"])
    fp = tuple(round(c_ / s) for c_ in comp["focal"])
    mp_ = tuple(round(c_ / s) for c_ in comp["mass"])
    rows = [
        ("lightness mean / std", f"{m['mean']} / {m['contrast']}", m["contrast"] < 12, "std <12 flat, 20+ punchy"),
        ("value range p2-p98", f"{m['range'][0]}-{m['range'][1]}", m["range"][1] - m["range"][0] < 45, "0-100; want ~10-90"),
        ("darks / lights %", f"{v[0] + v[1]:.0f} / {v[3] + v[4]:.0f}", v[0] + v[1] < 3 or v[3] + v[4] < 3, "2 darkest / 2 lightest bands"),
        ("saturation / colourful", f"{m['saturation']} / {m['colorfulness']:.0f}", False, "chroma 10+ vivid; HS <15 dull"),
        ("noise", f"{m['noise']} lvl", m["noise"] > lim["noise"], "<1 clean, >4 grainy"),
        ("aliasing", f"{m['aliasing']:.0f}% hard", m["aliasing"] > lim["aliasing"] and not m["pixel_art"],
         "pixel art: ignore" if m["pixel_art"] else "AA ~5-25, jaggy >35"),
        ("banding", f"{m['banding']}% area", m["banding"] > lim["banding"], ">2 visible steps"),
        ("clip black/white/chan", f"{m['clip_shadows']}/{m['clip_highlights']}/{m['clip_channels']}%", clip_bad, "want <1 / <1 / <5"),
        ("sharpness", f"{m['sharpness']:.0f} lvl/px", False, "<20 soft, >80 crisp"),
        ("focal / mass", f"{fp[0]},{fp[1]} / {mp_[0]},{mp_[1]}", False, "canvas px"),
        ("balance x,y / horizon", f"{comp['balance'][0]:+.2f},{comp['balance'][1]:+.2f} / " + ("-" if tilt is None else f"{tilt:+.1f}deg"),
         tilt is not None and 0.2 < abs(tilt) < 5, "-1 left/top .. +1"),
    ]
    x = M + histw + 2 * G
    ty = y + 2
    for lbl, val, bad, hint in rows:
        d.text((x, ty), lbl, font=_font(18, "sans"), fill=DIM)
        d.text((x + 215, ty), val, font=_font(18, "mono"), fill=RED if bad else FG)
        d.text((x + 455, ty + 2), hint, font=_font(15, "sans"), fill=(115, 115, 122))
        ty += 27
    ty += 8
    if not flags:
        _text(d, (x, ty), "no automatic flags - still judge the zooms yourself", 18, "sans", OK)
    for f in flags:
        ty = _text(d, (x, ty), "! " + f, 17, "sans", RED) - 2
    _save(sheet, path)
    crops_out = [(nm, (round(x0 / s), round(y0 / s), round(cw_ / s), round(ch_ / s)), v_ if v_ is None or isinstance(v_, tuple) else round(float(v_), 2))
                 for nm, (x0, y0, cw_, ch_), v_ in picks]
    comp_out = {k_: v_ for k_, v_ in comp.items() if not k_.startswith("_")}
    comp_out["focal"], comp_out["mass"] = fp, mp_
    return {"path": str(path), "metrics": {**m, "flags": flags}, "crops": crops_out, "composition": comp_out, "palette": pal}


# ---------------------------------------------------------------- compare
def compare(a, b, path="compare.jpg", labels=("A", "B"), amp=4.0, split=True, width=1280) -> dict:
    """A | B side by side, amplified difference map (|A-B| x amp, inferno) and a split view
    (left half A, right half B). B is resized to A if needed. Returns diff stats:
    mean_diff (8-bit levels), max_diff, changed (% px differing > 2 levels), psnr, bbox (canvas px of A)."""
    ra, sa = _src(a)
    rb, _ = _src(b)
    H, W = ra.shape[:2]
    if rb.shape[:2] != (H, W):
        rb = cv2.resize(rb, (W, H), interpolation=cv2.INTER_AREA if rb.shape[1] > W else cv2.INTER_LINEAR)
    diff = np.abs(ra - rb).max(-1) * 255
    mse = float(((ra - rb) ** 2).mean())
    ch = diff > 2
    bb = None
    if ch.any():
        ys, xs = np.nonzero(ch)
        bb = (round(xs.min() / sa), round(ys.min() / sa), round((xs.max() - xs.min() + 1) / sa), round((ys.max() - ys.min() + 1) / sa))
    stats = {"mean_diff": round(float(diff.mean()), 2), "max_diff": round(float(diff.max()), 1),
             "changed": round(float(ch.mean() * 100), 2),
             "psnr": round(10 * math.log10(1 / mse), 2) if mse > 0 else float("inf"), "bbox": bb}
    heat = cv2.applyColorMap(np.clip(diff * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    panels = [(ra, labels[0]), (rb, labels[1]), (heat, f"|{labels[0]}-{labels[1]}| x{amp:g}")]
    if split:
        sp = ra.copy()
        sp[:, W // 2:] = rb[:, W // 2:]
        sp[:, W // 2 - 1:W // 2 + 1] = (1, 0.85, 0)
        panels.append((sp, f"split: {labels[0]} | {labels[1]}"))
    M, G, lab_h = 16, 12, 28
    cols = 2
    cw = (width - 2 * M - G) // cols
    chh = min(round(cw * H / W), 700)
    rows = (len(panels) + 1) // cols
    head = 44
    foot = 40
    sheet = Image.new("RGB", (width, head + rows * (chh + lab_h + G) + foot), BG)
    d = ImageDraw.Draw(sheet)
    _text(d, (M, 10), f"compare {labels[0]} vs {labels[1]}   {W}x{H}", 22, "bold")
    for i, (p, lbl) in enumerate(panels):
        x = M + (i % cols) * (cw + G)
        y = head + (i // cols) * (chh + lab_h + G)
        im, _ = _fit(p, cw, chh)
        _text(d, (x, y), lbl, 17, "sans", DIM)
        sheet.paste(im, (x, y + lab_h))
    _text(d, (M, sheet.size[1] - foot + 6),
          f"mean diff {stats['mean_diff']} lvl   max {stats['max_diff']}   changed {stats['changed']}% px   "
          f"PSNR {stats['psnr']} dB   bbox {bb}", 18, "mono", FG)
    _save(sheet, path)
    stats["path"] = str(path)
    return stats


# ---------------------------------------------------------------- placement grid
_BOX_COLS = [(255, 70, 70), (70, 200, 255), (255, 200, 40), (120, 255, 120), (255, 90, 230), (255, 150, 60),
             (160, 130, 255), (60, 255, 210)]


def _nice_step(span, target=10):
    raw = max(span, 1e-6) / target
    p = 10 ** math.floor(math.log10(raw))
    return next(m * p for m in (1, 2, 2.5, 5, 10) if raw <= m * p + 1e-9)


def layer_boxes(c) -> list:
    """Bounding box of every layer's visible pixels, in canvas px: [(index, name, (x, y, w, h) | None)]."""
    return [(i, L.name or "", L.bounds) for i, L in enumerate(c.layers)]


def grid_view(img, path=None, width=768, step=None, region=None, marks=(), boxes=False) -> dict:
    """Image with a coordinate grid labelled in canvas px (labels in margins, never over the art).
    region=(x, y, w, h) zooms on a part (nearest upscale, labels stay in canvas px).
    marks: (x, y) points or (x, y, w, h[, label[, rgb]]) rects to check a placement. boxes=True (Canvas only) outlines
    each layer's visible bounds with its index:name. Returns {"step", "region", "boxes"}."""
    rgb, s = _src(img)
    H, W = rgb.shape[:2]
    cw, ch = W / s, H / s
    rx, ry, rw, rh = region if region is not None else (0, 0, cw, ch)
    x0, y0 = max(0, int(rx * s)), max(0, int(ry * s))
    x1, y1 = min(W, max(x0 + 1, round((rx + rw) * s))), min(H, max(y0 + 1, round((ry + rh) * s)))
    rx, ry, rw, rh = x0 / s, y0 / s, (x1 - x0) / s, (y1 - y0) / s
    k = width / (x1 - x0)                                    # display px per raster px
    dh = max(1, round((y1 - y0) * k))
    crop = rgb[y0:y1, x0:x1]
    view = cv2.resize(crop, (width, dh), interpolation=cv2.INTER_NEAREST if k > 1 else cv2.INTER_AREA)
    step = step or _nice_step(max(rw, rh), 12)
    ML, MT, MR, MB = 46, 20, 8, 8
    im = Image.new("RGB", (ML + width + MR, MT + dh + MB), (24, 24, 28))
    im.paste(Image.fromarray(to_u8(view, False)), (ML, MT))
    d = ImageDraw.Draw(im, "RGBA")
    f = _font(11, "mono")
    px = lambda x: ML + (x - rx) * s * k   # canvas x -> display x
    py = lambda y: MT + (y - ry) * s * k
    lab = (lambda v: f"{v:g}") if step < 1 else (lambda v: str(int(round(v))))
    # minor grid (half step) when there is room
    if step * s * k / 2 >= 22:
        for v in np.arange(math.ceil(rx / (step / 2)) * step / 2, rx + rw + 1e-6, step / 2):
            d.line([(px(v), MT), (px(v), MT + dh)], fill=(255, 255, 255, 38))
        for v in np.arange(math.ceil(ry / (step / 2)) * step / 2, ry + rh + 1e-6, step / 2):
            d.line([(ML, py(v)), (ML + width, py(v))], fill=(255, 255, 255, 38))
    for v in np.arange(math.ceil(rx / step) * step, rx + rw + 1e-6, step):
        x = px(v)
        d.line([(x + 1, MT), (x + 1, MT + dh)], fill=(0, 0, 0, 70))
        d.line([(x, MT), (x, MT + dh)], fill=(255, 255, 255, 120))
        t = lab(v)
        tw = d.textlength(t, font=f)
        d.text((min(max(x - tw / 2, ML - 4), ML + width - tw), 3), t, fill=(220, 220, 220), font=f)
    for v in np.arange(math.ceil(ry / step) * step, ry + rh + 1e-6, step):
        y = py(v)
        d.line([(ML, y + 1), (ML + width, y + 1)], fill=(0, 0, 0, 70))
        d.line([(ML, y), (ML + width, y)], fill=(255, 255, 255, 120))
        t = lab(v)
        d.text((ML - 4 - d.textlength(t, font=f), min(max(y - 7, MT - 2), MT + dh - 12)), t, fill=(220, 220, 220), font=f)
    # centre lines of the whole canvas
    for v in (cw / 2,):
        if rx <= v <= rx + rw:
            d.line([(px(v), MT), (px(v), MT + dh)], fill=(255, 60, 200, 110))
    for v in (ch / 2,):
        if ry <= v <= ry + rh:
            d.line([(ML, py(v)), (ML + width, py(v))], fill=(255, 60, 200, 110))

    def tag(x, y, t, col):
        tw = d.textlength(t, font=f)
        x = min(max(x, ML), ML + width - tw - 4)
        y = min(max(y, MT), MT + dh - 14)
        d.rectangle([x, y, x + tw + 4, y + 13], fill=(0, 0, 0, 170))
        d.text((x + 2, y), t, fill=col, font=f)

    found = []
    if boxes and hasattr(img, "layers"):
        found = layer_boxes(img)
        for i, name, b in found:
            if b is None or (b[2] >= cw * 0.98 and b[3] >= ch * 0.98):
                continue                                      # empty or full-canvas (background) layer
            col = _BOX_COLS[i % len(_BOX_COLS)]
            bx0, by0, bx1, by1 = px(b[0]), py(b[1]), px(b[0] + b[2]), py(b[1] + b[3])
            d.rectangle([bx0, by0, bx1, by1], outline=col + (230,), width=2)
            tag(bx0 + 2, by0 + 2, f"{i}:{name}" if name else str(i), col)
    for m in marks:
        col = tuple(m[5]) if len(m) > 5 else (255, 255, 0)
        if len(m) >= 4:                                       # (x, y, w, h[, label[, colour]])
            d.rectangle([px(m[0]), py(m[1]), px(m[0] + m[2]), py(m[1] + m[3])], outline=col + (255,), width=2)
            t = str(m[4]) if len(m) > 4 else ",".join(f"{v:g}" for v in m[:4])
            tag(px(m[0]) + 2, py(m[1]) + 2 if len(m) > 4 else py(m[1] + m[3]) + 2, t, col)
        else:
            x, y = px(m[0]), py(m[1])
            d.line([(x - 9, y), (x + 9, y)], fill=(0, 0, 0, 200), width=3)
            d.line([(x, y - 9), (x, y + 9)], fill=(0, 0, 0, 200), width=3)
            d.line([(x - 8, y), (x + 8, y)], fill=col + (255,), width=1)
            d.line([(x, y - 8), (x, y + 8)], fill=col + (255,), width=1)
            tag(x + 6, y + 4, f"{m[0]:g},{m[1]:g}", col)
    if path:
        _save(im, path)
    return {"step": step, "region": (round(rx, 2), round(ry, 2), round(rw, 2), round(rh, 2)), "boxes": found}


# ---------------------------------------------------------------- critique
SEV_COL = {"high": (255, 70, 70), "mid": (255, 170, 40), "low": (110, 200, 255)}
_TEXTY = ("text", "title", "label", "caption", "headline", "logo")


def _lin(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _wcag(a, b):
    la, lb = (float(_lin(x) @ LUMA) for x in (a, b))
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _disk(r):
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


_HIER = {   # metric: (label, floor below which both sides count as "none", fix for the competitor, fix for the focus)
    "contrast": ("value contrast", 0.03, "L.apply(contrast(-.3)) or haze it", "push light/dark contrast at its edges"),
    "sharpness": ("edge sharpness", 0.02, "L.apply(blur(1.5)) / lose some edges", "keep its edges crisp"),
    "detail": ("texture detail", 0.004, "simplify or blur its texture", "add texture / small detail"),
    "saturation": ("saturation", 0.06, "L.apply(saturation(.6))", "saturate its accent"),
}


def _elements(img, H, W, s):
    """Named elements of a Canvas, including layers already merged by c.apply() (kept as alpha snapshots):
    [{"name", "a" (H x W coverage), "b" (x, y, w, h canvas px), "blend"}]. Hidden, empty and full-frame layers skipped."""
    out = []
    src = [(n, a.astype(F32) / 255, bl) for n, a, bl in getattr(img, "__dict__", {}).get("merged", []) if a.shape == (H, W)]
    src += [(L.name or f"#{i}", L.px[..., 3] * L.opacity, L.blend) for i, L in enumerate(getattr(img, "layers", []))
            if L.visible and L.opacity >= 0.3]
    for name, a, blend in src:
        bb = _bbox_canvas(a, s)
        if bb is None or bb[2] * bb[3] > 0.9 * (W / s) * (H / s):
            continue
        out.append({"name": name, "a": a, "b": bb, "blend": blend})
    return out


def _bbox_canvas(a, s, thresh=0.02):
    m = a > thresh
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(m.any(0))
    x0, y0, x1, y1 = cols[0] / s, rows[0] / s, (cols[-1] + 1) / s, (rows[-1] + 1) / s
    return round(x0), round(y0), round(x1 - x0), round(y1 - y0)


def _focus_mask(img, focus, comp, els, H, W, s):
    """Focus -> (bool mask H x W, label, layer names). focus: layer/group name | (x, y, w, h) | None (auto)."""
    if isinstance(focus, str):
        es = [e for e in els if e["name"] == focus or e["name"].startswith(focus + "/")]
        if not es:
            raise KeyError(f"focus layer {focus!r} not found; elements: {[e['name'] for e in els]}")
        m = np.zeros((H, W), bool)
        for e in es:
            m |= e["a"] > 0.5
        return m, f"'{focus}'", [e["name"] for e in es]
    if focus is not None:
        x, y, w, h = (round(v * s) for v in focus)
        m = np.zeros((H, W), bool)
        m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = True
        return m, f"region {','.join(str(round(v)) for v in focus)}", []
    fx_, fy_ = comp["focal"]
    yy, xx = min(H - 1, int(fy_ * s)), min(W - 1, int(fx_ * s))
    for e in reversed(els):
        if e["a"][yy, xx] > 0.5:
            return e["a"] > 0.5, f"{e['name']!r} (auto: holds the focal point)", [e["name"]]
    r = 0.1 * math.hypot(H, W)
    gy, gx = np.ogrid[:H, :W]
    return (gx - xx) ** 2 + (gy - yy) ** 2 <= r * r, f"focal area {fx_},{fy_} (auto)", []


def hierarchy(img, focus=None) -> dict:
    """Does the focus win? Compares the focus (layer/group name, (x, y, w, h) or auto) with every other
    named layer and the background on value contrast, edge sharpness, texture detail and saturation;
    plus a squint test (focus vs surroundings at 64 px) and light/shadow temperature split.
    Returns {"focus", "scores": {metric: {"focus", "best_other", "ratio"}}, "squint", "temperature", "issues"}.
    ratio < 1 means something else out-competes the focus on that metric."""
    rgb, s = _src(img)
    H, W = rgb.shape[:2]
    comp = composition(img)
    els = _elements(img, H, W, s)
    fm, flabel, fnames = _focus_mask(img, focus, comp, els, H, W, s)
    lab = rgb2oklab(rgb)
    Lc, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    sig = max(2.0, 0.015 * max(H, W))
    maps = {"contrast": np.abs(Lc - cv2.GaussianBlur(Lc, (0, 0), sig)),
            "sharpness": np.hypot(cv2.Sobel(Lc, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lc, cv2.CV_32F, 0, 1, ksize=3)) / 4 * s,
            # texture density: band-pass energy between ~0.2 % and 0.8 % of the image size
            "detail": np.abs(cv2.GaussianBlur(Lc, (0, 0), max(0.7, 0.002 * max(H, W)))
                             - cv2.GaussianBlur(Lc, (0, 0), max(2.0, 0.008 * max(H, W)))), "saturation": C}
    stat = {"contrast": lambda v: np.percentile(v, 95), "sharpness": lambda v: np.percentile(v, 98),
            "detail": np.mean, "saturation": lambda v: np.percentile(v, 90)}
    rr = max(1, round(sig))
    grow = lambda m: cv2.dilate(m.astype(np.uint8), _disk(rr)) > 0
    shrink = lambda m: cv2.erode(m.astype(np.uint8), _disk(rr)) > 0
    fd = grow(fm)
    regions, used = [], fd.copy()                            # (name, mask incl. edges, interior mask)
    for e in els:
        if e["name"] in fnames:
            continue
        m = grow(e["a"] > 0.5) & ~fd
        used |= m
        if m.mean() > 0.015:
            regions.append((f"'{e['name']}'", m, shrink(e["a"] > 0.5) & ~fd))
    bg = ~used
    if bg.mean() > 0.015:
        regions.append(("background" if els else "the rest of the image", bg, shrink(bg)))
    fcore = shrink(fm)
    out = {"focus": flabel, "scores": {}, "issues": []}
    if fd.sum() < 20 or not regions:
        return out
    for k, (label, floor, fix_other, fix_focus) in _HIER.items():
        core = k == "detail"                                   # texture density: interiors only, edges excluded
        fmask = fcore if core and fcore.sum() > 20 else fd
        cands = [(float(stat[k](maps[k][mi if core and mi.sum() > 20 else m])), n) for n, m, mi in regions]
        fv = float(stat[k](maps[k][fmask]))
        best = max(cands, key=lambda t: t[0])
        ratio = fv / max(best[0], 1e-6)
        out["scores"][k] = {"focus": round(fv, 4), "best_other": best[1], "other": round(best[0], 4), "ratio": round(ratio, 2)}
        if max(fv, best[0]) < floor or ratio >= 0.9:
            continue
        sev = "high" if k in ("contrast", "sharpness") and ratio < 0.6 else "mid" if k != "saturation" or ratio < 0.6 else "low"
        out["issues"].append((sev, f"{best[1]} has {1 / max(ratio, 1e-3):.1f}x the {label} of the focus {flabel}: "
                              f"there, {fix_other}; on the focus, {fix_focus}.", best[1]))
    # squint test
    k64 = 64 / max(H, W)
    sw, sh = max(4, round(W * k64)), max(4, round(H * k64))
    Ls = cv2.GaussianBlur(cv2.resize(Lc, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    fs = cv2.resize(fm.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    ring = (cv2.dilate(fs.astype(np.uint8), _disk(2)) > 0) & ~fs
    if fs.sum() >= 2 and ring.sum() >= 2:
        dL = abs(float(np.median(Ls[fs]) - np.median(Ls[ring])))
        out["squint"] = round(dL, 3)
        if dL < 0.08:
            out["issues"].append(("high" if dL < 0.04 else "mid", f"squint test: at 64 px the focus {flabel} merges with its "
                                  f"surroundings (dL {dL:.2f}, want >= 0.1): make it clearly lighter or darker than what is around it.", None))
    # light / shadow temperature
    warm = lab[..., 2] + 0.3 * lab[..., 1]
    lo, hi = np.percentile(Lc, 20), np.percentile(Lc, 80)
    if hi - lo > 0.1:
        wl, ws = float(warm[Lc >= hi].mean() * 100), float(warm[Lc <= lo].mean() * 100)
        out["temperature"] = {"lights": round(wl, 1), "shadows": round(ws, 1)}
        if abs(wl - ws) < 1.5 and C.mean() > 0.03:
            out["issues"].append(("low", f"lights and shadows share one temperature (warmth {wl:.1f} vs {ws:.1f}): "
                                  "cool the shadows (color_balance(shadows=(-.03, 0, .04))) or warm the lights, for depth.", None))
    return out


def critique(img, path=None, width=1024, max_align=6, focus=None) -> dict:
    """Self-critique: problems an eye would catch, each with a concrete fix, ranked high/mid/low.
    Image-level: quality flags, balance, focal point, horizon tilt. With a Canvas, per layer (use names):
    figure-ground contrast (OKLab + WCAG for text-like names), elements hidden by layers above,
    near-miss alignments, near-tangent elements, elements crowding the canvas edge.
    path: writes the preview with a grid and numbered issue boxes. Returns {"issues": [...], "metrics", "composition"}.
    Each issue: {"n", "sev", "kind", "msg", "box": (x, y, w, h) | None, "layers": [...]}. Coordinates in canvas px."""
    issues = []

    def add(sev, kind, msg, box=None, layers=()):
        issues.append({"sev": sev, "kind": kind, "msg": msg, "box": None if box is None else tuple(round(v) for v in box),
                       "layers": list(layers)})

    m = _analyze(img)
    for f in m["flags"]:
        add("mid", "quality", f)
    comp = composition(img)
    rgb, s = _src(img)
    H, W = rgb.shape[:2]
    cw, ch = W / s, H / s
    canvas = hasattr(img, "layers")
    # light / effect layers (screen, add, multiply...) count for hierarchy, not as objects
    els = [e for e in _elements(img, H, W, s) if e["blend"] == "normal"] if canvas else []
    q = lambda e: f"'{e['name']}'"

    # ---- balance & focal
    bx, by = comp["balance"]
    if abs(bx) > 0.3 or abs(by) > 0.45:
        side = ("right" if bx > 0 else "left") if abs(bx) > 0.3 else ("bottom" if by > 0 else "top")
        heavy = ""
        if els:
            sal = comp["_sal"]
            mx, my = comp["mass"]
            on = [e for e in els if (e["b"][0] + e["b"][2] / 2 - cw / 2) * (mx - cw / 2) > 0 or abs(bx) <= 0.3]
            on.sort(key=lambda e: -float((sal * (e["a"] > 0.5)).sum()))
            heavy = f"; heaviest there: {', '.join(q(e) for e in on[:2])}" if on else ""
        add("mid", "balance", f"visual weight leans {side} (balance {bx:+.2f},{by:+.2f}, mass at {comp['mass'][0]},"
            f"{comp['mass'][1]}){heavy}. Move one toward the centre or add a counterweight on the other side.")
    fx_, fy_ = comp["focal"]
    holder = None
    for e in reversed(els):
        yy, xx = min(H - 1, int(fy_ * s)), min(W - 1, int(fx_ * s))
        if e["a"][yy, xx] > 0.5:
            holder = e
            break
    if canvas and els and holder is None:
        add("low", "focal", f"the strongest eye-catcher ({fx_},{fy_}) is background, not a named element: "
            "raise the subject's contrast/saturation or calm that area.", (fx_ - cw * .05, fy_ - ch * .05, cw * .1, ch * .1))
    centre_d = math.hypot(fx_ - cw / 2, fy_ - ch / 2) / math.hypot(cw, ch)
    if comp["focal_thirds_dist"] > 0.12 and centre_d > 0.1:
        tx, ty = min(((cw * a, ch * b) for a in (1 / 3, 2 / 3) for b in (1 / 3, 2 / 3)),
                     key=lambda p: math.hypot(p[0] - fx_, p[1] - fy_))
        add("low", "focal", f"focal point ({fx_},{fy_}){' on ' + q(holder) if holder else ''} is neither on a third nor "
            f"centred; nearest third: {tx:.0f},{ty:.0f}.")
    if comp["tilt"] is not None and 0.2 < abs(comp["tilt"]) < 5:
        add("mid", "tilt", f"the main horizontal line is tilted {comp['tilt']} deg: level it or tilt it clearly (>5 deg).")

    if els:
        lab = rgb2oklab(rgb)
        r = max(2, round(0.012 * max(W, H)))
        # ---- hidden by layers above
        above = []                                             # (group, coverage) of layers above
        for e in reversed(els):
            a, g = e["a"], e["name"].split("/")[0]
            other = np.zeros((H, W), F32)
            for g2, ac2 in above:
                if g2 != g:                                    # covered by its own group (e.g. a rim) is fine
                    other = 1 - (1 - other) * (1 - ac2)
            e["vis"] = float((a * (1 - other)).sum() / max(a.sum(), 1e-6))
            if e["vis"] < 0.25 and a.sum() > 30:
                add("mid", "hidden", f"{q(e)} is {100 - e['vis'] * 100:.0f}% covered by layers above it: restack it "
                    f"(L.front()) or move it.", e["b"], [e["name"]])
            above.append((g, np.clip(a, 0, 1)))
        # ---- figure-ground contrast
        for e in els:
            if e["vis"] < 0.5:
                continue
            inside = (e["a"] > 0.5).astype(np.uint8)
            if inside.sum() < 30:
                continue
            band = inside & ~cv2.erode(inside, _disk(r))
            ring = (cv2.dilate(inside, _disk(r)) > 0) & (e["a"] < 0.05)
            if band.sum() < 10 or ring.sum() < 10:
                continue
            li, lo = np.median(lab[band > 0], 0), np.median(lab[ring], 0)
            dL, dE = abs(float(li[0] - lo[0])), float(np.linalg.norm(li - lo))
            ci, co = np.median(rgb[band > 0], 0), np.median(rgb[ring], 0)
            ratio = _wcag(ci, co)
            texty = any(t in e["name"].lower() for t in _TEXTY)
            tag = f"(edge {_hex(ci)} vs behind {_hex(co)}, dL {dL:.2f}, dE {dE:.2f}, contrast {ratio:.1f}:1)"
            if dE < 0.06:
                add("high", "contrast", f"{q(e)} blends into what is behind it {tag}: shift its lightness or the "
                    "background by >=0.15 L, or add an outline / shadow.", e["b"], [e["name"]])
            elif texty and ratio < 4.5:
                add("high" if ratio < 3 else "mid", "contrast", f"text {q(e)} is hard to read {tag}: aim for >=4.5:1 "
                    "(>=3:1 for large titles).", e["b"], [e["name"]])
            elif dL < 0.05 and dE < 0.12:
                add("mid", "contrast", f"{q(e)} separates by hue only {tag}: it vanishes in a value study; "
                    "make it lighter or darker than its surroundings.", e["b"], [e["name"]])
        # ---- edge crowding
        tol_e = max(3 / s, 0.02 * min(cw, ch))
        for e in els:
            x, y, w, h = e["b"]
            for side, gap in (("left", x), ("top", y), ("right", cw - x - w), ("bottom", ch - y - h)):
                if 1.5 / s < gap < tol_e:
                    add("mid", "edge", f"{q(e)} is {gap:.0f}px from the {side} edge (tension): bleed it off the "
                        f"edge or give it >= {2 * tol_e:.0f}px.", e["b"], [e["name"]])
        # ---- near-tangent elements
        tol_t = max(3 / s, 0.006 * max(cw, ch))
        small = [e for e in els if e["b"][2] * e["b"][3] < 0.6 * cw * ch]
        dts = {}
        for ia, A in enumerate(small):
            for B in small[ia + 1:]:
                (ax, ay, aw, ah), (bx_, by_, bw, bh) = A["b"], B["b"]
                if ax - tol_t > bx_ + bw or bx_ - tol_t > ax + aw or ay - tol_t > by_ + bh or by_ - tol_t > ay + ah:
                    continue
                ma, mb = A["a"] > 0.5, B["a"] > 0.5
                if (ma & mb).any() or not ma.any() or not mb.any():
                    continue
                if ia not in dts:
                    dts[ia] = cv2.distanceTransform((~ma).astype(np.uint8), cv2.DIST_L2, 5)
                gap = max(0.0, float(dts[ia][mb].min()) - 1) / s     # empty pixels between the two
                if 1.5 / s <= gap <= tol_t:                    # 0-1 px = attached on purpose
                    yy, xx = np.unravel_index(int(np.argmin(np.where(mb, dts[ia], 1e9))), mb.shape)
                    add("mid", "tangent", f"{q(A)} and {q(B)} almost touch ({gap:.0f}px gap near {xx / s:.0f},{yy / s:.0f}): "
                        "overlap them clearly or open the gap.", (xx / s - 15, yy / s - 15, 30, 30), [A["name"], B["name"]])
        # ---- near-miss alignment
        eps, tol_a = 1.5 / s, max(4 / s, 0.006 * max(cw, ch))
        feats = lambda e: {"x": {"left": e["b"][0], "centre x": e["b"][0] + e["b"][2] / 2, "right": e["b"][0] + e["b"][2]},
                           "y": {"top": e["b"][1], "centre y": e["b"][1] + e["b"][3] / 2, "bottom": e["b"][1] + e["b"][3]}}
        refs = [{"name": "canvas", "f": {"x": {"centre x": cw / 2}, "y": {"centre y": ch / 2}}}]
        objs = [{"name": q(e), "f": feats(e), "e": e} for e in els] + refs
        found = []
        for ia, A in enumerate(objs):
            for B in objs[ia + 1:]:
                if "e" not in A and "e" not in B:
                    continue
                if "e" in A and "e" in B:
                    ga, gb = A["e"]["name"].split("/")[0], B["e"]["name"].split("/")[0]
                    (ax0, ay0, aw0, ah0), (bx0, by0, bw0, bh0) = A["e"]["b"], B["e"]["b"]
                    ox = max(0, min(ax0 + aw0, bx0 + bw0) - max(ax0, bx0)) * max(0, min(ay0 + ah0, by0 + bh0) - max(ay0, by0))
                    if ga == gb or ox > 0.5 * min(aw0 * ah0, bw0 * bh0):
                        continue                               # parts of one group / nested elements
                for ax_ in ("x", "y"):
                    pairs = [(abs(va - B["f"][ax_][ka]), ka, va, ka, B["f"][ax_][ka]) for ka, va in A["f"][ax_].items() if ka in B["f"][ax_]]
                    if not pairs:
                        continue
                    if any(d <= eps for d, *_ in pairs):
                        continue                               # already aligned on this axis
                    d, ka, va, kb, vb = min(pairs)
                    if d <= tol_a:
                        found.append((d, f"nearly aligned: {A['name']} {ka} {va:.0f} vs {B['name']} {kb} {vb:.0f} "
                                      f"({d:.0f}px): snap to {round((va + vb) / 2)} or offset clearly.",
                                      [o["e"]["name"] for o in (A, B) if "e" in o], [o["e"]["b"] for o in (A, B) if "e" in o]))
        for d, msg, names, bs in sorted(found, key=lambda t: -t[0])[:max_align]:
            add("low", "align", msg, bs[0], names)

    hi = hierarchy(img, focus)
    boxes = {f"'{e['name']}'": (e["b"], e["name"]) for e in els}
    for sev, msg, who in hi["issues"]:
        b, n = boxes.get(who, (None, None))
        add(sev, "hierarchy", msg, b, [n] if n else [])

    rank = {"high": 0, "mid": 1, "low": 2}
    issues.sort(key=lambda t: rank[t["sev"]])
    for n, t in enumerate(issues, 1):
        t["n"] = n
    if path:
        grouped = {}
        for t in issues:                                       # one box per region, worst severity colour
            if t["box"] is not None:
                grouped.setdefault(t["box"], []).append(t)
        marks = [(*bx, ",".join(str(t["n"]) for t in ts), SEV_COL[ts[0]["sev"]]) for bx, ts in reversed(grouped.items())]
        grid_view(img, path, width=width, marks=marks)
    return {"issues": issues, "metrics": m, "composition": {k: v for k, v in comp.items() if not k.startswith("_")},
            "hierarchy": {k: v for k, v in hi.items() if k != "issues"}}


def critique_score(r) -> int:
    """One number from a critique result (100 = nothing found): -15 per high, -5 per mid, -1 per low issue."""
    return 100 - sum({"high": 15, "mid": 5, "low": 1}[t["sev"]] for t in r["issues"])


def variant_sheet(items, path="variants.jpg", labels=None, focus=None, tile=420) -> list:
    """Side-by-side critique of variants / thumbnails: each tile = image, 5-value study (squint view), score,
    issue counts, hierarchy ratios and the worst issues; the best score gets a green frame.
    items: Canvases / images; focus: one value or a list (per item). Returns [{label, score, issues, hierarchy}]."""
    n = len(items)
    labels = labels or [chr(65 + i) for i in range(n)]
    focs = focus if isinstance(focus, list) else [focus] * n
    res = []
    for it, lb, fo in zip(items, labels, focs):
        r = critique(it, focus=fo)
        res.append({"label": lb, "score": critique_score(r), "issues": r["issues"], "hierarchy": r["hierarchy"], "_img": it})
    best = max(range(n), key=lambda i: res[i]["score"])
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    ih, th = round(tile * 0.72), 190
    pad = 12
    sheet = Image.new("RGB", (cols * (tile + pad) + pad, rows * (ih + th + pad) + pad), BG)
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(res):
        x0, y0 = pad + (i % cols) * (tile + pad), pad + (i // cols) * (ih + th + pad)
        rgb, _ = _src(r.pop("_img"))
        im, _ = _fit(rgb, tile, ih)
        sheet.paste(im, (x0 + (tile - im.width) // 2, y0))
        if i == best and n > 1:
            d.rectangle([x0 - 4, y0 - 4, x0 + tile + 3, y0 + ih + th - 6], outline=OK, width=3)
        vs, _ = _fit(_value_study(rgb2oklab(rgb)[..., 0]), round(tile * 0.34), th - 50)
        sheet.paste(vs, (x0, y0 + ih + 8))
        tx, ty = x0 + vs.width + 10, y0 + ih + 6
        c = {k: sum(t["sev"] == k for t in r["issues"]) for k in ("high", "mid", "low")}
        ty = _text(d, (tx, ty), f"{r['label']}   score {r['score']}", 17, "bold", OK if i == best else FG)
        ty = _text(d, (tx, ty), f"{c['high']} high  {c['mid']} mid  {c['low']} low", 13, "mono", DIM)
        sc = r["hierarchy"].get("scores", {})
        if sc:
            ratios = " ".join(f"{'sat' if k == 'saturation' else k[0]}{v['ratio']:.1f}" for k, v in sc.items())
            ty = _text(d, (tx, ty), f"focus/rest {ratios}", 13, "mono",
                       RED if min(v["ratio"] for v in sc.values()) < 0.9 else DIM)
        if "squint" in r["hierarchy"]:
            sq = r["hierarchy"]["squint"]
            ty = _text(d, (tx, ty), f"squint dL {sq:.2f}", 13, "mono",
                       RED if sq < 0.08 else DIM)
        for t in [t for t in r["issues"] if t["kind"] != "quality"][:3]:
            msg = f"{t['n']}. {t['kind']}: {t['msg']}"
            msg = msg if len(msg) < 52 else msg[:50] + ".."
            ty = _text(d, (tx, ty), msg, 12, "sans", SEV_COL[t["sev"]])
    _save(sheet, path)
    return res


__all__ = ["metrics", "contact_sheet", "compare", "guides", "composition", "saliency",
           "analyze_reference", "match_palette", "transfer_tone", "grid_view", "layer_boxes", "critique", "hierarchy",
           "critique_score", "variant_sheet"]
