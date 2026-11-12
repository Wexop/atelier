"""Atmospheric effects painted on the current layer: smoke, rain, snowfall, mist, lightning.

All take canvas px and `seed`; each paints into the current layer (put it on its own named layer)."""
import math

import cv2
import numpy as np

from .core import Layer, color, resolve_mask
from .field import _noise2
from .fx import _gblur

F32 = np.float32


def _rgb(c):
    return color(c)[:3].astype(F32)


def _fbm(u, v, seed, octaves=4, gain=0.5):
    """fbm in ~[-1, 1] at arbitrary lattice coords."""
    t, amp, norm = 0.0, 1.0, 0.0
    for i in range(octaves):
        a = 0.5 + i * 1.7
        ca, sa = math.cos(a), math.sin(a)
        f = 2.0 ** i
        t = t + amp * _noise2("perlin", (ca * u - sa * v) * f + i * 17.31, (sa * u + ca * v) * f + i * 9.73, seed)
        norm += amp
        amp *= gain
    return np.tanh(1.6 * t / norm).astype(F32)


def _sstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _box(self, x0, y0, x1, y1):
    """Canvas-px box -> clipped raster box (X0, Y0, X1, Y1) and pixel-centre grids in canvas px (or None)."""
    s = self.s
    X0, Y0 = max(int(math.floor(x0 * s)), 0), max(int(math.floor(y0 * s)), 0)
    X1, Y1 = min(int(math.ceil(x1 * s)), self.W), min(int(math.ceil(y1 * s)), self.H)
    if X0 >= X1 or Y0 >= Y1:
        return None
    xs = (np.arange(X0, X1, dtype=F32) + 0.5) / s
    ys = (np.arange(Y0, Y1, dtype=F32) + 0.5) / s
    gx, gy = np.meshgrid(xs, ys)
    return (X0, Y0, X1, Y1), gx, gy


def _put(self, rgb, a, box):
    """Paint straight colour rgb (h,w,3) with coverage a (h,w) into raster box."""
    col = np.concatenate([np.broadcast_to(rgb, a.shape + (3,)), np.ones(a.shape + (1,), F32)], -1)
    self.paint(np.clip(a, 0, 1).astype(F32), col, box)
    return self


def _region(self, region):
    if region is None:
        return None
    m = resolve_mask(region, self.H, self.W, None, self.s)
    return np.broadcast_to(m, (self.H, self.W))


# ------------------------------------------------------------------ smoke
def _layer_smoke(self, x, y, height, width=None, drift=0.3, spread=3.0, color="#d9d9d9", shadow=None,
                 density=0.8, curl=1.0, wisps=0.6, light=(120, 40), t=0.0, seed=0):
    """Rising smoke plume from (x, y) (source, canvas px) up `height` px. width = source width (default
    height/8), spread = top width / source width, drift = sideways lean at the top (fraction of height,
    + = right, wind), curl = turbulence, wisps = breakup into strands. Lit side `color`, far side `shadow`
    (default: darker, cooler). light = (azimuth, elev) like the anime helpers (0 = from right, 90 = top).
    t = time 0..1 (animation: billows churn, loops at t=1)."""
    w0 = width or height / 8
    rng = np.random.default_rng(seed)
    top_w = w0 * spread
    lean = drift * height
    pad = top_w * 0.9 + abs(lean) + height * 0.1 * curl
    b = _box(self, x - pad, y - height * 1.08, x + pad, y + w0 * 0.6)
    if b is None:
        return self
    box, gx, gy = b
    u = np.clip((y - gy) / height, -0.2, 1.3)
    uc = np.clip(u, 0, 1)
    wob = _noise2("perlin", uc * 2.2 + 3.1, np.full_like(uc, 0.5 + seed % 7), seed + 1)
    cx = x + lean * uc ** 1.6 + wob * height * 0.08 * curl * uc
    prof = lambda q: w0 / 2 * (0.6 + (spread - 0.6) * q ** 0.8)
    hw = prof(uc)
    # coordinates that stretch with the plume (dv = dy / half-width): billows grow as it rises, stay round
    q = np.linspace(0, 1, 257)
    vt = np.concatenate([[0], np.cumsum(height / 256 / prof((q[1:] + q[:-1]) / 2))])
    sx = (gx - cx) / hw
    sv = np.interp(u, q, vt)
    turb = curl * (0.2 + 1.3 * uc)
    oy = rng.uniform(0, 50)
    tu, tv = 0.8 * (math.cos(2 * math.pi * t) - 1), 0.8 * math.sin(2 * math.pi * t)   # 0 at t = 0
    sx_, sv_ = sx, sv
    sx, sv = sx + tu * uc, sv + tv
    wx = _fbm(sx * 0.3, sv * 0.3 + oy, seed + 2, 3) * turb
    wy = _fbm(sx * 0.3 + 40, sv * 0.3 + oy, seed + 3, 3) * turb
    n = _fbm(sx * 0.5 + wx * 1.2, sv * 0.5 + wy * 1.2 + oy, seed + 4, 5) * 0.5 + 0.5
    sx, sv = sx_, sv_
    br = wisps * (0.35 + 0.9 * uc)                          # breakup grows with height
    d = sx + wx * 0.7 + (0.5 - n) * 1.4 * br                # billowy, ragged outline; interior stays filled
    core = np.exp(-d * d * 1.3)
    env = _sstep(-0.01, 0.08, u) * (1 - _sstep(0.6, 1.05, u))
    dens = core * env * (1 - 0.6 * br * (1 - n))
    a = _sstep(0.04, 0.5, dens) * density * (1 - 0.4 * uc)         # thins out as it rises
    # lighting: lit side faces the light (gradient of blurred density)
    bump = cv2.GaussianBlur((dens * (0.6 + 0.8 * n)).astype(F32), (0, 0), max(w0 * self.s * 0.12, 0.7))
    dyv, dxv = np.gradient(bump)
    az = math.radians(light[0])
    lx, ly = math.cos(az), -math.sin(az)
    g = -(dxv * lx + dyv * ly)
    g = g / (np.percentile(np.abs(g), 99) + 1e-6)
    lit = np.clip(0.5 + 0.7 * g + 0.3 * (n - 0.5), 0, 1)[..., None]
    c1 = _rgb(color)
    c0 = _rgb(shadow) if shadow is not None else c1 * np.array([0.5, 0.52, 0.6], F32)
    rgb = c0 + (c1 - c0) * lit
    return _put(self, rgb, a, box)


# ------------------------------------------------------------------ rain
def _layer_rain(self, n=700, angle=12.0, length=(14, 46), width=1.0, color="#cfdbe8", opacity=0.55, region=None,
                depth=3, splashes=0, ground=None, t=0.0, seed=0):
    """Rain streaks over the layer: n drops, angle = lean from vertical (deg, + = falling to the right),
    length = (far, near) streak px, width = near streak width px, depth = 1-4 distance bands (far = short,
    thin, faint). region = mask where it rains (e.g. not under a roof). splashes = number of small crown
    splashes on `ground` (mask; default bottom 30%). t = time 0..1 (animation: drops fall, loops at t=1).
    Pair with wet reflections and a cool, low-contrast grade."""
    s, H, W = self.s, self.H, self.W
    rng = np.random.default_rng(seed)
    a = np.zeros((H, W), F32)
    th = math.radians(angle)
    dx, dy = math.sin(th), math.cos(th)
    depth = int(np.clip(depth, 1, 4))
    for band in range(depth):
        z = (band + 1) / depth                         # 1 = nearest
        m = int(n * (0.5 ** (depth - 1 - band)) * (1.6 if band < depth - 1 else 1) / 1.6 + 0.5)
        buf = np.zeros((H, W), np.uint8)
        L = (length[0] + (length[1] - length[0]) * z ** 1.5) * s
        thick = max(1, int(round(width * s * (0.5 + 0.5 * z))))
        xs = rng.uniform(-L * dx - 10, W + 10, m)
        ys = rng.uniform(-L, H + L, m)
        if t:                                           # fall: near bands faster, whole cycles -> seamless loop
            span = H + 2 * L
            y2 = (ys + L + t * (band + 1) * span) % span - L
            xs, ys = xs + (y2 - ys) * dx / max(dy, 1e-3), y2
        ls = L * rng.uniform(0.6, 1.2, m)
        vals = (255 * rng.uniform(0.45, 1.0, m)).astype(int)
        sh = 4                                          # sub-pixel line coords (1/16 px)
        for px, py, l, v in zip(xs, ys, ls, vals):
            p0 = (int(px * 16), int(py * 16))
            p1 = (int((px + dx * l) * 16), int((py + dy * l) * 16))
            cv2.line(buf, p0, p1, int(v), thick, cv2.LINE_AA, sh)
        b = buf.astype(F32) / 255 * (0.35 + 0.65 * z)
        if z > 0.99 and depth > 1:
            b = cv2.GaussianBlur(b, (0, 0), 0.6 * s)
        a = np.maximum(a, b)
    reg = _region(self, region)
    if reg is not None:
        a = a * reg
    rgb = np.broadcast_to(_rgb(color), (H, W, 3))
    _put(self, rgb, a * opacity, (0, 0, W, H))
    if splashes:
        gm = _region(self, ground if ground is not None else (lambda x, y: (y > 0.7).astype(F32)))
        p = gm.ravel() / max(gm.sum(), 1e-6)
        idx = rng.choice(gm.size, int(splashes), p=p)
        buf = np.zeros((H, W), np.uint8)
        for i in idx:
            py, px = divmod(int(i), W)
            near = 0.5 + py / H
            r = rng.uniform(2.0, 4.5) * near * s
            cv2.ellipse(buf, (int(px * 16), int(py * 16)), (int(r * 16), int(r * 0.32 * 16)), 0, 180, 360,
                        int(rng.uniform(140, 255)), 1, cv2.LINE_AA, 4)
            for k in (-1, 0, 1):                        # crown droplets
                qx, qy = px + k * r * 0.7, py - r * (1.1 if k == 0 else 0.8)
                cv2.circle(buf, (int(qx * 16), int(qy * 16)), max(int(0.7 * s * 16), 8), 200, -1, cv2.LINE_AA, 4)
        _put(self, rgb, buf.astype(F32) / 255 * min(1, opacity * 1.4), (0, 0, W, H))
    return self


# ------------------------------------------------------------------ snowfall
def _layer_snowfall(self, n=500, size=(1.2, 5.0), color="#ffffff", opacity=0.9, region=None, bokeh=0.06, t=0.0,
                    seed=0):
    """Falling snow: n soft flakes, size = (far, near) radius px; nearer flakes are bigger and brighter.
    bokeh = fraction of very near, large, blurred out-of-focus flakes. region = mask where it snows.
    t = time 0..1 (animation: flakes fall and sway, loops at t=1)."""
    s, H, W = self.s, self.H, self.W
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, n) ** 2.2                       # most flakes far away
    big = rng.uniform(0, 1, n) < bokeh
    r = (size[0] + (size[1] - size[0]) * z) * s
    r = np.where(big, size[1] * s * rng.uniform(2.2, 3.5, n), r)
    alpha = np.where(big, rng.uniform(0.25, 0.45, n), 0.45 + 0.55 * z * rng.uniform(0.8, 1, n))
    xs, ys = rng.uniform(0, W, n), rng.uniform(0, H, n)
    ph = rng.uniform(0, 1, n)
    k = 1 + np.floor(z * 1.999) + big                         # whole fall cycles per clip: near flakes faster
    span = H + 4 * r
    ys = (ys + 2 * r + t * k * span) % span - 2 * r
    xs = xs + np.sin(2 * np.pi * (t * k + ph)) * (2 + 6 * z) * s   # sway (same at t = 0 and t = 1)
    a = np.zeros((H, W), F32)
    edges = [0, 1.2, 2, 3.5, 6, 10, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi)
        if not sel.any():
            continue
        buf = np.zeros((H, W), F32)
        rm = float(np.median(r[sel]))
        for px, py, rr, al in zip(xs[sel], ys[sel], r[sel], alpha[sel]):
            rd = max(rr * 0.7, 0.75)                   # sub-pixel flakes would twinkle as they move: keep >= 0.75 px,
            al = al * min(1.0, (rr * 0.7 / rd) ** 2)   # same light energy
            cv2.circle(buf, (int(px * 16), int(py * 16)), int(rd * 16), float(al), -1, cv2.LINE_AA, 4)
        sig = 0.35 * rm + 0.35 if rm < 10 else 0.5 * rm
        buf = cv2.GaussianBlur(buf, (0, 0), sig) * (1.6 if rm < 10 else 1.2)
        a = np.maximum(a, np.clip(buf, 0, 1))
    reg = _region(self, region)
    if reg is not None:
        a = a * reg
    return _put(self, np.broadcast_to(_rgb(color), (H, W, 3)), a * opacity, (0, 0, W, H))


# ------------------------------------------------------------------ mist
def _layer_mist(self, y0, y1, color="#e6ecf0", density=0.75, wisps=0.6, scale=1.0, x0=None, x1=None, t=0.0, seed=0):
    """Low fog bank: transparent at y0 (top), thickest at y1 (canvas px), fading out just below y1.
    wisps = horizontal streak breakup, scale = streak size. Put it between depth planes (behind the
    foreground, over the far hills) to separate them; stack two banks for depth. t = time 0..1 (wisps drift, loops)."""
    x0 = 0 if x0 is None else x0
    x1 = self.w if x1 is None else x1
    hgt = max(y1 - y0, 1)
    b = _box(self, x0, y0, x1, y1 + hgt * 0.25)
    if b is None:
        return self
    box, gx, gy = b
    v = (gy - y0) / hgt
    prof = _sstep(0, 0.85, v) ** 1.3 * (1 - _sstep(1.0, 1.25, v))
    k = 1 / (hgt * 1.4 * scale)
    ou, ov = 0.9 * (math.cos(2 * math.pi * t) - 1), 0.9 * math.sin(2 * math.pi * t)   # loops; 0 at t = 0
    wy = _fbm(gx * k * 0.5 + ou, gy * k * 0.5 + ov, seed + 1, 2) * 0.6
    n = _fbm(gx * k * 0.35 + ou, (gy * k + wy) * 2.6 + ov * 0.3, seed + 2, 5) * 0.5 + 0.5
    m = _fbm(gx * k * 1.2 + 7 - ou, (gy * k + wy) * 5.0 - ov * 0.3, seed + 3, 3) * 0.5 + 0.5
    br = 1 - wisps + wisps * np.clip(1.7 * n * (0.6 + 0.6 * m) - 0.15, 0, 1.3)
    side = _sstep(x0, x0 + hgt * 0.8, gx) * (1 - _sstep(x1 - hgt * 0.8, x1, gx)) if (x0 > 0 or x1 < self.w) else 1
    a = prof * br * density * side
    rgb = _rgb(color) * (0.94 + 0.06 * n[..., None])
    return _put(self, rgb, a, box)


# ------------------------------------------------------------------ lightning
def _bolt(p0, p1, jag, rng, levels=7):
    pts = [np.asarray(p0, float), np.asarray(p1, float)]
    off = jag * np.linalg.norm(pts[1] - pts[0])
    for _ in range(levels):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            m = (a + b) / 2 + nrm * rng.normal(0, off * 0.5)
            out += [m, b]
        pts = out
        off *= 0.55
    return np.array(pts)


def _layer_lightning(self, x0, y0, x1, y1, branches=4, jag=0.16, width=2.5, color="#bcd2ff", glow=1.0, seed=0):
    """Lightning bolt from (x0, y0) to (x1, y1) (canvas px): jagged main channel, `branches` forks thinning
    out, white-hot core, coloured glow. Brighten the sky / cloud undersides around it yourself."""
    s, H, W = self.s, self.H, self.W
    rng = np.random.default_rng(seed)
    main = _bolt((x0, y0), (x1, y1), jag, rng)
    paths = [(main, 1.0)]
    L = math.hypot(x1 - x0, y1 - y0)
    for _ in range(int(branches)):
        i = int(rng.uniform(0.12, 0.75) * len(main))
        a = main[i]
        dirv = main[min(i + 8, len(main) - 1)] - a
        ang = math.atan2(dirv[1], dirv[0]) + rng.choice([-1, 1]) * math.radians(rng.uniform(20, 50))
        ln = L * rng.uniform(0.12, 0.35) * (1 - i / len(main))
        b = a + ln * np.array([math.cos(ang), math.sin(ang)])
        br = _bolt(a, b, jag * 1.2, rng, 6)
        paths.append((br, 0.55))
        if rng.uniform() < 0.5:                         # sub-fork
            j = int(rng.uniform(0.3, 0.7) * len(br))
            ang2 = ang + rng.choice([-1, 1]) * math.radians(rng.uniform(20, 45))
            c2 = br[j] + ln * 0.4 * np.array([math.cos(ang2), math.sin(ang2)])
            paths.append((_bolt(br[j], c2, jag * 1.3, rng, 5), 0.35))
    core = np.zeros((H, W), np.uint8)
    for pts, k in paths:
        n = len(pts)
        P = (pts * s * 16).astype(np.int32)
        for i in range(n - 1):                          # taper along each path
            t = i / n
            th = max(1, int(round(width * s * k * (1 - 0.6 * t))))
            cv2.line(core, tuple(P[i]), tuple(P[i + 1]), int(255 * (1 - 0.35 * t * (k < 1))), th, cv2.LINE_AA, 4)
    c = core.astype(F32) / 255
    g = _gblur(c, width * 3 * s) * 2.2 + _gblur(c, width * 14 * s) * 3.0
    a = np.clip(c + np.clip(g, 0, 1) * glow * 0.75, 0, 1)
    tint = _rgb(color)
    wcore = np.clip(c / np.maximum(a, 1e-4), 0, 1)[..., None]
    rgb = tint + (_rgb("#ffffff") - tint) * wcore
    return _put(self, rgb, a, (0, 0, W, H))


Layer.smoke = _layer_smoke
Layer.rain = _layer_rain
Layer.snowfall = _layer_snowfall
Layer.mist = _layer_mist
Layer.lightning = _layer_lightning

__all__ = []
