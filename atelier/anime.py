"""Anime-style painted elements: cel-shaded cumulus clouds with rim light.

c.anime_cloud(x, y, w)                               # one cumulus, base centred at (x, y), w px wide
c.anime_cloud(x, y, w, palette="sunset", light=(160, 25), seed=3)
"""
from __future__ import annotations

import math

import numpy as np
import cv2

from .core import F32, Layer, color as _col

CLOUD_PALETTES = {  # shadow, mid, light, rim
    "day": ("#9fb2d9", "#d6e0f2", "#ffffff", "#ffffff"),
    "sunset": ("#8a6fa8", "#e89a9a", "#ffd7b5", "#fff2d6"),
    "dusk": ("#4b4a86", "#8c7cb8", "#e7a7b8", "#ffd2a8"),
    "storm": ("#4c5670", "#7f8aa3", "#c7cfdc", "#eef2f7"),
}


def _rgb(c):
    return np.asarray(_col(c)[:3], F32)


def _noise(h, w, cell, rng):
    """Smooth value noise in [-1, 1], about one feature per `cell` px."""
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    g = rng.uniform(-1, 1, (gh, gw)).astype(F32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _puffs(w, h, rng, wisps):
    """Circles (cx, cy, r) in local px; base line at y = 0, cloud grows upward (negative y)."""
    P = []
    n = max(5, int(w / h * 4))
    for i in range(n):                                     # bottom row: flattish base
        t = (i + 0.5) / n * 2 - 1
        r = h * rng.uniform(0.22, 0.32) * (1 - 0.35 * abs(t))
        P.append((t * w * 0.46, -r * 0.75, r))
    for i in range(n + 2):                                 # main body: domed silhouette
        t = rng.uniform(-1, 1) * 0.9
        top = h * (1 - abs(t) ** 1.6) ** 0.8
        r = h * rng.uniform(0.2, 0.36) * (1 - 0.4 * abs(t))
        P.append((t * w * 0.45, -min(top - r * 0.6, top * rng.uniform(0.35, 0.8)) - r * 0.3, r))
    for _ in range(3):                                     # towering top puffs
        t = rng.normal(0, 0.25)
        r = h * rng.uniform(0.22, 0.32)
        P.append((t * w * 0.4, -h + r * rng.uniform(0.9, 1.3), r))
    for _ in range(int(wisps * 8)):                        # small puffs hugging the flanks
        side = rng.choice([-1, 1])
        r = h * rng.uniform(0.07, 0.13)
        P.append((side * w * rng.uniform(0.44, 0.54), -r * rng.uniform(0.6, 1.6), r))
    return np.array(P, F32)


def _light_vec(light):
    az, el = math.radians(light[0]), math.radians(light[1])
    return np.array([math.cos(az) * math.cos(el), -math.sin(az) * math.cos(el), math.sin(el)], F32)


def _cel(layer, P, ox, oy, H_, pal, light, bands, rim, edge, bottom, opacity, rng, flat_base, leafy=0.0, front=0.15, vert_w=0.35):
    """Cel-shade a smooth union of puffs P (cx, cy, r in raster px relative to raster point (ox, oy);
    the shape spans y in [-H_, 0]) and composite it onto `layer`."""
    pad = int(H_ * 0.25 + 6)
    x0 = int(ox + (P[:, 0] - P[:, 2]).min() - pad); x1 = int(ox + (P[:, 0] + P[:, 2]).max() + pad)
    y0 = int(oy + (P[:, 1] - P[:, 2]).min() - pad); y1 = int(oy + (P[:, 1] + P[:, 2]).max() + pad)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, layer.W), min(y1, layer.H)
    if X0 >= X1 or Y0 >= Y1:
        return
    bw, bh = x1 - x0, y1 - y0
    yy, xx = np.mgrid[0:bh, 0:bw].astype(F32)
    lx, ly = xx + x0 - ox, yy + y0 - oy

    # silhouette: smooth union of domes; the front-most puff gives the normal
    k = 0.08
    acc = np.zeros((bh, bw), F32)
    dome = np.full((bh, bw), -1e9, F32)
    nx, ny, nz = (np.zeros((bh, bw), F32) for _ in range(3))
    for cx, cy, r in P:
        dx, dy = (lx - cx) / r, (ly - cy) / r
        d2 = dx * dx + dy * dy
        acc += np.exp((1 - np.sqrt(d2)) / k)
        z = np.sqrt(np.clip(1 - d2, 0, 1))
        top = (z + 0.15) * r + cy * front
        m = (top > dome) & (d2 < 1.2)
        dome = np.where(m, top, dome)
        nx, ny, nz = np.where(m, dx, nx), np.where(m, dy, ny), np.where(m, z, nz)
    dens = k * np.log(acc + 1e-20)
    wob = _noise(bh, bw, H_ * 0.15, rng) * 0.8 + _noise(bh, bw, H_ * 0.05, rng) * 0.2
    dens = dens + wob * 0.06 * edge + _noise(bh, bw, max(H_ * 0.018, 2), rng) * 0.09 * leafy
    alpha = np.clip(dens / (1.2 / max(float(P[:, 2].mean()), 1)) + 0.5, 0, 1)
    if flat_base:
        base_y = -_noise(1, bw, H_ * 0.3, rng)[0] * H_ * 0.015 * edge - H_ * 0.01
        alpha *= np.clip((base_y[None, :] - ly) / 1.2 + 0.5, 0, 1)

    # cel-quantised lighting
    L = _light_vec(light)
    sg = max(H_ * 0.03, 0.7)
    nx, ny, nz = (cv2.GaussianBlur(v, (0, 0), sg) for v in (nx, ny, nz + 0.3))
    lam = (nx * L[0] + ny * L[1] + nz * L[2]) / (np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6)
    vert = np.clip(-ly / H_, 0, 1)
    t = lam * 0.75 + vert * vert_w - (1 - vert) ** 3 * bottom * 0.5
    t = t + _noise(bh, bw, H_ * 0.1, rng) * 0.05 * edge + _noise(bh, bw, max(H_ * 0.03, 2), rng) * 0.08 * leafy
    t = cv2.GaussianBlur(t.astype(F32), (0, 0), max(H_ * 0.008, 0.6))
    sh, md, lt, rc = (_rgb(c) for c in pal)
    bands = int(np.clip(bands, 2, 4))
    steps = [sh, md, lt] if bands == 3 else [sh, lt] if bands == 2 else [sh, (sh + md) / 2, md, lt]
    col = np.broadcast_to(steps[0], (bh, bw, 3)).copy()
    for e, c in zip(np.linspace(0.12, 0.58, len(steps) - 1), steps[1:]):
        m = np.clip((t - e) / 0.012 + 0.5, 0, 1)[..., None]
        col = col * (1 - m) + c * m

    if rim > 0:   # thin highlight just inside the silhouette, on the side facing the light
        inner = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=max(1, int(H_ * 0.018)))
        band = np.clip(alpha - inner, 0, 1)
        ga_y, ga_x = np.gradient(cv2.GaussianBlur(alpha, (0, 0), max(H_ * 0.02, 1.0)))
        face = np.clip(-(ga_x * L[0] + ga_y * L[1]) / (np.sqrt(ga_x ** 2 + ga_y ** 2) + 1e-6), 0, 1)
        m = (band * face ** 0.7 * rim * np.clip(vert * 2, 0, 1))[..., None]
        col = col * (1 - m) + rc * m

    a = (alpha * opacity)[..., None]
    src = np.concatenate([col * a, a], -1)[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    dst = layer.px[Y0:Y1, X0:X1]
    dst[:] = src + dst * (1 - src[..., 3:4])
    layer.touch()


def _layer_anime_cloud(self, x, y, w, h=None, palette="day", light=(120, 35), bands=3, rim=0.7, wisps=0.4,
                       edge=0.5, bottom=0.6, opacity=1.0, seed=0):
    """Paint a cel-shaded cumulus whose flat base is centred at canvas (x, y), `w` px wide, `h` px tall
    (default 0.55 w). palette: day | sunset | dusk | storm | (shadow, mid, light, rim) colours.
    light = (azimuth deg, 0 = from right, 90 = from top; elevation deg). bands = tone steps (2-4),
    rim = sun-side edge highlight, wisps = small loose puffs, edge = painterly wobble of the
    silhouette and tone borders, bottom = darkening toward the base."""
    s = self.s
    h = w * 0.55 if h is None else h
    rng = np.random.default_rng(seed)
    P = _puffs(w * s, h * s, rng, wisps)
    pal = CLOUD_PALETTES[palette] if isinstance(palette, str) else palette
    _cel(self, P, x * s, y * s, h * s, pal, light, bands, rim, edge, bottom, opacity, rng, True)
    return self


TREE_PALETTES = {  # canopy shadow, mid, light, rim, trunk
    "summer": ("#1f5a2e", "#3f8f3a", "#86c94f", "#d8f27a", "#4a3226"),
    "spring": ("#3f7f3a", "#6fb44a", "#b4e070", "#f0ffb0", "#5a3e2c"),
    "autumn": ("#8a2f22", "#d0552e", "#f39a3c", "#ffd27a", "#4a2e22"),
    "sakura": ("#c4708f", "#ef9fb8", "#ffd3e0", "#fff2f6", "#4b3030"),
    "dusk": ("#1c1a3c", "#2b2958", "#4a3f78", "#f6b27c", "#1c1a3c"),
}
_CROWNS = {  # canopy centre y (x height), half width (x width), half height (x height), clumps
    "round": (0.66, 0.42, 0.28, 8), "tall": (0.62, 0.26, 0.34, 7), "umbrella": (0.8, 0.5, 0.12, 11)}


def _canopy(Hs, Ws, kind, rng):
    """Leaf clumps (cx, cy, r) relative to the trunk base."""
    fy, fx, fh, n = _CROWNS[kind]
    cy0, rx, ry = -Hs * fy, Ws * fx, Hs * fh
    big = []
    if kind == "umbrella":                                 # wide flat crown: clumps along a drooping arc
        for i in range(n):
            u = i / (n - 1) * 2 - 1
            big.append((u * rx, cy0 + ry * 0.9 * u * u + rng.uniform(-0.3, 0.3) * ry, ry * rng.uniform(1.0, 1.35)))
        for i in range(n // 2):
            u = rng.uniform(-0.7, 0.7)
            big.append((u * rx, cy0 - ry * 0.7, ry * rng.uniform(0.9, 1.2)))
    else:
        for i in range(n):                                 # big clumps on an ellipse + one core
            a = 2 * math.pi * i / n + rng.uniform(-0.25, 0.25)
            big.append((math.cos(a) * rx * 0.7, cy0 + math.sin(a) * ry * 0.7 - ry * 0.1, min(rx, ry) * rng.uniform(0.55, 0.8)))
        big.append((0, cy0, min(rx, ry) * 0.9))
    P = list(big)
    for cx, cy, cr in big:                                 # leafy scallops on the outer edge of each clump
        for _ in range(3):
            a = rng.uniform(-math.pi, 0.3) if cy < cy0 else rng.uniform(-math.pi, math.pi)
            P.append((cx + math.cos(a) * cr * 0.75, cy + math.sin(a) * cr * 0.75, cr * rng.uniform(0.32, 0.48)))
    return np.array(P, F32), cy0


def _layer_anime_tree(self, x, y, height=300.0, kind="round", palette="summer", light=(120, 40), width=1.0,
                      bands=3, rim=0.8, leafy=0.6, edge=0.5, opacity=1.0, seed=0):
    """Paint a cel-shaded tree standing at canvas (x, y) (trunk base), `height` px tall.
    kind: round | tall | umbrella. palette: summer | spring | autumn | sakura | dusk | (shadow, mid, light,
    rim, trunk). width scales the crown; leafy = jagged leaf-cluster edges; other args as anime_cloud."""
    s = self.s
    Hs, Ws = height * s, height * s * width * (1.4 if kind == "umbrella" else 1.0)
    rng = np.random.default_rng(seed)
    pal = TREE_PALETTES[palette] if isinstance(palette, str) else palette
    P, cy0 = _canopy(Hs, Ws, kind, rng)
    bx, by = x * s, y * s
    top_y = by + cy0
    tw = Hs * 0.045
    lean = rng.uniform(-0.06, 0.06) * Hs
    trunk = pal[4] if len(pal) > 4 else "#4a3226"
    pts = [(bx - tw * 1.4, by), (bx - tw * 0.8, by - Hs * 0.2), (bx + lean - tw * 0.45, top_y),
           (bx + lean + tw * 0.45, top_y), (bx + tw * 0.8, by - Hs * 0.2), (bx + tw * 1.4, by)]
    self.ellipse(x, y, Ws * 0.32 / s, Hs * 0.025 / s, fill="#000000", opacity=0.18)   # ground contact shadow
    self.polygon([(px / s, py / s) for px, py in pts], fill=trunk)
    for side in (-1, 1):                                   # two branches reaching into the crown
        sy = by - Hs * rng.uniform(0.38, 0.5)
        if kind == "umbrella":
            ex, ey = bx + side * Ws * 0.33, top_y + Hs * 0.06
        else:
            ex, ey = bx + lean * 0.5 + side * Ws * rng.uniform(0.18, 0.3), sy - Hs * rng.uniform(0.12, 0.2)
        self.line(bx / s, sy / s, ex / s, ey / s, stroke=trunk, width=tw * 0.7 / s, cap="round")
    lx = float(_light_vec(light)[0])
    if abs(lx) > 0.1:                                      # sunlit stripe on the trunk
        off = np.sign(lx) * tw * 0.5
        self.line((bx + off) / s, by / s, (bx + lean * 0.8 + off * 0.5) / s, (top_y + Hs * 0.1) / s,
                  stroke=pal[2], width=tw * 0.35 / s, opacity=0.35)
    _cel(self, P, bx, by, Hs, pal[:4], light, bands, rim, edge, 0.4, opacity, rng, False, leafy, front=0.0, vert_w=0.1)
    return self


Layer.anime_cloud = _layer_anime_cloud
Layer.anime_tree = _layer_anime_tree

# ------------------------------------------------------------------ light effects (for c.apply / L.apply)
def _screen_add(img, add):
    out = img.copy()
    out[..., :3] = 1 - (1 - img[..., :3]) * (1 - np.clip(add, 0, 1))
    return out


def _radial_blur(a, cx, cy, passes=6, k=1.06):
    """Streak `a` away from (cx, cy) by repeated zoom-and-max (log-time radial blur)."""
    h, w = a.shape[:2]
    out = a.copy()
    for i in range(passes):
        sc = k ** (2 ** i)
        M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
        z = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out = np.maximum(out, z * 0.92)
    return out


def god_rays(x, y, rays=14, length=0.9, strength=0.45, color="#fff0c8", threshold=0.7, occlude=True, seed=0):
    """Sun beams from canvas point (x, y): bright sky behind the scene is streaked outward and cut into
    `rays` beams. occlude=True: beams only come from what is brighter than `threshold` (dark trees and
    hills block them). length = reach as a fraction of the image diagonal."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        cx, cy = x * s, y * s
        lum = img[..., :3] @ np.array([0.3, 0.59, 0.11], F32)
        src = np.clip((lum - threshold) / max(1 - threshold, 1e-3), 0, 1) if occlude else np.ones((h, w), F32)
        yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(w, h)
        src = src * np.exp(-d / 0.12)                                  # bright sky near the source seeds rays;
        streak = _radial_blur(src, cx, cy)                             # dark shapes in between cast shadow shafts
        ang = (np.arctan2(yy - cy, xx - cx) / (2 * math.pi) + 0.5) * rays
        rng = np.random.default_rng(seed)
        g = rng.uniform(0.15, 1, rays + 1); g[-1] = g[0]
        i0 = np.floor(ang).astype(int) % rays
        fr = ang - np.floor(ang)
        beam = g[i0] * (1 - fr) + g[i0 + 1] * fr
        beam = np.clip((np.sin(fr * math.pi) ** 0.6) * beam, 0, 1)        # soft gaps between beams
        fall = np.clip(1 - d / max(length, 1e-3), 0, 1) ** 1.5
        if occlude:                                                    # beams pass behind dark silhouettes
            fall = fall * (0.2 + 0.8 * np.clip((lum - 0.35) / 0.3, 0, 1))
        add = (streak * beam * fall * strength)[..., None] * _rgb(color)
        return _screen_add(img, add)
    return f


def lens_flare(x, y, strength=0.6, color="#ffd9a0", ghosts=6, streak=0.5, seed=0):
    """Lens flare for a light at canvas (x, y): halo, horizontal streak and tinted ghost discs along the
    line through the image centre."""
    def f(img, s=1.0):
        h, w = img.shape[:2]
        cx, cy = x * s, y * s
        D = math.hypot(w, h)
        yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        c = _rgb(color)
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (D * D)
        add = (np.exp(-r2 / 0.004) * 0.9 + np.exp(-r2 / 0.03) * 0.35)[..., None] * c
        if streak > 0:
            add += (np.exp(-((yy - cy) / (D * 0.0035)) ** 2) * np.exp(-np.abs(xx - cx) / (D * 0.25)) * streak)[..., None] * c
        rng = np.random.default_rng(seed)
        tints = [_rgb(t) for t in ("#9fd8ff", "#c9a8ff", "#ffb3c8", "#b8ffcf", "#ffe6a0")]
        ox, oy = w / 2 - cx, h / 2 - cy
        for i in range(ghosts):
            t = rng.uniform(0.5, 2.1)
            gx, gy = cx + ox * t, cy + oy * t
            gr = D * rng.uniform(0.012, 0.06)
            d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2) / gr
            disc = np.clip(1 - d, 0, 1) ** 0.4 * (0.5 + 0.5 * np.clip((d - 0.75) / 0.25, 0, 1))  # brighter rim
            add += (disc * rng.uniform(0.15, 0.35))[..., None] * tints[i % len(tints)]
        return _screen_add(img, add * strength)
    return f


def _layer_sparkles(self, n=40, region=None, size=(6, 16), color="#fff6d8", rays=4, glow=0.6, seed=0, opacity=1.0):
    """Star glints (water, snow, magic): n sparkles with `rays` thin spikes, scattered where `region`
    (mask) is bright. size = (min, max) spike length in canvas px."""
    s = self.s
    rng = np.random.default_rng(seed)
    H, W = self.H, self.W
    if region is None:
        ys, xs = rng.uniform(0, H, n), rng.uniform(0, W, n)
    else:
        from .core import resolve_mask
        m = resolve_mask(region, H, W, s=s)
        m = np.broadcast_to(m, (H, W)) if m.shape != (H, W) else m
        p = m.ravel() / max(m.sum(), 1e-6)
        idx = rng.choice(m.size, n, p=p)
        ys, xs = idx // W + rng.uniform(0, 1, n), idx % W + rng.uniform(0, 1, n)
    c = _rgb(color)
    for px, py in zip(xs, ys):
        L = rng.uniform(*size) * s
        R = int(L + 3)
        x0, y0 = int(px) - R, int(py) - R
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
        if X0 >= X1 or Y0 >= Y1:
            continue
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
        dx, dy = xx - px, yy - py
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        a = np.zeros_like(r)
        rot = rng.uniform(0, math.pi / rays) * (rays > 4)
        for k in range(rays):
            th = rot + math.pi * k / rays * (2 if rays % 2 else 1)
            along = dx * math.cos(th) + dy * math.sin(th)
            across = -dx * math.sin(th) + dy * math.cos(th)
            ln = L if k % 2 == 0 else L * 0.55
            a = np.maximum(a, np.exp(-(across / (0.6 * s + 0.04 * L)) ** 2) * np.clip(1 - np.abs(along) / ln, 0, 1) ** 1.5)
        a = np.maximum(a, np.exp(-(r / (0.12 * L + 0.5)) ** 2) * 1.0 + np.exp(-(r / (0.35 * L)) ** 2) * glow * 0.5)
        a = np.clip(a * opacity, 0, 1)[..., None]
        dst = self.px[Y0:Y1, X0:X1]
        dst[:] = np.concatenate([c * a, a], -1) + dst * (1 - a)
    self.touch()
    return self


Layer.sparkles = _layer_sparkles

# ------------------------------------------------------------------ painted mountain range
MOUNTAIN_PALETTES = {  # shadow, mid, light, snow, snow shadow, haze
    "day": ("#4a6ea8", "#6c93cc", "#8fb2e0", "#f6f9ff", "#b9cbe8", "#cfe3f6"),
    "dusk": ("#3a3470", "#5c5596", "#8a7fbf", "#ffd9c4", "#a99ad0", "#e8b9c8"),
    "autumn": ("#5a4a6e", "#86708a", "#c4a08e", "#fff4ea", "#c8bcd6", "#f0d8c0"),
}


def _layer_anime_mountains(self, y, height, peaks=4, x0=None, x1=None, palette="day", light=(150, 35), snow=0.35,
                           haze=0.6, sharp=1.0, depth=0.35, bands=3, seed=0, opacity=1.0):
    """Paint a cel-shaded mountain range whose foot lies on canvas row y; the highest summit rises `height` px.
    Built from an eroded height map seen from the front, so ridges, gullies and snow streaks are real
    relief, then flattened into `bands` tones. peaks ~ number of main summits between x0 and x1 (default
    full width); depth = how much the far ranges rise behind (0 = flat wall); snow = top fraction under
    snow (it settles on gentle slopes); sharp > 1 = spikier summits.
    palette: day | dusk | autumn | (shadow, mid, light, snow, snow_shadow, haze). haze fades far and low parts."""
    from .terrain import terrain, normals
    s = self.s
    x0 = 0 if x0 is None else x0
    x1 = self.w if x1 is None else x1
    rng = np.random.default_rng(seed)
    pal = MOUNTAIN_PALETTES[palette] if isinstance(palette, str) else palette
    sh, md, lt, sn, sns, hz = (_rgb(c) for c in pal)

    n = 160
    hm = terrain(n, "alpine", seed=seed, scale=max(peaks * 0.55, 1.0))
    zf = np.linspace(0, 1, n, dtype=F32)[:, None]                 # 0 = back row, 1 = front row
    xf = np.linspace(0, 1, n, dtype=F32)[None, :]
    env = np.clip(1 - (2 * zf - 1.1) ** 2 * 1.6, 0, 1) * np.clip(np.sin(np.pi * xf) * 1.3, 0, 1)
    hm = cv2.GaussianBlur(np.clip(hm, 0, 1).astype(F32), (0, 0), 1.2) ** sharp * env   # summits mid-depth
    hm = (hm / max(float(hm.max()), 1e-6)).astype(F32)

    span = x1 - x0
    X0, X1 = max(int(x0 * s), 0), min(int(x1 * s) + 1, self.W)
    Y0, Y1 = max(int((y - height * (1 + depth) * 1.02) * s), 0), min(int(y * s) + 1, self.H)
    if X0 >= X1 or Y0 >= Y1:
        return self
    bw, bh = X1 - X0, Y1 - Y0
    full = max(int(span * s), 2)                                   # resample columns to screen pixels
    hx = cv2.resize(hm, (full, n), interpolation=cv2.INTER_CUBIC)[:, max(X0 - int(x0 * s), 0):][:, :bw]
    if hx.shape[1] < bw:
        hx = np.pad(hx, ((0, 0), (0, bw - hx.shape[1])), mode="edge")
    N3 = normals(hx, size=bw / (height * s), height=1.0 / max(depth, 0.15) * 0.5)
    alt = (y - (np.arange(Y0, Y1, dtype=F32)[:, None] + 0.5) / s) / height    # 0 at the foot
    # front view, painter's order: far rows sit higher on screen (camera slightly above)
    top = np.where(hx > 0.01, hx + (1 - zf) * depth, -1.0)        # bare ground behind does not form a wall
    idx = np.full((bh, bw), -1, np.int32)
    for z in range(n):
        idx = np.where(top[z][None, :] > alt, z, idx)
    sil = top.max(0)
    alpha = np.clip((sil[None, :] - alt) * height * s / 1.2 + 0.5, 0, 1) * (idx >= 0)
    zi = np.maximum(idx, 0)
    cols = np.arange(bw)[None, :].repeat(bh, 0)
    Nn = N3[zi, cols]
    hh = hx[zi, cols]
    zz = zi / (n - 1)

    az, el = math.radians(light[0]), math.radians(light[1])
    Lv = np.array([math.cos(az) * math.cos(el), math.sin(el), 0.45 * math.sin(az) * math.cos(el) + 0.3], F32)
    Lv /= np.linalg.norm(Lv)
    sg = max(1.5, height * s * 0.008)                              # smooth normals on screen (no row combing)
    Nn = np.stack([cv2.GaussianBlur(np.ascontiguousarray(Nn[..., i]), (0, 0), sg) for i in range(3)], -1)
    lam = (Nn @ Lv).astype(F32) / (np.linalg.norm(Nn, axis=-1) + 1e-6)
    steps = [sh, md, lt] if bands == 3 else [sh, lt] if bands == 2 else [sh, (sh + md) / 2, md, lt]
    col = np.broadcast_to(steps[0], (bh, bw, 3)).copy()
    vis = lam[alpha > 0.5]
    lo, hi = np.percentile(vis, [30, 72]) if vis.size else (0.35, 0.75)   # every range gets all tones
    for e, c in zip(np.linspace(lo, hi, len(steps) - 1), steps[1:]):
        m = np.clip((lam - e) / 0.012 + 0.5, 0, 1)[..., None]
        col = col * (1 - m) + c * m
    # snow: high ground on gentle slopes; lit and shadowed tones
    line = 1 - snow + _noise(bh, bw, max(height * s * 0.05, 2), rng) * 0.04
    flat = np.clip((Nn[..., 1] - 0.05) / 0.12, 0, 1)
    sm = np.clip((hh - line) / 0.02, 0, 1) * flat
    sm = cv2.GaussianBlur(sm.astype(F32), (0, 0), max(0.6, s * 0.6))
    sm = np.clip((sm - 0.5) / 0.1 + 0.5, 0, 1)[..., None]
    col = col * (1 - sm) + np.where((lam > (lo + hi) / 2)[..., None], sn, sns) * sm
    if haze > 0:   # aerial perspective: far rows and the foot fade into the haze colour
        k = np.clip((1 - zz) * 0.55 + np.clip(1 - alt / 0.35, 0, 1) ** 2 * 0.6, 0, 1) * haze
        col = col * (1 - k[..., None]) + hz * k[..., None]
    a = (alpha * opacity)[..., None]
    dst = self.px[Y0:Y1, X0:X1]
    dst[:] = np.concatenate([col * a, a], -1) + dst * (1 - a)
    self.touch()
    return self


Layer.anime_mountains = _layer_anime_mountains

__all__ = ["CLOUD_PALETTES", "TREE_PALETTES", "MOUNTAIN_PALETTES", "god_rays", "lens_flare"]
