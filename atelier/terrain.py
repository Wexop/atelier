"""Terrain: heightfield generation, hydraulic + thermal erosion, materials, map styles.

h = terrain(384, kind="alpine", seed=1)          # (n, n) float32 in [0, 1], eroded
c.terrain_map(h, style="relief", water=.18)      # top-down shaded map on the current layer
The 3D view (sky, aerial perspective, lake) lives in nature.py: c.landscape(h, ...).
World convention for 3D: x, z in [0, size] metres, altitude = h * height metres, y up.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from .core import F32, Layer, color as _col, to_linear, to_srgb
from .field import fbm, make_lut, _apply_lut


# ------------------------------------------------------------------ sampling helpers
def _mirror(g, n):
    """Mirror-repeat grid coordinate g into [0, n-1] (continuous, no seams)."""
    p = 2 * (n - 1)
    g = np.mod(g, p)
    return np.where(g > n - 1, p - g, g)


def bilinear(img, gx, gy, mirror=True):
    """Bilinear sample of img (n, m[, C]) at float grid coords gx (cols), gy (rows). Exact float math."""
    n, m = img.shape[:2]
    if mirror:
        gx, gy = _mirror(gx, m), _mirror(gy, n)
    else:
        gx, gy = np.clip(gx, 0, m - 1), np.clip(gy, 0, n - 1)
    ix = np.minimum(gx.astype(np.int32), m - 2)
    iy = np.minimum(gy.astype(np.int32), n - 2)
    fx = (gx - ix).astype(F32)
    fy = (gy - iy).astype(F32)
    i = iy * m + ix
    flat = img.reshape(n * m, -1) if img.ndim == 3 else img.ravel()
    a, b, c, d = flat[i], flat[i + 1], flat[i + m], flat[i + m + 1]
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _norm01(h):
    lo, hi = float(h.min()), float(h.max())
    return ((h - lo) / max(hi - lo, 1e-9)).astype(F32)


# ------------------------------------------------------------------ generation
def _bands(n, seed, f0, octaves, lac=2.0):
    """Isotropic band-limited noise octaves (spectral synthesis), each ~N(0, 1), (octaves, n, n)."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    f = np.fft.fftfreq(n) * n
    k = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2) + 1e-6
    out = []
    for i in range(octaves):
        fi = f0 * lac ** i
        if fi > n * 0.45:
            break
        g = np.exp(-np.log2(k / fi) ** 2 / (2 * 0.45 ** 2))
        b = np.real(np.fft.ifft2(F * g))
        out.append((b / (b.std() + 1e-12)).astype(F32))
    return out


def heightfield(n=384, kind="alpine", seed=0, scale=2.0, warp=0.15, octaves=8, gain=0.5):
    """Raw (uneroded) height map (n, n) in [0, 1] from isotropic spectral noise (no lattice artifacts).
    kind: alpine (ridged crests) | hills (soft fbm) | mesa (terraced) | island (falloff to sea) | dunes.
    scale = big features across the map."""
    B = _bands(n, seed, scale, octaves)
    def fbm_(gn=gain, start=0):
        t, a, nn = 0, 1.0, 0
        for b in B[start:]:
            t, nn, a = t + a * b, nn + a, a * gn
        return t / nn
    def ridged_():
        t, a, nn, w = 0, 1.0, 0, 1.0
        for b in B:
            r = (1 - np.abs(np.tanh(b * 0.7))) ** 2 * w
            w = np.clip(r * 1.6, 0, 1)
            t, nn, a = t + a * r, nn + a, a * gain
        return t / nn
    if warp:
        W = _bands(n, seed + 1, scale * 1.5, 2)
        wx = W[0] * warp * n * 0.04
        wy = W[1] * warp * n * 0.04
        gy, gx = np.mgrid[0:n, 0:n].astype(F32)
        mapx, mapy = (gx + wx).astype(F32), (gy + wy).astype(F32)
        B = [cv2.remap(b, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for b in B]
    if kind == "dunes":
        x, y = np.meshgrid(np.linspace(0, 1, n, dtype=F32), np.linspace(0, 1, n, dtype=F32))
        ph = (x * 0.4 + y) * scale * 9 + fbm_() * 2.5
        crest = 1 - np.abs(((ph / (2 * np.pi)) % 1) * 2 - 1) ** 1.6
        return _norm01(crest * 0.4 + fbm_() * 0.6)
    base = fbm_(0.45)
    if kind == "hills":
        return _norm01(base)
    big = B[0] * 0.6 + B[1] * 0.4
    mask = np.clip(0.5 + big * 0.35, 0, 1)                   # where the range rises
    h = ridged_() * (0.3 + 0.7 * mask) + base * 0.08 + big * 0.12
    if kind == "mesa":
        h = _norm01(h)
        t = h * 6
        h = (np.floor(t) + np.clip((t % 1 - 0.75) * 4, 0, 1)) / 6 * 0.8 + h * 0.2
    elif kind == "island":
        x, y = np.meshgrid(np.linspace(-1, 1, n, dtype=F32), np.linspace(-1, 1, n, dtype=F32))
        h = _norm01(h) * np.clip(1.25 - np.sqrt(x * x + y * y) * 1.15, 0, 1) ** 0.7
    return _norm01(h)


# ------------------------------------------------------------------ erosion
def _brush(radius):
    r = int(math.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(ox * ox + oy * oy)
    m = d < radius
    w = (radius - d[m]).astype(F32)
    return oy[m].astype(np.int32), ox[m].astype(np.int32), w / w.sum()


def _splat(px, py, amt, by, bx, bw, n, m):
    cy = np.clip(np.round(py).astype(np.int32)[:, None] + by, 0, n - 1)
    cx = np.clip(np.round(px).astype(np.int32)[:, None] + bx, 0, m - 1)
    return np.bincount((cy * m + cx).ravel(), (amt[:, None] * bw).ravel(), n * m)


def erode(h, droplets=1.0, seed=0, steps=40, radius=2.5, inertia=0.05, capacity=4.0, min_capacity=0.01,
          erosion=0.3, deposition=0.3, evaporation=0.02, gravity=4.0, batch=24000, flow=False):
    """Hydraulic erosion by water droplets (vectorised batches, seeded). h: (n, m) heights (any unit,
    normalised internally). droplets = count per cell (1 ≈ well eroded). Returns new heights
    (same range units), and the flow map (n, m) in [0, 1] if flow=True. Mass is conserved
    except what droplets carry off the map edge."""
    h0 = np.asarray(h, F32)
    lo, hi = float(h0.min()), float(h0.max())
    rng_h = max(hi - lo, 1e-9)
    H = ((h0 - lo) / rng_h).astype(np.float64)
    n, m = H.shape
    flat = H.ravel()
    rng = np.random.default_rng(seed)
    by, bx, bw = _brush(radius)
    total = int(droplets * n * m)
    fl = np.zeros(n * m)
    def sample(px, py):
        ix = np.clip(px.astype(np.int32), 0, m - 2)
        iy = np.clip(py.astype(np.int32), 0, n - 2)
        fx, fy = px - ix, py - iy
        i = iy * m + ix
        a, b, c, d = flat[i], flat[i + 1], flat[i + m], flat[i + m + 1]
        gx = (b - a) * (1 - fy) + (d - c) * fy
        gy = (c - a) * (1 - fx) + (d - b) * fx
        hh = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
        return hh, gx, gy, i, fx, fy
    batch = max(64, min(batch, int(n * m * 0.08)))     # concurrent droplets must stay sparse (stability)
    done = 0
    while done < total:
        N = min(batch, total - done)
        done += N
        px = rng.uniform(0, m - 1.001, N)
        py = rng.uniform(0, n - 1.001, N)
        dx = np.zeros(N); dy = np.zeros(N)
        speed = np.ones(N); water = np.ones(N); sed = np.zeros(N)
        alive = np.ones(N, bool)
        for _ in range(steps):
            hh, gx, gy, i, fx, fy = sample(px, py)
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            ln = np.sqrt(dx * dx + dy * dy)
            alive &= ln > 1e-12
            ln = np.maximum(ln, 1e-12)
            dx /= ln; dy /= ln
            nx, ny = px + dx, py + dy
            inside = (nx >= 0) & (nx < m - 1) & (ny >= 0) & (ny < n - 1)
            off = alive & ~inside            # leaving the map: sediment is lost
            alive &= inside
            hn = sample(np.clip(nx, 0, m - 1.001), np.clip(ny, 0, n - 1.001))[0]
            dh = np.where(alive, hn - hh, 0.0)
            cap = np.maximum(-dh * speed * water * capacity, min_capacity)
            dep = alive & ((sed > cap) | (dh > 0))
            ero = alive & ~dep
            dep_amt = np.where(dep, np.where(dh > 0, np.minimum(dh, sed), (sed - cap) * deposition), 0.0)
            ero_amt = np.where(ero, np.minimum((cap - sed) * erosion, -dh), 0.0)
            ero_amt = np.maximum(ero_amt, 0)
            sed = sed + ero_amt - dep_amt
            sed[off] = 0
            # erode / deposit with a radial brush around the old position (smooth, no speckle)
            amt = ero_amt - dep_amt
            sel = np.nonzero(amt)[0]
            if sel.size:
                flat -= _splat(px[sel], py[sel], amt[sel], by, bx, bw, n, m)
            if flow:
                fl += np.bincount(i[alive], (water * speed)[alive], n * m)
            speed = np.sqrt(np.maximum(speed * speed + np.where(alive, -dh, 0) * gravity, 0)) + 1e-6
            speed = np.where(alive, speed, 0)
            water *= 1 - evaporation
            px = np.where(alive, nx, px); py = np.where(alive, ny, py)
            if not alive.any():
                break
        # droplets that evaporated/stopped drop their remaining load
        flat += _splat(px, py, sed, by, bx, bw, n, m)
    out = (flat.reshape(n, m) * rng_h + lo).astype(F32)
    if flow:
        f = fl.reshape(n, m)
        f = np.log1p(f / max(np.median(f[f > 0]) if (f > 0).any() else 1, 1e-9))
        f = cv2.GaussianBlur(f.astype(F32), (0, 0), 0.7)
        return out, np.clip(f / max(float(np.percentile(f, 99.5)), 1e-9), 0, 1).astype(F32)
    return out


def thermal(h, iters=40, talus=None, rate=0.5):
    """Thermal erosion (scree): material slides downhill where the slope per cell exceeds
    `talus` (height units per cell; default 4 / n of the height range). Exactly mass-conserving."""
    H = np.asarray(h, np.float64).copy()
    n, m = H.shape
    if talus is None:
        talus = 4.0 / n * float(H.max() - H.min())
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for _ in range(iters):
        P = np.pad(H, 1, mode="edge")
        flows = []
        for dy, dx in dirs:
            nb = P[1 + dy:1 + dy + n, 1 + dx:1 + dx + m]
            flows.append(np.maximum(H - nb - talus, 0) * (rate * 0.125))
        for (dy, dx), f in zip(dirs, flows):
            H -= f
            # f moves from (i, j) to (i + dy, j + dx)
            ys, yd = (slice(0, n - dy), slice(dy, n)) if dy >= 0 else (slice(-dy, n), slice(0, n + dy))
            xs, xd = (slice(0, m - dx), slice(dx, m)) if dx >= 0 else (slice(-dx, m), slice(0, m + dx))
            H[yd, xd] += f[ys, xs]
    return H.astype(F32)


def terrain(n=384, kind="alpine", seed=0, erosion=1.0, scale=2.0, flow=False):
    """Generated + eroded height map (n, n) in [0, 1] (thermal, then hydraulic, then light thermal).
    erosion = droplets per cell (0 = raw). flow=True also returns the water-flow map (gullies)."""
    h = heightfield(n, kind, seed, scale)
    fl = None
    if erosion > 0:
        h = thermal(h, 20)
        h, fl = erode(h, erosion, seed=seed + 11, flow=True)
        h = thermal(h, 6, talus=6.0 / n)
    h = _norm01(h)
    if flow:
        return h, (fl if fl is not None else np.zeros_like(h))
    return h


# ------------------------------------------------------------------ analysis
def normals(h, size=1.0, height=1.0):
    """Unit normals (n, m, 3) (x, y up, z) of a height map spanning size x size world units,
    altitude = h * height. Row index = +z, column = +x."""
    n, m = h.shape
    cell = size / (m - 1)
    gz, gx = np.gradient(np.asarray(h, F32) * F32(height), cell)
    N = np.stack([-gx, np.ones_like(gx), -gz], -1)
    return (N / np.linalg.norm(N, axis=-1, keepdims=True)).astype(F32)


def ambient_occlusion(h, size=1.0, height=1.0):
    """Cheap multi-scale height-map AO in [0, 1] (1 = open)."""
    n = h.shape[0]
    cell = size / (n - 1)
    H = np.asarray(h, F32) * F32(height)
    occ = np.zeros_like(H)
    for r in (2, 6, 18):
        b = cv2.GaussianBlur(H, (0, 0), r, borderType=cv2.BORDER_REFLECT)
        occ += np.clip((b - H) / (r * cell), 0, None)
    return np.clip(1 - occ * 0.9, 0.15, 1).astype(F32)


def shadow_map(h, sun_dir, size=1.0, height=1.0, soft=0.03, steps=96):
    """Soft cast shadows (n, m) in [0, 1] (1 = lit) for a directional sun (x, y, z toward sun)."""
    n, m = h.shape
    cell = size / (m - 1)
    sd = np.asarray(sun_dir, np.float64)
    hor = math.hypot(sd[0], sd[2])
    if sd[1] <= 0:
        return np.zeros((n, m), F32)
    if hor < 1e-6:
        return np.ones((n, m), F32)
    tan_e = sd[1] / hor
    ux, uz = sd[0] / hor, sd[2] / hor
    H = (np.asarray(h, F32) * F32(height)).astype(F32)
    gy, gx = np.mgrid[0:n, 0:m].astype(F32)
    res = np.ones((n, m), F32)
    dist = 0.0
    step = 0.7
    for _ in range(steps):
        dist += step
        step *= 1.045
        mx = (gx + F32(ux * dist)).astype(F32)
        my = (gy + F32(uz * dist)).astype(F32)
        hk = cv2.remap(H, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        d_w = dist * cell
        rise = H + F32(d_w * tan_e)
        res = np.minimum(res, (rise - hk) / F32(d_w * soft) + 1)
        if d_w * tan_e > height * 1.05:
            break
    r = np.clip(res, 0, 1)
    return (r * r * (3 - 2 * r)).astype(F32)


# ------------------------------------------------------------------ materials
MATERIALS = {   # sRGB albedo
    "rock": "#7a7470", "rock2": "#5e5550", "scree": "#958c82", "grass": "#5b7a33", "dry": "#9a8f55",
    "forest": "#2c4423", "snow": "#f2f4f8", "sand": "#c9b88e", "mud": "#6b5a45",
}


def materials(h, size=1.0, height=1.0, water=None, snow=0.72, flow=None, seed=0, nrm=None, trees=0.5):
    """Albedo map (n, m, 3) LINEAR by altitude/slope rules: sand near water, grass/forest low and
    gentle, rock on steep slopes (scree where flow deposits), snow above `snow` (fraction of height)."""
    n, m = h.shape
    N = normals(h, size, height) if nrm is None else nrm
    flat = N[..., 1]
    rnd = np.random.default_rng(seed)
    ox, oy = rnd.uniform(-40, 40, 2)
    nz = fbm(6, 5, seed=seed + 5).move(ox, oy).render(n, m)
    nz2 = fbm(18, 3, seed=seed + 6).move(oy, ox).render(n, m)
    L = {k: to_linear(_col(v)[:3]) for k, v in MATERIALS.items()}
    lerp = lambda a, b, t: a * (1 - t[..., None]) + b * t[..., None]
    sm = lambda e0, e1, v: np.clip((v - e0) / (e1 - e0), 0, 1) ** 2 * (3 - 2 * np.clip((v - e0) / (e1 - e0), 0, 1))
    hh = h + (nz - 0.5) * 0.08
    rock = lerp(L["rock"], L["rock2"], nz2)
    if flow is not None:
        rock = lerp(rock, L["scree"], np.clip(flow * 1.4, 0, 1) * 0.6)
    grass = lerp(L["grass"], L["dry"], sm(0.35, 0.75, nz) * 0.7)
    grass = lerp(grass, L["forest"], sm(0.45, 0.62, nz2 * 0.6 + nz * 0.4) * trees * (1 - sm(0.45, 0.6, hh)))
    veg = sm(0.82, 0.9, flat) * (1 - sm(snow - 0.2, snow + 0.02, hh))   # vegetation on gentle, low ground
    col = lerp(rock, grass, veg)
    if water is not None:
        col = lerp(col, L["sand"], (1 - sm(water + 0.004, water + 0.02 + nz * 0.01, h)) * sm(0.7, 0.85, flat))
    sn = sm(snow - 0.04, snow + 0.06, hh + (flat - 0.8) * 0.25) * sm(0.55, 0.75, flat + nz2 * 0.15)
    col = lerp(col, L["snow"], sn)
    return col.astype(F32)


# ------------------------------------------------------------------ top-down map
def _terrain_map(self, h, style="relief", water=None, light=(135, 45), contours=0.05, relief=6.0,
                 snow=0.75, x=0, y=0, w=None, hgt=None, seed=0):
    """Top-down map of a height map on this layer (box x, y, w, hgt in canvas px; default full).
    style: relief (hypsometric tint + hillshade) | topo (paper, contour lines) | shaded (materials + sun).
    water = sea level (fraction) or None."""
    W = int(round((w or self.w) * self.s)); Hh = int(round((hgt or self.h) * self.s))
    hh = cv2.resize(np.asarray(h, F32), (W, Hh), interpolation=cv2.INTER_CUBIC)
    az, el = math.radians(light[0]), math.radians(light[1])
    Ld = np.array([math.cos(az) * math.cos(el), math.sin(el), -math.sin(az) * math.cos(el)], F32)
    N = normals(hh, relief, 1.0)
    lam = np.clip(N @ Ld, 0, 1) / max(Ld[1], 1e-3)
    if style == "topo":
        col = np.ones((Hh, W, 3), F32) * to_srgb(to_linear(_col("#f1ead8")[:3]))
        tint = _apply_lut(hh, make_lut(["#e9e2c8", "#e6d6b0", "#d9c3a0", "#cdb7a4", "#f4f2ee"]))[..., :3]
        col = col * 0.55 + tint * 0.45
        col *= (0.82 + 0.18 * np.clip(lam, 0, 1.3))[..., None]
        k = hh / contours
        fw = np.fwidth if hasattr(np, "fwidth") else None
        gy, gx = np.gradient(k)
        g = np.sqrt(gx * gx + gy * gy) + 1e-6
        d = np.abs(k - np.round(k)) / g
        major = (np.round(k) % 5 == 0)
        line = np.clip(1 - (d - np.where(major, 0.9, 0.4)), 0, 1)
        col = col * (1 - line[..., None] * 0.75) + np.array([0.45, 0.3, 0.18], F32) * line[..., None] * 0.75
    elif style == "shaded":
        alb = to_srgb(materials(hh, relief, 1.0, water, snow, seed=seed, nrm=N))
        col = alb * (0.35 + 0.65 * np.clip(lam, 0, 1.4))[..., None]
    else:
        stops = ["#5f8a4a", "#8fae62", "#d9cf8a", "#c79a62", "#9a7458", "#e8e6e2"]
        col = _apply_lut(hh, make_lut(stops))[..., :3]
        col = col * (0.5 + 0.5 * np.clip(lam, 0, 1.4))[..., None]
    if water is not None:
        dep = np.clip((water - hh) / max(water, 1e-3), 0, 1)
        wc = _apply_lut(dep, make_lut(["#9cc6d8", "#5c93b5", "#2f5f8a"]))[..., :3]
        if style == "topo":
            wc = wc * 0.5 + 0.5 * np.array([0.82, 0.88, 0.9], F32)
        wm = np.clip((water - hh) * 400, 0, 1)[..., None]
        shore = np.clip(1 - np.abs(hh - water) * 300, 0, 1)[..., None] * 0.5
        col = col * (1 - wm) + wc * wm
        col = col * (1 - shore) + np.array([0.15, 0.25, 0.35], F32) * shore
    x0, y0 = int(round(x * self.s)), int(round(y * self.s))
    rgba = np.concatenate([np.clip(col, 0, 1), np.ones((Hh, W, 1), F32)], -1)
    x1, y1 = min(self.W, x0 + W), min(self.H, y0 + Hh)
    self.px[y0:y1, x0:x1] = rgba[:y1 - y0, :x1 - x0]
    return self.touch()


Layer.terrain_map = _terrain_map

__all__ = ["heightfield", "erode", "thermal", "terrain", "normals", "ambient_occlusion", "shadow_map",
           "materials", "bilinear", "MATERIALS"]
