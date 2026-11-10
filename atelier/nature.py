"""Natural things: physical sky + clouds, terrain views with aerial perspective, water, plants.

c.sky(sun_elev=8, clouds=.4)                         # physical sky background (Rayleigh + Mie + ozone)
c.sea(sun_elev=5)                                    # open sea: FFT-like waves, Fresnel sky reflection, foam
v = c.landscape(terrain(384, seed=2), water=.18)     # eroded range, lake, clouds, aerial perspective
c.tree(x, y, 300, kind="oak", seed=1)                # 2D trees (space colonisation), grass, flowers, ferns
trunk, leaves = tree3d("oak", 4, seed=1)             # SDFs for Scene
Sun position: elevation and azimuth in degrees; azimuth is in the view frame (0 = straight ahead,
+ = to the right) for c.sky/c.sea, and a world compass angle (0 = +z) for c.landscape(yaw=...).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import cv2
import skia

from .core import F32, Layer, CONFIG, color as _col, to_linear, to_srgb
from .fx import tonemap_linear
from .terrain import (heightfield, erode, thermal, terrain, normals, ambient_occlusion, shadow_map, materials,
                      bilinear, MATERIALS)

# ------------------------------------------------------------------ atmosphere (single scattering + cheap multiple)
_RE, _RA = 6360e3, 6460e3
_BR = np.array([5.802e-6, 13.558e-6, 33.1e-6])          # Rayleigh scattering
_BO = np.array([0.650e-6, 1.881e-6, 0.085e-6])          # ozone absorption
_BM = 4.0e-6                                            # Mie (aerosol) scattering at haze = 1
_HR, _HM = 8000.0, 1200.0
SUN_POWER = 22.0                                        # sun irradiance scale (linear units)


def _workers():
    return CONFIG.get("threads") or min(os.cpu_count() or 2, 4)


def sun_direction(elev=10.0, az=0.0):
    """Unit vector toward the sun (y up; az 0 = +z, 90 = +x)."""
    e, a = math.radians(elev), math.radians(az)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def _ext(h, haze):
    hc = np.maximum(h, 0)[..., None]
    r, m = np.exp(-hc / _HR), np.exp(-hc / _HM)
    o = np.clip(1 - np.abs(hc - 25000) / 15000, 0, 1)
    return _BR * r + _BM * haze * 1.11 * m + _BO * o, r, m


def _sun_T(P, s, haze, K=12):
    """Transmittance (M, 3) from points P (M, 3) (planet-centred) to the sun along s."""
    b = P @ s
    PP = (P * P).sum(-1)
    t = -b + np.sqrt(np.maximum(b * b - (PP - _RA ** 2), 0))
    u = (np.arange(K) + 0.5) / K
    tt = t[:, None] * u * u
    ds = t[:, None] * (2 * u / K)
    pts = P[:, None, :] + tt[..., None] * s
    h = np.sqrt((pts * pts).sum(-1)) - _RE
    od = (_ext(h, haze)[0] * ds[..., None]).sum(1)
    # planet shadow, softened over ~0.5 deg so dusk does not cut abruptly
    r = np.sqrt(PP)
    mu = b / r
    mu_h = -np.sqrt(np.maximum(1 - (_RE / r) ** 2, 0))
    vis = np.clip((mu - mu_h) / 0.01 + 0.5, 0, 1)
    return np.exp(-od) * vis[:, None]


@lru_cache(maxsize=8)
def _sky_lut(sun_elev, haze, alt, ne=96, na=48, K=40):
    """Sky radiance LUT (ne, na, 3) over view elevation (sqrt-warped) x azimuth from sun, and
    the total view transmittance (ne, na, 3). Linear, scaled by SUN_POWER."""
    v = (np.arange(ne) + 0.5) / ne
    w = 2 * v - 1
    e = np.sign(w) * w * w * (math.pi / 2)
    ph = np.linspace(0, math.pi, na)
    E, PH = np.meshgrid(e, ph, indexing="ij")
    d = np.stack([np.cos(E) * np.sin(PH), np.sin(E), np.cos(E) * np.cos(PH)], -1).reshape(-1, 3)
    s = sun_direction(sun_elev, 0)
    O = np.array([0, _RE + max(alt, 2.0), 0.0])                 # never exactly on the ground
    b = d @ O
    OO = O @ O
    tmax = -b + np.sqrt(np.maximum(b * b - (OO - _RA ** 2), 0))
    disc = b * b - (OO - _RE ** 2)
    tg = -b - np.sqrt(np.maximum(disc, 0))
    hitg = (disc > 0) & (tg > 0)
    tmax = np.where(hitg, tg, tmax)
    u = (np.arange(K) + 0.5) / K
    t = tmax[:, None] * u * u
    ds = tmax[:, None] * (2 * u / K)
    P = O + t[..., None] * d[:, None, :]
    h = np.sqrt((P * P).sum(-1)) - _RE
    ext, r, m = _ext(h, haze)
    od = np.cumsum(ext * ds[..., None], 1) - 0.5 * ext * ds[..., None]
    Tv = np.exp(-od)
    Ts = _sun_T(P.reshape(-1, 3), s, haze).reshape(P.shape)
    mu = (d @ s)[:, None, None]
    PR = 3 / (16 * math.pi) * (1 + mu * mu)
    g = 0.78
    PM = 3 / (8 * math.pi) * (1 - g * g) * (1 + mu * mu) / ((2 + g * g) * (1 + g * g - 2 * g * mu) ** 1.5)
    sr, sm = _BR * r, _BM * haze * m
    single = (Tv * Ts * (sr * PR + sm * PM) * ds[..., None]).sum(1)
    # cheap multiple scattering: isotropic re-scattering of the light already in the air
    ms = (Tv * (sr + sm) * ds[..., None]).sum(1) / (4 * math.pi) * Ts.mean(1) * 0.3 \
        + (Tv * (sr + sm) * ds[..., None]).sum(1) * max(math.sin(math.radians(sun_elev)) + 0.08, 0.0) * 0.006
    L = single + ms
    Tend = np.exp(-(ext * ds[..., None]).sum(1))
    # ground seen below the horizon: dull albedo lit by the sun + sky
    Pg = P[:, -1]
    ng = Pg / np.linalg.norm(Pg, axis=-1, keepdims=True)
    gl = 0.08 * Ts[:, -1] * np.clip(ng @ s, 0, 1)[:, None] / math.pi + 0.08 * 0.15 * Ts[:, -1]
    L = L + np.where(hitg[:, None], Tend * gl, 0)
    return (L * SUN_POWER).reshape(ne, na, 3).astype(F32), Tend.reshape(ne, na, 3).astype(F32)


def _lut_coords(D, s, ne, na):
    e = np.arcsin(np.clip(D[:, 1], -1, 1))
    w = np.sign(e) * np.sqrt(np.abs(e) / (math.pi / 2))
    gy = (0.5 + 0.5 * w) * ne - 0.5
    hx, hz = D[:, 0], D[:, 2]
    sh = math.hypot(s[0], s[2])
    if sh < 1e-6:
        ph = np.zeros(len(D))
    else:
        cs = (hx * s[0] + hz * s[2]) / (np.sqrt(hx * hx + hz * hz) + 1e-9) / sh
        ph = np.arccos(np.clip(cs, -1, 1))
    gx = ph / math.pi * (na - 1)
    return gx.astype(F32), gy.astype(F32)


def sky_radiance(D, sun_elev=10.0, sun_az=0.0, haze=1.0, alt=0.0, sun_size=0.6, with_T=False):
    """Linear sky radiance (N, 3) for unit directions D (N, 3) (y up). Includes the sun disc."""
    D = np.asarray(D, F32).reshape(-1, 3)
    L, T = _sky_lut(round(float(sun_elev), 3), round(float(haze), 3), round(float(alt), -1))
    s = sun_direction(sun_elev, sun_az)
    gx, gy = _lut_coords(D, s, *L.shape[:2])
    out = bilinear(L, gx, gy, mirror=False)
    if sun_size:
        cs = D @ s.astype(F32)
        r = np.sqrt(np.clip(1 - cs * cs, 0, 1)) / math.sin(math.radians(sun_size * 0.5))
        disk = np.clip((1 - r) * 6, 0, 1) * (1 - 0.55 * (1 - np.sqrt(np.clip(1 - r * r, 0, 1))))
        if disk.max() > 0:
            Ts = sun_color(sun_elev, haze, alt)
            out = out + disk[:, None] * (Ts * F32(SUN_POWER * 9000.0 / max(sun_size, 0.1) ** 2))
    if with_T:
        return out.astype(F32), bilinear(T, gx, gy, mirror=False).astype(F32)
    return out.astype(F32)


@lru_cache(maxsize=32)
def _sun_color_c(sun_elev, haze, alt):
    s = sun_direction(sun_elev, 0)
    return _sun_T(np.array([[0, _RE + alt, 0.0]]), s, haze, K=48)[0].astype(F32)


def sun_color(sun_elev=10.0, haze=1.0, alt=0.0):
    """Sun transmittance (rgb, linear 0-1) seen from altitude `alt` metres: white at noon, orange at dusk."""
    return _sun_color_c(round(float(sun_elev), 3), round(float(haze), 3), round(float(alt), -1))


@lru_cache(maxsize=8)
def _irr_fit(sun_elev, haze, alt):
    """Sky irradiance E(n) ~ a + b . n (linear rgb), least-squares fit of the cosine integral."""
    k = 600
    i = np.arange(k) + 0.5
    phi = math.pi * (3 - math.sqrt(5)) * i
    y = 1 - 2 * i / k
    rr = np.sqrt(1 - y * y)
    D = np.stack([rr * np.cos(phi), y, rr * np.sin(phi)], -1)
    L = sky_radiance(D, sun_elev, 0, haze, alt, sun_size=0).astype(np.float64)
    Ns = D[::12]
    E = (np.clip(Ns @ D.T, 0, None) @ L) * (4 * math.pi / k)
    A = np.concatenate([np.ones((len(Ns), 1)), Ns], 1)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return coef.astype(F32)                                  # (4, 3)


def sky_irradiance(n, sun_elev=10.0, sun_az=0.0, haze=1.0, alt=0.0):
    """Diffuse sky irradiance (N, 3) on surfaces with unit normals n (N, 3)."""
    coef = _irr_fit(round(float(sun_elev), 3), round(float(haze), 3), round(float(alt), -1))
    a = math.radians(sun_az)
    ca, sa = math.cos(a), math.sin(a)
    n = np.asarray(n, F32)
    # rotate normals into the sun-at-az-0 frame of the fit
    nx = n[:, 0] * ca - n[:, 2] * sa
    nz = n[:, 0] * sa + n[:, 2] * ca
    E = coef[0] + nx[:, None] * coef[1] + n[:, 1:2] * coef[2] + nz[:, None] * coef[3]
    return np.maximum(E, coef[0] * 0.05).astype(F32)


def _air_T(dist, y0, y1, haze):
    """Transmittance (N, 3) along straight paths of length dist from altitude y0 to y1 (metres)."""
    def I(H):
        dy = y1 - y0
        e0, e1 = np.exp(-np.maximum(y0, 0) / H), np.exp(-np.maximum(y1, 0) / H)
        return np.where(np.abs(dy) < 1, dist * e0, dist * H * (e0 - e1) / np.where(np.abs(dy) < 1, 1, dy))
    od = _BR * I(_HR)[:, None] + (_BM * haze * 1.11) * I(_HM)[:, None]
    return np.exp(-od).astype(F32)


def aerial(L, D, dist, y0, y1, sun_elev=10.0, sun_az=0.0, haze=1.0, alt=0.0):
    """Aerial perspective: attenuate linear radiance L (N, 3) seen along D over dist metres and add
    the in-scattered sky light (matches the sky colour at the horizon for distant objects)."""
    D = np.asarray(D, F32).copy()
    D[:, 1] = np.maximum(D[:, 1], 0.012)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Ls, Tinf = sky_radiance(D, sun_elev, sun_az, haze, alt, sun_size=0, with_T=True)
    T = _air_T(dist, y0, y1, haze)
    a = np.clip((1 - T) / np.maximum(1 - Tinf, 1e-3), 0, 1)
    return (L * T + Ls * a).astype(F32)


# ------------------------------------------------------------------ periodic spectral noise
@lru_cache(maxsize=16)
def _spectral(shape, beta=1.0, seed=0, kmin=1.0):
    """Tileable noise of any dimension (spectral synthesis, 1/k^beta), ~N(0, 1)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)
    F = np.fft.fftn(w)
    k2 = 0
    for i, n in enumerate(shape):
        f = np.fft.fftfreq(n) * n
        k2 = k2 + (f.reshape([-1 if j == i else 1 for j in range(len(shape))])) ** 2
    k = np.sqrt(k2)
    amp = np.where(k >= kmin, np.maximum(k, 1e-9) ** -beta, 0)
    out = np.real(np.fft.ifftn(F * amp))
    out = (out - out.mean()) / (out.std() + 1e-12)
    return out.astype(F32)


def _wrap_bilinear(img, gx, gy, smooth=False):
    n, m = img.shape[:2]
    gx = np.mod(gx, m); gy = np.mod(gy, n)
    ix = gx.astype(np.int32); iy = gy.astype(np.int32)
    fx = (gx - ix).astype(F32); fy = (gy - iy).astype(F32)
    ix %= m; iy %= n
    ix1 = (ix + 1) % m; iy1 = (iy + 1) % n
    if smooth:                                       # C1 (smoothstep) weights: no straight creases
        fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    a, b, c, d = img[iy, ix], img[iy, ix1], img[iy1, ix], img[iy1, ix1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ------------------------------------------------------------------ clouds (raymarched slab)
def _cloud_maps(seed):
    cov = _spectral((512, 512), 1.7, seed + 101, 3.0)
    vol = _spectral((24, 64, 64), 1.15, seed + 102, 1.0)
    return cov, vol


def _cloud_density(P, cov, vol, alt, cover, fine=1.0, period=34000.0):
    """Cloud density (N,) at world points P (N, 3)."""
    from scipy.ndimage import map_coordinates
    f = np.clip((P[:, 1] - alt[0]) / (alt[1] - alt[0]), 0, 1)
    c2 = _wrap_bilinear(cov, P[:, 0] / period * 512, P[:, 2] / period * 512, True)
    c2 = 0.5 + 0.21 * c2
    raw = np.maximum(c2 - (1 - cover), 0) / 0.2
    base = 1 - np.exp(-raw * 1.6)                         # gentle rise + soft saturation: domes, no lids
    d0 = base * 0.92 - f ** 1.2 - (1 - np.clip(f / 0.06, 0, 1)) * 0.6
    out = np.zeros(len(P), F32)
    m = (d0 > -0.35) & (f >= 0) & (f <= 1)
    if m.any():
        Q = P[m]
        crd = np.stack([Q[:, 1] / 1400.0 * 24, Q[:, 2] / 2600.0 * 64, Q[:, 0] / 2600.0 * 64])
        n1 = map_coordinates(vol, crd, order=1, mode="grid-wrap")
        n2 = map_coordinates(vol, crd * np.array([[3.1], [3.3], [3.3]]) + 7.3, order=1, mode="grid-wrap")
        fn = fine[m] if np.ndim(fine) else fine
        det = 0.5 + 0.17 * (n1 * 0.7 + n2 * 0.45 * fn)
        dm = d0[m]
        dd = dm + n1 * (0.1 + 0.12 * f[m]) * np.clip((dm + 0.12) / 0.2, 0, 1)   # bulges (no stray islands)
        out[m] = np.clip((dd - np.clip(det - 0.3, 0, 1) * 0.6 * (1.1 - np.clip(dd, 0, 1))) * 3, 0, 1) * (dd > 0)
    return out


def _clouds(D, cam, s, sun_rgb, amb_top, amb_bot, cover, alt, seed, steps=56, sigma=0.035):
    """March cloud slab along D from cam. Returns (radiance (N, 3), transmittance (N,), dist (N,))."""
    N = len(D)
    L = np.zeros((N, 3), F32); T = np.ones(N, F32); dist = np.full(N, 1e9, F32)
    dy = D[:, 1]
    up = dy > 0.006
    if cam[1] > alt[0] or not up.any() or cover <= 0:
        return L, T, dist
    cov, vol = _cloud_maps(seed)
    idx = np.nonzero(up)[0]
    t0 = (alt[0] - cam[1]) / dy[idx]
    t1 = (alt[1] - cam[1]) / dy[idx]
    t1 = np.minimum(t1, t0 + 14000)
    keep = t0 < 70000
    idx, t0, t1 = idx[keep], t0[keep], t1[keep]
    dl = (t1 - t0) / steps
    rng = np.random.default_rng(seed + 5)
    jit = rng.random(len(idx)).astype(F32)
    mu = (D[idx] @ s).astype(F32)
    hg = lambda g: (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * mu) ** 1.5)
    phase = (0.6 * hg(0.6) + 0.4 * hg(-0.2)) * 4 * math.pi * 0.6 + 0.2
    Ti = np.ones(len(idx), F32)
    Li = np.zeros((len(idx), 3), F32)
    dw = np.zeros(len(idx), F32); ww = np.zeros(len(idx), F32)
    lsteps = np.array([60, 180, 420, 900], F32)
    sF = s.astype(F32)
    alive = np.arange(len(idx))
    for k in range(steps):
        if alive.size == 0:
            break
        t = t0[alive] + (k + jit[alive]) * dl[alive]
        P = cam + D[idx[alive]] * t[:, None]
        fine = np.clip(1.6 - dl[alive] / 80.0, 0, 1)
        den = _cloud_density(P, cov, vol, alt, cover, fine)
        hit = den > 0
        if hit.any():
            a = alive[hit]
            Ph = P[hit]
            odl = np.zeros(len(a), F32)
            prev = 0.0
            for ls in lsteps:
                odl += _cloud_density(Ph + sF * ls, cov, vol, alt, cover, 0.0) * F32(ls - prev) * sigma
                prev = ls
            Tl = np.exp(-odl) + np.exp(-odl * 0.12) * 0.45      # direct + cheap multiple scattering
            f = np.clip((Ph[:, 1] - alt[0]) / (alt[1] - alt[0]), 0, 1)[:, None]
            sg = den[hit] * sigma
            powder = 1 - np.exp(-sg * 250)
            S = sun_rgb * (phase[a] * Tl * (0.55 + 0.45 * powder))[:, None] + (amb_bot * (1 - f) + amb_top * f) * 1.25
            Ts = np.exp(-sg * dl[a])
            contrib = Ti[a] * (1 - Ts)
            Li[a] += S * contrib[:, None]
            dw[a] += contrib * t[hit]; ww[a] += contrib
            Ti[a] *= Ts
        alive = alive[Ti[alive] > 0.02]
    L[idx], T[idx] = Li, Ti
    dist[idx] = np.where(ww > 1e-4, dw / np.maximum(ww, 1e-6), 1e9)
    return L, T, dist


def _cloud_shadow(P, s, cover, alt, seed):
    """Fraction of sunlight passing the cloud layer at points P (N, 3)."""
    if cover <= 0 or s[1] <= 0.02:
        return np.ones(len(P), F32)
    cov, _ = _cloud_maps(seed)
    mid = (alt[0] + alt[1]) * 0.5
    t = (mid - P[:, 1]) / s[1]
    Q = P + t[:, None] * s
    c2 = 0.5 + 0.21 * _wrap_bilinear(cov, Q[:, 0] / 34000.0 * 512, Q[:, 2] / 34000.0 * 512, True)
    base = np.clip((c2 - (1 - cover)) / max(cover, 0.05) * 1.3, 0, 1)
    k = float(np.clip((s[1] - 0.12) / 0.2, 0, 1))          # thin-layer approximation fails at grazing sun
    return np.clip(1 - base * 1.4 * k, 0.3, 1).astype(F32)


# ------------------------------------------------------------------ water waves
@lru_cache(maxsize=8)
def _ocean_tile(n=256, wind_dir=0.0, swell=0.6, seed=0, chop=1.0):
    """Periodic wave tile from a Phillips-like spectrum (unit period). Returns a mip pyramid of
    (sx, sz, sx^2, sz^2, foam) arrays; slopes normalised to rms 1."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n) * n
    kx, kz = np.meshgrid(f, f)
    k = np.sqrt(kx * kx + kz * kz) + 1e-9
    a = math.radians(wind_dir)
    wd = np.array([math.sin(a), math.cos(a)])
    cosw = (kx * wd[0] + kz * wd[1]) / k
    kp = 6.0                                          # spectral peak (waves per tile)
    P = np.exp(-(kp / k) ** 2 * 1.25) / k ** 4 * np.abs(cosw) ** (2 + 6 * swell) * np.exp(-(k / (n * 0.35)) ** 2)
    P[0, 0] = 0
    h0 = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * np.sqrt(P)
    sx = np.real(np.fft.ifft2(1j * kx * h0))
    sz = np.real(np.fft.ifft2(1j * kz * h0))
    dxx = np.real(np.fft.ifft2(-kx * kx / k * h0))
    dzz = np.real(np.fft.ifft2(-kz * kz / k * h0))
    dxz = np.real(np.fft.ifft2(-kx * kz / k * h0))
    rms = math.sqrt(float((sx * sx + sz * sz).mean()) + 1e-12)
    sx, sz = sx / rms, sz / rms
    lam = chop / max(float(np.abs(dxx + dzz).std()), 1e-12) * 0.6
    J = (1 + lam * dxx) * (1 + lam * dzz) - (lam * dxz) ** 2
    foam = np.clip((0.35 - J) * 1.6, 0, 1)
    lvl = [np.stack([sx, sz, sx * sx, sz * sz, foam], -1).astype(F32)]
    while lvl[-1].shape[0] > 4:
        a_ = lvl[-1]
        lvl.append(((a_[0::2, 0::2] + a_[1::2, 0::2] + a_[0::2, 1::2] + a_[1::2, 1::2]) * 0.25).astype(F32))
    return lvl


def _wave_sample(lvl, x, z, period, foot):
    """Sample the pyramid at world x, z with footprint foot (m). Returns (N, 5)."""
    n = lvl[0].shape[0]
    cell = period / n
    lv = np.clip(np.log2(np.maximum(foot / cell, 1e-6)), 0, len(lvl) - 1.001)
    out = np.zeros((len(x), 5), F32)
    l0 = np.floor(lv).astype(np.int32)
    fr = (lv - l0).astype(F32)
    for L in np.unique(l0):
        m = l0 == L
        a = _wrap_bilinear(lvl[L], x[m] / period * lvl[L].shape[0] - 0.5, z[m] / period * lvl[L].shape[0] - 0.5)
        b = _wrap_bilinear(lvl[L + 1], x[m] / period * lvl[L + 1].shape[0] - 0.5,
                           z[m] / period * lvl[L + 1].shape[0] - 0.5)
        out[m] = a * (1 - fr[m, None]) + b * fr[m, None]
    return out


def gerstner(x, z, t=0.0, waves=None, seed=0, n=8, amp=0.4, length=12.0, wind=0.0, steep=0.6):
    """Sum of Gerstner waves at world x, z (arrays). Returns (dx, y, dz) displacements; waves =
    list of (amplitude, wavelength, direction deg, phase) or generated from seed/n/amp/length/wind."""
    x = np.asarray(x, F32); z = np.asarray(z, F32)
    if waves is None:
        rng = np.random.default_rng(seed)
        waves = [(amp * 0.7 ** i, length * 0.72 ** i, wind + rng.normal(0, 35), rng.uniform(0, 2 * math.pi))
                 for i in range(n)]
    X = np.zeros_like(x); Y = np.zeros_like(x); Z = np.zeros_like(x)
    for A, wl, ang, ph in waves:
        k = 2 * math.pi / wl
        dxw, dzw = math.sin(math.radians(ang)), math.cos(math.radians(ang))
        w = math.sqrt(9.81 * k)
        th = k * (dxw * x + dzw * z) - w * t + ph
        Q = steep / (k * A * len(waves))
        X += Q * A * dxw * np.cos(th); Z += Q * A * dzw * np.cos(th); Y += A * np.sin(th)
    return X, Y, Z


def wave_field(n=256, seed=0, wind=0.0, swell=0.6):
    """Tileable wave height map (n, n) ~N(0, 1) (FFT synthesis of a wind-sea spectrum), e.g. for
    shade() or displace()."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n) * n
    kx, kz = np.meshgrid(f, f)
    k = np.sqrt(kx * kx + kz * kz) + 1e-9
    a = math.radians(wind)
    cosw = (kx * math.sin(a) + kz * math.cos(a)) / k
    P = np.exp(-(6 / k) ** 2 * 1.25) / k ** 4 * np.abs(cosw) ** (2 + 6 * swell)
    P[0, 0] = 0
    h = np.real(np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * np.sqrt(P)))
    return ((h - h.mean()) / (h.std() + 1e-12)).astype(F32)


def _remap_pts(img, px, py):
    """cv2 bilinear lookup of img at scattered pixel coords (any count), edge clamped."""
    N = len(px)
    k = 4096
    m = -(-N // k) * k
    mx = np.zeros(m, F32); my = np.zeros(m, F32)
    mx[:N], my[:N] = px, py
    r = cv2.remap(img, mx.reshape(-1, k), my.reshape(-1, k), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return r.reshape(m, -1)[:N]


# ------------------------------------------------------------------ height-map raster ("voxel space", exact for level cameras)
def _raster(Hm, size, cam, fwd, right, f, cx, cy, W, rows, far, step_k=1.0):
    """Forward depth z (len(rows), W) of the first height-map hit for each pixel (inf = none).
    Columns are vertical planes (camera has no pitch; horizon set by lens shift)."""
    n = Hm.shape[0]
    cell = size / (n - 1)
    zs = [2.0]
    while zs[-1] < far:
        zs.append(zs[-1] + max(zs[-1] * step_k / f, cell * 0.45))
    zs = np.asarray(zs, F32)
    S = len(zs)
    rows = np.asarray(rows, F32) + F32(0.5)
    out = np.full((len(rows), W), np.inf, F32)
    top = np.full(W, np.inf, F32)
    def job(c0):
        c1 = min(W, c0 + 160)
        u = (np.arange(c0, c1, dtype=F32) + 0.5 - cx) / f
        dx = fwd[0] + u * right[0]; dz = fwd[2] + u * right[2]
        X = cam[0] + zs[:, None] * dx[None]
        Z = cam[2] + zs[:, None] * dz[None]
        hs = bilinear(Hm, X / cell, Z / cell)
        ys = cy - (hs - cam[1]) / zs[:, None] * f
        ys = np.clip(ys, -4.0 * len(rows) - 10, 1e7)
        ymin = np.minimum.accumulate(ys, 0)
        cw = c1 - c0
        lo, hi = float(rows.min()) - 2, float(rows.max()) + 2
        a = -np.clip(ymin, lo, hi).T                                  # (cw, S) non-decreasing
        off = (hi - lo + 10)
        A = (a + (np.arange(cw, dtype=np.float64)[:, None] * off)).ravel()
        q = (-rows[None, :] + np.arange(cw, dtype=np.float64)[:, None] * off).ravel()
        k = np.searchsorted(A, q, side="left").reshape(cw, len(rows)) - (np.arange(cw)[:, None] * S)
        hit = k < S
        kk = np.clip(k, 1, S - 1)
        cols = np.arange(cw)[:, None]
        yprev = ymin[kk - 1, cols]
        ycur = ys[kk, cols]
        rr = rows[None, :]
        t = np.clip((yprev - rr) / np.maximum(yprev - ycur, 1e-6), 0, 1)
        z = zs[kk - 1] + (zs[kk] - zs[kk - 1]) * t
        z = np.where(k == 0, zs[0], z)
        out[:, c0:c1] = np.where(hit, z, np.inf).T
        top[c0:c1] = ymin[-1]
    with ThreadPoolExecutor(_workers()) as ex:
        list(ex.map(job, range(0, W, 160)))
    return out, top


# ------------------------------------------------------------------ the landscape pipeline
def _layer_landscape(self, h=None, size=16000.0, height=2600.0, cam=None, yaw=0.0, fov=50.0, horizon=0.58,
                     sun_elev=10.0, sun_az=30.0, haze=1.0, clouds=0.35, cloud_alt=(1600.0, 3000.0),
                     water=None, waves=0.06, wave_len=None, wind=None, foam=0.5, water_color="#0b2a3a",
                     snow=0.72, flow=None, albedo=None, far=60000.0, exposure=0.0, look="punchy", sun_size=0.6,
                     aa=1, seed=0, tonemap="agx", trees=0.5):
    """Render a physically lit landscape on this layer (replaces its pixels).
    h: height map [0,1] (None = no terrain), spans size x size metres, altitude = h*height.
    cam=(x, altitude, z) metres (default: south edge, above the ground); yaw = view compass angle (deg,
    0 = +z); horizon = horizon height as a fraction of the canvas (lens shift, no tilt).
    sun_az is a compass angle relative to yaw (0 = in front of the camera, 180 = behind).
    water = lake/sea level as a fraction of height (None = no water; with h=None: sea at 0).
    waves = rms wave slope (0.02 calm lake .. 0.2 windy sea); clouds = coverage 0..1.
    Returns a namespace: depth (H, W metres, inf = sky), water (mask), project(x, y, z) -> (px, py, px_per_m)."""
    if CONFIG.get("preview"):
        aa = 1
    W, H = int(self.W * aa), int(self.H * aa)
    vf = math.radians(fov) / 2
    f = (H / 2) / math.tan(vf)
    cx, cy = W / 2, H * horizon
    yawr = math.radians(yaw)
    fwd = np.array([math.sin(yawr), 0, math.cos(yawr)], F32)
    right = np.array([math.cos(yawr), 0, -math.sin(yawr)], F32)
    s = sun_direction(sun_elev, yaw + sun_az).astype(F32)
    saz = yaw + sun_az
    if h is not None:
        h = np.asarray(h, F32)
        n = h.shape[0]
        cell = size / (n - 1)
    if cam is None:
        if h is None:
            cam = (0.0, 3.0, 0.0)
        else:
            x0, z0 = size * 0.5, -size * 0.05
            gx, gz = np.meshgrid(np.linspace(-1500, 1500, 7) + x0, np.linspace(-1500, 1500, 7) + z0)
            g = float(bilinear(h, gx.ravel() / cell, gz.ravel() / cell).max()) * height
            cam = (x0, max(g, (water or 0) * height) + 120, z0)
    cam = np.asarray(cam, F32)
    wl = None if water is None else float(water) * (height if h is not None else 1.0)
    if h is None and water is not None:
        wl = 0.0 if water == 0 or water is True else float(water)
    ctx = dict(sun_elev=sun_elev, sun_az=saz, haze=haze, alt=0.0)
    sun_rgb = sun_color(sun_elev, haze, 0.0) * F32(SUN_POWER)
    sun_cloud = sun_color(sun_elev, haze, (cloud_alt[0] + cloud_alt[1]) / 2) * F32(SUN_POWER)

    def dirs(rows, cols=None):
        rr = np.asarray(rows, F32)[:, None] + 0.5
        cc = (np.arange(W, dtype=F32) + 0.5)[None, :]
        u = (cc - cx) / f
        v = (cy - rr) / f
        D = fwd[None, None] + u[..., None] * right[None, None] + v[..., None] * np.array([0, 1, 0], F32)
        D = D.reshape(-1, 3)
        return (D / np.linalg.norm(D, axis=1, keepdims=True)).astype(F32)

    # ---- sky buffer (rows r0 .. H) incl. clouds; also serves reflections (mirrored rows)
    r0 = int(min(0, math.floor(2 * cy - H) - 2))
    sky_rows = np.arange(r0, H)
    Dsky = dirs(sky_rows)
    if h is None and wl is None:                 # plain sky: below the horizon continue the horizon colour
        Dsky[:, 1] = np.maximum(Dsky[:, 1], 0)
        Dsky /= np.linalg.norm(Dsky, axis=1, keepdims=True)
    sky = sky_radiance(Dsky, sun_elev, saz, haze, 0.0, sun_size).reshape(len(sky_rows), W, 3)
    if clouds and clouds > 0:
        q = 2 if not CONFIG.get("preview") else 1
        cr = sky_rows[(sky_rows <= cy + 2)]
        sub_r = cr[::q]
        Wc = (W + q - 1) // q
        uu = ((np.arange(Wc, dtype=F32) * q + 0.5 * q) - cx) / f
        vv = (cy - (sub_r.astype(F32) + 0.5 * q)) / f
        Dc = fwd + uu[None, :, None] * right + vv[:, None, None] * np.array([0, 1, 0], F32)
        Dc = (Dc / np.linalg.norm(Dc, axis=-1, keepdims=True)).reshape(-1, 3).astype(F32)
        amb_top = sky_irradiance(np.array([[0, 1, 0]], F32), **ctx)[0] / math.pi
        amb_bot = sky_irradiance(np.array([[0, -1, 0]], F32), **ctx)[0] / math.pi * 0.6 + amb_top * 0.25
        parts = np.array_split(np.arange(len(Dc)), _workers() * 3)
        Lc = np.zeros((len(Dc), 3), F32); Tc = np.ones(len(Dc), F32); dc = np.zeros(len(Dc), F32)
        def cjob(ix):
            if ix.size:
                Lc[ix], Tc[ix], dc[ix] = _clouds(Dc[ix], cam, s, sun_cloud, amb_top, amb_bot, clouds, cloud_alt, seed)
        with ThreadPoolExecutor(_workers()) as ex:
            list(ex.map(cjob, parts))
        Ls_c, Tinf = sky_radiance(Dc, sun_elev, saz, haze, 0.0, sun_size=0, with_T=True)
        Ta = _air_T(np.minimum(dc, 2e5), np.full(len(dc), cam[1]), np.full(len(dc), np.mean(cloud_alt)), haze)
        a = np.clip((1 - Ta) / np.maximum(1 - Tinf, 1e-3), 0, 1)
        cl = Lc * (1 - a) + Ls_c * a * (1 - Tc[:, None])
        img = np.concatenate([cl, Tc[:, None]], 1).reshape(len(sub_r), Wc, 4)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        img = cv2.resize(img, (Wc * q, len(sub_r) * q), interpolation=cv2.INTER_LINEAR)[:len(cr), :W]
        k = len(cr)
        sky[:k] = sky[:k] * img[..., 3:4] + img[..., :3]
    out = sky[-r0:].copy() if r0 < 0 else sky.copy()          # main rows 0..H
    depth = np.full((H, W), np.inf, F32)
    wmask = np.zeros((H, W), F32)

    def shade_terrain(P, D, dist, cam_y):
        nrm = bilinear(Nmap, P[:, 0] / cell, P[:, 2] / cell)
        alb = bilinear(Amap, P[:, 0] / cell, P[:, 2] / cell)
        dt = _wrap_bilinear(detail, P[:, 0] / 9.0, P[:, 2] / 9.0) * 0.6 + \
            _wrap_bilinear(detail, P[:, 0] / 41.0 + 0.3, P[:, 2] / 41.0) * 0.4
        fade = np.clip(1 - dist / 2500, 0, 1)
        alb = alb * (1 + 0.22 * dt * (0.35 + 0.65 * fade))[:, None]
        # sub-cell rock relief: bump normals from a ridged detail tile at two scales, fading with distance
        rock = np.clip((1 - nrm[:, 1]) * 2.5, 0.25, 1)[:, None]
        foot = dist / f
        for per, amp in ((1600.0, 0.3), (500.0, 0.25)):
            g = _wrap_bilinear(dgrad, P[:, 0] / per * 256, P[:, 2] / per * 256)
            w = (amp * np.clip(per / 256 / np.maximum(foot, 1e-3) - 0.7, 0, 1))[:, None] * rock   # no aliasing
            nrm = nrm + np.stack([-g[:, 0], np.zeros(len(g), F32), -g[:, 1]], 1) * w
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        sh = bilinear(Smap, P[:, 0] / cell, P[:, 2] / cell) * _cloud_shadow(P, s, clouds, cloud_alt, seed)
        ao = bilinear(AOmap, P[:, 0] / cell, P[:, 2] / cell)
        ndl = np.clip(nrm @ s, 0, 1)
        Ed = sun_rgb * (ndl * sh)[:, None]
        Ea = sky_irradiance(nrm, **ctx) * ao[:, None]
        # warm bounce light from sunlit ground for shaded faces
        Eb = sun_rgb * F32(max(s[1], 0) * 0.12) * np.clip(-nrm[:, 1] * 0.5 + 0.5, 0, 1)[:, None] * alb
        L = alb * (Ed + Ea + Eb) / math.pi
        return aerial(L, D, dist, np.full(len(P), cam_y), P[:, 1], **ctx)

    if h is not None:
        Hm = h * F32(height)
        Nmap = normals(h, size, height)
        Amap = albedo if albedo is not None else materials(h, size, height, water, snow, flow, seed, Nmap, trees)
        Smap = shadow_map(h, s, size, height, soft=0.025)
        AOmap = ambient_occlusion(h, size, height)
        detail = _spectral((128, 128), 0.9, seed + 7, 1.0)
        rd = 1 - np.abs(np.tanh(_spectral((256, 256), 1.25, seed + 8, 2.0)))    # ridged strata tile
        gz_, gx_ = np.gradient(rd * 8.0)
        dgrad = np.stack([gx_, gz_], -1).astype(F32)
        Hr = Hm if wl is None else np.maximum(Hm, F32(wl))
        z, top = _raster(Hr, size, cam, fwd, right, f, cx, cy, W, np.arange(H), far)
        hit = np.isfinite(z)
        if hit.any():
            rr, cc = np.nonzero(hit)
            zz = z[rr, cc]
            u = (cc + 0.5 - cx) / f
            v = (cy - rr - 0.5) / f
            Dh = fwd + u[:, None] * right + v[:, None] * np.array([0, 1, 0], F32)
            dist = zz * np.linalg.norm(Dh, axis=1)
            P = cam + Dh * zz[:, None]
            Dn = (Dh / np.linalg.norm(Dh, axis=1, keepdims=True)).astype(F32)
            Ht = bilinear(Hm, P[:, 0] / cell, P[:, 2] / cell)
            isw = (Ht < wl - 0.05) if wl is not None else np.zeros(len(P), bool)
            col = np.zeros((len(P), 3), F32)
            land = ~isw
            if land.any():
                col[land] = shade_terrain(P[land], Dn[land], dist[land], cam[1])
            depth[rr, cc] = dist
            if isw.any():
                wmask[rr[isw], cc[isw]] = 1
            out[rr, cc] = col
            # anti-aliased skyline: blend top pixel of each column with the sky by coverage
            ys = np.clip(top, -1, H)
            r_top = np.floor(ys).astype(np.int32)
            ok = (r_top >= 0) & (r_top < H)
            cols = np.nonzero(ok)[0]
            rt = r_top[ok]
            cov = np.clip(rt + 1 - ys[ok], 0, 1)[:, None]
            skyrow = sky[rt - r0, cols]
            cur = out[rt, cols]
            hasr = np.isfinite(z[rt, cols])
            blend_c = np.where(hasr[:, None], cur * cov + skyrow * (1 - cov), cur)
            out[rt, cols] = blend_c
            wsel = isw
            Pw, Dw, distw = P[wsel], Dn[wsel], dist[wsel]
            rw, cw_ = rr[wsel], cc[wsel]
            depth_w = np.maximum(wl - Ht[wsel], 0)
        else:
            Pw = None
    if wl is not None and (h is None):
        # open water: analytic plane hit for all rows below the horizon
        rows = np.arange(max(0, int(cy)), H)
        Dw = dirs(rows)
        rw = np.repeat(rows, W); cw_ = np.tile(np.arange(W), len(rows))
        ok = Dw[:, 1] < -1e-4
        Dw, rw, cw_ = Dw[ok], rw[ok], cw_[ok]
        tw = (cam[1] - wl) / -Dw[:, 1]
        Pw = cam + Dw * tw[:, None]
        distw = tw
        depth_w = np.full(len(Pw), 1e4, F32)
        depth[rw, cw_] = distw
        wmask[rw, cw_] = 1
    if wl is not None and Pw is not None and len(Pw):
        # ---- reflection buffer: mirrored camera, terrain above the water only, + sky
        refl = sky.copy()
        if h is not None:
            camm = cam.copy(); camm[1] = 2 * wl - cam[1]
            Hmir = np.where(Hm > wl, Hm, F32(camm[1] - 1e6))
            mrows = np.arange(r0, min(H, int(cy) + 3))
            zm, _ = _raster(Hmir, size, camm, fwd, right, f, cx, cy, W, mrows, far)
            hm = np.isfinite(zm)
            if hm.any():
                rr, cc = np.nonzero(hm)
                zz = zm[rr, cc]
                u = (cc + 0.5 - cx) / f
                v = (cy - (rr + r0) - 0.5) / f
                Dh = fwd + u[:, None] * right + v[:, None] * np.array([0, 1, 0], F32)
                dist = zz * np.linalg.norm(Dh, axis=1)
                P = camm + Dh * zz[:, None]
                Dn = (Dh / np.linalg.norm(Dh, axis=1, keepdims=True)).astype(F32)
                Dv = Dn.copy(); Dv[:, 1] = -Dv[:, 1]
                refl[rr, cc] = shade_terrain(P, Dv, dist, cam[1])
        # ---- water surface
        wave_len = wave_len or (60.0 if h is None else 18.0)
        lvl = _ocean_tile(256, float(yaw + (wind or 0) + 15), 0.55 if h is None else 0.3, seed + 3)
        fl = distw / f
        grazing = np.maximum(-Dw[:, 1], 0.01)
        foot = fl / np.sqrt(grazing)
        w1 = _wave_sample(lvl, Pw[:, 0], Pw[:, 2], wave_len, foot)
        w2 = _wave_sample(lvl, Pw[:, 2] * 0.83 + 13.1, -Pw[:, 0] * 0.83 + 7.7, wave_len * 0.31, foot)
        sx = w1[:, 0] + w2[:, 0] * 0.45
        sz = w1[:, 1] + w2[:, 1] * 0.45
        var = np.maximum(w1[:, 2] - w1[:, 0] ** 2, 0) + np.maximum(w1[:, 3] - w1[:, 1] ** 2, 0) + \
            0.2 * (np.maximum(w2[:, 2] - w2[:, 0] ** 2, 0) + np.maximum(w2[:, 3] - w2[:, 1] ** 2, 0))
        k = waves / 1.1
        nW = np.stack([-sx * k, np.ones_like(sx), -sz * k], 1)
        nW /= np.linalg.norm(nW, axis=1, keepdims=True)
        cosv = np.clip(-(Dw * nW).sum(1), 0.02, 1)
        R = Dw + 2 * cosv[:, None] * nW
        R[:, 1] = np.maximum(R[:, 1], 0.004)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        rf = R @ fwd
        pc = cx + (R @ right) / np.maximum(rf, 1e-3) * f - 0.5
        pr = cy - R[:, 1] / np.maximum(rf, 1e-3) * f - 0.5 - r0
        # glossy reflection: blur the reflected image by the sub-pixel slope spread (+ a little curvature)
        sig = np.clip(1.1 * np.sqrt(var) * k * f + 0.8, 0, 40)
        levels = np.array([0.0, 3.0, 9.0, 24.0, 40.0])
        li = np.clip(np.searchsorted(levels, sig, side="right") - 1, 0, len(levels) - 2)
        t = np.clip((sig - levels[li]) / (levels[li + 1] - levels[li]), 0, 1)
        Lr = np.zeros((len(pc), 3), F32)
        for j in range(len(levels)):
            wgt = np.where(li == j, 1 - t, 0) + np.where(li == j - 1, t, 0)
            sel = wgt > 0
            if sel.any():
                # grazing view: reflections smear vertically, barely sideways
                img = refl if j == 0 else cv2.GaussianBlur(refl, (0, 0), sigmaX=float(levels[j]) * 0.2 + 0.5,
                                                           sigmaY=float(levels[j]))
                Lr[sel] += _remap_pts(img, pc[sel], pr[sel]) * wgt[sel, None].astype(F32)
        F = 0.02 + 0.98 * (1 - cosv) ** 5
        F = np.clip(F, 0, 1)[:, None]
        Esky = sky_irradiance(np.array([[0, 1, 0]], F32), **ctx)[0]
        csh = _cloud_shadow(Pw, s, clouds, cloud_alt, seed)
        wc = to_linear(_col(water_color)[:3])
        scat = wc * (Esky + sun_rgb * F32(max(s[1], 0)) * csh[:, None]) / math.pi * 0.4
        # translucent crests toward a low sun
        sss = np.clip(sx * s[0] + sz * s[2], 0, None) * (1 - max(s[1], 0)) * 0.25
        scat = scat + (wc * 2.2 + 0.02) * sun_rgb * (sss * csh)[:, None] * k * 3
        if h is not None:
            Pb = Pw[:, [0, 2]]
            ab = bilinear(Amap, Pb[:, 0] / cell, Pb[:, 1] / cell)
            tdep = np.exp(-depth_w * (1 + 1 / cosv) * 0.35)[:, None]
            bottom = ab * (Esky + sun_rgb * F32(max(s[1], 0))) / math.pi * np.array([0.55, 0.85, 0.85], F32)
            scat = scat * (1 - tdep) + bottom * tdep
        Lw = Lr * F + scat * (1 - F)
        # sun glint (Beckmann lobe widened by the sub-pixel slope variance)
        Hh = s[None, :] - Dw
        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
        ch = np.clip((Hh * nW).sum(1), 1e-3, 1)
        m2 = np.clip(0.0006 + var * k * k * 0.5, 0.0006, 0.5)
        Dg = np.exp(-(1 - ch * ch) / (ch * ch * m2)) / (math.pi * m2 * ch ** 4)
        Fh = 0.02 + 0.98 * (1 - np.clip((Hh * s).sum(1), 0, 1)) ** 5
        glint = Dg * Fh / (4 * cosv) * (s[1] > 0) * csh
        Lw = Lw + sun_rgb * glint[:, None]
        # foam: wave crests + shoreline band
        fm = w1[:, 4] * foam * np.clip(waves * 8, 0, 1)
        if h is not None:
            band = np.clip(1 - depth_w / 3.0, 0, 1) ** 2
            nn = _wrap_bilinear(detail, Pw[:, 0] / 6.0, Pw[:, 2] / 6.0)
            fm = np.maximum(fm, band * np.clip(0.55 + 0.5 * nn, 0, 1) * foam)
        fm = np.clip(fm, 0, 1)[:, None]
        Lf = 0.8 * (Esky + sun_rgb * F32(max(s[1], 0.05)) * csh[:, None]) / math.pi
        Lw = Lw * (1 - fm) + Lf * fm
        Lw = aerial(Lw, Dw, distw, np.full(len(Pw), cam[1]), np.full(len(Pw), wl), **ctx)
        out[rw, cw_] = Lw
    if wl is None and h is None:
        pass
    rgb = tonemap_linear(out, tonemap, exposure, look)
    if aa > 1:
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(np.where(np.isfinite(depth), depth, 1e9), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[depth >= 1e9] = np.inf
        wmask = cv2.resize(wmask, (self.W, self.H), interpolation=cv2.INTER_AREA)
    self.rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), F32)], -1)
    sc = self.s * aa

    def project(x, y, z):
        """World point -> (canvas x, canvas y, canvas px per metre at that depth)."""
        p = np.asarray([x, y, z], F32) - cam
        zf = float(p @ fwd)
        if zf <= 0:
            return None
        return (float(cx + (p @ right) / zf * f) / sc, float(cy - p[1] / zf * f) / sc, f / zf / sc)

    def ground(x, z):
        """Terrain altitude (metres) at world x, z."""
        if h is None:
            return 0.0
        return float(bilinear(h, np.array([x / cell], F32), np.array([z / cell], F32))[0]) * height

    return SimpleNamespace(depth=depth, water=wmask, project=project, ground=ground, cam=cam, sun=s,
                           sun_color=sun_rgb, f=f / sc, horizon=cy / sc)


def _layer_sky(self, sun_elev=10.0, sun_az=0.0, horizon=0.62, fov=55.0, haze=1.0, clouds=0.0,
               cloud_alt=(1600.0, 3000.0), exposure=0.0, sun_size=0.6, seed=0, look="punchy", alt=2.0):
    """Physically based sky background on this layer (Rayleigh + Mie + ozone single scattering,
    sun disc, optional raymarched clouds). sun_az = degrees right of the view centre."""
    return _layer_landscape(self, None, cam=(0.0, alt, 0.0), yaw=0.0, fov=fov, horizon=horizon,
                            sun_elev=sun_elev, sun_az=sun_az, haze=haze, clouds=clouds, cloud_alt=cloud_alt,
                            exposure=exposure, sun_size=sun_size, seed=seed, look=look)


def _layer_sea(self, sun_elev=6.0, sun_az=0.0, horizon=0.5, fov=55.0, haze=1.0, clouds=0.3, waves=0.12,
               wave_len=60.0, wind=0.0, foam=0.5, water_color="#0b2a3a", cam_height=4.0, exposure=0.0,
               sun_size=0.6, seed=0, look="punchy", cloud_alt=(1600.0, 3000.0), aa=1):
    """Open sea to the horizon under a physical sky: FFT-like wave normals (mip-filtered so the
    distance stays smooth), Fresnel sky + cloud reflection, sun glitter path, crest foam."""
    return _layer_landscape(self, None, cam=(0.0, cam_height, 0.0), yaw=0.0, fov=fov, horizon=horizon,
                            sun_elev=sun_elev, sun_az=sun_az, haze=haze, clouds=clouds, cloud_alt=cloud_alt,
                            water=0, waves=waves, wave_len=wave_len, wind=wind, foam=foam, water_color=water_color,
                            exposure=exposure, sun_size=sun_size, seed=seed, look=look, aa=aa)


Layer.landscape = _layer_landscape
Layer.sky = _layer_sky
Layer.sea = _layer_sea


# ------------------------------------------------------------------ Scene integration (no sdf.py change needed)
def scene_sky(scene, sun_elev=25.0, sun_az=30.0, haze=1.0, power=1.0, sun_power=1.0, soft=0.03):
    """Give an sdf Scene the physical sky: sets its sun (direction + colour) and overrides the sky
    radiance / irradiance seen by rays (instance attributes; Scene units: 1 = 1/SUN_POWER)."""
    s = sun_direction(sun_elev, sun_az)
    sc = sun_color(sun_elev, haze)
    scene.lights = [L for L in scene.lights if L.get("kind") != "sun"]
    k = 3.0 * sun_power / max(float(sc.max()), 1e-3)
    scene.lights.append(dict(kind="sun", dir=s.astype(F32), col=(sc * F32(k * max(sc.max(), 1e-3))).astype(F32),
                             soft=max(soft, 1e-3)))
    norm = F32(3.0 / SUN_POWER * power)
    sunL = scene.lights[-1]

    def _sky(d):
        L = sky_radiance(d, sun_elev, sun_az, haze, 0.0, sun_size=0.6) * norm
        cs = d @ s.astype(F32)
        return (L + sunL["col"] * (np.clip(cs, 0, 1) ** 200 * 0.3)[:, None]).astype(F32)

    def _irr(n):
        return (sky_irradiance(n, sun_elev, sun_az, haze) / F32(math.pi) * norm).astype(F32)
    scene._sky = _sky
    scene._irr = _irr
    return scene


# ------------------------------------------------------------------ plants: skeletons
TREES = {   # crown envelope (centre y, radii x/y as fractions of height), growth params, look
    "oak": dict(trunk=0.32, cy=0.62, rx=0.52, ry=0.36, n=1500, step=0.018, infl=0.18, kill=0.03, trop=0.05,
                bark="#4a3b2e", leaf=("#3c5a25", "#56732f", "#6f8a3a"), leaf_size=0.016, clump=0.06, r0=0.0035,
                pipe=2.4),
    "birch": dict(trunk=0.3, cy=0.63, rx=0.24, ry=0.38, n=900, step=0.016, infl=0.16, kill=0.028, trop=-0.12,
                  bark="#e9e4da", leaf=("#4f7028", "#6f8f34", "#98ad4c"), leaf_size=0.011, clump=0.055,
                  r0=0.0016, pipe=2.6),
    "dead": dict(trunk=0.28, cy=0.6, rx=0.45, ry=0.36, n=320, step=0.024, infl=0.3, kill=0.06, trop=0.12,
                 bark="#5b5148", leaf=None, leaf_size=0.0, clump=0, r0=0.003, pipe=2.2),
    "bush": dict(trunk=0.0, cy=0.42, rx=0.62, ry=0.42, n=1000, step=0.022, infl=0.2, kill=0.035, trop=0.1,
                 bark="#4a3d30", leaf=("#35522a", "#4d6b33", "#6a8640"), leaf_size=0.03, clump=0.1, r0=0.004,
                 pipe=2.3),
    "pine": dict(trunk=0.15, bark="#4b3a2e", leaf=("#1f3a26", "#2c4a2e", "#3e5e36"), leaf_size=0.02),
    "palm": dict(trunk=0.8, bark="#8a7458", leaf=("#3f6a2a", "#5a8236", "#7c9c48"), leaf_size=0.02),
}
SEASONS = {
    "spring": ("#7fa23a", "#9cbc4a", "#c3d36a"),
    "autumn": ("#a4461f", "#cf7a26", "#e2ad3c"),
}


def _envelope(sp, rng, dim):
    """Attractor points inside the crown ellipsoid, tree space (height 1)."""
    q = rng.uniform(-1, 1, (sp["n"] * 3, dim))
    q = q[(q * q).sum(1) < 1][:sp["n"]]
    out = np.zeros((len(q), dim))
    out[:, 0] = q[:, 0] * sp["rx"]
    out[:, 1] = sp["cy"] + q[:, 1] * sp["ry"]
    if dim == 3:
        out[:, 2] = q[:, 2] * sp["rx"]
    return out[out[:, 1] > sp["trunk"] * 0.9]


@lru_cache(maxsize=64)
def _skeleton_c(kind, seed, dim):
    from scipy.spatial import cKDTree
    sp = TREES[kind]
    rng = np.random.default_rng(seed)
    if kind == "pine":
        return _pine_skeleton(rng, dim)
    if kind == "palm":
        return _palm_skeleton(rng, dim)
    att = _envelope(sp, rng, dim)
    up = np.zeros(dim); up[1] = 1
    hz = np.ones(dim); hz[1] = 0
    pos = [np.zeros(dim)]
    par = [-1]
    if kind == "bush":                       # several stems from the ground
        for _ in range(int(rng.integers(3, 6))):
            a = rng.uniform(-0.5, 0.5, dim) * hz * 0.25 + up * 0.1
            pos.append(a); par.append(0)
    else:
        lean = rng.normal(0, 0.15, dim) * hz
        y = 0.0
        while y < sp["trunk"]:
            y += sp["step"]
            p = pos[-1] + sp["step"] * (up + lean * 0.4 + rng.normal(0, 0.12, dim) * hz)
            pos.append(p); par.append(len(pos) - 2)
    trop = up * sp["trop"]
    for _ in range(400):
        if len(att) == 0:
            break
        P = np.asarray(pos)
        tree = cKDTree(P)
        d, j = tree.query(att, distance_upper_bound=sp["infl"])
        ok = np.isfinite(d)
        if not ok.any():
            break
        v = att[ok] - P[j[ok]]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        acc = np.zeros_like(P); cnt = np.zeros(len(P))
        np.add.at(acc, j[ok], v); np.add.at(cnt, j[ok], 1)
        grow = np.nonzero(cnt > 0)[0]
        dirs = acc[grow] / cnt[grow, None] + trop + rng.normal(0, 0.18, (len(grow), dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        new = P[grow] + dirs * sp["step"]
        dn, _ = tree.query(new)
        keep = dn > sp["step"] * 0.3
        if not keep.any():
            break
        pos.extend(new[keep]); par.extend(grow[keep].tolist())
        hits = cKDTree(np.asarray(pos)).query_ball_point(att, sp["kill"])
        att = att[np.array([len(hh) == 0 for hh in hits], bool)]
    pos = np.asarray(pos); par = np.asarray(par)
    return pos, par, _pipe_radii(par, sp["r0"], sp["pipe"])


def tree_skeleton(kind="oak", seed=0, dim=2):
    """Branch graph of a tree of height 1 (y up): (pos (N, dim), parent (N,), radius (N,)).
    Space colonisation (oak, birch, dead, bush) or procedural whorls (pine) / arching fronds (palm)."""
    if kind not in TREES:
        raise ValueError(f"tree kind: {', '.join(TREES)}")
    pos, par, rad = _skeleton_c(kind, int(seed), int(dim))
    return pos.copy(), par.copy(), rad.copy()


def _pipe_radii(par, r0, e):
    """Leonardo / pipe model: r_parent^e = sum r_child^e, leaves r0."""
    n = len(par)
    acc = np.zeros(n)
    child = np.zeros(n, int)
    np.add.at(child, par[1:], 1)
    acc[child == 0] = r0 ** e
    for i in range(n - 1, 0, -1):
        acc[par[i]] += acc[i]
    return acc ** (1 / e)


def _pine_skeleton(rng, dim):
    pos = [np.zeros(dim)]; par = [-1]
    nseg = 40
    trunk = [0]
    for i in range(1, nseg + 1):
        p = np.zeros(dim); p[1] = i / nseg
        p[0] = math.sin(i * 0.3) * 0.004
        pos.append(p); par.append(len(pos) - 2); trunk.append(len(pos) - 1)
    y = 0.14
    while y < 0.97:
        t = (y - 0.14) / 0.86
        L = 0.34 * (1 - t) ** 0.95 + 0.03
        k = int(rng.integers(4, 7)) if dim == 3 else 2
        a0 = rng.uniform(0, 6.3)
        for j in range(k):
            ang = a0 + (j / k) * 2 * math.pi + rng.uniform(-0.3, 0.3)
            base = trunk[min(nseg, int(round(y * nseg)))]
            if dim == 2:
                dirx = np.array([1.0 if j == 0 else -1.0, 0])
                L2 = L * rng.uniform(0.7, 1.15)
            else:
                dirx = np.array([math.cos(ang), 0, math.sin(ang)])
                L2 = L * rng.uniform(0.8, 1.1)
            prev = base
            droop = rng.uniform(0.15, 0.35) * (1 - t * 0.5)
            for s_ in range(1, 7):
                u = s_ / 6
                p = pos[base] + dirx * L2 * u
                p = p.copy(); p[1] += (0.06 - droop * u * u) * L2 + u * 0.02
                pos.append(p); par.append(prev); prev = len(pos) - 1
        y += rng.uniform(0.035, 0.055) * (1.2 - t * 0.5)
    pos = np.asarray(pos); par = np.asarray(par)
    rad = _pipe_radii(par, 0.0022, 2.3)
    rad[trunk] = np.maximum(rad[trunk], 0.018 * (1.05 - pos[trunk, 1]) + 0.003)
    return pos, par, rad


def _palm_skeleton(rng, dim):
    pos = [np.zeros(dim)]; par = [-1]
    bend = rng.uniform(0.12, 0.25) * (1 if rng.random() < 0.5 else -1)
    nseg = 24
    for i in range(1, nseg + 1):
        t = i / nseg
        p = np.zeros(dim); p[1] = t * 0.86; p[0] = bend * t * t
        pos.append(p); par.append(len(pos) - 2)
    top = len(pos) - 1
    k = int(rng.integers(9, 13))
    for j in range(k):
        ang = j / k * 2 * math.pi + rng.uniform(-0.2, 0.2)
        L = rng.uniform(0.32, 0.42)
        lift = rng.uniform(0.1, 0.45)
        if dim == 2:
            d = np.array([math.cos(ang), 0])
            if abs(d[0]) < 0.25:
                d[0] = 0.25 * (1 if d[0] >= 0 else -1)
        else:
            d = np.array([math.cos(ang), 0, math.sin(ang)])
        prev = top
        for s_ in range(1, 9):
            u = s_ / 8
            p = pos[top] + d * L * u
            p = p.copy(); p[1] += L * (lift * u - (0.55 + lift) * u * u * 0.9)
            pos.append(p); par.append(prev); prev = len(pos) - 1
    pos = np.asarray(pos); par = np.asarray(par)
    rad = np.full(len(pos), 0.004)
    rad[:nseg + 1] = 0.028 - 0.006 * np.arange(nseg + 1) / nseg
    return pos, par, rad


# ------------------------------------------------------------------ plants: 2D drawing
def _skia_surface(w, h):
    info = skia.ImageInfo.Make(int(w), int(h), skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
    surf = skia.Surface.MakeRaster(info)
    return surf, surf.getCanvas()


def _blit(layer, surf, x0, y0, opacity=1.0):
    """Composite a premultiplied skia surface at raster (x0, y0) onto a layer (normal blend)."""
    img = surf.makeImageSnapshot().toarray().astype(F32) / 255.0
    h, w = img.shape[:2]
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, layer.W), min(y0 + h, layer.H)
    if X0 >= X1 or Y0 >= Y1:
        return
    src = img[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * F32(opacity)
    src[..., :3] *= src[..., 3:4]                      # skia's toarray() is unpremultiplied
    dst = layer.px[Y0:Y1, X0:X1]
    dst[:] = src + dst * (1 - src[..., 3:4])
    layer.touch()


def _rgb(c):
    return np.asarray(_col(c)[:3], F32)


def _c4(c, a=1.0):
    return skia.Color4f(float(c[0]), float(c[1]), float(c[2]), float(a))


def _paint(c, a=1.0):
    p = skia.Paint(AntiAlias=True)
    p.setColor(_c4(c, a))
    return p


def _light3(light):
    """2D light (azimuth deg: 0 = from right, 90 = from top; elevation) -> (x right, y up, z to viewer)."""
    az, el = math.radians(light[0]), math.radians(light[1])
    return np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])


def _haze(col, haze, hc):
    return col * (1 - haze) + hc * haze


def _seg_quad(a, b, ra, rb):
    d = b - a
    ln = math.hypot(d[0], d[1]) + 1e-9
    nx, ny = -d[1] / ln, d[0] / ln
    return [(a[0] + nx * ra, a[1] + ny * ra), (b[0] + nx * rb, b[1] + ny * rb),
            (b[0] - nx * rb, b[1] - ny * rb), (a[0] - nx * ra, a[1] - ny * ra)]


def _add_poly(path, pts):
    path.moveTo(*pts[0])
    for q in pts[1:]:
        path.lineTo(*q)
    path.close()


def _draw_branches(cv, P, par, R, bark, L, haze, hc, ox, oy):
    """Bark: base silhouette, a shadow strip and a lit strip on the light side (round shading)."""
    base = skia.Path(); lit = skia.Path(); dark = skia.Path()
    o = np.array([ox, oy])
    for i in range(1, len(P)):
        j = par[i]
        a, b = P[j] - o, P[i] - o
        ra, rb = max(R[j], 0.35), max(R[i], 0.3)
        _add_poly(base, _seg_quad(a, b, ra, rb))
        base.addCircle(float(b[0]), float(b[1]), float(rb))
        if ra > 1.2:
            d = b - a
            ln = math.hypot(*d) + 1e-9
            nx, ny = -d[1] / ln, d[0] / ln
            side = 1 if (nx * L[0] - ny * L[1]) > 0 else -1           # screen normal facing the light
            sh = np.array([nx, ny]) * side
            _add_poly(lit, _seg_quad(a + sh * ra * 0.4, b + sh * rb * 0.4, ra * 0.4, rb * 0.4))
            _add_poly(dark, _seg_quad(a - sh * ra * 0.55, b - sh * rb * 0.55, ra * 0.42, rb * 0.42))
    bk = _rgb(bark)
    cv.drawPath(base, _paint(_haze(bk * 0.8, haze, hc)))
    cv.drawPath(dark, _paint(_haze(bk * 0.62, haze, hc)))
    cv.drawPath(lit, _paint(_haze(np.clip(bk * 1.2 + 0.03, 0, 1), haze, hc)))


def _leaf_path():
    p = skia.Path()
    p.moveTo(0, -0.5); p.quadTo(0.36, -0.05, 0, 0.5); p.quadTo(-0.36, -0.05, 0, -0.5); p.close()
    return p


_LEAF = _leaf_path()


def _shade_col(pal, lam, rng, n):
    """Foliage colours by light level lam (0 dark .. 1 lit) through a 3-colour palette."""
    lam = np.clip(np.asarray(lam, float) + rng.normal(0, 0.08, n), 0, 1)
    p = np.asarray(pal)
    t = lam * 2
    lo = np.clip(t.astype(int), 0, 1)
    f = (t - lo)[:, None]
    col = p[lo] * (1 - f) + p[lo + 1] * f
    return col * (0.55 + 0.6 * lam)[:, None]


def _tree2d(self, x, y, height=300.0, kind="oak", seed=0, season="summer", light=(135, 45), haze=0.0,
            haze_color="#b9c6d3", leaves=1.0, colors=None, width=1.0, shadow=0.35, bark=None, lean=0.0):
    """Draw a tree standing at canvas (x, y) (trunk base), `height` px tall.
    kind: oak | pine | birch | palm | bush | dead. season: summer | spring | autumn | winter (bare,
    except pine/palm). light = (azimuth deg, 0 = from right, 90 = from top; elevation deg).
    haze 0..1 blends toward haze_color (aerial perspective for depth layers); leaves = density;
    colors = 3 foliage colours (dark, mid, light); width scales the crown horizontally."""
    sp = TREES[kind]
    s = self.s
    Hs = height * s
    pos, par, rad = tree_skeleton(kind, seed, 2)
    P = np.stack([x * s + pos[:, 0] * Hs * width + pos[:, 1] * Hs * lean, y * s - pos[:, 1] * Hs], 1)
    R = rad * Hs
    rng = np.random.default_rng(seed + 1)
    L = _light3(light)
    hc = _rgb(haze_color)
    pad = int(Hs * 0.15 + 8)
    x0 = int(min(P[:, 0].min(), x * s - Hs * 0.6) - pad); x1 = int(max(P[:, 0].max(), x * s + Hs * 0.6) + pad)
    y0 = int(P[:, 1].min() - pad); y1 = int(y * s + pad)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
    if X0 >= X1 or Y0 >= Y1:
        return self
    surf, cv = _skia_surface(X1 - X0, Y1 - Y0)
    ox, oy = X0, Y0
    if shadow:
        sx = -L[0] * Hs * 0.25 / max(math.tan(math.radians(light[1])), 0.4)
        shp = skia.Paint(AntiAlias=True, Color=_c4((0.06, 0.07, 0.04), shadow))
        shp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, float(Hs * 0.03 + 0.5)))
        cw = Hs * (sp["rx"] if "rx" in sp else 0.25) * width
        cv.drawOval(skia.Rect.MakeXYWH(x * s - ox + sx - cw, y * s - oy - Hs * 0.03, cw * 2, Hs * 0.07), shp)
    if colors is not None:
        pal = colors
    elif season == "winter":
        pal = sp["leaf"] if kind in ("pine", "palm") else None
    elif season in SEASONS and kind not in ("pine", "palm"):
        pal = SEASONS[season]
    else:
        pal = sp["leaf"]
    pal = None if pal is None else [_rgb(c) for c in pal]
    bark = bark or sp["bark"]
    child = np.zeros(len(P), int)
    np.add.at(child, par[1:], 1)
    if kind == "palm":
        _draw_palm(cv, P, par, R, rng, pal, _rgb(bark), L, haze, hc, ox, oy, Hs)
    elif kind == "pine":
        _draw_branches(cv, P, par, R, bark, L, haze, hc, ox, oy)
        if pal is not None:
            _draw_pine_foliage(cv, P, par, pos, rng, pal, L, haze, hc, ox, oy, Hs, leaves, season)
    else:
        if kind == "birch":
            R = R * 0.85
        _draw_branches(cv, P, par, R, bark, L, haze, hc, ox, oy)
        if kind == "birch":
            _birch_marks(cv, P, par, R, rng, haze, hc, ox, oy)
        if pal is not None and sp["leaf"] is not None:
            _draw_foliage(cv, P, par, R, child, pos, rng, pal, L, haze, hc, ox, oy, Hs * width, sp, leaves, kind)
    _blit(self, surf, ox, oy)
    return self


def _birch_marks(cv, P, par, R, rng, haze, hc, ox, oy):
    pm = _paint(_haze(np.array([0.12, 0.11, 0.1]), haze, hc), 0.85)
    for i in range(1, len(P)):
        if R[i] < 2.0 or rng.random() > 0.55:
            continue
        a, b = P[par[i]] - (ox, oy), P[i] - (ox, oy)
        c = a + (b - a) * rng.random()
        r = R[i]
        w = r * rng.uniform(0.5, 1.4)
        hgt = max(r * rng.uniform(0.08, 0.25), 0.6)
        x0 = c[0] - r if rng.random() < 0.5 else c[0] + r - w
        cv.drawRect(skia.Rect.MakeXYWH(float(x0), float(c[1]), float(w), float(hgt)), pm)


def _draw_foliage(cv, P, par, R, child, pos, rng, pal, L, haze, hc, ox, oy, Hs, sp, density, kind):
    tips = np.nonzero((child == 0) | (R < sp["r0"] * Hs * 2.2))[0]
    if sp["trunk"] > 0:
        tips = tips[pos[tips, 1] > sp["trunk"] * 0.85]
    if len(tips) == 0:
        return
    cen = np.array([P[tips, 0].mean(), P[tips, 1].mean()])
    ext = np.array([np.ptp(P[tips, 0]) / 2 + 1e-3, np.ptp(P[tips, 1]) / 2 + 1e-3])
    ls0 = sp["leaf_size"] * Hs
    ls = min(ls0, 3.0 + Hs * 0.006)                # big trees: smaller leaves, more of them
    clump = sp["clump"] * Hs
    per = max(3, int(round(16 * density * (sp["clump"] / 0.06) ** 1.5 * min((ls0 / ls) ** 2, 4))))
    if ls < 2.2:                                    # tiny leaves: fewer, bigger
        per = max(3, int(per * (ls / 2.2) ** 2 * 1.4)); ls = 2.2
    m = len(tips) * per
    k = np.repeat(tips, per)
    ang = rng.uniform(0, 2 * math.pi, m)
    rr = np.sqrt(rng.random(m)) * clump
    xy = P[k] + np.stack([np.cos(ang) * rr, np.sin(ang) * rr * 0.8], 1)
    if kind == "birch":                             # hanging foliage
        xy[:, 1] += rr * 0.6
    # sub-crowns (leaf clumps): leaves grouped by a coarse grid, each group shaded as its own ball
    cs = max(ext.max() * 0.42, clump * 1.5)
    key = np.floor((P[k] - cen) / cs + rng.random(2)).astype(np.int64)
    _, g = np.unique(key[:, 0] * 100003 + key[:, 1], return_inverse=True)
    g = g.ravel()
    cnt = np.bincount(g)
    gc = np.stack([np.bincount(g, xy[:, 0]) / cnt, np.bincount(g, xy[:, 1]) / cnt], 1)
    gr = np.sqrt(np.bincount(g, ((xy - gc[g]) ** 2).sum(1)) / cnt) * 1.5 + 1e-3
    ql = (xy - gc[g]) / gr[g][:, None]
    q = (xy - cen) / ext                            # pseudo 3D normal from the crown ellipsoid
    q = q * 0.55 + ql * 0.45
    q[:, 1] = -q[:, 1]
    z2 = np.clip(1 - (q * q).sum(1), 0, 1)
    zs = np.where(rng.random(m) < 0.6, 1, -1) * np.sqrt(z2)
    nrm = np.stack([q[:, 0], q[:, 1], zs], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    lam = np.clip(nrm @ L, 0, 1)
    depth = 1 - np.clip((zs + 1) / 2, 0, 1)        # back / inner leaves are occluded
    rim = np.clip(np.sqrt((ql ** 2).sum(1)), 0, 1)  # clump centres a bit darker than their lit rims
    lam = lam * (1 - 0.55 * depth) * (0.8 + 0.25 * rim) + 0.06
    cols = _haze(_shade_col(pal, lam, rng, m), haze, hc)
    order = np.argsort(zs + rng.normal(0, 0.15, m))  # back to front
    rot = rng.uniform(0, 360, m)
    sz = ls * rng.uniform(0.75, 1.3, m)
    paint = skia.Paint(AntiAlias=True)
    for i in order:
        paint.setColor(_c4(cols[i]))
        cv.save()
        cv.translate(float(xy[i, 0] - ox), float(xy[i, 1] - oy))
        cv.rotate(float(rot[i]))
        cv.scale(float(sz[i]), float(sz[i]))
        cv.drawPath(_LEAF, paint)
        cv.restore()


def _draw_pine_foliage(cv, P, par, pos, rng, pal, L, haze, hc, ox, oy, Hs, density, season):
    """Needle tufts: short strokes fanning along every branch, darker underneath. Vectorised, drawn
    as one path per shade level (dark first, lit on top)."""
    idx = np.nonzero(par >= 0)[0]
    idx = idx[pos[idx, 1] > 0.12]
    a, b = P[par[idx]], P[idx]
    pts = np.concatenate([a, (a + b) / 2])
    d = np.concatenate([b - a, b - a])
    hy = np.concatenate([pos[idx, 1], pos[idx, 1]])
    n = int(np.clip(Hs / 30, 2, 7) * density) + 1
    p = np.repeat(pts, n, 0); d = np.repeat(d, n, 0); hy = np.repeat(hy, n)
    m = len(p)
    ang = np.arctan2(d[:, 1], d[:, 0]) + rng.normal(0, 0.9, m)
    ln = Hs * rng.uniform(0.018, 0.04, m)
    e = p + np.stack([np.cos(ang), np.abs(np.sin(ang)) * 0.6 + 0.4], 1) * (ln[:, None] * np.array([1, 0.7]))
    side = (p[:, 0] - P[0, 0]) / (Hs * 0.3)
    lam = np.clip(0.45 + 0.4 * side * L[0] + 0.35 * (hy - 0.5) + rng.normal(0, 0.12, m), 0, 1)
    lam = np.where(e[:, 1] > p[:, 1], lam * 0.6, lam)
    lv = np.clip((lam * 11).astype(int), 0, 11)
    snow = (rng.random(m) < 0.35) if season == "winter" else np.zeros(m, bool)
    paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap)
    paint.setStrokeWidth(max(1.0, Hs * 0.006))
    o = np.array([ox, oy])
    for sn in (False, True):
        for k in range(12):
            sel = np.nonzero((lv == k) & (snow == sn))[0]
            if sel.size == 0:
                continue
            l0 = (k + 0.5) / 12
            c = (np.array([0.9, 0.92, 0.96]) * (0.7 + 0.3 * l0) if sn else
                 _shade_col(pal, [l0], np.random.default_rng(0), 1)[0])
            paint.setColor(_c4(_haze(c, haze, hc)))
            path = skia.Path()
            for q0, q1 in zip(p[sel] - o, e[sel] - o):
                path.moveTo(float(q0[0]), float(q0[1])); path.lineTo(float(q1[0]), float(q1[1]))
            cv.drawPath(path, paint)


def _draw_palm(cv, P, par, R, rng, pal, bark, L, haze, hc, ox, oy, Hs):
    nseg = 24
    o = np.array([ox, oy])
    path = skia.Path(); rings = skia.Path()
    for i in range(1, nseg + 1):
        a, b = P[i - 1] - o, P[i] - o
        _add_poly(path, _seg_quad(a, b, R[i - 1], R[i]))
        q = _seg_quad(a, b, R[i - 1] * 1.08, R[i] * 1.08)
        rings.moveTo(*q[0]); rings.lineTo(*q[3])
    cv.drawPath(path, _paint(_haze(bark * 0.85, haze, hc)))
    pr = _paint(_haze(bark * 0.5, haze, hc), 0.8)
    pr.setStyle(skia.Paint.kStroke_Style); pr.setStrokeWidth(max(0.8, Hs * 0.003))
    cv.drawPath(rings, pr)
    if pal is None:
        return
    pf = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap)
    kids = [i for i in range(nseg + 1, len(P)) if par[i] == nseg]
    for k0 in kids:
        chain = [nseg, k0]
        while chain[-1] + 1 < len(P) and par[chain[-1] + 1] == chain[-1]:
            chain.append(chain[-1] + 1)
        C = P[chain]
        for j in range(1, len(C)):
            a, b = C[j - 1], C[j]
            d = b - a
            ln = math.hypot(*d) + 1e-9
            nx, ny = -d[1] / ln, d[0] / ln
            for t in np.linspace(0, 1, 5, endpoint=False):
                pnt = a + d * t
                u = (j - 1 + t) / (len(C) - 1)
                Lf = Hs * 0.09 * math.sin(math.pi * min(u * 1.1 + 0.08, 1)) + 1
                for sgn in (-1, 1):
                    e = pnt + np.array([nx, ny]) * sgn * Lf * 0.8 + d / ln * Lf * 0.5 + np.array([0, Lf * 0.35])
                    lam = np.clip(0.5 + 0.5 * sgn * (nx * L[0] - ny * L[1]) + rng.normal(0, 0.12), 0, 1)
                    pf.setColor(_c4(_haze(_shade_col(pal, [lam], rng, 1)[0], haze, hc)))
                    pf.setStrokeWidth(max(0.8, Hs * 0.006 * (1 - u * 0.6)))
                    cv.drawLine(float(pnt[0] - ox), float(pnt[1] - oy), float(e[0] - ox), float(e[1] - oy), pf)
        pf.setColor(_c4(_haze(np.asarray(pal[0]) * 0.8, haze, hc)))
        pf.setStrokeWidth(max(1.0, Hs * 0.006))
        pth = skia.Path(); pth.moveTo(*(C[0] - o))
        for q in C[1:]:
            pth.lineTo(*(q - o))
        cv.drawPath(pth, pf)


def _grass2d(self, x, y, width=60.0, height=40.0, n=40, color=None, seed=0, light=(135, 45), haze=0.0,
             haze_color="#b9c6d3", bend=0.3, dry=0.0):
    """Grass tuft: n tapered curved blades rooted along [x - width/2, x + width/2] at y (dark base,
    lit tips). color = 3 greens (dark → light); dry 0..1 mixes in straw tones."""
    s = self.s
    rng = np.random.default_rng(seed)
    pal = [_rgb(c) for c in (color or ["#2f4a1e", "#4f6e2a", "#7c9640"])]
    straw = [_rgb(c) for c in ("#6b5a32", "#a08a52", "#cdb87a")]
    pal = [p * (1 - dry) + q * dry for p, q in zip(pal, straw)]
    hc = _rgb(haze_color)
    L = _light3(light)
    pad = int((height + width) * s) + 4
    X0, Y0 = int(x * s - width * s / 2 - pad), int(y * s - height * s * 1.2 - pad)
    surf, cv = _skia_surface(max(int(width * s + 2 * pad), 1), max(int(height * s * 1.2 + pad + 4), 1))
    bx = x * s - X0 + (rng.random(n) - 0.5) * width * s
    by = np.full(n, y * s - Y0) + rng.random(n) * 1.5
    hh = height * s * rng.uniform(0.55, 1.1, n)
    lean = rng.normal(0, 0.35, n) + (bx - (x * s - X0)) / max(width * s, 1) * 0.8
    paint = skia.Paint(AntiAlias=True)
    for i in rng.permutation(n):
        w = max(0.7, hh[i] * 0.045) * rng.uniform(0.7, 1.3)
        tipx = bx[i] + lean[i] * hh[i] * (0.5 + bend)
        tipy = by[i] - hh[i] * (1 - abs(lean[i]) * 0.25 * bend)
        cx_, cy_ = bx[i] + lean[i] * hh[i] * 0.1, by[i] - hh[i] * 0.6
        p = skia.Path()
        p.moveTo(bx[i] - w, by[i]); p.quadTo(cx_ - w * 0.6, cy_, tipx, tipy)
        p.quadTo(cx_ + w * 0.6, cy_, bx[i] + w, by[i]); p.close()
        lam = 0.5 + 0.35 * lean[i] * L[0] + rng.normal(0, 0.18)
        top = _haze(_shade_col(pal, [lam], rng, 1)[0], haze, hc)
        bot = _haze(np.asarray(pal[0]) * 0.55, haze, hc)
        paint.setShader(skia.GradientShader.MakeLinear(
            [skia.Point(float(bx[i]), float(by[i])), skia.Point(float(tipx), float(tipy))],
            [_c4(bot).toColor(), _c4(top).toColor()]))
        cv.drawPath(p, paint)
    _blit(self, surf, X0, Y0)
    return self


_FLOWERS = {"daisy": ("#f6f3ea", "#e7b52c"), "poppy": ("#d8321f", "#1d1612"), "tulip": ("#e0485a", "#f0c040"),
            "bell": ("#7c6bd0", "#e8e0f0"), "buttercup": ("#f2c81e", "#c98d10")}


def _flower2d(self, x, y, size=40.0, kind="daisy", color=None, seed=0, light=(135, 45), haze=0.0,
              haze_color="#b9c6d3", stem="#4a6b2a", lean=0.0):
    """Flower rooted at (x, y), `size` px tall: kind daisy | poppy | tulip | bell | buttercup."""
    s = self.s
    rng = np.random.default_rng(seed)
    pet, mid = _FLOWERS[kind]
    pet = _rgb(color or pet); mid = _rgb(mid); stc = _rgb(stem)
    hc = _rgb(haze_color)
    L = _light3(light)
    la = math.atan2(L[1], L[0])
    S = size * s
    pad = int(S * 0.8) + 4
    X0, Y0 = int(x * s - pad), int(y * s - S - pad)
    surf, cv = _skia_surface(2 * pad, int(S + 2 * pad))
    bx, by = x * s - X0, y * s - Y0
    hx = bx + (lean + rng.normal(0, 0.12)) * S * 0.4
    hy = by - S
    ps = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap)
    ps.setStrokeWidth(max(0.8, S * 0.035)); ps.setColor(_c4(_haze(stc, haze, hc)))
    p = skia.Path(); p.moveTo(bx, by); p.quadTo(bx + (hx - bx) * 0.1, by - S * 0.5, hx, hy)
    cv.drawPath(p, ps)
    pl = _paint(_haze(stc * 0.9, haze, hc))
    for sgn in (-1, 1):
        cv.save(); cv.translate(bx + sgn * S * 0.06, by - S * 0.18); cv.rotate(sgn * 40.0)
        cv.scale(S * 0.12, S * 0.3); cv.drawPath(_LEAF, pl); cv.restore()
    pc = lambda lam: _haze(pet * (0.6 + 0.45 * lam), haze, hc)
    if kind in ("daisy", "buttercup"):
        k = 13 if kind == "daisy" else 5
        r = S * (0.16 if kind == "daisy" else 0.12)
        for j in range(k):
            a = j / k * 360 + rng.uniform(-6, 6)
            lam = 0.6 + 0.4 * math.cos(math.radians(a) - la)
            cv.save(); cv.translate(hx, hy); cv.scale(1, 0.55); cv.rotate(a)
            pp = skia.Path(); pp.addOval(skia.Rect.MakeXYWH(-r * 0.2, -r, r * 0.4, r))
            cv.drawPath(pp, _paint(pc(lam))); cv.restore()
        cv.drawCircle(hx, hy, r * 0.3, _paint(_haze(mid, haze, hc)))
        cv.drawCircle(hx - r * 0.08, hy - r * 0.08, r * 0.13, _paint(_haze(np.clip(mid * 1.25, 0, 1), haze, hc)))
    elif kind == "poppy":
        r = S * 0.16
        for j in range(4):
            a = j * 90 + 45 + rng.uniform(-10, 10)
            lam = 0.55 + 0.45 * math.cos(math.radians(a) - la)
            cv.save(); cv.translate(hx, hy); cv.scale(1, 0.7); cv.rotate(a)
            cv.drawCircle(0, -r * 0.55, r * 0.6, _paint(pc(lam), 0.95)); cv.restore()
        cv.drawCircle(hx, hy, r * 0.22, _paint(_haze(mid, haze, hc)))
    elif kind == "tulip":
        r = S * 0.13
        for dx, lam in [(-0.5, 0.55), (0.5, 0.75), (0.0, 1.0)]:
            pp = skia.Path()
            pp.moveTo(hx + dx * r - r * 0.5, hy)
            pp.quadTo(hx + dx * r - r * 0.6, hy - r * 1.6, hx + dx * r, hy - r * 1.9)
            pp.quadTo(hx + dx * r + r * 0.6, hy - r * 1.6, hx + dx * r + r * 0.5, hy); pp.close()
            cv.drawPath(pp, _paint(pc(lam)))
    else:   # bell: drooping bells along the stem tip
        for j in range(3):
            t = 0.55 + j * 0.2
            cxp, cyp = bx + (hx - bx) * t, by - S * t
            r = S * 0.07
            pp = skia.Path()
            pp.moveTo(cxp - r, cyp + r * 1.6); pp.quadTo(cxp - r * 0.9, cyp, cxp, cyp)
            pp.quadTo(cxp + r * 0.9, cyp, cxp + r, cyp + r * 1.6); pp.close()
            cv.drawPath(pp, _paint(pc(0.7 + 0.1 * j)))
    _blit(self, surf, X0, Y0)
    return self


def _fern2d(self, x, y, size=120.0, fronds=7, seed=0, color=None, light=(135, 45), haze=0.0,
            haze_color="#b9c6d3", spread=150.0):
    """Fern clump rooted at (x, y): `fronds` arching fronds with alternating tapered pinnae."""
    s = self.s
    rng = np.random.default_rng(seed)
    pal = [_rgb(c) for c in (color or ["#24401d", "#3d6326", "#6c9038"])]
    hc = _rgb(haze_color)
    L = _light3(light)
    S = size * s
    pad = int(S * 1.3) + 4
    X0, Y0 = int(x * s - pad), int(y * s - pad)
    surf, cv = _skia_surface(2 * pad, pad + 6)
    bx, by = x * s - X0, y * s - Y0
    paint = skia.Paint(AntiAlias=True)
    pr = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap)
    angs = rng.uniform(-spread / 2, spread / 2, fronds)
    for a0 in angs[np.argsort(-np.abs(angs))]:              # outer fronds first (behind)
        Lf = S * rng.uniform(0.7, 1.05) * (1 - abs(a0) / spread * 0.4)
        a = math.radians(a0 - 90)
        curl = rng.uniform(0.9, 1.6) * (1 if a0 > 0 else -1)
        n = 22
        pts = []
        p = np.array([bx, by]); d = np.array([math.cos(a), math.sin(a)])
        for i in range(n):
            pts.append(p.copy())
            ang = math.atan2(d[1], d[0]) + curl * 0.045
            d = np.array([math.cos(ang), math.sin(ang) + 0.03 * i / n])
            d /= np.linalg.norm(d)
            p = p + d * Lf / n
        pts = np.asarray(pts)
        front = abs(a0) < spread * 0.2
        for i in range(2, n - 1):
            u = i / n
            dd = pts[i + 1] - pts[i]
            th = math.degrees(math.atan2(dd[1], dd[0]))
            pl = Lf * 0.2 * math.sin(math.pi * min(u * 1.05 + 0.04, 1)) ** 0.8
            for sgn in (-1, 1):
                lam = (0.55 + 0.3 * sgn * math.sin(math.radians(th)) * L[0] + 0.25 * u
                       - (0 if front else 0.15))
                paint.setColor(_c4(_haze(_shade_col(pal, [lam], rng, 1)[0], haze, hc)))
                cv.save(); cv.translate(float(pts[i, 0]), float(pts[i, 1]))
                cv.rotate(th + sgn * 62 + 90)
                cv.translate(0, -pl * 0.5)
                cv.scale(max(pl * 0.32, 0.8), max(pl, 0.8))
                cv.drawPath(_LEAF, paint)
                cv.restore()
        pr.setColor(_c4(_haze(pal[0] * 0.8, haze, hc)))
        pr.setStrokeWidth(max(0.7, S * 0.012))
        pth = skia.Path(); pth.moveTo(*pts[0])
        for q in pts[1:]:
            pth.lineTo(*q)
        cv.drawPath(pth, pr)
    _blit(self, surf, X0, Y0)
    return self


Layer.tree = _tree2d
Layer.grass = _grass2d
Layer.flower = _flower2d
Layer.fern = _fern2d


# ------------------------------------------------------------------ plants: 3D (sdf)
def tree3d(kind="oak", height=4.0, seed=0, leaf_noise=0.35, twig=0.006, season="summer"):
    """3D tree for an sdf Scene: returns (trunk, foliage) SDFs (trunk base at origin, y up, `height`
    units tall). Branches = tapered capsules culled by bounding spheres (groups of 24);
    foliage = smooth-union blobs displaced by noise3 (None for dead trees / winter)."""
    from .sdf import SDF, noise3
    sp = TREES[kind]
    pos, par, rad = tree_skeleton(kind, seed, 3)
    P = pos * height
    R = rad * height
    n = len(P)
    child = np.zeros(n, int)
    np.add.at(child, par[1:], 1)
    depth = np.zeros(n, int)
    for i in range(1, n):
        depth[i] = depth[par[i]] + 1
    keep = (child != 1) | (depth % 3 == 0)            # merge straight chains into longer capsules
    keep[0] = True
    keep &= (R > twig * height) | (np.arange(n) == 0)
    anc = par.copy()
    for i in range(1, n):
        a = par[i]
        while a > 0 and not keep[a]:
            a = par[a]
        anc[i] = max(a, 0)
    idx = np.nonzero(keep[1:])[0] + 1
    A, B, RA, RB = P[anc[idx]], P[idx], R[anc[idx]], R[idx]
    mid = (A + B) / 2
    leaves_ix = []

    def split(ix):                                    # kd-split into compact groups of <= 12 capsules
        if len(ix) <= 12:
            leaves_ix.append(ix)
            return
        ax = int(np.argmax(np.ptp(mid[ix], 0)))
        o = ix[np.argsort(mid[ix, ax])]
        split(o[:len(o) // 2]); split(o[len(o) // 2:])
    split(np.arange(len(A)))
    groups = []
    for sl in leaves_ix:
        c = np.concatenate([A[sl], B[sl]]).mean(0)
        r = float(max((np.linalg.norm(A[sl] - c, axis=1) + RA[sl]).max(),
                      (np.linalg.norm(B[sl] - c, axis=1) + RB[sl]).max()))
        groups.append((c.astype(F32), r, A[sl].astype(F32), (B[sl] - A[sl]).astype(F32),
                       RA[sl].astype(F32), (RB[sl] - RA[sl]).astype(F32)))
    GC = np.stack([g[0] for g in groups]); GR = np.array([g[1] for g in groups], F32)

    def caps(p):
        D = np.linalg.norm(p[:, None, :] - GC[None], axis=-1)
        T = (D + GR).min(1)                               # upper bound of the true distance
        best = np.full(len(p), np.inf, F32)
        for gi, (c, r, a, ba, ra, dr) in enumerate(groups):
            m = (D[:, gi] - r < np.minimum(T, best))
            if not m.any():
                continue
            q = p[m][:, None, :] - a[None]
            bb = (ba * ba).sum(-1)
            hh = np.clip((q * ba[None]).sum(-1) / np.maximum(bb, 1e-12), 0, 1)
            d = np.linalg.norm(q - ba[None] * hh[..., None], axis=-1) - (ra + dr * hh)
            best[m] = np.minimum(best[m], d.min(1))
        return np.minimum(best, T)

    allp = np.concatenate([A, B])
    c0 = allp.mean(0)
    r0 = float(np.linalg.norm(allp - c0, axis=1).max() + RA.max())
    trunk = SDF(caps, (c0.astype(F32), r0), 0.9)
    if sp.get("leaf") is None or (season == "winter" and kind not in ("pine", "palm")):
        return trunk, None
    tips = P[(child == 0) & (pos[:, 1] > sp.get("trunk", 0.2) * 0.9)]
    cell = height * (0.16 if kind not in ("pine", "palm") else 0.09)
    _, inv = np.unique(np.round(tips / cell), axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    blobs = np.stack([np.bincount(inv, tips[:, k]) / cnt for k in range(3)], 1).astype(F32)
    br = (cell * (0.55 + 0.25 * np.sqrt(cnt / cnt.max()))).astype(F32)
    nz = noise3(1.8 / cell, 2, seed + 9)
    amp = F32(cell * leaf_noise)
    k = F32(cell * 0.35)

    def leaves(p):
        d = np.linalg.norm(p[:, None, :] - blobs[None], axis=-1) - br[None]
        m = d.min(1)
        out = m - k * np.log(np.exp(-(d - m[:, None]) / k).sum(1))     # smooth union (log-sum-exp)
        near = out < amp * 2.5
        if near.any():
            out[near] = out[near] + (nz(p[near]) - 0.5) * 2 * amp
        return out.astype(F32)

    lc = blobs.mean(0)
    lr = float(np.linalg.norm(blobs - lc, axis=1).max() + br.max() + amp * 2)
    return trunk, SDF(leaves, (lc.astype(F32), lr), 0.6)


__all__ = ["tree_skeleton", "tree3d", "TREES", "SEASONS",
           "sun_direction", "sun_color", "sky_radiance", "sky_irradiance", "aerial", "gerstner", "wave_field",
           "scene_sky", "SUN_POWER",
           "heightfield", "erode", "thermal", "terrain", "normals", "ambient_occlusion", "shadow_map", "materials"]
