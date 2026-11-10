"""Progressive Monte Carlo path tracer over sdf.Scene (numpy, CPU, deterministic).

r = c.pathtrace(s, spp=32, bounces=4)   # same Render as c.render: rgb, linear, depth, normal, id, alpha
Diffuse + GGX glossy / metal (VNDF importance sampled), rough dielectric glass (Fresnel, any number of
refractions/reflections, Beer absorption), wrap-lighting SSS, sheen, emissive objects, next-event
estimation + MIS toward sun / point / area lights (area lights sampled over their surface and visible in
reflections), sky, Russian roulette. Jittered adaptive sampling (= anti-aliasing) and an AOV-guided
a-trous denoiser. A scene distance grid accelerates sphere tracing away from surfaces (exact SDF near them).
Randomness is a counter-based hash of (pixel, sample, dimension, seed): results do not depend on threads.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from .core import F32, Layer, CONFIG, color as _col, to_linear
from .fx import tonemap_linear
from .sdf import Render, _norm, _v

_LUM = np.array([0.2126, 0.7152, 0.0722], F32)
_PI = F32(math.pi)
_TETRA = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], F32)


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


# ------------------------------------------------------------------ counter-based RNG
_M1, _M2 = np.uint32(0x7FEB352D), np.uint32(0x846CA68B)


def _hash(x):
    x = x ^ (x >> np.uint32(16))
    x = x * _M1
    x = x ^ (x >> np.uint32(15))
    x = x * _M2
    return x ^ (x >> np.uint32(16))


def _rnd(key, dim):
    h = _hash(key ^ np.uint32((dim * 0x9E3779B9 + 0x632BE5AB) & 0xFFFFFFFF))
    return (h >> np.uint32(8)).astype(F32) * F32(1.0 / 16777216)


def _onb(n):
    s = np.where(n[:, 2] >= 0, F32(1), F32(-1))
    a = -1 / (s + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t1 = np.stack([1 + s * n[:, 0] ** 2 * a, s * b, -s * n[:, 0]], 1)
    t2 = np.stack([b, s + n[:, 1] ** 2 * a, -n[:, 1]], 1)
    return t1.astype(F32), t2.astype(F32)


# ------------------------------------------------------------------ geometry acceleration
class _Accel:
    """Exact SDF evaluation over object subsets + a trilinear distance grid of the opaque objects
    (stored as a 2D atlas, looked up with cv2.remap). Glass and emitters are always evaluated exactly
    (shadow rays skip them)."""

    def __init__(self, scene, cells=3e6, workers=2):
        self.objs = objs = scene.objects
        isx = lambda m: m.glass > 0.5 or m.emit is not None
        self.special = [k for k, (_, m) in enumerate(objs) if isx(m)]
        self.solid = [k for k, (_, m) in enumerate(objs) if not isx(m)]
        self.glass = [k for k in self.special if objs[k][1].glass > 0.5]
        self.unb = [k for k in self.solid if objs[k][0].bounds is None]
        self.all = list(range(len(objs)))
        lip = lambda ks: min([objs[k][0].lip for k in ks] + [1.0])
        self.lip, self.lip_x = lip(self.solid), lip(self.special)
        self.G, self.h = None, F32(0)
        self.C = np.array([o[0].bounds[0] if o[0].bounds is not None else (0, 0, 0) for o in objs], F32).reshape(-1, 3)
        self.R = np.array([o[0].bounds[1] if o[0].bounds is not None else 0 for o in objs], F32)
        bk = [k for k in self.solid if objs[k][0].bounds is not None]
        if bk and cells and cells > 0:
            self._build(bk, int(cells), workers, lip(bk))

    def dist(self, ks, p, want_id=False, scaled=False, cap=None):
        """min SDF over objects ks (bounding-sphere culled). scaled: each object's value times its
        Lipschitz factor (= a lower bound of the true distance, for safe marching steps).
        cap: known upper bound per point (objects whose bound is farther are skipped)."""
        best = np.full(len(p), np.inf, F32) if cap is None else np.array(cap, F32)
        ids = np.full(len(p), -1, np.int32) if want_id else None
        bk = [k for k in ks if self.objs[k][0].bounds is not None]
        if bk:
            C = self.C[bk]
            LB = np.sqrt(np.maximum(_dot(p, p)[:, None] - 2 * (p @ C.T) + (C * C).sum(1), 0)) - self.R[bk]
            col = {k: i for i, k in enumerate(bk)}
        for k in ks:
            sdf = self.objs[k][0]
            if sdf.bounds is not None:
                lb = LB[:, col[k]]
                m = lb < best
                if not m.any():
                    continue
                if m.all():
                    d = np.asarray(sdf.f(p), F32)
                else:
                    d = lb.astype(F32)
                    d[m] = sdf.f(p[m])
            else:
                d = np.asarray(sdf.f(p), F32)
            if scaled and sdf.lip < 1:
                d = d * F32(sdf.lip)
            if want_id:
                ids = np.where(d < best, k, ids)
            best = np.minimum(best, d)
        return (best, ids) if want_id else best

    def _build(self, bk, cells, workers, lipb):
        objs = self.objs
        C = np.array([objs[k][0].bounds[0] for k in bk], F32)
        R = np.array([objs[k][0].bounds[1] for k in bk], F32)
        lo, hi = (C - R[:, None]).min(0), (C + R[:, None]).max(0)
        h0 = float((hi - lo).max()) / 40
        axes = [np.arange(lo[i] + h0 / 2, hi[i], h0, dtype=F32) for i in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        occ = np.abs(self.dist(bk, P)) < F32(0.9 * h0 / lipb)
        if occ.any():      # tight box around the surfaces, padded so no surface touches its faces
            Q = P[occ]
            lo, hi = Q.min(0) - 1.5 * h0, Q.max(0) + 1.5 * h0
        ext = np.maximum(hi - lo, 1e-3)
        h = float((np.prod(ext) / cells) ** (1 / 3))
        dims = np.maximum(np.ceil(ext / h).astype(int) + 1, 2)
        nx, ny, nz = (int(v) for v in dims)
        xs, ys, zs = (lo[i] + np.arange(dims[i], dtype=F32) * F32(h) for i in range(3))
        G = np.empty((nz, ny, nx), F32)
        X, Y = np.meshgrid(xs, ys)
        X, Y = X.ravel(), Y.ravel()
        def job(z):
            pts = np.stack([X, Y, np.full(X.size, zs[z], F32)], 1)
            G[z] = self.dist(self.solid, pts, scaled=True).reshape(ny, nx)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(job, range(nz)))
        tx = max(1, min(32000 // nx, int(math.ceil(math.sqrt(nz * ny / nx)))))
        ty = -(-nz // tx)
        if ty * ny > 32000 or nx > 32000:
            return
        At = np.zeros((ty * ny, tx * nx), F32)
        for z in range(nz):
            At[(z // tx) * ny:(z // tx + 1) * ny, (z % tx) * nx:(z % tx + 1) * nx] = G[z]
        self.G, self.At, self.tx = G, At, tx
        self.lo, self.hi = lo.astype(F32), (lo + (dims - 1) * h).astype(F32)
        self.h, self.dims = F32(h), (nx, ny, nz)

    def _grid(self, q):
        nx, ny, nz = self.dims
        g = (q - self.lo) / self.h
        gx, gy = np.clip(g[:, 0], 0, nx - 1), np.clip(g[:, 1], 0, ny - 1)
        gz = np.clip(g[:, 2], 0, nz - 1.001)
        z0 = gz.astype(np.int32)
        fz = gz - z0
        n = len(q)
        cols = 1024 if n > 1024 else max(n, 1)
        rows = -(-n // cols)
        pad = rows * cols - n
        out = []
        mx = np.zeros(rows * cols, F32)
        my = np.zeros(rows * cols, F32)
        for zz in (z0, z0 + 1):
            mx[:n] = gx + (zz % self.tx) * nx
            my[:n] = gy + (zz // self.tx) * ny
            out.append(cv2.remap(self.At, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()[:n])
        return out[0] + (out[1] - out[0]) * fz

    def query(self, p, cmask=None, jump=None):
        """Opaque objects: (safe step, hit distance, coarse flag). cmask: points allowed the coarse
        grid-only test (their hit distance is then the raw grid value, flagged in the third array)."""
        L = F32(1)
        N = len(p)
        cg = np.zeros(N, bool)
        if self.G is None:
            d = np.abs(self.dist(self.solid, p, scaled=True))
            return d * F32(0.95) * L, d, cg
        stp = np.empty(N, F32)
        hd = np.full(N, np.inf, F32)
        ins = ((p >= self.lo) & (p <= self.hi)).all(1)
        out = ~ins
        h = self.h
        if out.any():
            q = p[out]
            if jump is None:
                e = np.maximum(np.maximum(self.lo - q, q - self.hi), 0)
                bd = np.sqrt(_dot(e, e)) + h * F32(0.5)
            else:      # exact distance along the ray to the grid box (inf once the ray has left it)
                bd = jump[out]
            if self.unb:
                du = np.abs(self.dist(self.unb, q, scaled=True))
                bd = np.minimum(bd, du * F32(0.95) * L)
                hd[out] = du
            stp[out] = bd
        if ins.any():
            q = p[ins]
            a = np.abs(self._grid(q))
            near = a < 3 * h / L
            s = L * a - F32(1.85) * h
            hh = np.full(len(q), np.inf, F32)
            en = near
            if cmask is not None:
                cn = near & cmask[ins]
                s[cn] = np.maximum(a[cn] * L * F32(0.95), h * F32(0.3))
                hh[cn] = a[cn]
                cg[np.flatnonzero(ins)[cn]] = True
                en = near & ~cmask[ins]
            if en.any():
                e = np.abs(self.dist(self.solid, q[en], scaled=True, cap=a[en] + 2 * h))
                s[en] = e * L * F32(0.95)
                hh[en] = e
            stp[ins] = s
            hd[ins] = hh
        return stp, hd, cg


def _march(A, o, d, tmax, eps0, epsk, coarse=False, shadow=False, steps=160, t0=None, cone=None):
    """Sphere-trace rays (o, d) from t0 up to tmax. coarse: grid-only hits once a ray is clear of its
    start. shadow: ignore glass and emitters. Returns (t, hit) [, safe]. cone = (pixk, aperture, focus):
    also return the distance up to which every ray of the pixel cone around (o, d) is in empty space."""
    N = len(o)
    t = np.zeros(N, F32) if t0 is None else np.array(t0, F32)
    if cone is not None:
        safe = np.zeros(N, F32)
        ok = np.ones(N, bool)
    hit = np.zeros(N, bool)
    idx = np.arange(N)
    sp = [] if shadow else A.special
    lx = F32(0.95)
    cmask = np.broadcast_to(np.asarray(coarse, bool), (N,))
    if A.G is not None:     # ray / grid-box slab intersection
        inv = 1 / np.where(np.abs(d) < 1e-9, F32(1e-9), d)
        t1, t2 = (A.lo - o) * inv, (A.hi - o) * inv
        tn = np.maximum(np.minimum(t1, t2).max(1), 0).astype(F32)
        tf = np.maximum(t1, t2).min(1).astype(F32)
        tf = np.where(tf >= tn, tf, F32(-1))
    use_c = A.G is not None and cmask.any()
    if use_c:
        aprev = np.full(N, -np.inf, F32)
        hc = F32(0.5) * A.h
    for _ in range(steps):
        if idx.size == 0:
            break
        ti = t[idx]
        p = o[idx] + d[idx] * ti[:, None]
        jump = None
        if A.G is not None:
            tn_, tf_ = tn[idx], tf[idx]
            jump = np.where(ti < tn_, tn_ - ti + F32(1e-4), np.where(ti > tf_, F32(1e9), F32(1e-4)))
        st, hd, cg = A.query(p, cmask[idx] if use_c else None, jump)
        eps = np.maximum(eps0[idx], ti * F32(epsk))
        if use_c and cg.any():
            # coarse grid test: close to a surface AND getting closer (rays leaving their start surface pass)
            hcg = cg & (hd < hc) & (hd < aprev[idx] - F32(1e-5))
            aprev[idx] = np.where(cg, hd, F32(-np.inf))
            h = np.where(cg, hcg, hd < eps)
        else:
            h = hd < eps
        if sp:
            e = np.abs(A.dist(sp, p, scaled=True, cap=st * F32(1.06) + F32(1e-4)))
            st = np.minimum(st, e * lx)
            h |= e < eps
        if cone is not None:
            dl = F32(cone[0]) * ti + F32(cone[1]) * np.abs(1 - ti / F32(cone[2]))
            good = ok[idx] & (st > 3 * dl + F32(2e-3))
            safe[idx[good]] = ti[good]
            ok[idx[~good]] = False
        hit[idx[h]] = True
        t[idx] = ti + np.where(h, F32(0), st)
        idx = idx[~h & (t[idx] < tmax[idx])]
    return (t, hit) if cone is None else (t, hit, safe * F32(0.97))


def _march_set(A, ks, o, d, tmax, steps=96):
    """Exact march against a few objects (glass in shadow rays). Returns hit, ids."""
    N = len(o)
    t = np.full(N, F32(1e-3))
    hit = np.zeros(N, bool)
    idx = np.arange(N)
    for _ in range(steps):
        if idx.size == 0:
            break
        e = np.abs(A.dist(ks, o[idx] + d[idx] * t[idx][:, None], scaled=True))
        h = e < 2e-4
        hit[idx[h]] = True
        t[idx] += e * F32(0.95)
        idx = idx[~h & (t[idx] < tmax[idx])]
    return hit, t


def _normal(A, p, eps):
    e = eps[:, None]
    n = 0
    for k in _TETRA:
        n = n + k * A.dist(A.all, p + k * e, cap=eps * 12 + F32(0.02))[:, None]
    return _norm(n).astype(F32)


# ------------------------------------------------------------------ lights
def _pattern(kind, u, v):
    if kind is None:
        return np.ones(len(u), F32)
    if callable(kind):
        return np.clip(np.asarray(kind(u, v), F32), 0, None)
    if kind == "window":   # 2 x 3 panes, mullions and frame
        bu = np.minimum(np.abs(u - 0.5), np.minimum(u, 1 - u))
        bv = np.minimum(np.minimum(np.abs(v - 1 / 3), np.abs(v - 2 / 3)), np.minimum(v, 1 - v))
        bar = np.clip((np.minimum(bu - 0.022, bv - 0.018)) / 0.004, 0, 1)
        return (bar * (0.85 + 0.3 * v)).astype(F32)
    # softbox: bright centre, soft fall-off to the edges
    e = (1 - np.abs(2 * u - 1) ** 6) * (1 - np.abs(2 * v - 1) ** 6)
    return (0.55 + 0.45 * e).astype(F32)


def _prep_lights(scene):
    out = []
    for L in scene.lights:
        if L["kind"] == "sun":
            ang = float(np.clip(L["soft"], 0.005, 0.3))
            cm = math.cos(ang)
            out.append(dict(kind=0, dir=_norm(_v(L["dir"])), col=_v(L["col"]), cosm=cm, om=2 * math.pi * (1 - cm)))
        elif L["kind"] == "point":
            out.append(dict(kind=1, pos=_v(L["pos"]), col=_v(L["col"]), r=float(L["soft"])))
        else:
            sx, sy = L.get("size", (1.0, 1.0))
            pat = L.get("pattern")
            g = (np.arange(64, dtype=F32) + 0.5) / 64
            U, V = np.meshgrid(g, g)
            mean = float(_pattern(pat, U.ravel(), V.ravel()).mean()) or 1.0
            rad = _v(L["rad"]) if "rad" in L else _v(L["col"]) * len(L.get("pts", [0]))
            nv = _norm(_v(L.get("nv", (0, -1, 0))))
            out.append(dict(kind=2, pos=_v(L["pos"]), t1=_v(L["t1"]), t2=_v(L["t2"]), nv=nv, sx=sx, sy=sy,
                            area=sx * sy, Le=rad / F32(sx * sy * mean), pat=pat, vis=bool(L.get("visible"))))
    return out


# ------------------------------------------------------------------ materials & BSDF
def _mats(S, p, ng, ids, fw, V, bump=True):
    N = len(p)
    z = lambda: np.zeros(N, F32)
    M = dict(alb=np.zeros((N, 3), F32), rough=z(), metal=z(), glass=z(), ior=np.ones(N, F32), sss=z(), sheen=z(),
             spec=z(), emit=np.zeros((N, 3), F32), sssc=np.ones((N, 3), F32), shc=np.zeros((N, 3), F32),
             sig=np.zeros((N, 3), F32), n=ng.copy())
    has_emit = False
    for k in np.unique(ids):
        m = ids == k
        sdf, mat = S.scene.objects[max(int(k), 0)]
        ev = mat.eval(p[m], ng[m], sdf, fw[m], bump)
        a = ev["alb"]
        M["alb"][m], M["rough"][m], M["n"][m] = a, ev["rough"], ev["n"]
        for key in ("metal", "glass", "ior", "sss", "sheen", "spec"):
            M[key][m] = getattr(mat, key)
        if mat.emit is not None:
            M["emit"][m] = to_linear(_col(mat.emit)[:3]) * F32(mat.power)
            has_emit = True
        if mat.sss:
            if mat.sss_color is not None:
                sc = np.broadcast_to(to_linear(_col(mat.sss_color)[:3]).astype(F32), a.shape)
            else:
                sc = a * a
            M["sssc"][m] = sc / np.maximum(sc.max(1, keepdims=True), 1e-4)
        if mat.sheen:
            M["shc"][m] = (to_linear(_col(mat.sheen_color)[:3]) if mat.sheen_color is not None
                           else np.sqrt(a)) * F32(mat.sheen)
        if mat.glass:
            M["sig"][m] = -np.log(np.clip(a, 1e-3, 1)) / F32(0.1)
    # keep the bumped normal on the viewer's side of the geometric one
    nb = M["n"]
    bad = _dot(nb, V) * np.sign(_dot(ng, V)) < 0.02
    nb[bad] = ng[bad]
    M["ng"] = ng
    alb, metal = M["alb"], M["metal"][:, None]
    M["a"] = np.clip(M["rough"], 0.02, 1) ** 2
    M["F0"] = F32(0.08) * M["spec"][:, None] * (1 - metal) + alb * metal
    ndv = np.clip(np.abs(_dot(nb, V)), 1e-4, 1)
    fd = F32(0.04) + F32(0.96) * (1 - ndv) ** 5
    M["kd"] = ((1 - metal[:, 0]) * (1 - fd))[:, None]
    F = M["F0"] + (1 - M["F0"]) * ((1 - ndv) ** 5)[:, None]
    Fs, Fd = F @ _LUM, (alb * M["kd"]) @ _LUM
    M["ps"] = np.where(Fd < 1e-4, F32(1), np.clip(Fs / (Fs + Fd + 1e-6), 0.1, 1)).astype(F32)
    M["nonspec"] = (M["glass"] < 0.5) & ((M["metal"] < 0.5) | (M["rough"] > 0.3))
    M["has_emit"] = has_emit
    return M


def _sub(M, m):
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in M.items()}


def _bsdf(M, V, l, nee=False):
    """f*cos (N,3) and the sampling pdf (N,) for light direction l."""
    n = M["n"]
    ndl = _dot(n, l)
    ndv = np.clip(_dot(n, V), 1e-4, 1)
    cl = np.clip(ndl, 0, 1)
    a2 = M["a"] ** 2
    hv = _norm(l + V)
    ndh = np.clip(_dot(n, hv), 0, 1)
    vdh = np.clip(_dot(V, hv), 0, 1)
    F0 = M["F0"]
    F = F0 + (1 - F0) * ((1 - vdh) ** 5)[:, None]
    D = a2 / (_PI * (ndh * ndh * (a2 - 1) + 1) ** 2)
    G1v = 2 * ndv / (ndv + np.sqrt(a2 + (1 - a2) * ndv * ndv))
    G1l = 2 * cl / (cl + np.sqrt(a2 + (1 - a2) * cl * cl) + 1e-9)
    spec = (D * G1v * G1l / (4 * ndv))[:, None] * F
    c3 = cl[:, None]
    if nee and M["sss"].any():
        s = M["sss"][:, None]
        W = np.clip((ndl + 0.5) / 1.5, 0, 1)[:, None] ** 1.5
        Tt = M["sssc"] + (1 - M["sssc"]) * c3
        c3 = (1 - s) * c3 + s * W * Tt
    fc = M["alb"] * M["kd"] / _PI * c3 + spec
    if M["sheen"].any():
        inv = F32(1 / 0.35)
        Dc = (2 + inv) * np.maximum(1 - ndh * ndh, 0) ** (inv / 2) / (2 * _PI)
        Vc = 1 / (4 * (cl + ndv - cl * ndv) + 1e-3)
        fc = fc + M["shc"] * (Dc * Vc * cl)[:, None]
    ps = M["ps"]
    pdf = ps * G1v * D / (4 * ndv) + (1 - ps) * cl / _PI
    return fc.astype(F32), pdf.astype(F32)


def _vndf(Vl, a, u1, u2):
    """GGX visible-normal sample (Heitz 2018), local frame, isotropic alpha a."""
    vh = _norm(np.stack([a * Vl[:, 0], a * Vl[:, 1], np.maximum(Vl[:, 2], 1e-4)], 1))
    ls = vh[:, 0] ** 2 + vh[:, 1] ** 2
    inv = 1 / np.sqrt(np.maximum(ls, 1e-12))
    T1 = np.where((ls > 1e-7)[:, None], np.stack([-vh[:, 1] * inv, vh[:, 0] * inv, np.zeros_like(ls)], 1),
                  np.array([1, 0, 0], F32))
    T2 = np.cross(vh, T1)
    r = np.sqrt(u1)
    ph = 2 * _PI * u2
    p1, p2 = r * np.cos(ph), r * np.sin(ph)
    s = 0.5 * (1 + vh[:, 2])
    p2 = (1 - s) * np.sqrt(np.maximum(1 - p1 * p1, 0)) + s * p2
    Nh = p1[:, None] * T1 + p2[:, None] * T2 + np.sqrt(np.maximum(1 - p1 * p1 - p2 * p2, 0))[:, None] * vh
    return _norm(np.stack([a * Nh[:, 0], a * Nh[:, 1], np.maximum(Nh[:, 2], 1e-6)], 1))


def _to_world(v, t1, t2, n):
    return v[:, :1] * t1 + v[:, 1:2] * t2 + v[:, 2:3] * n


def _sample_opaque(M, V, u1, u2, u3):
    n = M["n"]
    t1, t2 = _onb(n)
    spec = u3 < M["ps"]
    Vl = np.stack([_dot(V, t1), _dot(V, t2), _dot(V, n)], 1)
    hl = _vndf(Vl, M["a"], u1, u2)
    h = _to_world(hl, t1, t2, n)
    ls = 2 * _dot(V, h)[:, None] * h - V
    r = np.sqrt(u1)
    ph = 2 * _PI * u2
    ld = _to_world(np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(np.maximum(1 - u1, 0))], 1), t1, t2, n)
    l = _norm(np.where(spec[:, None], ls, ld)).astype(F32)
    fc, pdf = _bsdf(M, V, l)
    ok = (_dot(l, n) > 1e-4) & (_dot(l, M["ng"]) * np.sign(_dot(V, M["ng"])) > 1e-4) & (pdf > 1e-7)
    w = np.where(ok[:, None], fc / np.maximum(pdf, 1e-7)[:, None], 0).astype(F32)
    return l, w, pdf, ok, spec


def _sample_glass(M, d, u1, u2, u3):
    n = M["n"]
    ent = _dot(d, M["ng"]) < 0
    nn = np.where(ent[:, None], n, -n)
    ior = M["ior"]
    eta = np.where(ent, 1 / ior, ior).astype(F32)
    a = M["a"]
    rough = a > 2e-3
    m = nn
    if rough.any():
        t1, t2 = _onb(nn)
        V = -d
        Vl = np.stack([_dot(V, t1), _dot(V, t2), _dot(V, nn)], 1)
        mh = _to_world(_vndf(Vl, a, u1, u2), t1, t2, nn)
        m = np.where(rough[:, None], mh, nn)
    cosi = np.clip(-_dot(d, m), 1e-4, 1)
    k = 1 - eta * eta * (1 - cosi * cosi)
    cost = np.sqrt(np.maximum(k, 0))
    rs = (eta * cosi - cost) / (eta * cosi + cost)
    rp = (cosi - eta * cost) / (cosi + eta * cost)
    F = np.where(k <= 0, F32(1), 0.5 * (rs * rs + rp * rp))
    refl = u3 < F
    lr = d + 2 * cosi[:, None] * m
    lt = eta[:, None] * d + (eta * cosi - cost)[:, None] * m
    l = _norm(np.where(refl[:, None], lr, lt)).astype(F32)
    side = _dot(l, nn)
    ok = np.where(refl, side > 0, side < 0)
    w = np.where(ok, F32(1), F32(0))
    if rough.any():   # Smith G1 of the outgoing direction (VNDF weight)
        c = np.clip(np.abs(side), 1e-4, 1)
        a2 = a * a
        w = w * np.where(rough, 2 * c / (c + np.sqrt(a2 + (1 - a2) * c * c)), 1)
    # absorption state: entering -> this object's sigma, leaving -> none, reflection -> unchanged
    return l, w.astype(F32), refl, ent, ok, F, _norm(lt).astype(F32)


def _clampc(c, on, clamp):
    if clamp is None or not on.any():
        return c
    y = c @ _LUM
    s = np.where(on & (y > clamp), clamp / np.maximum(y, 1e-9), 1).astype(F32)
    return c * s[:, None]


def _nee(S, M, p, ng, V, key, b):
    """Next-event estimation: one light sample per vertex, MIS-weighted against BSDF sampling."""
    m = len(p)
    nl = len(S.lights)
    li = np.minimum((_rnd(key, 40 + b * 16) * nl).astype(np.int32), nl - 1)
    u1, u2, u3 = _rnd(key, 41 + b * 16), _rnd(key, 42 + b * 16), _rnd(key, 43 + b * 16)
    ldir = np.zeros((m, 3), F32)
    ldir[:, 1] = 1
    dist = np.full(m, 1e4, F32)
    Li = np.zeros((m, 3), F32)
    pl = np.zeros(m, F32)
    for j, L in enumerate(S.lights):
        s = li == j
        if not s.any():
            continue
        if L["kind"] == 0:
            ct = 1 - u1[s] * (1 - L["cosm"])
            st = np.sqrt(np.maximum(1 - ct * ct, 0))
            ph = 2 * _PI * u2[s]
            D = np.broadcast_to(L["dir"], (s.sum(), 3)).astype(F32)
            t1, t2 = _onb(D)
            ldir[s] = _norm(t1 * (st * np.cos(ph))[:, None] + t2 * (st * np.sin(ph))[:, None] + D * ct[:, None])
            Li[s] = L["col"] * F32(nl)
            pl[s] = 1 / (L["om"] * nl)
        elif L["kind"] == 1:
            z = 1 - 2 * u1[s]
            r = np.sqrt(np.maximum(1 - z * z, 0))
            ph = 2 * _PI * u2[s]
            q = L["pos"] + np.stack([r * np.cos(ph), r * np.sin(ph), z], 1) * (F32(L["r"]) * np.cbrt(u3[s]))[:, None]
            v = q - p[s]
            dd = np.sqrt(_dot(v, v))
            ldir[s] = v / dd[:, None]
            dist[s] = dd
            Li[s] = L["col"] * (F32(nl) / np.maximum(dd * dd, 1e-4))[:, None]
        else:
            q = L["pos"] + L["t1"] * ((u1[s] - 0.5) * L["sx"])[:, None] + L["t2"] * ((u2[s] - 0.5) * L["sy"])[:, None]
            v = q - p[s]
            dd = np.sqrt(_dot(v, v))
            ln = v / dd[:, None]
            cosl = np.clip(-(ln @ L["nv"]), 0, 1)
            ldir[s] = ln
            dist[s] = dd
            Li[s] = L["Le"] * (_pattern(L["pat"], u1[s], u2[s]) * L["area"] * cosl * nl / np.maximum(dd * dd, 1e-4))[:, None]
            pl[s] = np.where(cosl > 1e-4, dd * dd / (L["area"] * np.maximum(cosl, 1e-4) * nl), 0)
    fc, pb = _bsdf(M, V, ldir, nee=True)
    wm = np.where(pl > 0, pl * pl / (pl * pl + pb * pb + 1e-12), 1).astype(F32)
    c = fc * Li * wm[:, None]
    act = c.max(1) > 1e-7
    if act.any():
        a = np.flatnonzero(act)
        sd = np.sign(_dot(ldir[a], ng[a]))
        off = F32(1e-3)
        o = p[a] + ng[a] * (sd * off)[:, None]
        tmax = dist[a] - 2 * off
        eps0 = np.full(len(a), F32(1e-4))
        t, hit = _march(S.A, o, ldir[a], tmax, eps0, S.pixang * 0.5, coarse=True, shadow=True, steps=S.steps)
        vis = np.repeat((~hit).astype(F32)[:, None], 3, 1)
        if S.A.glass:
            g = ~hit
            if g.any():
                gi = np.flatnonzero(g)
                gh, _ = _march_set(S.A, S.A.glass, o[gi], ldir[a][gi], tmax[gi])
                if gh.any():
                    vis[gi[gh]] *= S.glass_tr
        c[a] *= vis
        c[~act] = 0
    else:
        c[:] = 0
    return c


def _seg_emission(S, o, d, tt, pb, delta, camera=False):
    """Radiance picked up along ray segments (o, d, length tt; inf = escaped): area lights crossed
    (one-sided) and, for escaped rays, sky + sun discs. MIS-weighted against light sampling unless the
    previous vertex was specular (delta)."""
    N = len(o)
    Le = np.zeros((N, 3), F32)
    nl = max(len(S.lights), 1)
    for L_ in S.lights:
        if L_["kind"] != 2 or (camera and not L_["vis"]):
            continue
        den = d @ L_["nv"]
        fr = den < -1e-6
        tl = ((L_["pos"] - o) @ L_["nv"]) / np.where(fr, den, F32(-1))
        ok = fr & (tl > 1e-4) & (tl < tt)
        if not ok.any():
            continue
        a = np.flatnonzero(ok)
        q = o[a] + d[a] * tl[a][:, None] - L_["pos"]
        uu, vv = q @ L_["t1"] / F32(L_["sx"]) + 0.5, q @ L_["t2"] / F32(L_["sy"]) + 0.5
        inr = (uu >= 0) & (uu <= 1) & (vv >= 0) & (vv <= 1)
        a, uu, vv = a[inr], uu[inr], vv[inr]
        if not a.size:
            continue
        plp = tl[a] ** 2 / (L_["area"] * -den[a] * nl)
        wm = np.where(delta[a], F32(1), pb[a] ** 2 / (pb[a] ** 2 + plp * plp + 1e-12))
        Le[a] += L_["Le"] * (_pattern(L_["pat"], uu, vv) * wm)[:, None]
    ms = ~np.isfinite(tt)
    if ms.any():
        a = np.flatnonzero(ms)
        dm = d[a]
        E = S.scene._env(dm)
        for L_ in S.lights:
            if L_["kind"] != 0:
                continue
            inc = dm @ L_["dir"] >= L_["cosm"]
            if inc.any():
                plp = F32(1 / (L_["om"] * nl))
                wm = np.where(delta[a], F32(1), pb[a] ** 2 / (pb[a] ** 2 + plp * plp))
                E = E + (L_["col"] / F32(L_["om"])) * (inc * wm)[:, None]
        Le[a] += E
    return Le


# ------------------------------------------------------------------ path integrator (one ray batch)
def _paths(S, pix, smp):
    N = len(pix)
    W, H = S.W, S.H
    pk = _hash(pix.astype(np.uint32) ^ np.uint32(S.seed_mix))
    key = _hash(pk ^ _hash(smp.astype(np.uint32) * np.uint32(0x2545F491) + np.uint32(0x1234567)))
    sm = smp.astype(np.float64)
    jx = ((sm * 0.7548776662 + _rnd(pk, 0)) % 1).astype(F32)
    jy = ((sm * 0.5698402910 + _rnd(pk, 1)) % 1).astype(F32)
    x = (pix % W).astype(F32) + jx
    y = (pix // W).astype(F32) + jy
    u = (x / W * 2 - 1) * S.th * W / H
    v = (1 - y / H * 2) * S.th
    d = _norm(S.fwd + u[:, None] * S.right + v[:, None] * S.up).astype(F32)
    o = np.broadcast_to(S.pos, (N, 3)).astype(F32)
    if S.aperture > 0:
        r = np.sqrt(_rnd(key, 2)) * F32(S.aperture)
        ph = 2 * _PI * _rnd(key, 3)
        fp = S.pos + d * (F32(S.focus) / (d @ S.fwd))[:, None]
        o = (S.pos + S.right * (r * np.cos(ph))[:, None] + S.up * (r * np.sin(ph))[:, None]).astype(F32)
        d = _norm(fp - o).astype(F32)
    L = np.zeros((N, 3), F32)
    T = np.ones((N, 3), F32)
    sig = np.zeros((N, 3), F32)
    wg = np.zeros(N, F32)               # geometric cone width (pixel footprint)
    wt = np.zeros(N, F32)               # texture footprint width (grows with roughness)
    spr = np.full(N, S.pixang, F32)
    lpdf = np.zeros(N, F32)
    ldelta = np.ones(N, bool)
    dif = np.zeros(N, bool)             # path went through a rough bounce (-> clamp fireflies)
    aov_a = np.ones((N, 3), F32)
    aov_n = np.zeros((N, 3), F32)
    aov_done = np.zeros(N, bool)
    hit0 = np.zeros(N, bool)
    t0 = np.full(N, np.inf, F32)
    alive = np.arange(N)
    nl = max(len(S.lights), 1)
    depth = np.zeros(N, np.int32)       # non-glass vertices so far (glass hops do not use up bounces)
    maxb = S.bounces + 8
    for b in range(maxb + 1):
        if alive.size == 0:
            break
        oa, da = o[alive], d[alive]
        eps0 = np.clip(wg[alive] * F32(0.5), F32(1e-4), F32(2e-4) if b else F32(1e9))
        t, hit = _march(S.A, oa, da, np.full(len(alive), S.max_dist, F32), eps0, S.pixang * 0.5,
                        coarse=dif[alive] if b >= 1 else False, steps=S.steps,
                        t0=S.safe[pix[alive]] if b == 0 else None)
        tt = np.where(hit, t, np.inf).astype(F32)
        sg = sig[alive]
        ins = sg.max(1) > 0
        if ins.any():
            T[alive[ins]] *= np.exp(-sg[ins] * np.minimum(t[ins], 5)[:, None])
        Ta = T[alive]
        cl_on = dif[alive]
        Le = _seg_emission(S, oa, da, tt, lpdf[alive], ldelta[alive], b == 0)
        L[alive] += _clampc(Ta * Le, cl_on, S.clamp)
        ms = ~hit
        if ms.any():
            a = np.flatnonzero(ms)
            nd = a[~aov_done[alive[a]]]
            aov_done[alive[nd]] = True
        if b == 0:
            hit0[alive] = hit
            t0[alive] = tt
        if not hit.any():
            break
        hs = np.flatnonzero(hit)
        hidx = alive[hs]
        dh = da[hs]
        th_ = t[hs]
        ph = oa[hs] + dh * th_[:, None]
        wv = wg[hidx] + th_ * F32(S.pixang)
        wtx = wt[hidx] + th_ * spr[hidx]
        epsn = np.clip(wv * F32(0.5), F32(2e-4), F32(1.5e-3))
        ng = _normal(S.A, ph, epsn)
        sd, ids = S.A.dist(S.A.all, ph, want_id=True, cap=epsn * 8 + F32(0.03))
        ph = (ph - ng * np.clip(sd, -0.05, 0.05)[:, None]).astype(F32)
        V = -dh
        cosv = np.abs(_dot(ng, V))
        M = _mats(S, ph, ng, ids, wtx / np.sqrt(np.maximum(cosv, 0.15)), V, b < 2)
        if M["has_emit"]:
            L[hidx] += _clampc(T[hidx] * M["emit"], dif[hidx], S.clamp)
        base = 100 + b * 16
        isg = _rnd(key[hidx], base) < M["glass"]
        term = (~isg & (depth[hidx] >= S.bounces)) | (b == maxb)
        rec = ~aov_done[hidx] & (M["nonspec"] | term)
        if rec.any():
            r = hidx[rec]
            aov_a[r] = np.clip(T[r] * M["alb"][rec], 0, 1)
            aov_n[r] = M["n"][rec]
            aov_done[r] = True
        if term.all():
            break
        depth[hidx[~isg]] += 1
        isg &= ~term
        ne = ~isg & ~term
        if ne.any() and S.lights:
            q = np.flatnonzero(ne)
            c = _nee(S, _sub(M, q), ph[q], ng[q], V[q], key[hidx[q]], b)
            L[hidx[q]] += _clampc(T[hidx[q]] * c, dif[hidx[q]], S.clamp)
        u1, u2, u3 = _rnd(key[hidx], base + 1), _rnd(key[hidx], base + 2), _rnd(key[hidx], base + 3)
        nl_dir = np.zeros((len(hidx), 3), F32)
        wgt = np.zeros((len(hidx), 3), F32)
        okv = np.zeros(len(hidx), bool)
        pdf = np.zeros(len(hidx), F32)
        if ne.any():
            q = np.flatnonzero(ne)
            l, w, pd, ok, spec = _sample_opaque(_sub(M, q), V[q], u1[q], u2[q], u3[q])
            nl_dir[q], wgt[q], okv[q], pdf[q] = l, w, ok, pd
            a_ = M["a"][q]
            rough_b = ~spec | (a_ > 0.1)
            dif[hidx[q]] |= rough_b
            spr[hidx[q]] = np.maximum(spr[hidx[q]], np.where(spec, a_ * 0.5, F32(0.35)))
            ldelta[hidx[q]] = False
        if isg.any():
            q = np.flatnonzero(isg)
            Mg = _sub(M, q)
            l, w, refl, ent, ok, F, lt = _sample_glass(Mg, dh[q], u1[q], u2[q], u3[q])
            # smooth glass seen through a specular chain: if the mirror direction escapes, add its
            # light now (weight F) and always refract (weight 1-F) -> far less noise than a coin flip
            sm = ldelta[hidx[q]] & (Mg["a"] <= 2e-3) & (F < 0.999) & (b < 3)
            if sm.any():
                k = np.flatnonzero(sm)
                lr = _norm(dh[q][k] - 2 * _dot(dh[q][k], Mg["n"][k])[:, None] * Mg["n"][k]).astype(F32)
                sgn = np.sign(_dot(lr, ng[q][k]))
                orr = ph[q][k] + ng[q][k] * (sgn * F32(6e-4))[:, None]
                tr_, hr_ = _march(S.A, orr, lr, np.full(len(k), S.max_dist, F32),
                                  np.maximum(F32(1e-4), wv[q][k] * F32(0.5)), S.pixang * 0.5, steps=S.steps)
                esc = ~hr_
                if esc.any():
                    ke = k[esc]
                    Le = _seg_emission(S, orr[esc], lr[esc], np.full(esc.sum(), np.inf, F32),
                                       np.zeros(esc.sum(), F32), np.ones(esc.sum(), bool))
                    hq_ = hidx[q][ke]
                    L[hq_] += T[hq_] * Le * F[ke][:, None]
                    l[ke] = lt[ke]
                    refl[ke] = False
                    w[ke] = np.where(F[ke] < 1, 1 - F[ke], 0)
                    ok[ke] = _dot(lt[ke], np.where(ent[ke][:, None], Mg["n"][ke], -Mg["n"][ke])) < 0
            nl_dir[q], okv[q] = l, ok
            wgt[q] = w[:, None]
            hq = hidx[q]
            tr = ~refl
            sig[hq[tr & ent]] = Mg["sig"][tr & ent]
            sig[hq[tr & ~ent]] = 0
            ldelta[hq] = True
        T[hidx] *= wgt
        side = np.sign(_dot(nl_dir, ng))
        off = np.full(len(hidx), F32(6e-4))      # < half the thinnest glass wall, > next-segment eps
        o[hidx] = ph + ng * (side * off)[:, None]
        d[hidx] = nl_dir
        wg[hidx] = wv
        wt[hidx] = wtx
        lpdf[hidx] = pdf
        surv = okv & (T[hidx].max(1) > 1e-5) & ~term
        if b >= 2:
            qv = np.clip(T[hidx].max(1), 0.05, 0.95)
            rr = _rnd(key[hidx], base + 4) < qv
            T[hidx] /= qv[:, None]
            surv &= rr
        alive = hidx[surv]
    if S.fog is not None:
        fc, dens = S.fog
        f = np.exp(-dens * np.where(hit0, t0, 0))[:, None]
        L = np.where(hit0[:, None], L * f + fc * (1 - f), L)
    return L, aov_a, aov_n, hit0


# ------------------------------------------------------------------ denoiser
def denoise(img, albedo=None, normal=None, depth=None, var=None, iters=5, sigma_l=4.0, sigma_n=64.0):
    """Edge-avoiding a-trous wavelet filter (SVGF-style, spatial) for a noisy linear HDR image (H×W×3).
    Guides (optional, H×W): albedo (textures are kept: the filter runs on img / albedo), normal, depth,
    var = per-pixel variance of the mean luminance (else estimated locally)."""
    img = np.asarray(img, F32)
    H, W = img.shape[:2]
    A = (np.asarray(albedo, F32) if albedo is not None else np.ones_like(img)) + F32(0.02)
    I = img / A
    Y = I @ _LUM
    if var is None:
        m = cv2.blur(Y, (3, 3))
        v = np.maximum(cv2.blur(Y * Y, (3, 3)) - m * m, 0)
    else:
        v = np.asarray(var, F32) / np.maximum(A @ _LUM, 0.02) ** 2
    v = cv2.GaussianBlur(v.astype(F32), (3, 3), 0)
    Z = np.zeros((H, W), F32) if depth is None else np.where(np.isfinite(depth), depth, 1e4).astype(F32)
    gz = np.maximum(np.abs(np.gradient(Z)[0]), np.abs(np.gradient(Z)[1])).astype(F32)
    gz = np.minimum(gz, 1e3)
    Nn = None if normal is None else np.asarray(normal, F32)
    k1 = (1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16)
    sa = F32(1 / 0.004)
    for it in range(iters):
        s = 2 ** it
        pd = 2 * s
        P = lambda a: np.pad(a, ((pd, pd), (pd, pd)) + ((0, 0),) * (a.ndim - 2), mode="edge")
        Ip, Yp, Vp, Zp, Ap = P(I), P(Y), P(v), P(Z), P(A)
        Np = P(Nn) if Nn is not None else None
        acc = np.zeros_like(I)
        ws = np.zeros((H, W), F32)
        va = np.zeros((H, W), F32)
        sl = F32(sigma_l) * np.sqrt(v) + F32(1e-3)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ys, xs = pd + dy * s, pd + dx * s
                sl_ = (slice(ys, ys + H), slice(xs, xs + W))
                if dx == 0 and dy == 0:
                    w = np.full((H, W), F32(k1[2] * k1[2]))
                else:
                    e = np.abs(Y - Yp[sl_]) / sl
                    e += np.abs(Z - Zp[sl_]) / (gz * F32(s * math.hypot(dx, dy)) + F32(1e-3) * Z + F32(1e-4))
                    da = A - Ap[sl_]
                    e += (da * da).sum(2) * sa
                    w = F32(k1[dy + 2] * k1[dx + 2]) * np.exp(-e)
                    if Np is not None:
                        w *= np.clip((Nn * Np[sl_]).sum(2), 0, 1) ** F32(sigma_n)
                acc += w[..., None] * Ip[sl_]
                ws += w
                va += w * w * Vp[sl_]
        I = acc / ws[..., None]
        v = va / (ws * ws)
        Y = I @ _LUM
    return (I * A).astype(F32)


_atrous = denoise


# ------------------------------------------------------------------ driver
class _S:
    pass


def _center_pass(S, workers, only_safe=False):
    """Deterministic pinhole ray per pixel centre: depth (z), normal, id for the Render and the guides,
    + the empty-space distance shared by all jittered rays of the pixel (primary rays start there)."""
    W, H = S.W, S.H
    o, d, _ = S.scene.rays(W, H)
    N = len(d)
    safe = np.zeros(N, F32)
    cone = (S.pixang * 0.75, S.aperture, max(S.focus, 1e-3))
    dep = np.full(N, np.inf, F32)
    nrm = np.zeros((N, 3), F32)
    ids = np.full(N, -1, np.int32)
    parts = [(i, min(N, i + 16384)) for i in range(0, N, 16384)]
    def job(ab):
        a, b = ab
        oo, dd = np.ascontiguousarray(o[a:b]), np.ascontiguousarray(d[a:b])
        t, hit, safe[a:b] = _march(S.A, oo, dd, np.full(b - a, S.max_dist, F32), np.full(b - a, F32(1e-4)),
                                   S.pixang * 0.5, steps=S.steps, cone=cone)
        if hit.any() and not only_safe:
            h = np.flatnonzero(hit)
            p = oo[h] + dd[h] * t[h][:, None]
            nrm[a + h] = _normal(S.A, p, np.maximum(F32(2e-4), t[h] * F32(S.pixang * 0.5)))
            ids[a + h] = S.A.dist(S.A.all, p, want_id=True)[1]
            dep[a + h] = t[h] * (dd[h] @ S.fwd)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, parts))
    return dep.reshape(H, W), nrm.reshape(H, W, 3), ids.reshape(H, W), safe


def pathtrace(scene, w=640, h=400, spp=32, bounces=4, denoise=True, adaptive=True, seed=0, exposure=0.0,
              tonemap="agx", look=None, aperture=0.0, focus=None, clamp=20.0, threads=None, grid=3e6,
              verbose=False):
    """Path-trace a Scene -> Render (rgb sRGB, linear, alpha, depth, normal, id; + .spp map, .time s).
    spp = average samples per pixel (adaptive: more on edges / noisy pixels, jittered = anti-aliasing);
    bounces = max path vertices after the first hit; denoise = AOV-guided a-trous filter;
    aperture = lens radius (scene units, 0 = pinhole) focused at `focus` (distance, (x, y) pixel or
    None = camera target); clamp = max luminance of indirect samples (fireflies); grid = distance-grid cells."""
    t_start = time.time()
    if not scene.lights:
        scene.sun()
    if CONFIG.get("preview"):
        spp, bounces, grid = max(4, spp // 4), min(bounces, 3), min(grid or 0, 1e6)
    W, H = int(w), int(h)
    P = W * H
    workers = threads or CONFIG.get("threads") or os.cpu_count() or 2
    S = _S()
    S.scene, S.W, S.H, S.bounces, S.clamp = scene, W, H, int(bounces), clamp
    S.A = _Accel(scene, grid, workers)
    cam = scene.cam
    S.pos = _v(cam["pos"])
    S.fwd = _norm(_v(cam["target"]) - S.pos)
    S.right = _norm(np.cross(S.fwd, cam["up"])).astype(F32)
    S.up = np.cross(S.right, S.fwd).astype(F32)
    S.th = F32(math.tan(math.radians(cam["fov"]) / 2))
    S.pixang = float(2 * S.th / H)
    S.max_dist, S.steps, S.coarse_from = F32(scene.max_dist), scene.steps, 2
    S.seed_mix = (int(seed) * 0x9E3779B9 + 0x7F4A7C15) & 0xFFFFFFFF
    S.lights = _prep_lights(scene)
    S.fog = scene.fog_cfg
    gl = [scene.objects[k][1] for k in S.A.glass]
    S.glass_tr = (np.clip(np.mean([to_linear(_col(m.color)[:3]) if not callable(m.color) else np.ones(3)
                                   for m in gl], 0), 0, 1) ** 0.5 * 0.85).astype(F32) if gl else np.ones(3, F32)
    t_grid = time.time() - t_start
    S.aperture, S.focus = 0.0, 1.0
    dep, nrm0, ids0, S.safe = _center_pass(S, workers)
    S.aperture = float(aperture)
    if focus is None:
        S.focus = float(np.linalg.norm(_v(cam["target"]) - S.pos))
    elif np.ndim(focus):
        x, y = int(np.clip(focus[0], 0, W - 1)), int(np.clip(focus[1], 0, H - 1))
        dd = dep[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        dd = dd[np.isfinite(dd)]
        S.focus = float(np.median(dd)) if dd.size else 10.0
    else:
        S.focus = float(focus)
    if S.aperture > 0:   # lens rays spread more: recompute the shared empty-space distance
        S.safe = _center_pass(S, workers, only_safe=True)[3]
    # accumulators
    sL = np.zeros((P, 3))
    sY = np.zeros(P)
    sY2 = np.zeros(P)
    sA = np.zeros((P, 3))
    sN = np.zeros((P, 3))
    sH = np.zeros(P)
    n = np.zeros(P, np.int64)

    def run(cnt):
        cs = np.cumsum(cnt)
        tot = int(cs[-1]) if P else 0
        if tot == 0:
            return
        CH = 32768
        cuts = np.searchsorted(cs, np.arange(CH, tot, CH), side="left") + 1
        bnd = [0] + [int(c) for c in cuts if 0 < c < P] + [P]
        bnd = sorted(set(bnd))
        jobs = [(a, b) for a, b in zip(bnd[:-1], bnd[1:]) if cs[b - 1] - (cs[a - 1] if a else 0) > 0]
        def job(ab):
            a, b = ab
            c = cnt[a:b]
            pix = np.repeat(np.arange(a, b), c)
            st = np.cumsum(c) - c
            smp = n[pix] + (np.arange(len(pix)) - np.repeat(st, c))
            L, al, nr, hh = _paths(S, pix, smp)
            rel = pix - a
            m = b - a
            Y = (L @ _LUM).astype(np.float64)
            bc = lambda wv: np.bincount(rel, wv, m)
            return (a, b, np.stack([bc(L[:, i]) for i in range(3)], 1), bc(Y), bc(Y * Y),
                    np.stack([bc(al[:, i]) for i in range(3)], 1), np.stack([bc(nr[:, i]) for i in range(3)], 1),
                    bc(hh.astype(np.float64)))
        with ThreadPoolExecutor(workers) as ex:
            for a, b, l3, y1, y2, a3, n3, hh in ex.map(job, jobs):
                sL[a:b] += l3; sY[a:b] += y1; sY2[a:b] += y2
                sA[a:b] += a3; sN[a:b] += n3; sH[a:b] += hh
        n[:] += cnt

    def var_mean():
        nn = np.maximum(n, 1)
        m = sY / nn
        return np.maximum(sY2 / nn - m * m, 0) / np.maximum(nn - 1, 1), m

    spp = max(1, int(spp))
    if adaptive and spp >= 4:
        n1 = max(2, int(round(spp * 0.4)))
        run(np.full(P, n1, np.int64))
        # edge map from the centre pass (ids, normals, depth jumps) -> anti-aliasing priority
        e = np.zeros((H, W), bool)
        for ax in (0, 1):
            dI = np.diff(ids0, axis=ax) != 0
            dN = (np.diff(nrm0, axis=ax) ** 2).sum(-1) > 0.02
            z = np.where(np.isfinite(dep), dep, 1e4)
            dZ = np.abs(np.diff(z, axis=ax)) > 0.03 * np.minimum(z[:-1] if ax == 0 else z[:, :-1],
                                                                  z[1:] if ax == 0 else z[:, 1:])
            j = dI | dN | dZ
            if ax == 0:
                e[:-1] |= j; e[1:] |= j
            else:
                e[:, :-1] |= j; e[:, 1:] |= j
        edge = e.ravel().astype(np.float64)
        budget = spp * P - n1 * P
        cap = 4 * spp
        pk = _hash(np.arange(P, dtype=np.uint32) ^ np.uint32(S.seed_mix ^ 0x5bd1e995))
        for r in range(2):
            rb = budget / (2 - r)
            vm, m = var_mean()
            err = np.sqrt(vm) / (m + 0.04) * (1 + 1.5 * edge) + 0.02 * edge
            err = cv2.blur(err.reshape(H, W).astype(F32), (3, 3)).ravel().astype(np.float64)
            al = rb * err / max(err.sum(), 1e-12)
            al = np.minimum(al, np.maximum(cap - n, 0))
            cnt = np.floor(al + _rnd(pk, 900 + r)).astype(np.int64)
            run(cnt)
            budget -= int(cnt.sum())
    else:
        run(np.full(P, spp, np.int64))
    nn = np.maximum(n, 1)[:, None]
    lin = (sL / nn).reshape(H, W, 3).astype(F32)
    alb = (sA / nn).reshape(H, W, 3).astype(F32)
    nrm = _norm((sN / nn).astype(F32)).reshape(H, W, 3)
    alpha = (sH / nn[:, 0]).reshape(H, W).astype(F32)
    if denoise:
        vm, _ = var_mean()
        lin = _atrous(lin, alb, nrm, dep, vm.reshape(H, W).astype(F32))
    rgb = tonemap_linear(lin, tonemap, exposure, look)
    r = Render(rgb.astype(F32), lin, alpha, dep, nrm0, ids0)
    r.spp = n.reshape(H, W)
    r.albedo, r.time = alb, time.time() - t_start
    if verbose:
        print(f"pathtrace {W}x{H}: {n.mean():.1f} spp avg (max {n.max()}), grid {t_grid:.1f}s, "
              f"total {r.time:.1f}s")
    return r


def _layer_pathtrace(self, scene, spp=32, bounces=4, denoise=True, alpha=False, focus=None, **kw):
    """Path-trace a Scene into this layer at the canvas raster size (focus=(x, y) in canvas px)."""
    if focus is not None and np.ndim(focus):
        focus = (focus[0] * self.s, focus[1] * self.s)
    r = pathtrace(scene, self.W, self.H, spp, bounces, denoise, focus=focus, **kw)
    a = r.alpha if alpha else np.ones_like(r.alpha)
    self.rgba = np.concatenate([r.rgb, a[..., None]], -1)
    return r


Layer.pathtrace = _layer_pathtrace

__all__ = ["pathtrace", "denoise"]
