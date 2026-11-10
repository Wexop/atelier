"""Vector shapes and text, rasterised with skia (anti-aliased coverage, any paint source).

Shapes are in canvas px. A Shape is also a Field (its coverage), so it works as a mask,
combines with noise (fbm() * shape), and can be warped. Boolean ops: | & - ^.
Drawing: c.rect/ellipse/circle/polygon/line/spline/path/text/draw(shape, fill=, stroke=, ...).
"""
from __future__ import annotations

import math
import os
import re
import warnings
import numpy as np
import cv2
import skia

from .core import Layer, F32
from .field import Field

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
WEIGHTS = {"thin": 100, "light": 300, "regular": 400, "normal": 400, "medium": 500,
           "semibold": 600, "bold": 700, "heavy": 800, "black": 900}
_SERIF = ["Georgia", "Times New Roman", "DejaVu Serif", "Liberation Serif", "Noto Serif", "Times"]
_TF_CACHE: dict = {}
_DEF = object()


# ------------------------------------------------------------------ fonts
def _embedded(name):
    return skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, name))


def typeface(font="sans", weight=400, italic=False):
    """-> (skia.Typeface, embolden, skew). font: 'sans' (embedded Inter), 'mono'
    (embedded JetBrains Mono), 'serif' (system), a system family name, or a .ttf/.otf path."""
    w = WEIGHTS.get(weight, weight) if isinstance(weight, str) else int(weight)
    key = (font, w, italic)
    if key in _TF_CACHE:
        return _TF_CACHE[key]
    tf, emb, skew = None, False, 0.0
    if isinstance(font, str) and font.lower().endswith((".ttf", ".otf")):
        tf = skia.Typeface.MakeFromFile(font)
        if tf is None:
            raise FileNotFoundError(f"font file not found: {font}")
    elif font == "mono":
        tf, emb = _embedded("JetBrainsMono-Regular.ttf"), w >= 600
    elif font in ("sans", "inter", None):
        tf = _embedded("Inter-SemiBold.otf" if w >= 550 else "Inter-Regular.otf")
        emb = w >= 750
    else:
        names = _SERIF if font == "serif" else [font]
        style = skia.FontStyle(w, 5, skia.FontStyle.kItalic_Slant if italic else skia.FontStyle.kUpright_Slant)
        for n in names:
            t = skia.FontMgr().matchFamilyStyle(n, style)
            if t is not None:
                tf = t
                break
        if tf is None:
            warnings.warn(f"font '{font}' not found, using embedded sans")
            return typeface("sans", w, italic)
        if italic and not tf.isItalic():
            skew = -0.2
        if w >= 700 and not tf.isBold():
            emb = True
        _TF_CACHE[key] = (tf, emb, skew)
        return _TF_CACHE[key]
    if italic:
        skew = -0.2
    _TF_CACHE[key] = (tf, emb, skew)
    return _TF_CACHE[key]


def fonts():
    """System font families available to skia."""
    m = skia.FontMgr()
    return sorted({m.getFamilyName(i) for i in range(m.countFamilies())})


# ------------------------------------------------------------------ Shape
def _grid_box(x, y, W, H):
    """If x, y are a pixel-centre grid, return its pixel box, else None."""
    if x.ndim != 2 or x.size == 0:
        return None
    if x.shape[0] > 1 and x[0, 0] != x[-1, 0]:
        return None
    if x.shape[1] > 1 and y[0, 0] != y[0, -1]:
        return None
    x0, y0 = int(round(x[0, 0] * W - 0.5)), int(round(y[0, 0] * H - 0.5))
    return x0, y0, x0 + x.shape[1], y0 + x.shape[0]


def _cover(path, s, W, H, aa=True, feather=0.0, box=None):
    """Rasterise a fill path (canvas px) at scale s -> (coverage float32 (h,w), pixel box) or (None, None)."""
    p = skia.Path(path)
    if s != 1:
        p.transform(skia.Matrix.Scale(s, s))
    sig = feather * s if aa else 0
    if box is None:
        b = p.computeTightBounds()
        m = 3 * sig + 2
        box = (max(0, math.floor(b.left() - m)), max(0, math.floor(b.top() - m)),
               min(W, math.ceil(b.right() + m)), min(H, math.ceil(b.bottom() + m)))
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None, None
    surf = skia.Surface.MakeRaster(skia.ImageInfo.MakeA8(w, h))
    cv = surf.getCanvas()
    cv.translate(-x0, -y0)
    cv.drawPath(p, skia.Paint(AntiAlias=aa))
    cov = surf.makeImageSnapshot().toarray(colorType=skia.kAlpha_8_ColorType).astype(F32) * F32(1 / 255)
    if cov.ndim == 3:
        cov = cov[..., 0]
    if sig > 0:
        cov = cv2.GaussianBlur(cov, (0, 0), sig)
    return cov, box


_CAPS = {"butt": skia.Paint.kButt_Cap, "round": skia.Paint.kRound_Cap, "square": skia.Paint.kSquare_Cap}
_JOINS = {"miter": skia.Paint.kMiter_Join, "round": skia.Paint.kRound_Join, "bevel": skia.Paint.kBevel_Join}


def _stroke(path, width, cap="round", join="round", dash=None, miter=4.0):
    """Stroke outline of a path as a fill path (canvas px)."""
    paint = skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=float(width),
                       StrokeCap=_CAPS[cap], StrokeJoin=_JOINS[join], StrokeMiter=miter)
    if dash:
        paint.setPathEffect(skia.DashPathEffect.Make(list(map(float, dash)), 0))
    out = skia.Path()
    paint.getFillPath(path, out, None, 4.0)
    return out


class Shape(Field):
    """A vector path (canvas px). Also a Field = its anti-aliased coverage."""

    def __init__(self, path=None, feather=0.0):
        self.path = path if path is not None else skia.Path()
        self.feather = feather
        super().__init__(self._fn)

    def _fn(self, x, y, k):
        H = int(round(1 / k.px))
        W = int(round(k.ar * H))
        box = _grid_box(x, y, W, H)
        if box is not None:
            cov, b = _cover(self.path, k.s, W, H, True, self.feather, box)
            return cov if cov is not None else np.zeros(x.shape, F32)
        cov, b = _cover(self.path, k.s, W, H, True, self.feather, (0, 0, W, H))
        return cv2.remap(cov, (x * W - 0.5).astype(F32), (y * H - 0.5).astype(F32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # booleans
    def _op(self, o, op):
        return Shape(skia.Op(self.path, o.path, op))

    def __or__(self, o):
        return self._op(o, skia.PathOp.kUnion_PathOp)

    def __and__(self, o):
        return self._op(o, skia.PathOp.kIntersect_PathOp)

    def __sub__(self, o):
        if isinstance(o, Shape):
            return self._op(o, skia.PathOp.kDifference_PathOp)
        return Field.__sub__(self, o)

    def __xor__(self, o):
        return self._op(o, skia.PathOp.kXOR_PathOp)

    # transforms (return new shapes)
    def _tf(self, m):
        p = skia.Path(self.path)
        p.transform(m)
        return Shape(p, self.feather)

    def move(self, dx=0.0, dy=0.0):
        return self._tf(skia.Matrix.Translate(dx, dy))

    def rotate(self, deg, cx=None, cy=None):
        if cx is None:
            cx, cy = self.center
        return self._tf(skia.Matrix.RotateDeg(deg, skia.Point(cx, cy)))

    def scale(self, sx, sy=None, cx=None, cy=None):
        if cx is None:
            cx, cy = self.center
        m = skia.Matrix()
        m.setScale(sx, sx if sy is None else sy, cx, cy)
        return self._tf(m)

    def soft(self, feather):
        """Same shape with blurred edges (feather in canvas px) when used as a mask."""
        return Shape(self.path, feather)

    def outline(self, width=2.0, cap="round", join="round", dash=None):
        """Stroke outline as a fillable Shape."""
        return Shape(_stroke(self.path, width, cap, join, dash))

    @property
    def bounds(self):
        b = self.path.computeTightBounds()
        return b.left(), b.top(), b.right(), b.bottom()

    @property
    def center(self):
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _contours(self):
        pm = skia.PathMeasure(self.path, False)
        lens = []
        while True:
            lens.append(pm.getLength())
            if not pm.nextContour():
                break
        return lens

    @property
    def length(self):
        return float(sum(self._contours()))

    def points(self, n=100, spacing=None):
        """Evenly spaced points along the path (all contours concatenated). (n, 2) array."""
        lens = self._contours()
        total = sum(lens)
        if total == 0:
            return np.zeros((0, 2), F32)
        if spacing:
            n = max(2, int(total / spacing) + 1)
        pm = skia.PathMeasure(self.path, False)
        out = []
        for L in lens:
            k = max(2, round(n * L / total))
            for d in np.linspace(0, L, k):
                pos, _ = pm.getPosTan(float(d))
                out.append((pos.x(), pos.y()))
            pm.nextContour()
        return np.array(out, F32)

    def contains(self, x, y):
        return self.path.contains(x, y)

    def __repr__(self):
        return f"<Shape bounds={tuple(round(v, 1) for v in self.bounds)}>"


# ------------------------------------------------------------------ constructors
def union(*shapes):
    """Union of shapes: union(a, b, c) or union(generator_of_shapes)."""
    if len(shapes) == 1 and not isinstance(shapes[0], Shape):
        shapes = tuple(shapes[0])
    out = Shape()
    for sh in shapes:
        out = out | sh
    return out


def rect(x, y, w, h, r=0.0):
    """Rectangle from top-left (x, y); r = corner radius."""
    p = skia.Path()
    if r:
        p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r))
    else:
        p.addRect(skia.Rect.MakeXYWH(x, y, w, h))
    return Shape(p)


def ellipse(cx, cy, rx, ry=None):
    p = skia.Path()
    ry = rx if ry is None else ry
    p.addOval(skia.Rect.MakeLTRB(cx - rx, cy - ry, cx + rx, cy + ry))
    return Shape(p)


def circle(cx, cy, r):
    return ellipse(cx, cy, r, r)


def polygon(points, closed=True):
    p = skia.Path()
    pts = [tuple(map(float, q)) for q in points]
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if closed:
        p.close()
    return Shape(p)


def line(*pts):
    """Open polyline: line(x1, y1, x2, y2) or line([(x, y), ...])."""
    if len(pts) == 4 and all(isinstance(v, (int, float, np.floating, np.integer)) for v in pts):
        pts = [(pts[0], pts[1]), (pts[2], pts[3])]
    elif len(pts) == 1:
        pts = pts[0]
    return polygon(pts, closed=False)


def regular(cx, cy, r, n=6, rot=0.0):
    """Regular n-gon; rot in degrees (0 = first vertex up)."""
    a = np.radians(rot) - np.pi / 2 + np.arange(n) * 2 * np.pi / n
    return polygon(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))


def star(cx, cy, r, r2=None, n=5, rot=0.0):
    """n-pointed star; r2 = inner radius (default 0.45 r)."""
    r2 = r * 0.45 if r2 is None else r2
    a = np.radians(rot) - np.pi / 2 + np.arange(2 * n) * np.pi / n
    rr = np.where(np.arange(2 * n) % 2 == 0, r, r2)
    return polygon(np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1))


def arc(cx, cy, r, start, sweep):
    """Open circular arc; angles in degrees (0 = right, clockwise)."""
    p = skia.Path()
    p.addArc(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), start, sweep)
    return Shape(p)


def spline(points, closed=False, tension=1.0):
    """Smooth Catmull-Rom curve through points (as cubic Beziers)."""
    P = np.asarray(points, np.float64)
    n = len(P)
    if n < 3:
        return polygon(P, closed)
    p = skia.Path()
    p.moveTo(*P[0])
    segs = n if closed else n - 1
    t = tension / 6
    for i in range(segs):
        p0 = P[(i - 1) % n] if (closed or i > 0) else P[0]
        p1, p2 = P[i % n], P[(i + 1) % n]
        p3 = P[(i + 2) % n] if (closed or i + 2 < n) else P[-1]
        c1, c2 = p1 + (p2 - p0) * t, p2 - (p3 - p1) * t
        p.cubicTo(*c1, *c2, *p2)
    if closed:
        p.close()
    return Shape(p)


_TOK = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def svg(d):
    """Shape from an SVG path 'd' string (M L H V C S Q T A Z, abs + relative)."""
    toks = _TOK.findall(d)
    p = skia.Path()
    i, cmd = 0, None
    cx = cy = sx = sy = 0.0
    lc = None  # last control point (for S/T)
    lcmd = ""

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def flag():
        nonlocal i
        t = toks[i]
        if len(t) > 1 and t[0] in "01" and not t.startswith(("0.", "1.")):
            toks[i] = t[1:]
            return t[0] == "1"
        i += 1
        return float(t) != 0

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                p.close()
                cx, cy = sx, sy
                lcmd = "Z"
                continue
        if cmd is None:
            raise ValueError("SVG path must start with a command")
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = (cx, cy) if rel else (0.0, 0.0)
        if C == "M":
            cx, cy = ox + num(), oy + num()
            p.moveTo(cx, cy)
            sx, sy = cx, cy
            cmd = "l" if rel else "L"
        elif C == "L":
            cx, cy = ox + num(), oy + num()
            p.lineTo(cx, cy)
        elif C == "H":
            cx = (cx if rel else 0) + num()
            p.lineTo(cx, cy)
        elif C == "V":
            cy = (cy if rel else 0) + num()
            p.lineTo(cx, cy)
        elif C == "C":
            x1, y1, x2, y2, x, y = ox + num(), oy + num(), ox + num(), oy + num(), ox + num(), oy + num()
            p.cubicTo(x1, y1, x2, y2, x, y)
            lc, cx, cy = (x2, y2), x, y
        elif C == "S":
            x1, y1 = (2 * cx - lc[0], 2 * cy - lc[1]) if lcmd in "CS" and lc else (cx, cy)
            x2, y2, x, y = ox + num(), oy + num(), ox + num(), oy + num()
            p.cubicTo(x1, y1, x2, y2, x, y)
            lc, cx, cy = (x2, y2), x, y
        elif C == "Q":
            x1, y1, x, y = ox + num(), oy + num(), ox + num(), oy + num()
            p.quadTo(x1, y1, x, y)
            lc, cx, cy = (x1, y1), x, y
        elif C == "T":
            x1, y1 = (2 * cx - lc[0], 2 * cy - lc[1]) if lcmd in "QT" and lc else (cx, cy)
            x, y = ox + num(), oy + num()
            p.quadTo(x1, y1, x, y)
            lc, cx, cy = (x1, y1), x, y
        elif C == "A":
            rx, ry, rot = num(), num(), num()
            large, sweep = flag(), flag()
            x, y = ox + num(), oy + num()
            p.arcTo(rx, ry, rot, skia.Path.ArcSize.kLarge_ArcSize if large else skia.Path.ArcSize.kSmall_ArcSize,
                    skia.PathDirection.kCW if sweep else skia.PathDirection.kCCW, x, y)
            cx, cy = x, y
        else:
            raise ValueError(f"unsupported SVG command '{cmd}'")
        lcmd = C
    return Shape(p)


# ------------------------------------------------------------------ kerning (tiny OpenType reader)
_KERN_CACHE: dict = {}


def _u16(b, o):
    return (b[o] << 8) | b[o + 1]


def _i16(b, o):
    v = _u16(b, o)
    return v - 65536 if v >= 32768 else v


def _coverage(b, o):
    """OpenType Coverage table -> {glyph: coverage index}."""
    fmt, n = _u16(b, o), _u16(b, o + 2)
    out = {}
    if fmt == 1:
        for i in range(n):
            out[_u16(b, o + 4 + 2 * i)] = i
    elif fmt == 2:
        for i in range(n):
            r = o + 4 + 6 * i
            s, e, idx = _u16(b, r), _u16(b, r + 2), _u16(b, r + 4)
            for g in range(s, e + 1):
                out[g] = idx + g - s
    return out


def _classdef(b, o):
    """OpenType ClassDef table -> {glyph: class} (missing = class 0)."""
    fmt = _u16(b, o)
    out = {}
    if fmt == 1:
        start, n = _u16(b, o + 2), _u16(b, o + 4)
        for i in range(n):
            out[start + i] = _u16(b, o + 6 + 2 * i)
    elif fmt == 2:
        for i in range(_u16(b, o + 2)):
            r = o + 4 + 6 * i
            s, e, c = _u16(b, r), _u16(b, r + 2), _u16(b, r + 4)
            for g in range(s, e + 1):
                out[g] = c
    return out


def _vsize(vf):
    return 2 * bin(vf & 0xFF).count("1")


def _xadv(b, o, vf):
    """XAdvance from a ValueRecord at o (0 if absent)."""
    if not vf & 4:
        return 0
    return _i16(b, o + 2 * bin(vf & 3).count("1"))


def _pairpos(b, o, subs):
    fmt, cov = _u16(b, o), _coverage(b, o + _u16(b, o + 2))
    vf1, vf2 = _u16(b, o + 4), _u16(b, o + 6)
    s1, s2 = _vsize(vf1), _vsize(vf2)
    if not vf1 & 4:
        return
    if fmt == 1:
        pairs = {}
        n = _u16(b, o + 8)
        for g, i in cov.items():
            if i >= n:
                continue
            ps = o + _u16(b, o + 10 + 2 * i)
            for k in range(_u16(b, ps)):
                r = ps + 2 + k * (2 + s1 + s2)
                v = _xadv(b, r + 2, vf1)
                pairs.setdefault((g, _u16(b, r)), v)
        subs.append(("p", cov, pairs))
    elif fmt == 2:
        cd1, cd2 = _classdef(b, o + _u16(b, o + 8)), _classdef(b, o + _u16(b, o + 10))
        n1, n2 = _u16(b, o + 12), _u16(b, o + 14)
        rec = s1 + s2
        m = np.zeros((n1, n2), np.int32)
        base = o + 16
        for c1 in range(n1):
            for c2 in range(n2):
                m[c1, c2] = _xadv(b, base + (c1 * n2 + c2) * rec, vf1)
        subs.append(("c", cov, (cd1, cd2, m)))


def _read_gpos(b):
    subs = []
    fl, ll = _u16(b, 6), _u16(b, 8)
    idx = set()
    for i in range(_u16(b, fl)):
        r = fl + 2 + 6 * i
        if b[r:r + 4] == b"kern":
            f = fl + _u16(b, r + 4)
            idx.update(_u16(b, f + 4 + 2 * k) for k in range(_u16(b, f + 2)))
    for li in sorted(idx):
        lo = ll + _u16(b, ll + 2 + 2 * li)
        typ, n = _u16(b, lo), _u16(b, lo + 4)
        for k in range(n):
            so = lo + _u16(b, lo + 6 + 2 * k)
            t = typ
            if t == 9:  # extension
                t = _u16(b, so + 2)
                so = so + ((_u16(b, so + 4) << 16) | _u16(b, so + 6))
            if t == 2:
                _pairpos(b, so, subs)
    return subs


def _read_kern(b):
    """Legacy 'kern' table, format 0 subtables (horizontal)."""
    pairs = {}
    if _u16(b, 0) != 0:
        return []
    o = 4
    for _ in range(_u16(b, 2)):
        ln, covf = _u16(b, o + 2), _u16(b, o + 4)
        if covf >> 8 == 0 and covf & 1:
            n = _u16(b, o + 6)
            for k in range(n):
                r = o + 14 + 6 * k
                pairs.setdefault((_u16(b, r), _u16(b, r + 2)), _i16(b, r + 4))
        o += ln
    return [("p", None, pairs)] if pairs else []


def kern_pairs(tf):
    """Kerning lookup for a skia Typeface (GPOS PairPos 1+2, else legacy 'kern'), cached.
    -> (subtables, units_per_em); use kern_value(tf, g1, g2) for one pair."""
    key = tf.uniqueID()
    if key in _KERN_CACHE:
        return _KERN_CACHE[key]
    subs = []
    tags = set(tf.getTableTags())
    try:
        tg = int.from_bytes(b"GPOS", "big")
        if tg in tags:
            subs = _read_gpos(bytes(tf.copyTableData(tg)))
        tk = int.from_bytes(b"kern", "big")
        if not subs and tk in tags:
            subs = _read_kern(bytes(tf.copyTableData(tk)))
    except Exception as e:  # malformed font: no kerning rather than a crash
        warnings.warn(f"kerning disabled for {tf.getFamilyName()}: {e}")
        subs = []
    _KERN_CACHE[key] = (subs, tf.getUnitsPerEm() or 1000)
    return _KERN_CACHE[key]


def _kern(subs, g1, g2):
    for kind, cov, data in subs:
        if cov is not None and g1 not in cov:
            continue
        if kind == "p":
            v = data.get((g1, g2))
            if v is not None:
                return v
        else:
            cd1, cd2, m = data
            c1, c2 = cd1.get(g1, 0), cd2.get(g2, 0)
            if c1 < m.shape[0] and c2 < m.shape[1]:
                return int(m[c1, c2])
            return 0
    return 0


def kern_value(tf, g1, g2):
    """Kerning between glyph ids g1, g2 in font units (negative = tighter)."""
    return _kern(kern_pairs(tf)[0], g1, g2)


# ------------------------------------------------------------------ text
def _font(size, font, weight, italic):
    tf, emb, skew = typeface(font, weight, italic)
    f = skia.Font(tf, float(size))
    f.setEmbolden(emb)
    f.setSkewX(skew)
    f.setSubpixel(True)
    f.setLinearMetrics(True)
    return f


def _run(f, txt, spacing, kern=True):
    """Glyph run -> (glyphs, x positions, advances, total width); kerning from GPOS/kern applied."""
    glyphs = list(f.textToGlyphs(txt))
    widths = list(f.getWidths(glyphs))
    xs, x = [], 0.0
    subs, upem = kern_pairs(f.getTypeface()) if kern and len(glyphs) > 1 else ((), 1)
    k = f.getSize() / upem
    for i, (g, w) in enumerate(zip(glyphs, widths)):
        if subs and i:
            x += _kern(subs, glyphs[i - 1], g) * k
        xs.append(x)
        x += w + spacing
    return glyphs, xs, widths, (x - spacing if glyphs else 0.0)


def _line_path(f, txt, spacing, kern=True):
    glyphs, xs, _, total = _run(f, txt, spacing, kern)
    p = skia.Path()
    for g, x in zip(glyphs, xs):
        gp = f.getPath(g)
        if gp is not None and not gp.isEmpty():
            p.addPath(gp, x, 0)
    return p, total


def text_shape(txt, x, y, size=32, font="sans", weight="regular", align="left", valign="baseline",
               spacing=0.0, line_height=1.2, italic=False, fit=None, kern=True):
    """Text as a Shape. align: left|center|right; valign: baseline|top|middle|bottom (cap-height based);
    spacing = extra letter spacing in em; fit = target width in px (sets size); kern = font pair kerning."""
    lines = str(txt).split("\n")
    f = _font(size, font, weight, italic)
    if fit:
        widest = max(_line_path(f, ln, spacing * size, kern)[1] for ln in lines) or 1
        size = size * fit / widest
        f = _font(size, font, weight, italic)
    sp = spacing * size
    met = f.getMetrics()
    cap = met.fCapHeight or size * 0.7
    lh = line_height * size
    block = cap + lh * (len(lines) - 1)
    oy = {"baseline": 0.0, "top": cap, "middle": cap - block / 2, "bottom": cap - block}[valign]
    out = skia.Path()
    for i, ln in enumerate(lines):
        p, w = _line_path(f, ln, sp, kern)
        dx = {"left": 0.0, "center": -w / 2, "right": -w}[align]
        out.addPath(p, x + dx, y + oy + i * lh)
    return Shape(out)


def text_size(txt, size=32, font="sans", weight="regular", spacing=0.0, italic=False, kern=True):
    """(width, cap_height) of single-line text in px."""
    f = _font(size, font, weight, italic)
    return _line_path(f, txt, spacing * size, kern)[1], f.getMetrics().fCapHeight


def _path_of(path):
    if isinstance(path, Shape):
        return path.path
    if isinstance(path, skia.Path):
        return path
    return spline(path).path


def text_on_path(txt, path, size=32, font="sans", weight="regular", offset=0.0, align="left", lift=0.0,
                 spacing=0.0, italic=False, kern=True, fit=False):
    """Text set along a path (Shape, skia.Path or points -> smooth spline), glyphs rotated to the tangent.
    offset = extra distance along the path in px; align: left|center|right within the path length;
    lift = baseline shift in px (positive = to the left of the travel direction, i.e. above a
    left-to-right path). Glyphs past either end are dropped. fit=True spreads the letter spacing so the
    text spans the whole path (closed rings: end the text with a separator such as '  ·  ')."""
    f = _font(size, font, weight, italic)
    pm = skia.PathMeasure(_path_of(path), False)
    L = pm.getLength() - offset
    glyphs, xs, ws, total = _run(f, str(txt), spacing * size, kern)
    if fit and len(glyphs) > 1:
        glyphs, xs, ws, total = _run(f, str(txt), spacing * size + (L - total - ws[-1] * .5) / len(glyphs), kern)
    L += offset
    d0 = offset + {"left": 0.0, "center": (L - total) / 2, "right": L - total}[align]
    out = skia.Path()
    for g, x, w in zip(glyphs, xs, ws):
        mid = d0 + x + w / 2
        if mid < 0 or mid > L:
            continue
        gp = f.getPath(g)
        if gp is None or gp.isEmpty():
            continue
        pos, tan = pm.getPosTan(float(mid))
        tx, ty = tan.x(), tan.y()
        m = skia.Matrix()
        m.setAll(tx, -ty, pos.x() + ty * lift, ty, tx, pos.y() - tx * lift, 0, 0, 1)
        m.preTranslate(-w / 2, 0)
        q = skia.Path(gp)
        q.transform(m)
        out.addPath(q)
    return Shape(out)


_RICH = re.compile(r"(\*\*|\*|\[/\]|\[[^\]/]+\])")


def _rich(txt):
    """Markup -> [(text, bold, italic, colour|None)]: **bold**, *italic*, [colour]text[/]."""
    out, b, it, col = [], False, False, []
    for tok in _RICH.split(str(txt)):
        if tok == "**":
            b = not b
        elif tok == "*":
            it = not it
        elif tok == "[/]":
            col = col[:-1]
        elif len(tok) > 2 and tok[0] == "[" and tok[-1] == "]":
            col.append(tok[1:-1])
        elif tok:
            out.append((tok, b, it, col[-1] if col else None))
    return out


def _bold_of(weight):
    w = WEIGHTS.get(weight, weight) if isinstance(weight, str) else int(weight)
    return 700 if w < 600 else 900


def _box_layout(txt, w, size, font, weight, align, line_height, spacing, kern, rich):
    """Greedy word wrap -> (items [(path, x, y, colour)], n_lines, height, a word overflows w)."""
    fonts = {}

    def fnt(b, it):
        if (b, it) not in fonts:
            fonts[(b, it)] = _font(size, font, _bold_of(weight) if b else weight, it)
        return fonts[(b, it)]

    f0 = fnt(False, False)
    sp = spacing * size
    space = f0.getWidths(f0.textToGlyphs(" "))[0] + sp
    cap = f0.getMetrics().fCapHeight or size * 0.7
    lh = line_height * size
    runs = _rich(txt) if rich else [(str(txt), False, False, None)]
    paras, words, cur = [], [], []
    for text, b, it, col in runs:  # paragraphs -> words -> pieces (path, width, colour)
        for chunk in re.split(r"(\n|\s+)", text):
            if chunk == "\n" or (chunk and chunk.isspace()):
                if cur:
                    words.append(cur)
                cur = []
                if "\n" in chunk:
                    paras.append(words)
                    words = []
            elif chunk:
                p, wd = _line_path(fnt(b, it), chunk, sp, kern)
                cur.append((p, wd, col))
    if cur:
        words.append(cur)
    paras.append(words)
    out, y, nl, over = [], cap, 0, False
    for words in paras:
        lines, ln, lw = [], [], 0.0
        for wd in words:
            ww = sum(q[1] for q in wd)
            over |= ww > w + 0.5
            if ln and lw + space + ww > w:
                lines.append((ln, lw))
                ln, lw = [], 0.0
            lw += (space if ln else 0.0) + ww
            ln.append((wd, ww))
        lines.append((ln, lw))
        for li, (ln, lw) in enumerate(lines):
            gap, x = space, 0.0
            if align == "justify" and li < len(lines) - 1 and len(ln) > 1:
                gap = space + (w - lw) / (len(ln) - 1)
            elif align == "center":
                x = (w - lw) / 2
            elif align == "right":
                x = w - lw
            for wd, ww in ln:
                xx = x
                for p, pw, col in wd:
                    out.append((p, xx, y, col))
                    xx += pw
                x += ww + gap
            y += lh
            nl += 1
    return out, nl, cap + lh * (nl - 1) + 0.28 * size, over


def text_box(txt, x, y, w, h=None, size=18, font="sans", weight="regular", align="left", line_height=1.4,
             spacing=0.0, fit=False, kern=True, rich=True, min_size=4.0):
    """Paragraphs word-wrapped into width w from top-left (x, y); '\\n' starts a new paragraph.
    align: left|right|center|justify (last line of a paragraph stays left); fit=True (needs h) shrinks
    size until the text fits the box. rich markup: **bold**, *italic*, [#c33]coloured[/].
    -> Shape with .runs [(colour|None, Shape)], .size (final size), .lines, .height (px)."""
    args = (font, weight, align, line_height, spacing, kern, rich)
    if fit and h:
        lay = _box_layout(txt, w, size, *args)
        if lay[2] > h or lay[3]:
            lo, hi = float(min_size), float(size)
            for _ in range(14):
                mid = (lo + hi) / 2
                lay = _box_layout(txt, w, mid, *args)
                lo, hi = (lo, mid) if (lay[2] > h or lay[3]) else (mid, hi)
            size = lo
    items, nl, height, _ = _box_layout(txt, w, size, *args)
    groups, allp = {}, skia.Path()
    for p, px, py, col in items:
        groups.setdefault(col, skia.Path()).addPath(p, x + px, y + py)
        allp.addPath(p, x + px, y + py)
    sh = Shape(allp)
    sh.runs = [(k, Shape(v)) for k, v in groups.items()]
    sh.size, sh.lines, sh.height = size, nl, height
    return sh


# ------------------------------------------------------------------ drawing on layers
def draw(self, shape, fill=_DEF, stroke=None, width=2.0, cap="round", join="round", dash=None,
         feather=0.0, opacity=1.0):
    """Paint a Shape: fill and/or stroke with any paint source (colour, gradient, field, array).
    width/dash/feather in canvas px. Returns the shape."""
    if fill is _DEF:
        fill = "black" if stroke is None else None
    aa = self.aa
    feather = feather or getattr(shape, "feather", 0.0)
    if fill is not None:
        cov, box = _cover(shape.path, self.s, self.W, self.H, aa, feather)
        if cov is not None:
            self.paint(cov * F32(opacity) if opacity != 1 else cov, fill, box)
    if stroke is not None:
        sp = skia.Path(shape.path)
        sp.transform(skia.Matrix.Scale(self.s, self.s))
        sp = _stroke(sp, width * self.s, cap, join, [d * self.s for d in dash] if dash else None)
        cov, box = _cover(sp, 1.0, self.W, self.H, aa, feather * self.s)
        if cov is not None:
            self.paint(cov * F32(opacity) if opacity != 1 else cov, stroke, box)
    return shape


def _m_rect(self, x, y, w, h, r=0.0, **st):
    return self.draw(rect(x, y, w, h, r), **st)


def _m_ellipse(self, cx, cy, rx, ry=None, **st):
    return self.draw(ellipse(cx, cy, rx, ry), **st)


def _m_circle(self, cx, cy, r, **st):
    return self.draw(circle(cx, cy, r), **st)


def _m_polygon(self, points, **st):
    return self.draw(polygon(points), **st)


def _m_line(self, *pts, stroke="black", **st):
    return self.draw(line(*pts), fill=None, stroke=stroke, **st)


def _m_spline(self, points, closed=False, stroke="black", fill=None, **st):
    return self.draw(spline(points, closed), fill=fill, stroke=stroke, **st)


def _m_path(self, d, **st):
    return self.draw(svg(d), **st)


def _m_text(self, txt, x, y, size=32, fill="black", font="sans", weight="regular", align="left",
            valign="baseline", spacing=0.0, line_height=1.2, italic=False, fit=None, kern=True, **st):
    sh = text_shape(txt, x, y, size, font, weight, align, valign, spacing, line_height, italic, fit, kern)
    return self.draw(sh, fill=fill, **st)


def _m_text_on_path(self, txt, path, size=32, fill="black", font="sans", weight="regular", offset=0.0,
                    align="left", lift=0.0, spacing=0.0, italic=False, kern=True, fit=False, **st):
    sh = text_on_path(txt, path, size, font, weight, offset, align, lift, spacing, italic, kern, fit)
    return self.draw(sh, fill=fill, **st)


def _m_text_box(self, txt, x, y, w, h=None, size=18, fill="black", font="sans", weight="regular",
                align="left", line_height=1.4, spacing=0.0, fit=False, kern=True, rich=True, **st):
    sh = text_box(txt, x, y, w, h, size, font, weight, align, line_height, spacing, fit, kern, rich)
    for col, part in sh.runs:
        self.draw(part, fill=fill if col is None else col, **st)
    return sh


def _darker(col, k=0.55):
    from .core import color as _color
    c = _color(col)
    return tuple(float(v) * k for v in c[:3])


def title(self, txt, x, y, size=96, style="shadow", fill="black", color2=None, depth=None, **kw):
    """Display-type presets (kw -> text_shape: font, weight, align, valign, spacing, fit, kern...).
    style: 'shadow' (hard offset shadow) | 'extrude' (3D block) | 'outline' (hollow letters) |
    'inline' (filled + inner line) | 'neon' (outline + glow). color2 = shadow / line colour.
    depth = shadow offset in px (default 6% of size)."""
    kw.setdefault("weight", "black" if style == "inline" else "bold")
    sh = text_shape(txt, x, y, size, **kw)
    d = size * 0.06 if depth is None else depth
    if style == "shadow":
        self.draw(sh.move(d, d), fill=color2 or "#00000066")
        self.draw(sh, fill=fill)
    elif style == "extrude":
        c2 = color2 or _darker(fill)
        n = max(2, int(d * self.s * 1.5))
        steps = skia.Path()
        for i in range(1, n + 1):
            steps.addPath(sh.path, d * i / n, d * i / n)
        self.draw(Shape(skia.Simplify(steps) or steps), fill=c2)
        self.draw(sh, fill=fill)
    elif style == "outline":
        self.draw(sh, fill=None, stroke=fill, width=max(1.0, size * 0.035), join="miter")
    elif style == "inline":
        inset = max(1.0, size * 0.03)
        self.draw(sh, fill=fill)
        core = Shape(skia.Op(sh.path, _stroke(sh.path, 2 * inset, "round", "round"), skia.PathOp.kDifference_PathOp))
        self.draw(core, fill=None, stroke=color2 or "white", width=max(0.8, size * 0.014))
    elif style == "neon":
        from .fx import glow
        L = self.canvas.layer(blend="screen")
        L.draw(sh, fill=None, stroke=fill, width=max(1.0, size * 0.03))
        L.apply(glow(size * 0.12, 1.2, color2 or fill))
        L.draw(sh, fill=None, stroke="white", width=max(0.6, size * 0.01))
        self.canvas.use(self)  # back to the caller's layer
    else:
        raise ValueError("title style: shadow | extrude | outline | inline | neon")
    return sh


Layer.draw = draw
Layer.rect = _m_rect
Layer.ellipse = _m_ellipse
Layer.circle = _m_circle
Layer.polygon = _m_polygon
Layer.line = _m_line
Layer.spline = _m_spline
Layer.path = _m_path
Layer.text = _m_text
Layer.text_on_path = _m_text_on_path
Layer.text_box = _m_text_box
Layer.title = title

__all__ = ["Shape", "union", "rect", "ellipse", "circle", "polygon", "line", "regular", "star", "arc", "spline",
           "svg", "text_shape", "text_size", "text_on_path", "text_box", "kern_pairs", "kern_value", "typeface",
           "fonts", "WEIGHTS"]
