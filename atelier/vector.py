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


# ------------------------------------------------------------------ text
def _font(size, font, weight, italic):
    tf, emb, skew = typeface(font, weight, italic)
    f = skia.Font(tf, float(size))
    f.setEmbolden(emb)
    f.setSkewX(skew)
    f.setSubpixel(True)
    f.setLinearMetrics(True)
    return f


def _line_path(f, txt, spacing):
    glyphs = f.textToGlyphs(txt)
    widths = f.getWidths(glyphs)
    p, x = skia.Path(), 0.0
    for g, w in zip(glyphs, widths):
        gp = f.getPath(g)
        if gp is not None and not gp.isEmpty():
            p.addPath(gp, x, 0)
        x += w + spacing
    return p, (x - spacing if glyphs else 0.0)


def text_shape(txt, x, y, size=32, font="sans", weight="regular", align="left", valign="baseline",
               spacing=0.0, line_height=1.2, italic=False, fit=None):
    """Text as a Shape. align: left|center|right; valign: baseline|top|middle|bottom (cap-height based);
    spacing = extra letter spacing in em; fit = target width in px (sets size)."""
    lines = str(txt).split("\n")
    f = _font(size, font, weight, italic)
    if fit:
        widest = max(_line_path(f, ln, spacing * size)[1] for ln in lines) or 1
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
        p, w = _line_path(f, ln, sp)
        dx = {"left": 0.0, "center": -w / 2, "right": -w}[align]
        out.addPath(p, x + dx, y + oy + i * lh)
    return Shape(out)


def text_size(txt, size=32, font="sans", weight="regular", spacing=0.0, italic=False):
    """(width, cap_height) of single-line text in px."""
    f = _font(size, font, weight, italic)
    return _line_path(f, txt, spacing * size)[1], f.getMetrics().fCapHeight


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
            valign="baseline", spacing=0.0, line_height=1.2, italic=False, fit=None, **st):
    sh = text_shape(txt, x, y, size, font, weight, align, valign, spacing, line_height, italic, fit)
    return self.draw(sh, fill=fill, **st)


Layer.draw = draw
Layer.rect = _m_rect
Layer.ellipse = _m_ellipse
Layer.circle = _m_circle
Layer.polygon = _m_polygon
Layer.line = _m_line
Layer.spline = _m_spline
Layer.path = _m_path
Layer.text = _m_text

__all__ = ["Shape", "union", "rect", "ellipse", "circle", "polygon", "line", "regular", "star", "arc", "spline",
           "svg", "text_shape", "text_size", "typeface", "fonts", "WEIGHTS"]
