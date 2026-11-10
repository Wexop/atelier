"""Canvas, layers, blend modes, import/export.

Conventions
- Coordinates passed to drawing calls are *logical* pixels of the canvas (w x h).
  The canvas may be rasterised at another resolution (render scale `s`, set by the
  CLI for previews / 2x finals); every module multiplies by `layer.s`.
- Layers store premultiplied RGBA float32 in sRGB. `layer.rgba` is straight alpha.
- Anything "paintable" (fill, stroke colour, ...) is resolved by `resolve()`:
  colour string/tuple, numpy array (H,W[,C]), a Field (has .render) or f(x, y)
  with x, y normalised to [0, 1].
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from PIL import Image, ImageColor

# Set by the CLI before running a scene: width -> render at that pixel width,
# scale -> render scale multiplier, nosave -> ignore save() calls in the script.
CONFIG = {"width": None, "scale": None, "nosave": False, "preview": False, "threads": None}
_LAST: list = [None]  # last Canvas created (the CLI picks it up)
_SAVED: list = []     # paths written by Canvas.save (reported by the CLI)

F32 = np.float32


# ---------------------------------------------------------------- colour utils
def color(c) -> np.ndarray:
    """Parse '#rgb', '#rrggbb[aa]', CSS name, (r,g,b[,a]) in 0-1 or 0-255, or gray float."""
    if isinstance(c, np.ndarray) and c.ndim == 1:
        c = tuple(c.tolist())
    if isinstance(c, str):
        v = ImageColor.getrgb(c)
        v = [x / 255 for x in v]
    elif isinstance(c, (int, float)):
        v = [float(c)] * 3
    else:
        v = list(c)
        if any(x > 1 for x in v):
            v = [x / 255 for x in v]
    if len(v) == 3:
        v.append(1.0)
    return np.array(v, F32)


def to_linear(c):
    c = np.asarray(c, F32)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4).astype(F32)


def to_srgb(c):
    c = np.maximum(np.asarray(c, F32), 0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(F32)


def rgba(a) -> np.ndarray:
    """Promote a (h,w), (h,w,1), (h,w,3) or (h,w,4) array to straight RGBA float32."""
    a = np.asarray(a, F32)
    if a.ndim == 2:
        a = a[..., None]
    c = a.shape[-1]
    if c == 1:
        return np.concatenate([a, a, a, np.ones_like(a)], -1)
    if c == 3:
        return np.concatenate([a, np.ones_like(a[..., :1])], -1)
    return a


def premul(a):
    out = a.copy()
    out[..., :3] *= a[..., 3:4]
    return out


def unpremul(p):
    out = p.copy()
    al = p[..., 3:4]
    inv = np.zeros_like(al)
    np.divide(1, al, out=inv, where=al > 1e-6)
    out[..., :3] *= inv
    return out


def coords(H, W, box=None):
    """Normalised pixel-centre grids x, y in [0,1] for the full raster or a sub-box."""
    x0, y0, x1, y1 = box or (0, 0, W, H)
    xs = ((np.arange(x0, x1, dtype=F32) + 0.5) / W)
    ys = ((np.arange(y0, y1, dtype=F32) + 0.5) / H)
    return np.meshgrid(xs, ys)


def resolve(src, H, W, box=None, s=1.0) -> np.ndarray:
    """Turn any paint source into straight RGBA (h,w,4) for `box`, or (1,1,4) if constant.
    s = render scale (raster px per canvas px), needed by Shapes used as paint/mask."""
    if src is None:
        return None
    if hasattr(src, "render"):
        return rgba(src.render(H, W, box, s))
    if isinstance(src, np.ndarray) and src.ndim >= 2:
        a = src
        x0, y0, x1, y1 = box or (0, 0, W, H)
        if a.shape[:2] == (y1 - y0, x1 - x0) and box is not None:
            return rgba(a)
        if a.shape[:2] != (H, W):
            a = cv2.resize(np.ascontiguousarray(a, F32), (W, H), interpolation=cv2.INTER_LINEAR)
        return rgba(a[y0:y1, x0:x1])
    if callable(src):
        x, y = coords(H, W, box)
        return rgba(src(x, y))
    return color(src)[None, None, :]


def resolve_mask(m, H, W, box=None, s=1.0) -> np.ndarray:
    """Like resolve() but returns a single (h,w) [0,1] channel (alpha if RGBA, else luminance)."""
    if m is None:
        return None
    if isinstance(m, (int, float)):
        return np.full((1, 1), m, F32)
    if hasattr(m, "render"):
        a = np.asarray(m.render(H, W, box, s), F32)
    elif callable(m) and not isinstance(m, np.ndarray):
        x, y = coords(H, W, box)
        a = np.asarray(m(x, y), F32)
    else:
        a = np.asarray(m, F32)
        x0, y0, x1, y1 = box or (0, 0, W, H)
        if not (box is not None and a.shape[:2] == (y1 - y0, x1 - x0)):
            if a.shape[:2] != (H, W):
                a = cv2.resize(np.ascontiguousarray(a), (W, H), interpolation=cv2.INTER_LINEAR)
            a = a[y0:y1, x0:x1]
    if a.ndim == 3:
        a = a[..., 3] if a.shape[-1] == 4 else a[..., :3].mean(-1)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    return np.clip(a, 0, 1)


# ---------------------------------------------------------------- blend modes
def _lum(c):
    return (0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2])[..., None]


def _clipcolor(c):
    L = _lum(c)
    n = c.min(-1, keepdims=True)
    x = c.max(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(n < 0, L + (c - L) * L / np.maximum(L - n, 1e-6), c)
        c = np.where(x > 1, L + (c - L) * (1 - L) / np.maximum(x - L, 1e-6), c)
    return c


def _setlum(c, L):
    return _clipcolor(c + (L - _lum(c)))


def _sat(c):
    return c.max(-1, keepdims=True) - c.min(-1, keepdims=True)


def _setsat(c, s):
    mn = c.min(-1, keepdims=True)
    d = _sat(c)
    return np.where(d > 1e-6, (c - mn) * s / np.maximum(d, 1e-6), 0)


def _softlight(b, s):
    d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.maximum(b, 0)))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))


def _dodge(b, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.minimum(1, b / np.maximum(1 - s, 1e-6))
    return np.where(b <= 0, 0, np.where(s >= 1, 1, r))


def _burn(b, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 1 - np.minimum(1, (1 - b) / np.maximum(s, 1e-6))
    return np.where(b >= 1, 1, np.where(s <= 0, 0, r))


def _hardlight(b, s):
    return np.where(s <= 0.5, 2 * s * b, 1 - 2 * (1 - s) * (1 - b))


# B(backdrop, source) on straight sRGB values in [0,1].
BLENDS = {
    "normal": lambda b, s: s,
    "multiply": lambda b, s: b * s,
    "screen": lambda b, s: b + s - b * s,
    "overlay": lambda b, s: _hardlight(s, b),
    "hard_light": _hardlight,
    "soft_light": _softlight,
    "add": lambda b, s: np.minimum(1, b + s),
    "subtract": lambda b, s: np.maximum(0, b - s),
    "color_dodge": _dodge,
    "color_burn": _burn,
    "darken": np.minimum,
    "lighten": np.maximum,
    "difference": lambda b, s: np.abs(b - s),
    "hue": lambda b, s: _setlum(_setsat(s, _sat(b)), _lum(b)),
    "saturation": lambda b, s: _setlum(_setsat(b, _sat(s)), _lum(b)),
    "color": lambda b, s: _setlum(s, _lum(b)),
    "luminosity": lambda b, s: _setlum(b, _lum(s)),
}


def blend(backdrop, source, mode="normal", opacity=1.0, mask=None):
    """Composite straight-RGBA `source` over straight-RGBA `backdrop` with a blend mode."""
    src = premul(rgba(source))
    dst = premul(rgba(backdrop)) * np.ones_like(src[..., :1])
    return unpremul(_composite(dst, src, mode, opacity, mask))


def _bbox(a):
    """Bounding box (y0, y1, x0, x1) of a > 0, or None if empty."""
    rows = np.flatnonzero(a.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(a.any(0))
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _composite(dst, src, mode, opacity=1.0, mask=None):
    """Composite premultiplied src onto premultiplied dst IN PLACE (only where src is non-empty)."""
    if mode not in BLENDS:
        raise ValueError(f"unknown blend '{mode}'. Use one of: {', '.join(BLENDS)}")
    if src.shape[0] > 1 and src.shape[1] > 1:
        bb = _bbox(src[..., 3] > 0)
        if bb is None:
            return dst
        y0, y1, x0, x1 = bb
        src = src[y0:y1, x0:x1]
        if mask is not None and mask.ndim == 2 and mask.shape[0] > 1:
            mask = mask[y0:y1, x0:x1]
        d = dst[y0:y1, x0:x1]
    else:
        d = dst
    k = F32(opacity)
    if mask is not None:
        k = k * mask[..., None]
    a_s = src[..., 3:4] * k
    if mode == "normal":
        d *= 1 - a_s
        d += src * k
        return dst
    cs = unpremul(src)[..., :3]
    ab = d[..., 3:4].copy()
    cb = unpremul(d)[..., :3]
    mixed = (1 - ab) * cs + ab * np.clip(BLENDS[mode](cb, cs), 0, 1)
    d[..., :3] = a_s * mixed + d[..., :3] * (1 - a_s)
    d[..., 3:] = a_s + ab * (1 - a_s)
    return dst


# ---------------------------------------------------------------- image io
def load(src) -> np.ndarray:
    """Path / PIL image / array -> straight RGBA float32."""
    if isinstance(src, (str, os.PathLike)):
        if not os.path.exists(src):
            raise FileNotFoundError(f"image not found: {src}")
        src = Image.open(src)
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGBA"), F32) / 255
    a = np.asarray(src)
    if a.dtype == np.uint8:
        a = a.astype(F32) / 255
    return rgba(a)


def resize(img, W, H, nearest=False):
    """Resize straight RGBA with alpha-correct filtering."""
    h, w = img.shape[:2]
    if (w, h) == (W, H):
        return img
    if nearest:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    interp = cv2.INTER_AREA if W < w else cv2.INTER_LANCZOS4
    out = cv2.resize(premul(img), (W, H), interpolation=interp)
    return unpremul(np.clip(out, 0, None))


def to_u8(img, dither=True):
    """Float [0,1] -> uint8 with tiny deterministic triangular dither (kills gradient banding)."""
    a = np.asarray(img, F32)
    if dither:
        rng = np.random.default_rng(0)
        a = a + (rng.random(a.shape, F32) - rng.random(a.shape, F32)) / 255
    return np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)


def write(img, path, quality=92, bg="white"):
    """Write straight RGBA/RGB float image; format from extension (.png .jpg .jpeg .webp)."""
    img = rgba(img)
    ext = os.path.splitext(str(path))[1].lower()
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    opaque = bool(img[..., 3].min() >= 0.999)
    if ext in (".jpg", ".jpeg") or (opaque and ext in (".png", ".webp")):
        if not opaque:
            img = blend(np.broadcast_to(color(bg), img.shape), img)
        im = Image.fromarray(to_u8(img[..., :3]), "RGB")
    else:
        im = Image.fromarray(to_u8(img), "RGBA")
    if ext in (".jpg", ".jpeg"):
        im.save(path, quality=quality, subsampling=0 if quality >= 90 else 2)
    elif ext == ".webp":
        im.save(path, quality=quality)
    elif ext == ".png":
        im.save(path, optimize=False, compress_level=6)
    else:
        raise ValueError(f"unsupported extension '{ext}' (use .png, .jpg or .webp)")
    return str(path)


# ---------------------------------------------------------------- layers
class Layer:
    """RGBA raster (premultiplied in .px) + opacity, blend mode and optional mask."""

    def __init__(self, canvas, blend="normal", opacity=1.0, mask=None, name=None):
        self.canvas = canvas
        self.px = np.zeros((canvas.H, canvas.W, 4), F32)
        self.blend, self.opacity, self.name = blend, opacity, name
        self.mask = mask
        self.height = None  # optional height map (oil impasto, etc.)
        self.v = 0  # bumped on every change (flatten cache); call touch() after editing .px by hand

    def touch(self):
        self.v += 1
        return self

    # geometry shortcuts
    s = property(lambda self: self.canvas.s)
    W = property(lambda self: self.canvas.W)
    H = property(lambda self: self.canvas.H)
    w = property(lambda self: self.canvas.w)
    h = property(lambda self: self.canvas.h)
    aa = property(lambda self: not self.canvas.pixel)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        c = self.canvas
        self._mask = None if m is None else np.broadcast_to(
            resolve_mask(m, c.H, c.W, None, c.s), (c.H, c.W)).astype(F32)

    @property
    def rgba(self):
        return unpremul(self.px)

    @rgba.setter
    def rgba(self, img):
        self.v += 1
        self.px = premul(np.ascontiguousarray(resolve(img, self.H, self.W, None, self.s) * np.ones((self.H, self.W, 1), F32)))

    @property
    def alpha(self):
        return self.px[..., 3]

    def paint(self, cov, src, box=None, erase=False):
        """Core compositing primitive: paint `src` through coverage `cov` (h,w) placed at pixel
        box (x0,y0,x1,y1). Box may overflow the raster; it is clipped."""
        H, W = self.H, self.W
        if box is None:
            box = (0, 0, W, H)
        x0, y0, x1, y1 = (int(v) for v in box)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if cx0 >= cx1 or cy0 >= cy1:
            return self
        self.v += 1
        if cov is not None:
            cov = np.asarray(cov, F32)
            if cov.ndim == 2 and cov.shape[0] > 1:
                cov = cov[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        else:
            cov = np.ones((1, 1), F32)
        dst = self.px[cy0:cy1, cx0:cx1]
        if erase:
            dst *= 1 - cov[..., None]
            return self
        col = resolve(src, H, W, (cx0, cy0, cx1, cy1), self.s)
        a = col[..., 3:4] * cov[..., None]
        dst[..., :3] = col[..., :3] * a + dst[..., :3] * (1 - a)
        dst[..., 3:] = a + dst[..., 3:] * (1 - a)
        return self

    def fill(self, src, mask=None):
        """Fill the whole layer with a colour / array / field, optionally through a mask.
        The source is only evaluated inside the mask's bounding box."""
        m = resolve_mask(mask, self.H, self.W, None, self.s)
        if m is None or m.shape[0] == 1:
            return self.paint(m, src)
        bb = _bbox(m > 0)
        if bb is None:
            return self
        y0, y1, x0, x1 = bb
        return self.paint(m[y0:y1, x0:x1], src, (x0, y0, x1, y1))

    def erase(self, mask=None):
        """Erase where mask is 1 (everything if None)."""
        return self.paint(resolve_mask(mask if mask is not None else 1.0, self.H, self.W, None, self.s), None, erase=True)

    def clear(self):
        self.v += 1
        self.px[:] = 0
        return self

    def apply(self, *fx):
        """Apply effects (from atelier.fx) to this layer only."""
        img = self.rgba
        for f in fx:
            img = f(img, self.s)
        self.rgba = img
        return self

    def copy(self):
        L = Layer(self.canvas, self.blend, self.opacity, None, self.name)
        L.px = self.px.copy()
        L._mask = None if self._mask is None else self._mask.copy()
        return L

    def __repr__(self):
        return f"<Layer {self.name or ''} {self.W}x{self.H} blend={self.blend} opacity={self.opacity}>"


class Canvas:
    """Stack of layers. Drawing methods called on the canvas go to the current layer (c.cur).

    Canvas(w, h, bg=None, pixel=False). pixel=True: no anti-aliasing, render scale fixed
    at 1, previews/exports upscale with nearest neighbour.
    """

    def __init__(self, w, h, bg=None, pixel=False):
        self.w, self.h, self.pixel = int(w), int(h), pixel
        s = 1.0
        if not pixel:
            if CONFIG.get("width"):
                # the first canvas of a scene fixes the preview scale; helper canvases created later
                # (sub-renders, tiles) reuse it instead of each being blown up to the preview width
                if CONFIG.get("_width_scale") is None:
                    CONFIG["_width_scale"] = CONFIG["width"] / w
                s = CONFIG["_width_scale"]
            elif CONFIG.get("scale"):
                s = float(CONFIG["scale"])
        self.s = s
        self.W, self.H = max(1, round(w * s)), max(1, round(h * s))
        self.layers: list[Layer] = []
        self.cur = None
        self.layer(name="base")
        if bg is not None:
            self.cur.fill(bg)
        _LAST[0] = self

    def __getattr__(self, name):
        cur = self.__dict__.get("cur")
        if cur is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(cur, name)

    # layers
    def layer(self, blend="normal", opacity=1.0, mask=None, name=None) -> Layer:
        """Add an empty layer on top, make it current, return it."""
        L = Layer(self, blend, opacity, mask, name)
        self.layers.append(L)
        self.cur = L
        return L

    def use(self, layer):
        """Make `layer` the current drawing target."""
        self.cur = layer
        return layer

    def flatten(self) -> np.ndarray:
        """Composite all layers -> straight RGBA float32 (H, W, 4). Cached until a layer changes."""
        key = tuple((id(L), id(L.px), L.v, L.blend, L.opacity, id(L._mask)) for L in self.layers)
        if self.__dict__.get("_fkey") == key:
            return self._flat.copy()
        dst = np.zeros((self.H, self.W, 4), F32)
        for L in self.layers:
            if L.opacity > 0:
                _composite(dst, L.px, L.blend, L.opacity, L._mask)
        self._fkey, self._flat = key, unpremul(dst)
        return self._flat.copy()

    def merge(self) -> Layer:
        """Collapse all layers into one normal base layer (made current)."""
        img = self.flatten()
        self.layers = []
        L = self.layer(name="base")
        L.px = premul(img)
        return L

    def apply(self, *fx):
        """Merge everything and apply effects to the whole image (final grading)."""
        return self.merge().apply(*fx)

    def place(self, src, x=None, y=None, scale=None, rot=0.0, fit="cover",
              blend="normal", opacity=1.0, layer=None) -> Layer:
        """Import an image (path/array/PIL) as a new layer. (x, y) = centre in canvas px
        (default: canvas centre); scale=None fits the canvas ('cover'|'contain');
        rot in degrees (counter-clockwise). Pass layer= to paint into an existing layer."""
        img = load(src)
        ih, iw = img.shape[:2]
        if scale is None:
            fx_, fy_ = self.w / iw, self.h / ih
            scale = max(fx_, fy_) if fit == "cover" else min(fx_, fy_)
        cx = (self.w / 2 if x is None else x) * self.s
        cy = (self.h / 2 if y is None else y) * self.s
        k = scale * self.s
        pm = premul(img)
        if k < 0.5 and not self.pixel:  # pre-shrink to avoid aliasing
            nw, nh = max(1, round(iw * k * 2)), max(1, round(ih * k * 2))
            pm = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_AREA)
            k *= iw / nw
            ih, iw = nh, nw
        M = cv2.getRotationMatrix2D((iw / 2, ih / 2), rot, k)
        M[0, 2] += cx - iw / 2
        M[1, 2] += cy - ih / 2
        interp = cv2.INTER_NEAREST if self.pixel else cv2.INTER_LINEAR
        warped = cv2.warpAffine(pm, M, (self.W, self.H), flags=interp,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        L = layer if layer is not None else self.layer(blend, opacity, name="placed")
        _composite(L.px, warped, "normal")
        L.touch()
        return L

    # export
    def _out(self, scale=1, smooth=None):
        img = self.flatten()
        if scale != 1:
            nearest = self.pixel if smooth is None else not smooth
            img = resize(img, round(self.W * scale), round(self.H * scale), nearest)
        return img

    def save(self, path, quality=92, scale=1, smooth=None, bg="white"):
        """Write PNG/JPG/WebP (from extension). scale=2 upsamples the raster (nearest if
        pixel canvas). For a true hi-res render use the CLI: atelier render --scale 2."""
        if CONFIG.get("nosave"):
            return str(path)
        out = write(self._out(scale, smooth), path, quality, bg)
        _SAVED.append(out)
        return out

    def preview(self, path="preview.jpg", width=512):
        """Small JPEG for quick look."""
        img = self.flatten()
        H = max(1, round(self.H * width / self.W))
        img = resize(img, width, H, nearest=self.pixel and width > self.W)
        return write(img, path, quality=85)

    def crop_preview(self, path, x, y, w, h, width=512):
        """Zoom on a region given in canvas px; upscales with nearest so real pixels show."""
        s = self.s
        x0, y0 = max(0, round(x * s)), max(0, round(y * s))
        x1, y1 = min(self.W, round((x + w) * s)), min(self.H, round((y + h) * s))
        img = self.flatten()[y0:y1, x0:x1]
        H = max(1, round((y1 - y0) * width / max(1, x1 - x0)))
        img = resize(img, width, H, nearest=width > (x1 - x0))
        return write(img, path, quality=88)

    def __repr__(self):
        return f"<Canvas {self.w}x{self.h} raster={self.W}x{self.H} layers={len(self.layers)}>"


__all__ = ["Canvas", "Layer", "CONFIG", "BLENDS", "color", "to_linear", "to_srgb", "rgba",
           "premul", "unpremul", "coords", "resolve", "resolve_mask", "blend", "load",
           "resize", "to_u8", "write"]
