"""Animation: a scene script reads T (0..1 over the clip) and is rendered once per frame.

    atelier animate scene.py --frames 24 --fps 12          -> scene.gif (also .webp / .mp4 via --out)
    atelier animate scene.py --check                       -> scene_frames.jpg: frames, onion skin, motion curve

T = FRAME / FRAMES, so T = 1 is the first frame again: anything periodic in T (wave, orbit, integer
cycles) loops seamlessly."""
import math

import cv2
import numpy as np

from .core import color as _color, to_u8

F32 = np.float32

# ------------------------------------------------------------------ timing helpers
_EASE = {
    "linear": lambda t: t,
    "in": lambda t: t ** 3,
    "out": lambda t: 1 - (1 - t) ** 3,
    "in_out": lambda t: 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2,
    "sine": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
    "back": lambda t: 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2,      # overshoots, settles
    "bounce": lambda t: 1 - abs(math.cos(t * math.pi * 2.5)) * (1 - t) ** 2,
}


def ease(t, kind="in_out"):
    """Remap t in [0,1]: linear in out in_out sine back bounce."""
    if kind not in _EASE:
        raise ValueError(f"ease: one of {' '.join(_EASE)}")
    return float(_EASE[kind](min(max(float(t), 0.0), 1.0)))


def _val(v):
    if isinstance(v, str):
        return np.asarray(_color(v)[:3], F32), True
    return np.asarray(v, F32), False


def key(t, keys, ease="in_out"):
    """Value at time t from keyframes [(time, value), ...] (times in [0,1], sorted). Values: numbers, tuples
    (positions) or colours ("#hex" -> (r,g,b) 0-1). ease applies on every segment. Before the first / after
    the last key the value holds."""
    keys = sorted(keys, key=lambda k: k[0])
    t = float(t)
    if t <= keys[0][0]:
        v, _ = _val(keys[0][1])
    elif t >= keys[-1][0]:
        v, _ = _val(keys[-1][1])
    else:
        for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
            if t0 <= t <= t1:
                u = _EASE[ease]((t - t0) / max(t1 - t0, 1e-9))
                va, _ = _val(a)
                vb, _ = _val(b)
                v = va + (vb - va) * u
                break
    return float(v) if v.ndim == 0 else tuple(float(x) for x in v)


def wave(t, cycles=1, phase=0.0, lo=0.0, hi=1.0):
    """Smooth back-and-forth between lo and hi, `cycles` times per clip (integer cycles loop seamlessly)."""
    return lo + (hi - lo) * (0.5 - 0.5 * math.cos(2 * math.pi * (cycles * t + phase)))


def pingpong(t):
    """0 -> 1 -> 0 over the clip (linear)."""
    return 1 - abs(2 * (t % 1) - 1)


def orbit(t, r=0.3, cycles=1, phase=0.0):
    """(dx, dy) on a circle of radius r: feed it to Field.move() to animate noise that loops seamlessly."""
    a = 2 * math.pi * (cycles * t + phase)
    return r * math.cos(a), r * math.sin(a)


# ------------------------------------------------------------------ rendering & export
def render_frames(path, frames=24, width=None, seed=0, preview=False, threads=None, log=None):
    """Run scene `path` once per frame with T = i / frames; returns list of straight RGBA float frames."""
    from .cli import run_scene
    out = []
    for i in range(frames):
        cv, _ = run_scene(path, width=width, seed=seed, preview=preview, nosave=True, threads=threads,
                          extra=dict(T=i / frames, FRAME=i, FRAMES=frames))
        out.append(cv.flatten())
        if log:
            log(i)
    return out


_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], F32) / 16 - 0.5


def _rgb8(f, bg=(1, 1, 1)):
    a = f[..., 3:4]
    return f[..., :3] * a + np.asarray(bg, F32) * (1 - a)


def save_gif(frames, path, fps=12, colors=256):
    """GIF with one shared palette and ordered dithering (stable from frame to frame: no crawling noise)."""
    from PIL import Image
    rgb = [_rgb8(f) for f in frames]
    H, W = rgb[0].shape[:2]
    pick = rgb[:: max(1, len(rgb) // 8)]
    sample = np.concatenate([to_u8(x[::2, ::2]) for x in pick], 0)
    pal_img = Image.fromarray(sample).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    th = np.tile(_BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W, None] * (1.5 / 32)
    ims = [Image.fromarray(to_u8(np.clip(x + th, 0, 1))).quantize(palette=pal_img, dither=Image.Dither.NONE)
           for x in rgb]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0, optimize=False,
                disposal=1)
    return str(path)


def save_webp(frames, path, fps=12, quality=85):
    from PIL import Image
    ims = [Image.fromarray(to_u8(f), "RGBA") for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0, quality=quality)
    return str(path)


def save_mp4(frames, path, fps=12):
    H, W = frames[0].shape[:2]
    W2, H2 = W - W % 2, H - H % 2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W2, H2))
    for f in frames:
        vw.write(cv2.cvtColor(to_u8(_rgb8(f))[:H2, :W2], cv2.COLOR_RGB2BGR))
    vw.release()
    return str(path)


def save_anim(frames, path, fps=12):
    """Write .gif / .webp / .mp4 by extension."""
    ext = str(path).lower().rsplit(".", 1)[-1]
    fn = {"gif": save_gif, "webp": save_webp, "mp4": save_mp4}.get(ext)
    if fn is None:
        raise ValueError("animation output: .gif, .webp or .mp4")
    return fn(frames, path, fps)


# ------------------------------------------------------------------ motion check (judge motion from stills)
def continuity_probes(path, frames, n=3, width=None, seed=0, preview=False, threads=None):
    """Re-render a few frames at T + 0.001 with FRAME + 1. A continuous animation barely changes; anything
    re-rolled per frame (seed=FRAME, unseeded randomness) changes completely. Returns [(i, probe_frame)]."""
    from .cli import run_scene
    out = []
    for i in sorted(set(np.linspace(0, frames - 1, n + 2)[1:-1].round().astype(int).tolist())):
        cv, _ = run_scene(path, width=width, seed=seed, preview=preview, nosave=True, threads=threads,
                          extra=dict(T=i / frames + 1e-3, FRAME=i + 1, FRAMES=frames))
        out.append((i, cv.flatten()))
    return out


def motion_metrics(frames, loop=True, probes=None):
    """Per-step change and red flags. change[i] = mean |frame i+1 - frame i| (0-255 scale); seam = change from the
    last frame back to the first relative to the typical step (>2: the loop jumps); flicker (needs probes from
    continuity_probes) = change after a 0.001 nudge of T relative to a normal step (~0 continuous, ~1 re-rolled)."""
    L = [cv2.cvtColor(np.ascontiguousarray(_rgb8(f), F32), cv2.COLOR_RGB2GRAY) for f in frames]
    n = len(L)
    steps = list(range(n - 1)) + ([n - 1] if loop and n > 2 else [])
    ch = [float(np.abs(L[(i + 1) % n] - L[i]).mean() * 255) for i in steps]
    inner = ch[:-1] if loop and n > 2 else ch
    med = float(np.median(inner)) if inner else 0.0
    flags = []
    if med < 0.05:
        flags.append("static: almost nothing moves (does the scene read T?)")
    # the wrap step (last -> first) should look like the steps around it (fast motion near the seam is fine)
    seam = ch[-1] / max(med, 0.5 * (ch[0] + ch[-2]), 1e-3) if loop and n > 2 else None
    if seam is not None and seam > 2.0 and med >= 0.05:
        flags.append(f"loop seam: last -> first frame jumps {seam:.1f}x a normal step (use wave/orbit/integer cycles in T)")
    jumps = [i for i, v in enumerate(inner) if med > 0.05 and v > 3 * med]
    if jumps:
        flags.append(f"jump at frame(s) {', '.join(str(i + 1) for i in jumps[:6])}: change {max(inner[i] for i in jumps):.1f} vs typical {med:.1f}")
    flick = None
    if probes:
        r = []
        for i, pf in probes:
            pl = cv2.cvtColor(np.ascontiguousarray(_rgb8(pf), F32), cv2.COLOR_RGB2GRAY)
            r.append(float(np.abs(pl - L[i]).mean() * 255) / max(med, 1e-3))
        flick = float(np.median(r))
        if flick > 0.3 and med >= 0.05:
            flags.append(f"flicker: nudging T by 0.001 changes the frame {flick:.0%} as much as a full step: something is "
                         "re-rolled per frame (seed=FRAME, unseeded random). Keep seeds fixed; move noise with orbit(T)")
    return dict(change=[round(v, 2) for v in ch], typical=round(med, 2), seam=None if seam is None else round(seam, 2),
                flicker=None if flick is None else round(flick, 2), flags=flags)


def motion_sheet(frames, path, fps=12, cols=6, tile=220, probes=None):
    """Contact sheet to judge motion from a still: up to 12 frames, onion skin (all frames averaged = motion
    trails), difference heat map (where things move) and the per-step change curve with flags."""
    m = motion_metrics(frames, probes=probes)
    n = len(frames)
    idx = sorted(set(np.linspace(0, n - 1, min(n, 12)).round().astype(int).tolist()))
    H, W = frames[0].shape[:2]
    th = max(40, round(tile * H / W))
    rgb = [_rgb8(f) for f in frames]
    tiles = [cv2.resize(rgb[i], (tile, th), interpolation=cv2.INTER_AREA) for i in idx]
    onion = cv2.resize(np.mean(rgb, 0), (tile * 2, th * 2), interpolation=cv2.INTER_AREA)
    diff = np.mean([np.abs(rgb[(i + 1) % n] - rgb[i]).mean(-1) for i in range(n)], 0)
    diff = diff / max(float(diff.max()), 1e-6)
    heat = cv2.applyColorMap(to_u8(diff), cv2.COLORMAP_INFERNO)[..., ::-1].astype(F32) / 255
    heat = cv2.resize(heat, (tile * 2, th * 2), interpolation=cv2.INTER_AREA)
    rows = -(-len(tiles) // cols)
    gap, lab = 6, 18
    Wt = cols * (tile + gap) + gap
    gh = 120
    Ht = gap + rows * (th + lab + gap) + th * 2 + lab + gap + gh + 40
    sheet = np.full((Ht, max(Wt, tile * 4 + gap * 3), 3), 0.1, F32)
    from PIL import Image, ImageDraw
    for k, (i, t) in enumerate(zip(idx, tiles)):
        x, y = gap + (k % cols) * (tile + gap), gap + (k // cols) * (th + lab + gap) + lab
        sheet[y:y + th, x:x + tile] = t
    y2 = gap + rows * (th + lab + gap) + lab
    sheet[y2:y2 + th * 2, gap:gap + tile * 2] = onion
    sheet[y2:y2 + th * 2, gap * 2 + tile * 2:gap * 2 + tile * 4] = heat
    yg = y2 + th * 2 + gap + 22
    img = Image.fromarray(to_u8(sheet))
    d = ImageDraw.Draw(img)
    for k, i in enumerate(idx):
        d.text((gap + (k % cols) * (tile + gap), gap + (k // cols) * (th + lab + gap) + 2),
               f"frame {i + 1}/{n}  T={i / n:.2f}", fill=(220, 220, 220))
    d.text((gap, y2 - lab + 2), "onion skin (all frames): motion trails", fill=(220, 220, 220))
    d.text((gap * 2 + tile * 2, y2 - lab + 2), "where it moves (bright = most change)", fill=(220, 220, 220))
    ch = m["change"]
    gw = sheet.shape[1] - 2 * gap
    top = max(max(ch) * 1.4 if ch else 1, 1e-3)
    pts = [(gap + gw * k / max(len(ch) - 1, 1), yg + gh - gh * v / top) for k, v in enumerate(ch)]
    d.rectangle([gap, yg, gap + gw, yg + gh], outline=(80, 80, 80))
    if len(pts) > 1:
        d.line(pts, fill=(255, 190, 80), width=2)
    d.text((gap, yg - 18), f"change per step (typical {m['typical']}, flicker {m['flicker']}, loop seam "
           f"{m['seam']}x; last point = last -> first)  {n} frames @ {fps} fps = {n / fps:.1f}s", fill=(220, 220, 220))
    d.text((gap, yg + gh + 6), "flags: " + ("; ".join(m["flags"]) if m["flags"] else "none"),
           fill=(255, 120, 100) if m["flags"] else (140, 220, 140))
    img.save(str(path), quality=88)
    return m


__all__ = ["ease", "key", "wave", "pingpong", "orbit", "render_frames", "save_anim", "save_gif", "save_webp",
           "save_mp4", "continuity_probes", "motion_metrics", "motion_sheet"]
