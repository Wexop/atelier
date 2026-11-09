"""atelier command line.

  atelier render scene.py                 run the scene (its own save() calls write the final)
  atelier render scene.py --preview 512   fast low-res render -> scene_prev.jpg (no supersampling, saves skipped)
  atelier render scene.py --crop x,y,w,h  full-res zoom on a region -> scene_crop.jpg
  atelier render scene.py --out a.png --scale 2 --seed 3
  atelier palette | brushes | fx          short reference lists
A scene is a Python script that builds a Canvas (named `canvas` or `c`, else the last one created).
`SEED` (from --seed, default 0) is available inside the script.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from . import core
from .core import Canvas


class SceneError(Exception):
    pass


def run_scene(path, width=None, scale=None, seed=0, preview=False, nosave=False, threads=None):
    """Execute a scene script with render settings; returns (canvas, namespace)."""
    if not os.path.exists(path):
        raise SceneError(f"scene not found: {path}")
    saved = dict(core.CONFIG)
    core.CONFIG.update(width=width, scale=scale, preview=preview, nosave=nosave, seed=seed, threads=threads)
    core._LAST[0] = None
    g = {"__name__": "__main__", "__file__": os.path.abspath(path), "SEED": seed}
    d = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, d)
    try:
        src = open(path, encoding="utf8").read()
        exec(compile(src, path, "exec"), g)
    finally:
        sys.path.remove(d)
        core.CONFIG.clear()
        core.CONFIG.update(saved)
    cv = next((g[k] for k in ("canvas", "c") if isinstance(g.get(k), Canvas)), None) or core._LAST[0]
    if cv is None:
        raise SceneError("the scene created no Canvas (name it `canvas` or `c`)")
    return cv, g


def _short_error(e, path):
    """One actionable line: file:line, exception and source line from the user's script."""
    tb = traceback.extract_tb(e.__traceback__)
    frames = [f for f in tb if os.path.abspath(f.filename) == os.path.abspath(path)]
    where = ""
    if frames:
        f = frames[-1]
        where = f"{path}:{f.lineno}: {f.line.strip() if f.line else ''}\n  "
    elif tb:
        f = tb[-1]
        where = f"{os.path.basename(f.filename)}:{f.lineno}\n  "
    return f"error: {where}{type(e).__name__}: {e}"


def _stem(path, suffix, ext):
    base = os.path.splitext(path)[0]
    return f"{base}{suffix}{ext}"


def cmd_render(a):
    t0 = time.time()
    crop = None
    if a.crop:
        try:
            crop = [float(v) for v in a.crop.split(",")]
            assert len(crop) == 4
        except Exception:
            raise SceneError("--crop expects x,y,w,h in canvas pixels, e.g. --crop 400,300,200,120")
    if a.preview:
        cv, _ = run_scene(a.scene, width=a.preview, seed=a.seed, preview=True, nosave=True, threads=a.threads)
        out = a.out or _stem(a.scene, "_prev", ".jpg")
        cv.preview(out, width=a.preview)
        kind = f"preview {a.preview}px"
    elif crop:
        cv, _ = run_scene(a.scene, scale=a.scale, seed=a.seed, nosave=True, threads=a.threads)
        out = a.out or _stem(a.scene, "_crop", ".jpg")
        cv.crop_preview(out, *crop, width=a.width or 512)
        kind = f"crop {a.crop}"
    else:
        nosave = a.out is not None
        before = set(core._SAVED)
        cv, _ = run_scene(a.scene, scale=a.scale, seed=a.seed, nosave=nosave, threads=a.threads)
        if a.out:
            out = cv.save(a.out, quality=a.quality)
        else:
            new = [p for p in core._SAVED if p not in before]
            out = ", ".join(new) if new else cv.save(_stem(a.scene, "", ".png"))
        kind = f"final {cv.W}x{cv.H}"
    print(f"{kind} -> {out} ({time.time() - t0:.1f}s)")


def cmd_palette(a):
    from .field import PALETTES
    for k, v in PALETTES.items():
        print(f"{k:9s} {' '.join(v)}")


def cmd_brushes(a):
    from .brush import BRUSHES, _SCATTER_DEF
    for k, v in BRUSHES.items():
        extra = ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ("tip", "spacing", "taper"))
        print(f"{k:11s} {extra}")
    print("scatter:   " + " ".join(_SCATTER_DEF) + " | any Shape | brush name (soft dab)")


def cmd_fx(a):
    from . import fx
    import inspect
    for n in fx.__all__:
        f = getattr(fx, n)
        try:
            sig = str(inspect.signature(f))
        except (TypeError, ValueError):
            sig = "(...)"
        print(f"{n}{sig}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="atelier", description="Procedural image studio for LLMs.")
    p.add_argument("--version", action="version", version=f"atelier {__import__('atelier').__version__}")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("render", help="run a scene script")
    r.add_argument("scene")
    r.add_argument("--out", help="output file (.png .jpg .webp)")
    r.add_argument("--preview", type=int, nargs="?", const=512, help="fast preview at this width (default 512)")
    r.add_argument("--crop", help="x,y,w,h zoom region in canvas px (full-res)")
    r.add_argument("--width", type=int, help="width of the crop image (default 512)")
    r.add_argument("--seed", type=int, default=0, help="value of SEED inside the script")
    r.add_argument("--scale", type=float, help="render scale for the final (2 = twice the pixels)")
    r.add_argument("--quality", type=int, default=92)
    r.add_argument("--threads", type=int, help="threads for 3D rendering (default: all cores)")
    r.add_argument("--debug", action="store_true", help="full Python traceback on error")
    sub.add_parser("palette", help="list built-in palettes")
    sub.add_parser("brushes", help="list brushes and scatter kinds")
    sub.add_parser("fx", help="list effects with signatures")
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 0
    try:
        {"render": cmd_render, "palette": cmd_palette, "brushes": cmd_brushes, "fx": cmd_fx}[a.cmd](a)
    except SceneError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        if getattr(a, "debug", False):
            raise
        print(_short_error(e, getattr(a, "scene", "")), file=sys.stderr)
        print("  (run with --debug for the full traceback)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
