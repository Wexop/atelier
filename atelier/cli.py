"""atelier command line.

  atelier render scene.py                 run the scene (its own save() calls write the final)
  atelier render scene.py --preview 512   fast low-res render -> scene_prev.jpg (no supersampling, saves skipped)
  atelier render scene.py --crop x,y,w,h  full-res zoom on a region -> scene_crop.jpg
  atelier render scene.py --grid [STEP] --boxes --mark x,y   preview/crop with canvas-px grid, layer bounds, marks
  atelier render scene.py --out a.png --scale 2 --seed 3
  atelier inspect scene.py|image.png      contact sheet (view, values, histogram, clipping, auto zooms, metrics)
  atelier critique scene.py|image.png     ranked problems + fixes (contrast, balance, alignment, tangents, edges)
  atelier sheet a.py b.py c.py [--seeds 1,2,3]   variants side by side, scored by critique
  atelier compare a.png b.png             A | B | amplified diff | split  (scenes render at preview size)
  atelier reference ref.jpg               palette, value key, light direction of a reference image
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
    core.CONFIG.update(width=width, scale=scale, preview=preview, nosave=nosave, seed=seed, threads=threads,
                       _width_scale=None)
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
    marks = []
    for m in a.mark or []:
        for part in m.split(";"):
            try:
                v = [float(t) for t in part.split(",")]
                assert len(v) in (2, 4)
            except Exception:
                raise SceneError("--mark expects x,y or x,y,w,h in canvas px (several: 'x,y;x,y,w,h')")
            marks.append(v)
    overlay = a.grid is not None or a.boxes or marks
    if overlay and not (a.preview or crop):
        a.preview = 512
    if a.preview or crop:
        if a.preview:
            cv, _ = run_scene(a.scene, width=a.preview, seed=a.seed, preview=True, nosave=True, threads=a.threads)
            out = a.out or _stem(a.scene, "_prev", ".jpg")
            kind = f"preview {a.preview}px"
        else:
            cv, _ = run_scene(a.scene, scale=a.scale, seed=a.seed, nosave=True, threads=a.threads)
            out = a.out or _stem(a.scene, "_crop", ".jpg")
            kind = f"crop {a.crop}"
        if overlay:
            from .inspect import grid_view
            w = a.width or (768 if crop else max(a.preview, 768))
            r = grid_view(cv, out, width=w, step=a.grid or None, region=crop, marks=marks, boxes=a.boxes)
            kind += f", grid {r['step']:g} px (canvas {cv.w}x{cv.h})"
        elif crop:
            cv.crop_preview(out, *crop, width=a.width or 512)
        else:
            cv.preview(out, width=a.preview)
        print(f"{kind} -> {out} ({time.time() - t0:.1f}s)")
        if overlay and a.boxes:
            for i, name, b in r["boxes"]:
                print(f"  layer {i}{' ' + name if name else ''}: " + ("empty" if b is None else
                      f"x={b[0]} y={b[1]} w={b[2]} h={b[3]}  (centre {b[0] + b[2] / 2:g},{b[1] + b[3] / 2:g})"))
        return
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


def _load_target(path, a, width=None):
    """Image path -> path; scene .py -> rendered Canvas (preview width unless --full)."""
    if not path.lower().endswith(".py"):
        if not os.path.exists(path):
            raise SceneError(f"image not found: {path}")
        return path
    if getattr(a, "full", False):
        cv, _ = run_scene(path, scale=getattr(a, "scale", None), seed=a.seed, nosave=True, threads=a.threads)
    else:
        w = width or a.preview
        cv, _ = run_scene(path, width=w, seed=a.seed, preview=True, nosave=True, threads=a.threads)
    return cv


def cmd_inspect(a):
    from .inspect import contact_sheet, guides
    t0 = time.time()
    img = _load_target(a.target, a)
    out = a.out or _stem(a.target, "_sheet", ".jpg")
    r = contact_sheet(img, out, crop_size=a.crop_size, title=os.path.basename(a.target))
    m = r["metrics"]
    print(f"sheet -> {out} ({time.time() - t0:.1f}s)")
    print(f"  lightness {m['mean']}/{m['contrast']} range {m['range'][0]}-{m['range'][1]} values {m['values']}  "
          f"sat {m['saturation']} colorful {m['colorfulness']}")
    print(f"  noise {m['noise']} aliasing {m['aliasing']}% banding {m['banding']}% sharp {m['sharpness']}  "
          f"clip black {m['clip_shadows']}% white {m['clip_highlights']}% chan {m['clip_channels']}%")
    c = r["composition"]
    print(f"  focal {c['focal']} mass {c['mass']} balance {c['balance']} tilt {c['tilt']}  palette {' '.join(h for h, _ in r['palette'])}")
    for f in m["flags"]:
        print(f"  ! {f}")
    print("  zooms: " + "  ".join(f"{n}=--crop {','.join(map(str, b))}" for n, b, _ in r["crops"]))
    if a.guides:
        g = _stem(out, "_guides", ".jpg").replace("_sheet_guides", "_guides")
        guides(img, g)
        print(f"guides -> {g}")


def _focus_arg(v):
    if v is None or isinstance(v, (tuple, list)):
        return v
    parts = str(v).split(",")
    if len(parts) == 4:
        try:
            return tuple(float(t) for t in parts)
        except ValueError:
            pass
    return str(v)


def _critique_target(path, a):
    """-> (image or Canvas, focus) ; a scene may define FOCUS = "layer" | (x, y, w, h)."""
    focus = _focus_arg(getattr(a, "focus", None))
    if not path.lower().endswith(".py"):
        return _load_target(path, a), focus
    if getattr(a, "full", False):
        cv, g = run_scene(path, scale=a.scale, seed=a.seed, nosave=True, threads=a.threads)
    else:
        cv, g = run_scene(path, width=a.preview, seed=a.seed, preview=True, nosave=True, threads=a.threads)
    return cv, focus if focus is not None else _focus_arg(g.get("FOCUS"))


def _hier_line(h):
    sc = h.get("scores", {})
    if not sc:
        return f"  hierarchy: focus {h['focus']} (not enough area to compare)"
    parts = [f"{k} {v['ratio']:.2f}x" + ("" if v["ratio"] >= 0.9 else f" (<{v['best_other']})") for k, v in sc.items()]
    sq = f"  squint dL {h['squint']}" if "squint" in h else ""
    return f"  hierarchy: focus {h['focus']} vs rest: {'  '.join(parts)}{sq}"


def cmd_critique(a):
    from .inspect import critique
    t0 = time.time()
    img, focus = _critique_target(a.target, a)
    out = a.out or _stem(a.target, "_critique", ".jpg")
    r = critique(img, out, focus=focus)
    iss = r["issues"]
    n = {k: sum(t["sev"] == k for t in iss) for k in ("high", "mid", "low")}
    print(f"critique -> {out} ({time.time() - t0:.1f}s)  {n['high']} high, {n['mid']} mid, {n['low']} low")
    print(_hier_line(r["hierarchy"]))
    for t in iss:
        print(f"  {t['n']:>2} [{t['sev']}] {t['kind']}: {t['msg']}" + (f"  box {','.join(map(str, t['box']))}" if t["box"] else ""))
    if not iss:
        print("  no issue found (still look at the image)")


def cmd_sheet(a):
    from .inspect import variant_sheet
    t0 = time.time()
    items, labels, focs = [], [], []
    seeds = [int(v) for v in a.seeds.split(",")] if a.seeds else [a.seed]
    for t in a.targets:
        for sd in seeds:
            a.seed = sd
            img, fo = _critique_target(t, a)
            items.append(img)
            focs.append(fo)
            labels.append(os.path.splitext(os.path.basename(t))[0] + (f" seed {sd}" if a.seeds else ""))
    if a.labels:
        labels = a.labels.split(",")
        if len(labels) != len(items):
            raise SceneError(f"--labels needs {len(items)} names")
    out = a.out or os.path.join(os.path.dirname(a.targets[0]), "variants.jpg")
    res = variant_sheet(items, out, labels=labels, focus=focs)
    print(f"sheet -> {out} ({time.time() - t0:.1f}s)")
    for r in sorted(res, key=lambda r: -r["score"]):
        c = {k: sum(t["sev"] == k for t in r["issues"]) for k in ("high", "mid", "low")}
        print(f"  {r['score']:>4}  {r['label']}  ({c['high']} high, {c['mid']} mid, {c['low']} low)")


def cmd_compare(a):
    from .inspect import compare
    ia, ib = _load_target(a.a, a), _load_target(a.b, a)
    labels = tuple(a.labels.split(",")) if a.labels else (os.path.basename(a.a), os.path.basename(a.b))
    if len(labels) != 2:
        raise SceneError("--labels expects two names, e.g. --labels before,after")
    out = a.out or os.path.join(os.path.dirname(a.a), f"{os.path.splitext(os.path.basename(a.a))[0]}_vs_"
                                f"{os.path.splitext(os.path.basename(a.b))[0]}.jpg")
    r = compare(ia, ib, out, labels=labels, amp=a.amp)
    print(f"compare -> {out}  mean diff {r['mean_diff']} max {r['max_diff']} changed {r['changed']}% "
          f"PSNR {r['psnr']} dB bbox {r['bbox']}")


def cmd_reference(a):
    from .inspect import analyze_reference
    r = analyze_reference(_load_target(a.image, a), k=a.colors)
    print(f"palette  {' '.join(f'{h}:{w * 100:.0f}%' for h, w in r['palette'])}")
    print(f"key {r['key']} (mean {r['mean']}, contrast {r['contrast']})  light from {r['light']} "
          f"(azimuth {r['light_azimuth']} deg, strength {r['light_strength']})  "
          f"warmth lights {r['temperature']['lights']} shadows {r['temperature']['shadows']}")


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
    r.add_argument("--grid", type=float, nargs="?", const=0, help="coordinate grid in canvas px (step, default auto); implies --preview")
    r.add_argument("--mark", action="append", help="check a placement: x,y or x,y,w,h (canvas px); repeat or ';'-separate")
    r.add_argument("--boxes", action="store_true", help="outline each layer's visible bounds and print them")
    r.add_argument("--seed", type=int, default=0, help="value of SEED inside the script")
    r.add_argument("--scale", type=float, help="render scale for the final (2 = twice the pixels)")
    r.add_argument("--quality", type=int, default=92)
    r.add_argument("--threads", type=int, help="threads for 3D rendering (default: all cores)")
    r.add_argument("--debug", action="store_true", help="full Python traceback on error")
    def common(q, preview=640):
        q.add_argument("--out", help="output image (.jpg/.png)")
        q.add_argument("--preview", type=int, default=preview, help=f"render width for scenes (default {preview})")
        q.add_argument("--full", action="store_true", help="render scenes at full resolution")
        q.add_argument("--scale", type=float, help="render scale with --full")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--threads", type=int)
        q.add_argument("--debug", action="store_true")
    i = sub.add_parser("inspect", help="contact sheet + metrics for a scene or image")
    i.add_argument("target")
    common(i)
    i.add_argument("--crop-size", type=int, default=100, help="zoom window size in raster px")
    i.add_argument("--guides", action="store_true", help="also write <name>_guides.jpg (composition overlay)")
    cr = sub.add_parser("critique", help="ranked problems with fixes: contrast, balance, alignment, tangents, edges")
    cr.add_argument("target")
    common(cr, preview=1024)
    cr.add_argument("--focus", help="intended focus: layer/group name or x,y,w,h (default: FOCUS in the scene, else auto)")
    sh = sub.add_parser("sheet", help="variants / thumbnails side by side with critique scores")
    sh.add_argument("targets", nargs="+")
    common(sh, preview=512)
    sh.add_argument("--seeds", help="render each scene with these seeds, e.g. 1,2,3,4")
    sh.add_argument("--labels", help="comma-separated tile names")
    sh.add_argument("--focus", help="intended focus for all (default: FOCUS in each scene, else auto)")
    cp = sub.add_parser("compare", help="A/B images or scenes side by side with a difference map")
    cp.add_argument("a")
    cp.add_argument("b")
    common(cp)
    cp.add_argument("--labels", help="two labels, e.g. before,after")
    cp.add_argument("--amp", type=float, default=4.0, help="difference amplification (default 4)")
    rf = sub.add_parser("reference", help="palette / value key / light direction of a reference image")
    rf.add_argument("image")
    common(rf)
    rf.add_argument("--colors", type=int, default=6)
    sub.add_parser("palette", help="list built-in palettes")
    sub.add_parser("brushes", help="list brushes and scatter kinds")
    sub.add_parser("fx", help="list effects with signatures")
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 0
    try:
        {"render": cmd_render, "inspect": cmd_inspect, "critique": cmd_critique, "sheet": cmd_sheet, "compare": cmd_compare, "reference": cmd_reference,
         "palette": cmd_palette, "brushes": cmd_brushes, "fx": cmd_fx}[a.cmd](a)
    except SceneError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        if getattr(a, "debug", False):
            raise
        scene = next((v for v in (getattr(a, k, None) for k in ("scene", "target", "a", "b")) if isinstance(v, str) and v.endswith(".py")), "")
        print(_short_error(e, getattr(a, "scene", None) or scene), file=sys.stderr)
        print("  (run with --debug for the full traceback)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
