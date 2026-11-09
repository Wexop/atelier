"""Benchmark: preview (512 px) and final render time of the demo scenes.

  python bench.py              # all scenes, preview + final, 2 threads
  python bench.py --quick      # previews only
  python bench.py --threads 6 --only 06_sdf 02_field
Nothing is written to disk (saves are disabled).
"""
import argparse
import glob
import os
import time

from atelier import core
from atelier.cli import run_scene


def bench(path, threads, final=True):
    out = {}
    modes = [("preview", dict(width=512, preview=True))] + ([("final", dict())] if final else [])
    for name, kw in modes:
        t = time.time()
        cv, _ = run_scene(path, nosave=True, threads=threads, **kw)
        cv.flatten()
        out[name] = (time.time() - t, f"{cv.W}x{cv.H}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="previews only")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", nargs="*", help="scene name prefixes (e.g. 06_sdf)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    scenes = sorted(glob.glob("demos/0*.py")) + sorted(glob.glob("examples/*.py"))
    scenes = [s for s in scenes if "sheet" not in s and "smoke" not in s]
    if a.only:
        scenes = [s for s in scenes if any(os.path.basename(s).startswith(o) for o in a.only)]
    print(f"{'scene':28s} {'preview':>14s} {'final':>20s}   (threads={a.threads})")
    for s in scenes:
        try:
            r = bench(s, a.threads, not a.quick)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(s):28s} error: {e}")
            continue
        p = f"{r['preview'][0]:6.1f}s {r['preview'][1]:>7s}"
        f = f"{r['final'][0]:6.1f}s {r['final'][1]:>10s}" if "final" in r else "-"
        print(f"{os.path.basename(s):28s} {p:>14s} {f:>20s}", flush=True)
    core.CONFIG.update(width=None, scale=None, preview=False, nosave=False)


if __name__ == "__main__":
    main()
