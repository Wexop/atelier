# Rules for this repository

Applies to every contributor, human or AI (Claude Code loads this file automatically).

## Language
- **English only** in the repo: code, comments, docstrings, docs, commit messages, CLI output, error messages,
  and any text rendered into example or demo images. Conversations with the maintainer can be in any language.

## Versioning
- The version lives in one place: `__version__` in `atelier/__init__.py` (`pyproject.toml` reads it).
- Bump it for every change pushed to `main` that touches the `atelier/` package or `SKILL.md`,
  so `pip install -U git+https://github.com/Wexop/atelier` picks it up.
  Semver: patch = fix, minor = new feature or API addition, major = breaking change.
- Docs-only or example-only changes do not need a bump.

## SKILL.md
- `SKILL.md` is the cheat sheet an AI loads to use the tool: keep it **≤ 200 lines**, dense, accurate.
- Any API change (new function, renamed parameter, new default) updates `SKILL.md` in the same commit.
- `.claude/skills/atelier/SKILL.md` must stay identical to the root `SKILL.md`:
  `cp SKILL.md .claude/skills/atelier/SKILL.md`.
- The install line in `SKILL.md` and `README.md` stays `pip install git+https://github.com/Wexop/atelier`.

## Code
- Constraints, in priority order: token-economical API (short, composable, sensible defaults), fast preview loop,
  deterministic (everything random takes `seed`), CPU only (must run on 2 cores), no mandatory external assets.
- Flat functions, no plugin systems, config systems or needless abstract classes.
- Dependencies: numpy, pillow, opencv-python-headless, scipy, skia-python. Anything else needs a written justification.
- Export names must be unique across modules (`from atelier import *` merges them); check with
  `python -c "import atelier, collections; print([k for k, v in collections.Counter(atelier.__all__).items() if v > 1])"`.
- Slow imports (e.g. `scipy.interpolate`) go inside the function that needs them.

## Tests & checks
- `pytest` must pass before every commit; new features come with a test (known values, seed determinism, or a tiny render).
- Visual changes are checked by looking at renders: `atelier render scene.py --preview`, then `--crop` on details.
- Performance-sensitive changes: run `python bench.py` (2 threads) and keep finals under ~60 s.

## Examples
- `examples/*.py` save next to themselves (`c.save(__file__[:-3] + ".png")`) and are portfolio quality:
  at least 3 look-and-fix iterations each.
- After changing an example image, rebuild the banner: `python examples/make_gallery.py`.
- Do not commit previews, crops or other intermediates (`*_prev.jpg`, `*_crop.jpg`, `demos/out/`).
