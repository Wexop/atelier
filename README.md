# atelier

**Let your LLM paint.**

LLMs can write, reason and code, but when you ask one for a picture it hands the job to an image model. atelier gives
the LLM itself the brush: it writes a short Python script, renders it, **looks at the result and fixes it**, until the
image is right. No image model is ever called: no diffusion, no image API, no key, no GPU. Every pixel comes from code
the LLM wrote, computed on the CPU.

That covers any style: photoreal 3D, oil and watercolour painting, comics and engravings, people, landscapes, posters
and typography, pixel art, generative abstracts. Under the hood: noise fields, vector shapes, a brush engine, effects,
an SDF path tracer, terrain erosion, a physical sky, posable figures, and tools that let the LLM inspect its own renders.

Website: <https://atelier.wexop.eu> · every image below was made by atelier from a script in [examples/](examples).

![gallery](examples/gallery.jpg)

## Install

```bash
pip install git+https://github.com/Wexop/atelier
```

Python ≥ 3.11. Dependencies: numpy, pillow, opencv-python-headless, scipy, skia-python.

## Install the skill (for AI agents)

[SKILL.md](SKILL.md) is a compact cheat sheet that teaches an agent the whole API and the render → look → fix loop.

**If you are an AI agent:** run `pip install git+https://github.com/Wexop/atelier`, then read
<https://raw.githubusercontent.com/Wexop/atelier/main/SKILL.md> and follow it.

**Claude Code** (personal skill, available in every project):

```bash
mkdir -p ~/.claude/skills/atelier && curl -fsSL https://raw.githubusercontent.com/Wexop/atelier/main/SKILL.md -o ~/.claude/skills/atelier/SKILL.md
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.claude\skills\atelier" | Out-Null; Invoke-WebRequest https://raw.githubusercontent.com/Wexop/atelier/main/SKILL.md -OutFile "$HOME\.claude\skills\atelier\SKILL.md"
```

For a single project, put it in `.claude/skills/atelier/SKILL.md` instead (this repo already has it).

**Claude.ai / Claude desktop:** download `SKILL.md`, put it in a folder named `atelier`, zip the folder and upload it in
*Settings → Capabilities → Skills*. The skill installs the package itself with the `pip install` line above.

**Updating:** nothing updates automatically. Package: `pip install -U git+https://github.com/Wexop/atelier`
(the version is bumped on every package change). Skill: re-run the `curl` / `Invoke-WebRequest` line above,
or re-upload the zip on Claude.ai.

## Quick start

```python
# night.py
from atelier import *

c = Canvas(1600, 1000)
c.fill(gradient([(0, "#0b1026"), (0.6, "#3b2a5a"), (1, "#f4a259")]))
c.layer(blend="screen").fill("white", mask=stars(70, seed=3) * top(0.6, soft=0.2))
c.layer().fill_below(ridged(2.2, seed=1).ridge_line(y=0.7, amp=0.15), fbm(6).ramp(["#1b1b2f", "#2e2e4d"]))
c.apply(bloom(0.8, 12), grain(0.03), vignette(0.3))
c.save("night.png")
```

```bash
atelier render night.py --preview          # fast 512 px look -> night_prev.jpg
atelier inspect night.py                   # contact sheet: value study, clipping, auto zooms on defects, metrics
atelier render night.py --crop 600,500,300,200   # full-res zoom on a region
atelier render night.py                    # final render
```

## What's inside

| module | contents |
|---|---|
| `core` | canvas, layers, 17 blend modes, masks, image import, export, previews |
| `field` | lazy procedural fields: perlin/simplex/value, fbm, ridged, worley, domain warp, patterns, gradients, palettes, 2.5D shading |
| `vector` | skia shapes, SVG paths, booleans, strokes/dashes, text with embedded fonts and GPOS kerning, text on a path, justified text boxes, title presets |
| `brush` | stroke engine (round, ink, chalk, bristle, oil with impasto, watercolour with glazing), scatter, hatching, pigment mixing, `paint_like` |
| `fx` | blur, bloom, glow, shadows, grading (curves, levels, colour balance, AgX), grain, halftone, pixelate + dithering, depth of field |
| `sdf` | 3D signed-distance primitives and ops, materials (bump, object-space textures, wear: smudges, scratches, dust, edge wear), fast raymarcher |
| `pt` | progressive path tracer: global illumination, multi-refraction glass and liquids, adaptive anti-aliasing, AOV-guided denoiser |
| `stylize` | 3D → illustration (toon shading, ink outlines, halftone shadows, surface-following hatching), Kuwahara, XDoG, pencil, watercolour, linocut, stained glass, mosaic |
| `terrain` · `nature` | eroded terrain, physical sky and raymarched clouds, aerial perspective, lakes and seas (FFT waves), 2D and 3D trees, grass, flowers, ferns |
| `figure` | posable people (15 poses, outfits, hair, body types) and animals in flat, line, ink or 3D mannequin styles; perspective crowds |
| `inspect` | quality metrics (noise, aliasing, banding, clipping), contact sheets, A/B compare, composition guides, reference palette and tone transfer |
| `cli` | `atelier render / inspect / compare / reference / palette / brushes / fx` |

The full API cheat sheet (made for an LLM) is [SKILL.md](SKILL.md). Gallery scripts are in [examples/](examples).

## Development

```bash
pip install -e ".[test]"
pytest
python bench.py        # preview / final timings of the demo scenes
```

Contribution rules (language, versioning, SKILL.md sync, tests) are in [CLAUDE.md](CLAUDE.md).

Fonts in `atelier/fonts` are distributed under the SIL Open Font License (see `atelier/fonts/LICENSE.md`).
Code: MIT.
