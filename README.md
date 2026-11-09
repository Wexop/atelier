# atelier

A procedural image studio designed to be driven by an LLM: short Python scripts produce PNG/JPG/WebP images in any
style (photoreal 3D, oil and watercolour painting, flat posters, pixel art, generative abstracts) with **no generative
AI and no image API**. Everything is computed: noise fields, vector shapes, a brush engine, effects and a small
SDF raymarcher, all on the CPU.

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
atelier render night.py --crop 600,500,300,200   # full-res zoom on a region
atelier render night.py                    # final render
```

## What's inside

| module | contents |
|---|---|
| `core` | canvas, layers, 17 blend modes, masks, image import, export, previews |
| `field` | lazy procedural fields: perlin/simplex/value, fbm, ridged, worley, domain warp, patterns, gradients, palettes, 2.5D shading |
| `vector` | skia shapes, SVG paths, booleans, strokes/dashes, text with embedded fonts (Inter, JetBrains Mono) |
| `brush` | stroke engine (round, ink, chalk, bristle, oil with impasto, watercolour with glazing), scatter, hatching, pigment mixing, `paint_like` |
| `fx` | blur, bloom, glow, shadows, grading (curves, levels, colour balance, AgX), grain, halftone, pixelate + dithering, depth of field |
| `sdf` | 3D signed-distance primitives and ops, PBR-ish materials, glass, subsurface, soft shadows, AO, reflections, fog |
| `cli` | `atelier render / palette / brushes / fx` |

The full API cheat sheet (made for an LLM) is [SKILL.md](SKILL.md). Gallery scripts are in [examples/](examples).

## Development

```bash
pip install -e ".[test]"
pytest
python bench.py        # preview / final timings of the demo scenes
```

Fonts in `atelier/fonts` are distributed under the SIL Open Font License (see `atelier/fonts/LICENSE.md`).
Code: MIT.
