# New style: invent one the package does not cover yet

Use when the request names or shows a look with no guide in `styles/` (a reference image, an art
movement, "like a stained-glass map", "ukiyo-e", "risograph"...). The package is a toolbox, not a catalogue:
almost every look is a combination of primitives. Work it out, paint it, then save it as a guide.

## 1. Study
- Reference image: `analyze_reference("ref.jpg")` → palette, 5-value thumbnail, key, contrast, light
  direction, light/shadow temperature. Also look at it yourself.
- No image: write the look down from what you know of it (era, medium, makers' habits).
- Describe it on six axes, one line each:
  **shapes** (geometric, organic, blobby, faceted) · **edges** (hard, soft, outlined, lost) ·
  **values** (number of tones, key, contrast) · **colour** (palette size, saturation, temperature of shadows) ·
  **texture** (none, paper, grain, strokes, halftone, hatching) · **light** (flat, cel, rim, glow, volumetric).

## 2. Map to the toolbox
For each axis pick the closest primitives (cheat sheet in SKILL.md). Typical mappings:
- flat tones → vector shapes, `posterize`, `toon`, cel quantise a Field with `aastep`;
- painterly → `paint_like` (oil), `watercolor` strokes, `kuwahara`, `watercolorize`;
- print → `halftone`, `linocut`, `xdog`, `c.hatch`, `pixelate(dither=...)`, `stained_glass`, `mosaic`;
- volume → `shade(height)`, 3D `Scene` + `toon` / `render_hatch`, `anime_*` cel helpers;
- light → `glow`, `bloom`, `god_rays`, `lens_flare`, `sparkles`, `color_balance`, `gradient_map`.
If something is missing, write a small helper **in the scene script** (numpy on `L.px` or a `Field(lambda x, y: ...)`).

## 3. Paint
- Start with a value study (greys only) to fix composition, then colour, then texture, then light.
- ≥ 3 look-and-fix rounds with `--preview`, `inspect`, `--crop`; compare against the reference side by side
  (`atelier compare ref.jpg scene.png`).

## 4. Save the style
When the result is good, write `styles/<name>.md` in the same format as the other guides
(title, one-line description, example script, Recipe, Rules, Pitfalls, ≤ 60 lines), and keep the script as its
example. If a helper proved reusable and generic, propose adding it to the package instead.
