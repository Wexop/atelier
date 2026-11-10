# Style: photoreal 3D

Examples: `examples/still_life_pt.py` (path traced), `examples/mountains.py`, `examples/seascape.py`, `examples/forest.py`.

## Still life / product
1. `s = Scene()`; objects from SDF primitives, `lathe` for vessels (`thickness=.02`), liquids = solid lathe 2 mm inside.
2. Materials: `Mat(color | Field, rough, metal, glass, ior, local=True, bump=.2, wear=[smudges(.3)])`;
   clear glass `rough=0`. Textures: `wood()`, `marble()`, `fabric()`.
3. Light: `s.studio(...)` or a window `s.area(pattern="window", visible=True)` + soft fill.
4. Draft with `c.render(s)`, final `r = c.pathtrace(s, spp=16)`, then
   `depth_of_field(r.depth, r.focus_at(x, y, c.s), 3)`, `bloom(.85, 10, .25)`, `grain(.012)`.

## Landscape
- `h = terrain(512, "alpine", seed)`, `v = c.landscape(h, sun_elev=6, water=.15, clouds=.25)`; trees on the
  terrain with `v.project(x, y, z)` + `c.tree(..., haze=...)`.
- Sea: `c.sea(sun_elev, waves, foam)`; sky alone: `c.sky(sun_elev, clouds)`.

## Rules
- Low sun (5–15°) for long shadows and colour; cloud cover ≤ .3 overhead.
- Imperfection sells it: wear, smudges, slight bump, DOF, grain.

## Pitfalls
- Pathtrace cost: draft with `c.render`; raise spp only when glossy metal / glass sparkles.
- `tone_map` is for these HDR renders only, not for 2D illustration.
