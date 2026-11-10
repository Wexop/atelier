# Style: flat vector, poster, typography

Examples: `examples/poster.py`, `examples/typography.py`, `examples/people.py`, `examples/dusk.py`.

## Recipe
1. Palette of 4–5 colours max (`cosine_palette` or a hand-picked list); one accent.
2. Geometric composition: `rect / circle / polygon / svg`, booleans (`shape & panel` to clip, `|`, `-`).
3. Gradients only inside shapes: `gradient(...).within(shape)`.
4. Shading without gradients: multiply layer + lighting Field + `halftone(...)`, then `c.layer()`.
5. Type: `c.text(..., fit=W - 2 * margin)` for titles, `c.text_box(..., align="justify")` for body,
   `c.title(style="shadow"|"extrude"|"outline"|"neon")`, `c.text_on_path`.
6. People: `c.figure(x, y, h, pose, style="flat")`, crowds with `c.crowd`.

## Rules
- Grid and margins first; align everything to it.
- Flat vector dusk / silhouette scenes: layered hills back to front, each lighter and hazier with distance,
  a warm rim stroke on crests facing the light (`shape.outline(4)` at low opacity).

## Pitfalls
- `c.layer(blend="multiply")` stays current: text drawn after it is multiplied too.
- Hard thresholds alias: use `aastep` / `smoothstep` for visible edges.
