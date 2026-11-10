# Style: oil and watercolour painting

Example: `examples/landscape.py` (oil). Paint from a flat colour study, coarse strokes first, fine last.

## Oil
1. Build `ref = Canvas(w, h)` as a flat colour study: gradient sky, `clouds()` ramped, `fill_below(ridge_line)`
   hills, svg river, simple tree blobs. Big shapes, no detail: the strokes add it.
2. `c.paint_like(ref, "oil", sizes=(64, 32, 16, 9), seed=SEED)`; sizes = coarse → fine passes.
3. `c.cur.relief = .6` (impasto lighting). Grade with `curves`, `vignette(.25)`, light `grain(.015)`.
- Keep the study's values clear: the strokes blur the value structure, they never fix it.
- Warm lights / cool shadows in the study; saturated accents only near the focal point.

## Watercolour
1. Paper background `#f4efe4`; paper tooth: `c.layer(blend="overlay", opacity=.5).fill(paper())`, then `c.layer()`.
2. Broad washes `c.stroke(points, "watercolor", size=40–90, color)`, light → dark; glazes multiply.
3. Leave the paper white for the lightest lights; never paint white.
4. Details last with `ink` brush, few and loose. Or render any image and `watercolorize(bleed, edge, granulation)`.

## Pitfalls
- Too many fine passes = muddy: stop at 9–12 px.
- Watercolour wash on a `multiply` layer then text: call `c.layer()` before drawing anything else.
