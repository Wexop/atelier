# Style: generative abstract

Example: `examples/abstract.py`.

## Recipe
1. Field: `domain_warp(domain_warp(fbm(1.3, 4), .45, 1.2), .1, 2)`; folds with `np.sin(f * np.pi * k)`.
2. Colour: `hue.ramp(curated palette)`; avoid rainbows, use 3–5 related hues + one complement.
3. Composition: a mask Field (S-curve, disc, band) on a dark background so the pattern has a shape.
4. Depth: `shade(f, color=..., strength=.5)` on an overlay layer, `bloom` on highlights, light `grain`.

## Rules
- Large calm areas next to dense detail; one focal zone.
- Seed exploration: render 4–6 seeds in preview, keep the best composition, then refine.
