# Style: pixel art

Example: `examples/pixel.py`.

## Recipe
1. `c = Canvas(320, 180, pixel=True)` (no anti-aliasing, scale fixed at 1).
2. Palette of 8–16 colours; sub-palettes per zone (sky, water, foliage).
3. Draw at native resolution; per layer `L.apply(pixelate(1, PAL, dither="bayer", strength=.3))`.
4. Export `c.save("x.png", scale=4)` (nearest neighbour).

## Rules
- Silhouettes readable at 1×: test with the preview, not zoomed.
- Dithering only on large gradients (sky, water); keep sprites and edges clean.
- Reflections: mirror rows, roll every other row by ±1 px, darken with depth.

## Pitfalls
- Shapes smaller than 2 px vanish; lines must be 1 px exactly (`width=1`).
