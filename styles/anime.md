# Style: anime painted landscape

Bright, saturated backgrounds in the spirit of hand-painted anime films. Big cel-shaded cumulus, crisp
flat tones with soft gradients only in the sky, strong rim light, warm/cool contrast. Example: `examples/anime_sky.py`
(day), `examples/dusk.py` (flat vector dusk with light effects).

## Recipe
1. Sky: `c.fill(gradient([...]))`. Day = deep cobalt top `#1f5fd0` → pale cyan near the horizon `#cfe8fb`.
   Dusk = indigo → violet → coral → peach (`#2a2350 #7a5fa8 #e0795f #ffd9a0`). Never grey.
2. Clouds: `c.anime_cloud(x, y_base, w, h, palette, light, seed)`: 1 hero tower (w ≈ 40–50 % of width), 2–4
   smaller ones, flat bases roughly on 2–3 horizontal lines (perspective). Low clouds smaller and paler.
3. Far to near, one layer each: far range `anime_mountains(..., haze=.55, paler palette)`, main range
   `anime_mountains(..., snow=.4)`, haze band `c.layer(opacity=.4).fill(haze, mask=band(...))`, rolling meadows
   (`fill_below(fbm(1.2, 2).ridge_line(y, amp), flat colour)`, two tones per band: lit crest + darker body).
4. Trees: `c.anime_tree(x, y, h, kind, palette, light)`: big ones framing the sides, tiny ones mid-ground
   for scale. Same `light` everywhere.
5. Foreground: grass tufts (`c.grass`) + flower dots (`c.scatter("dot")`); petals / fireflies with
   `scatter("petal")`, `scatter("dot") + L.apply(glow(...))`.
6. Light: `c.apply(god_rays(sx, sy), lens_flare(sx, sy), bloom(.7, 18, .5))`; glints on water with
   `c.sparkles(n, region=...)`. Grade: `curves` small lift, `saturation(1.05–1.15)`, light `grain(.01)`.

## Rules
- 3 tones per object (shadow, mid, light) + rim. Shadows are blue/violet, never black or grey.
- One light direction for every element (`light=(az, elev)`); sun side gets the rim light.
- Value plan: sky mid, clouds the lightest, mountains mid-light and hazy, foreground the most saturated.
- Hard edges everywhere except sky gradient, haze and glow.

## Pitfalls
- A cloud in front of a summit reads as fog: keep peaks against blue sky (move clouds up).
- `c.tree` is realistic: use `c.anime_tree` in this style.
- Too many sparkles / petals = noise; 40–120 is plenty on a 1600 px canvas.
