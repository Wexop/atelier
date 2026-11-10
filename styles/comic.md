# Style: comic, ligne claire, engraving

Examples: `examples/comic.py` (ligne claire), `examples/engraving.py` (copper plate). Model the subject in 3D,
then stylize the render: perspective and shading come for free.

## Ligne claire / toon
1. Build a `Scene` with simple SDF shapes; `r = c.render(s, alpha=True)`.
2. On a layer: `toon(r, bands=3, shadow=.45, colors={id: col})`, `shadow_dots(r, size=5, lo=.3)` (halftone in
   shadows), `outlines(r, width=2.5, inner=1)`.
3. Flat gradient sky, vector clouds (union of circles + stroke), speech balloon = ellipse | polygon tail + `c.text`.
- Flat saturated local colours, black outlines of constant width, no texture.

## Engraving
1. `L.fill(PAPER, mask=r.alpha)`; `L.apply(render_hatch(r, s, spacing=4.5, layers=3), outlines(r, width=2))`.
2. Cream paper, sepia or black ink, a ruled frame and a caption in serif small caps.

## Other ink looks
- `xdog(sigma, eps)` ink lines on any image; `pencil()`; `linocut(lines, angle)`; `c.hatch(region, tone=Field)`.

## Pitfalls
- Outlines too thin vanish in previews: check at final size with `--crop`.
