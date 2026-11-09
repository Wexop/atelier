---
name: atelier
description: Make images (PNG/JPG/WebP) yourself with code, no generative AI — photoreal 3D (SDF raymarching), oil/watercolour painting, flat posters with typography, pixel art, generative abstracts. Use when asked to create, draw, paint or render an image.
---
# atelier — image studio driven by short Python scripts

Install (Python ≥ 3.11): `pip install git+https://github.com/Wexop/atelier`
Check: `atelier palette` · `atelier brushes` · `atelier fx` (signatures).

## Loop (always)
1. Write `scene.py` (30–80 lines, `from atelier import *`, build a Canvas `c`, end with `c.save(__file__[:-3] + ".png")`).
2. `atelier render scene.py --preview` → `scene_prev.jpg` (512 px, real low-res render, saves skipped). **Read the image.**
3. Fix what looks wrong; zoom with `atelier render scene.py --crop x,y,w,h` (canvas px, full res).
4. ≥3 look-and-fix rounds, then `atelier render scene.py` (final; `--scale 2` = true 2× render, `--seed N` → `SEED`).
Errors print one line `scene.py:LINE: Type: message` (`--debug` for traceback).

## Conventions
- Drawing coords = canvas px (y down). Fields use normalised x, y ∈ [0,1]. Colours: "#hex", CSS names, (r,g,b) 0–1 or 0–255.
- Paint source (fill=, color=, stroke=) = colour | Field | f(x,y) | array H×W×3/4. Masks = Field | Shape | array | f(x,y).
- Everything random takes `seed=`. Previews render at low res: the picture is identical, only sharper at final.

## Canvas & layers (core)
```
c = Canvas(w, h, bg=None, pixel=False)      # pixel=True: no AA, fixed scale, nearest-neighbour export
L = c.layer(blend="normal", opacity=1, mask=None)   # new layer, BECOMES CURRENT; c.use(L) to switch back
blends: normal multiply screen overlay soft_light hard_light add subtract color_dodge color_burn darken lighten difference hue saturation color luminosity
c.fill(src, mask=None) · c.erase(mask) · c.place(img|path, x=None, y=None, scale=None(cover), rot=0, fit="cover"|"contain", blend, opacity) -> layer
c.apply(*fx, mask=None)   # MERGES all layers, then effects (final grading);  L.apply(*fx, mask=None) = one layer
c.save(path, quality=92, scale=1, smooth=None) · c.preview(path, width=512) · c.crop_preview(path, x, y, w, h)
c.flatten() -> H×W×4 float · L.rgba (get/set) · c.w c.h (logical) c.W c.H c.s (raster scale)
```

## Fields (field) — lazy, composable: + - * / ** > <, np.sin(f), f.ramp() f.warp() f.scale() f.rotate() f.move() f.remap(lo,hi) f.clip() f.within(shape|(x0,y0,x1,y1))
```
X Y radial(cx,cy) angle(cx,cy) · Field(lambda x, y: ...)
perlin/simplex/value(scale=4, seed) · fbm(scale=4, octaves=5, lacunarity=2, gain=.5, seed, base="perlin")
ridged(scale=3, ...) · turbulence(scale=4, ...) · worley(scale=6, mode="F1"|"F2"|"F2-F1"|"id", seed)
domain_warp(f, strength=.1, scale=2, seed) · aastep(f, edge=.5) · smoothstep(e0, e1, v) · mix(a, b, t)
patterns [0,1]: wood(rings=14, angle) marble() clouds(scale, cover=.5, sharp) terrain_height() water_ripples() fabric(scale=60) paper() stars(density=60, size)
gradient(stops, angle=90 (top→bottom; 0 = left→right), kind="linear"|"radial"|"conic", center, radius)
ramp(f, stops) / f.ramp("sunset") · stops = palette name | ["#..", ...] | [(pos, col), ...]   (OKLab interpolation)
palette("name") names: sunset night ocean forest desert earth fire ice pastel neon vapor viridis wood marble mono retro16 green4
cosine_palette(a, b, c, d, n=8) · rgb2hsv hsv2rgb rgb2oklab oklab2rgb
shade(height, light=(azimuth, elevation), strength=1, color=None, ambient=.25, spec=0)   # no color → grey, flat=0.5 (use on overlay layer)
masks: top(f) bottom(f) left(f) right(f) band(y0, y1) zone(x0, y0, x1, y1) disc(cx, cy, r)  (all soft=)
curves: f.ridge_line(y=.6, amp=.15) / curve(lambda x: y) → c.fill_below(curve, src) / fill_above · curve.below()
```

## Vector & text (vector) — Shape = skia path, also a Field mask; booleans | & - ^
```
rect(x, y, w, h, r=0) ellipse(cx, cy, rx, ry) circle(cx, cy, r) polygon(pts) line(x1,y1,x2,y2 | pts) regular(cx,cy,r,n)
star(cx, cy, r, r2, n=5) arc(cx, cy, r, start, sweep) spline(pts, closed) svg("M .. C .. A .. Z") union(*shapes)
shape.move/rotate(deg)/scale/outline(width, dash)/soft(feather)/points(n)/bounds/length
c.draw(shape, fill="black", stroke=None, width=2, cap="round", join="round", dash=(on, off), feather=0, opacity=1)
c.rect/ellipse/circle/polygon/line/spline/path(d)(..., **same style)
c.text(txt, x, y, size=32, fill, font="sans"|"mono"|"serif"|family|path.ttf, weight="regular"|"semibold"|"bold"|int,
       align="left"|"center"|"right", valign="baseline"|"top"|"middle"|"bottom", spacing=em, line_height=1.2, italic, fit=width_px)
text_shape(...) → Shape (text as mask/outline) · text_size(txt, size) → (width, cap_height)
```

## Brushes (brush)
```
c.stroke(points|Shape, brush="round", size=12, color, pressure=[...]|f(t)|float, opacity, jitter, spacing, taper=(a, b), seed, **params)
brushes: round round_soft airbrush ink(speed) chalk(tooth) bristle(n, var, gaps) oil(impasto, pickup) watercolor(edge, granulation, bleed)
c.paint_like(ref, brush="oil", sizes=(48,24,12), tol=.08, angle=0, seed)   # repaint a reference (Canvas/Field/image) with strokes
c.scatter(what, n, region=mask, size=(a, b), color=col|list|palette|Field, color_jitter, rot=(a, b), seed)
   what: dot soft star grass leaf petal square person | Shape (unit size at 0,0)
c.hatch(region, angle=45, spacing=8, width=1.2, color, tone=Field (0=dark → thick), cross=False, wobble)
km_mix(a, b, t) pigment-like mixing (blue+yellow = green) · L.relief = 1.0 (oil impasto lighting strength)
```

## Effects (fx) — c.apply(...) / L.apply(...); sizes in canvas px
```
blur(r) motion_blur(len, angle) sharpen(amt) bloom(threshold=.75, radius=12, strength=.8, tint) glow(r, strength, color)
drop_shadow(dx, dy, r, color, opacity) inner_shadow(...) vignette(amount=.35, softness=.6) grain(amount=.04, size=1, seed)
chromatic_aberration(px) lens_distortion(k) exposure(stops) contrast(amt) levels(black, white, gamma) curves([(in, out), ...], channel)
hue_sat(hue_deg, sat_mult, light) saturation(x) color_balance(shadows=(r,g,b), mids, highlights) gradient_map(stops) invert()
tone_map("agx"|"filmic"|"reinhard", exposure, look="punchy") displace(field, strength) posterize(levels)
pixelate(size, palette=name|list, colors=N, dither=None|"bayer"|"fs", strength=.6) halftone(size, angle, ink, paper, cmyk=False)
depth_of_field(depth, focus, aperture=8)   # depth from r = c.render(scene)
```

## 3D (sdf) — y up, units ≈ metres, primitives centred at origin
```
sphere(r) box(w, h, d) round_box(w, h, d, r) cylinder(r, h) capsule(a, b, r) torus(R, r) cone(r1, h, r2) plane(y, normal)
ellipsoid(rx, ry, rz) lathe([(radius, y), ...] bottom→top, thickness=None|shell) extrude([(x, y), ...], depth)
ops: a | b, a & b, a - b, a.union(b, k=smooth) .inter .sub · .move(x,y,z) .rotate(x,y,z deg) .scale(k|(kx,ky,kz))
     .onion(t) .round(r) .twist(k) .bend(k) .repeat(period, count) .mirror("x") .elongate() .displace(noise3(scale, octaves, seed), amount)
Mat(color | 2D Field (proj="tri"|"xz"|"xy"|"zy", tex_scale) | f(p, n)->(N,3), rough=.5, metal=0, glass=0, ior=1.45,
    sss=0, sss_color, emit=None, power=1)
s = Scene(); id = s.add(sdf, Mat(...)); s.camera(pos, target, fov=40)
s.sun(dir_toward_sun, color, power=3, soft=.04) · s.point(pos, color, power, radius) · s.area(pos, size=(w,h), normal, color, power)
s.sky(zenith, horizon, ground, power) · s.fog(color, density)
r = c.render(s, aa=1|2, exposure=0, tonemap="agx", look=None|"punchy", alpha=False)   # paints current layer
r.depth r.normal r.id r.mask(id) r.focus_at(x, y, c.s) r.linear
```

## Recipes
**Photoreal** — table `box` or `plane` with `Mat(wood().ramp([...]), proj="xz", tex_scale=.4)`, wall `plane(-1, (0,0,1))`;
vessels = `lathe(profile, thickness=.02)`; fruit = `sphere().scale((1,.88,1)).sub(dimple, k=.04)` + `sss=.3–.5` + texture `f(p, n)`
with `noise3`; key `s.area(...)` from the side + dim `s.sky(power=.15–.35)`; add an emissive panel (`emit=`) off-camera behind the light so
metal/glass have something to reflect; `c.render(s, aa=2, look="punchy")` then `depth_of_field(r.depth, r.focus_at(x, y, c.s), 3)`,
`bloom(.85, 10, .25)`, `vignette`, `grain(.012)`.
**Oil** — build a flat colour study `ref` (gradient sky, clouds, `fill_below(ridge_line)` hills, svg river, ellipses trees), then
`c.paint_like(ref, "oil", sizes=(64,32,16,9))`; `c.cur.relief = .6`; grade with `curves`, `vignette`, light `grain`.
**Watercolour** — paper bg `#f4efe4`, broad `stroke(..., "watercolor", size 40–90)` washes light → dark (glazes multiply),
details with `ink` or small `round`; paper tooth: `c.layer(blend="overlay", opacity=.5).fill(paper())` (0.5 = neutral), then `c.layer()`.
**Flat / poster** — 4–5 colour palette, shapes + booleans (`shape & panel` to clip), `gradient(...).within(shape)`,
title `c.text(..., fit=W-2*margin, weight="bold")`, small caps lines `spacing=.1–.2`; halftone shading: multiply layer filled with
a lighting Field, `L.apply(halftone(9, 30, ink, paper="white"))`, then `c.layer()` again.
**Pixel art** — `Canvas(320, 180, pixel=True)`, custom 8–16 colour list, draw shapes/fields, `L.apply(pixelate(1, PAL, dither="bayer",
strength=.3))` per layer (give water/sky their own sub-palette), `c.save("x.png", scale=4)` (nearest).
**Abstract** — `domain_warp(domain_warp(fbm(1.3, 4), .45, 1.2), .1, 2)`, colour via `hue.ramp(curated palette)`, folds
`np.sin(f*np.pi*k)`, composition mask (e.g. S-curve `Field`) on a dark bg, light `shade(f, color=..., strength=.5)`, `bloom`.

## Pitfalls
- `c.layer()` changes the current target: later `c.text/...` go to that layer (bad if it is `multiply`). Call `c.layer()` or `c.use(L)`.
- `c.apply()` merges all layers. Use `L.apply()` to keep layers.
- `shade(strength)` depends on field frequency: ≈1 for big forms, ≈.05 for fine textures (fabric).
- `tone_map` on 2D art greys pure white — it is for HDR/3D; use `curves`/`levels` on illustrations.
- Hard thresholds alias: use `aastep`/`smoothstep`, not `f > .5`, for visible edges.
- No kerning in text: prefer caps/tracking for big titles.
- Noise `scale` = features per canvas height. Shape coords = canvas px; Field coords = 0–1.
- Slow paths: `paint_like` scales with stroke count (~15–40 s preview); 3D with `aa=2` ≈ 4× (previews force aa=1).
  `lathe/extrude` profiles are pre-gridded (fast); many objects → union them into one `s.add` only if same material.
- Glass bends light once (thin shells look best); glass and emitters cast no shadows.
