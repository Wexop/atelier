---
name: atelier
description: Paint images (PNG/JPG/WebP) yourself with code instead of calling an image-generation model — photoreal 3D (path tracing), oil/watercolour painting, comics and engravings from 3D, people and animals, landscapes (terrain, sky, sea, trees), posters with typography, pixel art, generative abstracts. Use when asked to create, draw, paint or render an image.
---
# atelier — image studio driven by short Python scripts

Install (Python ≥ 3.11): `pip install git+https://github.com/Wexop/atelier` (update: `pip install -U git+...`)
Check: `atelier palette` · `atelier brushes` · `atelier fx` (signatures).

## Loop (always)
0. Pick the style guide (see Styles below) and read it; none fits → `styles/new-style.md`.
   Complex scene (subject + setting, 4+ elements, story, poster) → also `styles/planning.md`: PLAN + FOCUS, 3 grey thumbnails, passes.
1. Write `scene.py` (30–80 lines, `from atelier import *`, build a Canvas `c`, end with `c.save(__file__[:-3] + ".png")`).
2. `atelier render scene.py --preview` → `scene_prev.jpg` (512 px, real low-res render, saves skipped). **Read the image.**
   Placing things: `--grid [step]` labels canvas-px coords in the margins, `--boxes` outlines + prints each layer's bounds
   (name layers: `c.layer(name="sun")`), `--mark x,y` / `--mark x,y,w,h` checks a target spot. Works with `--crop` too.
3. `atelier inspect scene.py` → `scene_sheet.jpg`: view, 5-value study + thirds/focal, clip map, histogram, palette, 4 auto zooms
   (aliasing, banding, noise, highlight) labelled `--crop x,y,w,h`, metrics + red flags. **Read it.** Zoom: `atelier render scene.py --crop x,y,w,h`.
   `atelier critique scene.py` → ranked problems with fixes (contrast, balance, near-miss alignment, tangents, edge crowding,
   hidden layers, **hierarchy**: does `FOCUS = "layer"` (or `--focus`) win on contrast, sharpness, detail, saturation, squint test?)
   + `scene_critique.jpg` with numbered boxes. One element per named layer; names survive `c.apply()`.
   Variants: `atelier sheet a.py b.py` / `atelier sheet scene.py --seeds 1,2,3` → tiles + value studies + scores.
4. ≥3 look-and-fix rounds (`atelier compare old.png new.png` for A/B), then `atelier render scene.py` (final; `--scale 2`, `--seed N` → `SEED`).
Errors print one line `scene.py:LINE: Type: message` (`--debug` for traceback).

## Conventions
- Drawing coords = canvas px (y down). Fields use normalised x, y ∈ [0,1]. Colours: "#hex", CSS names, (r,g,b) 0–1 or 0–255.
- Paint source (fill=, color=, stroke=) = colour | Field | f(x,y) | array H×W×3/4. Masks = Field | Shape | array | f(x,y).
- Everything random takes `seed=`. Previews render at low res: the picture is identical, only sharper at final.

## Canvas & layers (core)
```
c = Canvas(w, h, bg=None, pixel=False)      # pixel=True: no AA, fixed scale, nearest-neighbour export
L = c.layer(blend="normal", opacity=1, mask=None, name=None)   # new layer, BECOMES CURRENT; c.use(L) to switch back
Name layers by role ("sun", "hero/body", "hero/shadow") and edit by name: c["sun"] · c.use("sun") · "sun" in c · c.group("hero") -> [layers]
L.bounds -> (x,y,w,h) · L.move(dx,dy) · L.move_to(x, y, anchor="center"|"bottom"|"topleft"..) · L.transform(dx,dy,scale,rot,flip="x"|"y",center)
L.front() .back() .above("x") .below("x") · L.hide() .show() · L.duplicate("hero/shadow") · L.remove()   (group: for L in c.group("hero"): L.move(40, 0))
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
       align="left"|"center"|"right", valign="baseline"|"top"|"middle"|"bottom", spacing=em, line_height=1.2, italic, fit=width_px, kern=True)
text_shape(...) → Shape · text_size(txt, size) → (width, cap_height) · c.text_on_path(txt, shape|points, size, align, lift, fit, fill)
c.text_box(txt, x, y, w, h=None, size=18, align="left"|"justify"|"center", line_height=1.4, fit=True)   # **b** *i* [#c33]colour[/]
c.title(txt, x, y, size=96, style="shadow"|"extrude"|"outline"|"inline"|"neon", fill, color2)
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
god_rays(x, y, rays=14, length=.9, strength=.45, color, threshold=.7, occlude=True) lens_flare(x, y, strength=.6, color, ghosts=6, streak=.5)
depth_of_field(depth, focus, aperture=8)   # depth from r = c.render(scene)
match_palette("ref.jpg", amount) transfer_tone(ref, amount)   # colour/tone transfer from a reference image
```

## Stylize (stylize) — 3D → illustration + painterly filters (effects for c.apply / L.apply)
```
r = c.render(s, alpha=True); c.cur.apply(toon(r, bands=3, shadow=.45, colors={id: col}), shadow_dots(r, size=5, lo=.3), outlines(r, width=2.5, inner=1))
engraving: L.fill(PAPER, mask=r.alpha); L.apply(render_hatch(r, s, spacing=4.5, layers=3), outlines(r, width=2))  # hatches follow 3D surfaces
kuwahara(radius=6) oil · xdog(sigma=1, eps=.9, ink, paper) ink lines · pencil() · linocut(lines=7, angle) · edge_smooth(4)
watercolorize(bleed=2, edge=1, granulation=.4, ink=0..1) · stained_glass(cell=26, lead=3) · mosaic(tile=10) · shading(r) → (H,W) light
```

## 3D (sdf) — y up, units ≈ metres, primitives centred at origin
```
sphere(r) box(w, h, d) round_box(w, h, d, r) cylinder(r, h) capsule(a, b, r) torus(R, r) cone(r1, h, r2) plane(y, normal)
ellipsoid(rx, ry, rz) lathe([(radius, y), ...] bottom→top, thickness=None|shell) extrude([(x, y), ...], depth)
ops: a | b, a & b, a - b, a.union(b, k=smooth) .inter .sub · .move(x,y,z) .rotate(x,y,z deg) .scale(k|(kx,ky,kz))
     .onion(t) .round(r) .twist(k) .bend(k) .repeat(period, count) .mirror("x") .elongate() .displace(noise3(scale, octaves, seed), amount)
Mat(color | 2D Field (proj="tri"|"xz"|"xy"|"zy", tex_scale) | f(p, n)->(N,3), rough=.5|Field|f(p,n), metal=0, glass=0, ior=1.45,
    sss=0, sss_color, emit=None, power=1, bump=.3, bump_scale=60, local=True (textures follow the object), sheen=.6,
    wear=[smudges(.5), scratches(.5, density=40), dust(.3), edge_wear(.5, width=.01, color)])
s = Scene(); id = s.add(sdf, Mat(...)); s.camera(pos, target, fov=40)
s.sun(dir_toward_sun, color, power=3, soft=.04) · s.point(pos, color, power, radius) · s.area(pos, size=(w,h), normal, color, power, pattern="window"|"softbox", visible)
s.studio(key, target, size, power=45, window=True, fill=.2, bg) · s.sky(zenith, horizon, ground, power) · s.fog(color, density)
r = c.render(s, aa=1|2, exposure=0, look=None|"punchy", alpha=False)          # fast raymarcher (direct light + AO + 1 reflection)
r = c.pathtrace(s, spp=16, bounces=4, denoise=True, aperture=0, focus=None|dist|(x,y), seed=SEED)  # photoreal: GI, multi-refraction glass, adaptive AA
r.depth r.normal r.id r.mask(id) r.focus_at(x, y, c.s) r.linear r.spp r.time · denoise(img, albedo, normal, depth)
```

## Nature (nature, terrain)
```
h = terrain(384, kind="alpine"|"hills"|"mesa"|"island"|"dunes", seed, erosion=1, flow=False)   # eroded heightmap [0,1]; erode() thermal()
c.terrain_map(h, style="relief"|"topo"|"shaded", water=.2)
v = c.landscape(h, cam=(x, alt, z), yaw, fov, horizon=.58, sun_elev=10, sun_az=30, haze=1, clouds=.35, water=.1, waves=.02, snow=.7, seed)
    v.depth v.water v.project(x, y, z) -> (px, py, px_per_m) v.ground(x, z)   # place 2D trees on the 3D terrain
c.sky(sun_elev=8, sun_az=0, horizon=.6, clouds=.3, haze=1) · c.sea(sun_elev=6, waves=.12, wave_len=60, foam=.5, clouds=.25)
c.tree(x, y, height_px, kind="oak"|"pine"|"birch"|"palm"|"bush"|"dead", season="summer"|"spring"|"autumn"|"winter", haze=0..1, seed)
c.grass(x, y, w, h, n) c.flower(x, y, size, kind="daisy"|"poppy"|"tulip"|"bell") c.fern(x, y, size) · tree3d("oak", 4, seed) → (trunk, leaves) SDFs
scene_sky(s, sun_elev, sun_az)   # physical sky on a 3D Scene
c.anime_cloud(x, y, w, h=.55w, palette="day"|"sunset"|"dusk"|"storm"|(shadow,mid,light,rim), light=(az, elev), bands=3, rim=.7, wisps, edge, bottom, seed)  # cel-shaded cumulus, flat base at (x, y)
c.anime_tree(x, y, height, kind="round"|"tall"|"umbrella", palette="summer"|"spring"|"autumn"|"sakura"|"dusk"|(sh,mid,lt,rim,trunk), light, width, leafy=.6, seed)  # cel-shaded tree
c.anime_mountains(y_foot, height, peaks=4, x0, x1, palette="day"|"dusk"|"autumn"|(sh,mid,lt,snow,snow_sh,haze), light, snow=.35, haze=.6, sharp=1, depth=.35, bands=3, seed)  # eroded relief, front view, cel tones
c.sparkles(n, region=mask, size=(6, 16), color, rays=4, glow=.6, seed)   # star glints (water, snow)
```

## Figures (figure) — feet at (x, y), height = standing px (animals: withers)
```
c.figure(x, y, h, pose="stand", style="flat"|"line"|"silhouette"|"stick"|"ink"|"chalk", kind="human"|"cat"|"dog"|"horse"|"bird", seed)
   poses: stand idle walk run sit wave point cheer jump dance think hips phone reach · phase=0-1 (walk cycle) view="front"|"3q"|"side"|"back"
   body=adult|teen|child|elder build=slim|heavy outfit="tee pants|coat skirt|dress" hair=short|bob|long|bun|curly skin=SKIN name colors={...}
figure_shape(...) → Shape · figure_joints(...) → {"hand_r": (x, y), ...} (attach props) · figure_sdf(pose, height=1.75, style="mannequin")
c.crowd(n, box=(x0, y0, x1, y1), horizon=y, height=near_px, tint=(haze, .5), dogs=.1, seed)   # perspective-scaled crowd
```

## Inspect (inspect) — judge numbers, not only the downscaled view
```
m = metrics(c|path|arr)   # noise (<1 clean, >4 grainy) aliasing (% hard steps, >35 jaggy) banding clip_shadows/highlights flags[...]
contact_sheet(c, "sheet.jpg") · compare(a, b, "ab.jpg") · guides(c) / composition(c) → focal, balance, tilt · analyze_reference("ref.jpg")
critique(c, "cr.jpg", focus=None) → {"issues": [{n, sev high|mid|low, kind, msg, box, layers}], metrics, composition, hierarchy}
hierarchy(c, focus) → focus/rest ratios + squint + temperature · variant_sheet([c1, c2], "v.jpg", labels, focus) · critique_score(r)
grid_view(c, "g.jpg", step=None, region=(x,y,w,h), marks=[(x,y)|(x,y,w,h)], boxes=True) · layer_boxes(c) → [(i, name, (x,y,w,h)|None)]
```

## Styles — read the matching guide before writing the scene
Guides live in `styles/` next to this file (if missing: `https://raw.githubusercontent.com/Wexop/atelier/main/styles/<name>.md`).
Each gives recipe, rules, pitfalls and an example script. Several can be mixed (e.g. anime + flat).
- `anime.md` — painted anime landscapes: cel clouds, trees, mountains, rim light, god rays, sparkles
- `oil.md` — oil (`paint_like`) and watercolour washes
- `photoreal.md` — path-traced still life / product, terrain landscapes, sea and sky
- `comic.md` — ligne claire / toon from 3D, engraving, ink filters
- `flat.md` — flat vector, posters, typography, people, silhouette scenes
- `pixel.md` — pixel art with palettes and dithering
- `abstract.md` — generative abstracts from warped fields
- `planning.md` — **process** for complex pictures: plan, thumbnails, passes, hierarchy rules (use with a style guide)
- `new-style.md` — **no guide fits**: study the look, map it to primitives, paint, then save a new `styles/<name>.md`

## Pitfalls
- `c.layer()` changes the current target: later `c.text/...` go to that layer (bad if it is `multiply`). Call `c.layer()` or `c.use(L)`.
- `c.apply()` merges all layers. Use `L.apply()` to keep layers.
- `shade(strength)` depends on field frequency: ≈1 for big forms, ≈.05 for fine textures (fabric).
- `tone_map` on 2D art greys pure white — it is for HDR/3D; use `curves`/`levels` on illustrations.
- Radial `gradient(center=(cx, cy))`: cx is a fraction of the width, cy of the height (radius in height units).
- Hard thresholds alias: use `aastep`/`smoothstep`, not `f > .5`, for visible edges.
- Noise `scale` = features per canvas height. Shape coords = canvas px; Field coords = 0–1.
- Cost: `pathtrace` ≈ 30–60 µs/path (1200×800 @ 8 spp ≈ 5 min on 2 cores, previews 4× fewer spp) — draft with `c.render`;
  low spp sparkles on glossy metal/glass (raise spp). `aa=2` ≈ 4×. Many SDF parts (figures, tree3d, displace) are slow to trace.
- Glass: `rough=0` for clear (default rough .5 = frosted). In `c.render` glass bends once; use `pathtrace` for real refraction.
- Landscape: keep cloud cover ≤ .3 overhead (low clouds look boxy); bloom near the sun haloes clouds.
