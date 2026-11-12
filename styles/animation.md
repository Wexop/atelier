# Animation: GIF / WebP / MP4 loops from one scene script

Examples: `examples/lava.py` (metaball blobs, glossy shading, `--frames 96 --fps 24`) and
`examples/daynight.py` (colour keyframes per phase, sun and moon on one wheel, `--frames 120 --fps 24`).

## How it works
The scene is rendered once per frame. Inside the script `T` goes 0 → 1 over the clip (`FRAME`, `FRAMES` too);
a still render sees `T = 0`, so the same file is also the poster frame. `T = 1` equals the first frame again:
anything periodic in T loops without a seam.

## Recipe
1. Build the still first (normal loop: preview, inspect, critique). Pick what moves: 1–3 things, not everything.
2. Drive each motion from T:
   - travel: `x = key(T, [(0, -60), (1, W + 60)], ease="linear")` (enter and leave off-screen: the loop hides the jump);
   - bounce / pop: `key(T, [(0, 0), (.3, 1), (1, 1)], ease="back")`; breathing light: `wave(T, cycles=2, lo=.4, hi=.6)`;
   - colour shift: `key(T, [(0, "#ffd080"), (.5, "#ff9050"), (1, "#ffd080")])`;
   - noise that drifts: `fbm(3, seed=1).move(*orbit(T, .3))` (a circle in noise space: loops);
   - weather: `c.rain(..., t=T)`, `c.snowfall(..., t=T)`, `c.mist(..., t=T)`, `c.smoke(..., t=T)`;
   - walk cycle: `c.figure(..., pose="walk", phase=(T * k) % 1)` with an integer k (strides per clip).
3. `atelier animate scene.py --frames 24 --fps 12 --check-only` → `scene_frames.jpg`. **Read it.**
4. Fix the flags, then `atelier animate scene.py --frames 36 --fps 12` (default 480 px wide; `--width 0` = canvas size).

## Reading the check sheet
- 12 frames with their T: poses, positions, timing.
- Onion skin (all frames averaged): the trail of every moving thing; a crisp object = it never moved.
- Motion map: where change happens. It should sit on what you meant to animate, not everywhere.
- Change curve: flat = constant speed; a spike = a jump; the last point (last → first) must look like the others.
- Flags: `static` (nothing reads T), `jump at frame n`, `loop seam` (T-driven value does not come back),
  `flicker` (something is re-rolled per frame: a nudge of T by 0.001 already changes the picture).

## Rules
- Keep every `seed=` constant. Never `seed=FRAME` or unseeded randomness: it flickers. Move things instead.
- Loops: use integer cycles (`wave(T, 2)`, `phase=(T * 3) % 1`), `orbit`, or travel from off-screen to off-screen.
- Speed: one clip second ≈ 12 frames. Slow ambient motion (mist, light) reads calm; fast motion needs 24 fps.
- Easing: `in_out` for moves that start and stop, `linear` for constant drift, `back` / `bounce` for playful pops.
- Hierarchy holds in time: the focus moves the most or the most clearly; background motion stays slow and small.
- Size: GIF is 256 colours, ordered dithering, ~3–8 MB for 4–5 s at 480 px (gradients cost most); WebP / MP4 are 5–10× smaller.
- Simple wins: one idea, one or two motions, strong colours. Weather + figure + mist + stars at once reads as noise.

## Pitfalls
- Heavy scenes (path tracing, big SDF) × 36 frames are slow: animate 2D layers, render 3D once and reuse it.
- `c.apply(grain(...))` with a fixed seed is a static texture over moving things: fine; with seed=FRAME it boils.
- Text and thin lines crawl when they move by sub-pixel steps at low width: move them by whole pixels or not at all.
