# Guide: planning a complex picture

Use for any scene with a subject + setting, 4+ elements, a story, or a poster. Example: `examples/planned.py`.
Decide the big things on thumbnails, where changes cost seconds; details come last and only after checks pass.

## 1. Write the plan at the top of scene.py (keep it updated)
```
PLAN = """
intent: lone lighthouse keeper at dusk, calm but lonely
read order: 1 keeper + lamp  2 lighthouse  3 sea / sky
values (3-5): sky light, sea mid, land + lighthouse dark, lamp = lightest small spot next to the darkest
light: low sun left, warm lights / cool violet shadows; accent colour only on the lamp
layers (back -> front): sky, sun, sea, cliff, lighthouse, lamp, keeper/body, keeper/shadow, fx
"""
FOCUS = "keeper"      # layer or group name (or x, y, w, h): `atelier critique` checks that it wins
```

## 2. Thumbnails: 3 different compositions, flat grey shapes only
- `thumb_a.py`, `thumb_b.py`, `thumb_c.py`: 400x(ratio) canvas, 3-5 flat greys, no texture, no colour, same layer names.
  Vary what matters: horizon height, subject size and side, big diagonal vs calm, near/far.
- `atelier sheet thumb_a.py thumb_b.py thumb_c.py` → one image with value studies + scores. Pick by eye **and** numbers:
  squint dL >= 0.1, no hierarchy/balance/tangent issue. A thumbnail that fails in grey will not be saved by colour.

## 3. Build in passes, back to front, big to small; check after each pass
| pass | what | check |
|---|---|---|
| masses | final composition as flat value shapes, one named layer per element | `critique`: no high, squint passes |
| light | key light, shadow shapes, cast shadows (`L.duplicate("x/shadow")`) | values still read in the value study |
| colour | palette, temperature split (warm light / cool shadow or reverse) | hierarchy saturation ratio >= 1 |
| texture | texture and detail, **mostly on the focus**; background simpler | detail and sharpness ratios >= 1 |
| finish | edges (lose some away from the focus), atmosphere, grading | final `critique` + `inspect` at full res |
Use `--grid --boxes` while placing; fix by name (`c["sun"].move_to(...)`), never by guessing new numbers blind.

## 4. Hierarchy rules (what `critique` measures as focus/rest ratios)
- The focus gets the strongest value contrast, the crispest edges, the most detail and the purest colour. Each ratio < 1
  means something else competes: calm it (blur, lower contrast, desaturate, haze) rather than shouting louder at the focus.
- Background detail density well below the subject; repeat shapes but vary their size and spacing (big/medium/small).
- Darkest dark next to the lightest light only at the focus. Keep 3-5 value groups; mid-tones carry most of the area.
- Avoid tangents and near-misses: overlap clearly or separate clearly; lead lines point at the focus, not out of frame.

## 5. Variants instead of guessing
Unsure between options (sun position, palette, crop)? Make them explicit: copy the scene or use `SEED`, then
`atelier sheet a.py b.py` or `atelier sheet scene.py --seeds 1,2,3,4` and keep the best score that also looks best.

## Pitfalls
- Skipping thumbnails because the subject is "simple": composition errors are the most expensive to fix late.
- Detailing everything evenly: the picture goes flat. Detail is a budget spent on the focus.
- One layer for everything: per-element checks (contrast, tangents, hierarchy) need one element per named layer.
- Trusting the score alone: it catches measurable faults, not taste. Always look at the image too.
