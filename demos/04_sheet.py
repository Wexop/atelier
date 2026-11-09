# Step 4 check: every effect on the same little scene (4 x 4 tiles).
from atelier import *

def scene(fx=(), layer_fx=()):
    c = Canvas(400, 260)
    c.fill(gradient(["#12203a", "#3f5f8a", "#f2b880"]))
    c.circle(290, 150, 46, fill="#fff3d6")                              # bright sun (bloom)
    c.draw(svg("M0 200 C 90 150 170 190 250 205 S 360 170 400 190 V 260 H 0 Z"), fill="#1d2a2a")
    c.rect(40, 60, 90, 90, r=10, fill=fbm(5).ramp("forest").within(rect(40, 60, 90, 90)))
    t = c.layer()
    t.text("FX", 85, 238, 58, fill="#ff5d8f", weight="bold", align="center")
    if layer_fx:
        t.apply(*layer_fx)
    if fx:
        c.apply(*fx)
    return c.flatten()

tiles = [
    ("original", ()), ("blur", (blur(4),)), ("motion", (motion_blur(30, 20),)), ("bloom", (bloom(0.7, 14, 1.0),)),
    ("glow+shadow", "L"), ("inner", "I"), ("vignette+grain", (vignette(0.6), grain(0.08))),
    ("chroma+lens", (lens_distortion(0.15), chromatic_aberration(4))),
    ("curves S", (curves([(0.25, 0.15), (0.75, 0.88)]),)), ("hue_sat", (hue_sat(120, 1.4),)),
    ("teal/orange", (color_balance(shadows=(-.06, .01, .08), highlights=(.08, .03, -.06)),)),
    ("duotone", (gradient_map(["#1b1035", "#ff7a59", "#fff1d0"]),)), ("agx", (tone_map("agx", 1.0),)),
    ("displace", (displace(fbm(6), 8),)), ("pixel bayer", (pixelate(4, "retro16", dither="bayer"),)),
    ("halftone cmyk", (halftone(5, cmyk=True),)),
]
sheet = Canvas(1640, 1080, bg="#0d0d0d")
for i, (name, fx) in enumerate(tiles):
    if fx == "L":
        img = scene(layer_fx=(drop_shadow(5, 6, 4, opacity=0.7), glow(8, 1.2, "#ff5d8f")))
    elif fx == "I":
        img = scene(layer_fx=(inner_shadow(3, 3, 3, opacity=0.8),))
    else:
        img = scene(fx)
    x, y = 10 + (i % 4) * 408 + 200, 10 + (i // 4) * 268 + 130
    sheet.place(img, x, y, scale=1, layer=sheet.cur)
    sheet.text(name, x - 192, y + 122, 15, fill="white", weight="semibold")
sheet.save("demos/out/04_sheet.png")
sheet.preview("demos/out/04_sheet_prev.jpg", 1400)
