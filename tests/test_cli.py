import os
from PIL import Image

from atelier import core
from atelier.cli import main, run_scene

SCENE = """from atelier import *
c = Canvas(400, 200, bg="#204060")
c.circle(200, 100, 40 + SEED, fill="white")
c.save(OUT)
"""


def write(tmp_path, out_name="final.png", body=SCENE):
    p = tmp_path / "scene.py"
    p.write_text(body.replace("OUT", repr(str(tmp_path / out_name))), encoding="utf8")
    return str(p)


def test_preview_does_not_save_and_resets_config(tmp_path, capsys):
    s = write(tmp_path)
    assert main(["render", s, "--preview", "256"]) == 0
    assert not (tmp_path / "final.png").exists()
    prev = tmp_path / "scene_prev.jpg"
    assert Image.open(prev).size == (256, 128)
    assert core.CONFIG["width"] is None and not core.CONFIG["nosave"]


def test_final_scale_seed_out(tmp_path):
    s = write(tmp_path)
    assert main(["render", s]) == 0
    assert Image.open(tmp_path / "final.png").size == (400, 200)
    out = tmp_path / "big.webp"
    assert main(["render", s, "--scale", "2", "--seed", "5", "--out", str(out)]) == 0
    assert Image.open(out).size == (800, 400)
    c1, _ = run_scene(s, seed=0, nosave=True)
    c2, _ = run_scene(s, seed=20, nosave=True)
    assert c2.flatten()[..., 0].sum() > c1.flatten()[..., 0].sum()   # SEED reached the script


def test_crop(tmp_path):
    s = write(tmp_path)
    assert main(["render", s, "--crop", "150,50,100,100", "--width", "200"]) == 0
    assert Image.open(tmp_path / "scene_crop.jpg").size == (200, 200)


def test_errors_are_short(tmp_path, capsys):
    s = write(tmp_path, body="from atelier import *\nc = Canvas(10, 10)\nc.fill(palette('nope'))\n")
    assert main(["render", s]) == 1
    err = capsys.readouterr().err
    assert "scene.py:3" in err and "unknown palette" in err and "Traceback" not in err
    assert main(["render", str(tmp_path / "missing.py")]) == 2
    assert main(["render", s, "--crop", "1,2"]) == 2


def test_lists(capsys):
    for cmd in ("palette", "brushes", "fx"):
        assert main([cmd]) == 0
    out = capsys.readouterr().out
    assert "sunset" in out and "watercolor" in out and "bloom(" in out


def test_inspect_compare_reference(tmp_path, capsys):
    s = write(tmp_path)
    assert main(["inspect", s, "--preview", "200", "--guides"]) == 0
    assert Image.open(tmp_path / "scene_sheet.jpg").size[0] == 1280
    assert (tmp_path / "scene_guides.jpg").exists()
    out = capsys.readouterr().out
    assert "--crop" in out and "noise" in out
    png = tmp_path / "x.png"
    Image.new("RGB", (120, 80), (200, 40, 40)).save(png)
    assert main(["inspect", str(png), "--out", str(tmp_path / "x_sheet.png")]) == 0
    assert main(["compare", str(png), s, "--labels", "img,scene", "--out", str(tmp_path / "ab.jpg")]) == 0
    assert (tmp_path / "ab.jpg").exists()
    assert main(["reference", str(png)]) == 0
    assert "#c82828" in capsys.readouterr().out
    assert main(["inspect", str(tmp_path / "missing.png")]) == 2
    assert main(["compare", str(png), str(png), "--labels", "one"]) == 2


def test_preview_scale_shared_by_helper_canvases(tmp_path):
    p = tmp_path / "multi.py"
    p.write_text("from atelier import *\nc = Canvas(2000, 1000)\nsub = Canvas(500, 500)\n", encoding="utf8")
    _, g = run_scene(str(p), width=500, preview=True, nosave=True)
    assert g["c"].W == 500 and g["sub"].W == 125        # same scale (0.25), not blown up to 500 px


def test_grid_overlay(tmp_path, capsys):
    s = write(tmp_path)
    assert main(["render", s, "--grid", "--boxes", "--mark", "200,100;10,10,40,20"]) == 0
    out = capsys.readouterr().out
    assert "grid 50 px" in out and "layer 0" in out
    assert (tmp_path / "scene_prev.jpg").exists() and not (tmp_path / "final.png").exists()
    assert main(["render", s, "--crop", "150,50,100,100", "--grid", "10"]) == 0
    assert (tmp_path / "scene_crop.jpg").exists()
    assert main(["render", s, "--grid", "--mark", "1,2,3"]) == 2


def test_critique_command(tmp_path, capsys):
    s = write(tmp_path)
    assert main(["critique", s, "--preview", "256"]) == 0
    assert "critique ->" in capsys.readouterr().out
    assert (tmp_path / "scene_critique.jpg").exists()


def test_sheet_and_focus_from_scene(tmp_path, capsys):
    body = 'from atelier import *\nFOCUS = "dot"\nc = Canvas(400, 200, bg="#204060")\nc.layer(name="dot")\n' \
           'c.circle(200, 100, 30 + SEED, fill="white")\nc.save(OUT)\n'
    s = write(tmp_path, body=body)
    assert main(["critique", s, "--preview", "256"]) == 0
    assert "focus 'dot'" in capsys.readouterr().out
    assert main(["sheet", s, "--seeds", "0,5", "--preview", "256"]) == 0
    out = capsys.readouterr().out
    assert "scene seed 0" in out and "scene seed 5" in out and (tmp_path / "variants.jpg").exists()
