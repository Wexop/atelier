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
