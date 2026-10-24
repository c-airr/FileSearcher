import json
import time

import pytest

from filesearcher import cli
from filesearcher.app import ensure_embedder, open_app
from filesearcher.config import Config
from filesearcher.indexer import run_phase1
from filesearcher.models import ModelBundle
from filesearcher.models.fake import FakeDescriber, FakeEmbedder
from filesearcher.search import Hit
from filesearcher.store import Store


def test_ensure_embedder_first_run(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    assert ensure_embedder(s, FakeEmbedder(), ask=lambda m: pytest.fail("nie powinno pytać"))
    assert s.get_meta("embedder_id") == "fake/embedder" and s.get_meta("dim") == "16"


def test_ensure_embedder_declined(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    ensure_embedder(s, FakeEmbedder(model_id="a"), ask=lambda m: True)
    assert ensure_embedder(s, FakeEmbedder(model_id="b"), ask=lambda m: False) is False
    assert s.get_meta("embedder_id") == "a"


def test_ensure_embedder_rebuild(tmp_path, make_image):
    s = Store(tmp_path / "i.sqlite")
    ensure_embedder(s, FakeEmbedder(model_id="a"), ask=lambda m: True)
    from filesearcher.store import NewImage
    import numpy as np

    s.add_images([NewImage("x.jpg", 1.0, 1, [("full", np.ones(16, dtype=np.float32))])])
    asked = []
    assert ensure_embedder(s, FakeEmbedder(model_id="b"), ask=lambda m: asked.append(m) or True)
    assert "a" in asked[0] and "b" in asked[0]
    assert s.counts().total == 0 and s.get_meta("embedder_id") == "b"


@pytest.fixture
def app(tmp_path, make_image):
    folder = tmp_path / "zdjecia"
    make_image(folder / "czerwone.jpg", color="red")
    make_image(folder / "niebieskie.jpg", color="blue")
    cfg = Config(folder=folder, baza_dir=tmp_path / "db", batch_size=2)
    app = open_app(cfg, FakeEmbedder(), FakeDescriber(), ask=lambda m: True)
    run_phase1(app.store, app.index, app.embedder, cfg, app.arbiter, show_progress=False)
    return app


def feed(lines):
    it = iter(lines)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _input


def test_cli_search_status_quit(app):
    app.background.start()
    out = []
    cli.run_cli(app, input_fn=feed(["czerwony", "s", "q"]), print_fn=lambda *a: out.append(" ".join(map(str, a))))
    app.background.stop(5)
    text = "\n".join(out)
    assert "[Opisy i OCR:" in text
    assert "[1]" in text and "czerwone.jpg" in text.split("[1]")[1].splitlines()[0]
    assert sum("[Opisy i OCR:" in line for line in out) >= 3


@pytest.mark.parametrize("lines", [[""], []])
def test_cli_exits_on_empty_line_or_eof(app, lines):
    cli.run_cli(app, input_fn=feed(lines), print_fn=lambda *a: None)


def test_cli_empty_database(tmp_path):
    cfg = Config(folder=tmp_path, baza_dir=tmp_path / "db")
    app = open_app(cfg, FakeEmbedder(), None, ask=lambda m: True)
    out = []
    cli.run_cli(app, input_fn=feed(["kot"]), print_fn=lambda *a: out.append(" ".join(map(str, a))))
    assert any("Baza jest pusta" in line for line in out)


def test_format_hit():
    hit = Hit(1, "F:/POZOSTALOSCI/2019/IMG_0412.jpg", 0.871, {"obraz": "wycinek prawy dół", "opis": "czerwony rower"})
    assert cli.format_hit(1, hit) == (
        "[1] 0.87  F:/POZOSTALOSCI/2019/IMG_0412.jpg\n    obraz: wycinek prawy dół · opis: „czerwony rower”"
    )


def test_format_hit_fts_only():
    assert cli.format_hit(2, Hit(2, "a.jpg", None, {"OCR": "ROWER"})) == "[2]    –  a.jpg\n    OCR: „ROWER”"


def test_main_smoke(tmp_path, make_image, monkeypatch, capsys):
    folder = tmp_path / "zdjecia"
    make_image(folder / "a.jpg", color="green")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"FOLDER_ZE_ZDJECIAMI": str(folder), "BAZA_DIR": str(tmp_path / "db")}), encoding="utf-8")
    from filesearcher import hardware, models

    monkeypatch.setattr(hardware, "detect", lambda: hardware.HardwareInfo("cpu", None, None, None, []))
    monkeypatch.setattr(models, "load_models", lambda profile, log=print: ModelBundle(FakeEmbedder(), FakeDescriber()))
    monkeypatch.setattr("builtins.input", feed(["zielony", "q"]))
    assert cli.main(["--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "lite" in out and "a.jpg" in out


def test_main_reports_torch_problem(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"FOLDER_ZE_ZDJECIAMI": str(tmp_path)}), encoding="utf-8")
    from filesearcher import hardware

    monkeypatch.setattr(hardware, "detect", lambda: hardware.HardwareInfo("cuda", "RTX 5070", 11.9, (12, 0), ["sm_90"]))
    assert cli.main(["--config", str(cfg)]) == 1
    assert "cu128" in capsys.readouterr().err


def test_main_missing_folder(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"FOLDER_ZE_ZDJECIAMI": str(tmp_path / "brak")}), encoding="utf-8")
    assert cli.main(["--config", str(cfg)]) == 1
    assert "nie istnieje" in capsys.readouterr().err
