import json
from pathlib import Path

import pytest

from filesearcher.config import ConfigError, load_config


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_defaults_applied(tmp_path):
    cfg = load_config(write(tmp_path, {"FOLDER_ZE_ZDJECIAMI": "F:/zdjecia"}))
    assert cfg.folder == Path("F:/zdjecia")
    assert cfg.baza_dir == Path("./filesearcher_db")
    assert cfg.wyniki == 5
    assert cfg.profil == "auto"
    assert cfg.opisy is True
    assert cfg.io_workers == 4 and cfg.batch_size == 16 and cfg.queue_maxsize == 8
    assert ".jpg" in cfg.rozszerzenia


def test_old_chroma_dir_replaced(tmp_path, capsys):
    cfg = load_config(write(tmp_path, {"FOLDER_ZE_ZDJECIAMI": "x", "BAZA_DIR": "./chroma_db"}))
    assert cfg.baza_dir == Path("./filesearcher_db")
    assert "chroma_db" in capsys.readouterr().err


def test_missing_folder_key_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, {}))


def test_bad_profile_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, {"FOLDER_ZE_ZDJECIAMI": "x", "PROFIL": "turbo"}))


def test_extensions_lowercased(tmp_path):
    cfg = load_config(write(tmp_path, {"FOLDER_ZE_ZDJECIAMI": "x", "ROZSZERZENIA": [".JPG", "png"]}))
    assert cfg.rozszerzenia == frozenset({".jpg", ".png"})


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "brak.json")
