import pytest

from filesearcher import models
from filesearcher.hardware import PROFILES
from dataclasses import replace


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(models, "_make_embedder", lambda model_id, device: log.append(("emb", model_id)) or "E")

    def vlm(model_id, device, four_bit):
        log.append(("vlm", model_id, four_bit))
        if four_bit:
            raise RuntimeError("bitsandbytes niedostępny")
        return "V"

    monkeypatch.setattr(models, "_make_vlm", vlm)
    monkeypatch.setattr(models, "_make_ocr", lambda device: log.append(("ocr",)) or "O")
    return log


def test_4bit_failure_falls_back_to_2b(calls):
    messages = []
    bundle = models.load_models(replace(PROFILES["high"], device="cuda"), log=messages.append)
    assert bundle.describer == "V"
    assert calls[-1] == ("vlm", "Qwen/Qwen3-VL-2B-Instruct", False)
    assert any("4-bit" in m for m in messages)


def test_lite_uses_ocr(calls):
    bundle = models.load_models(PROFILES["lite"], log=lambda m: None)
    assert bundle.embedder == "E" and bundle.describer == "O"
    assert ("emb", "jinaai/jina-clip-v2") in calls


def test_max_loads_bf16_4b(calls):
    bundle = models.load_models(replace(PROFILES["max"], device="cuda"), log=lambda m: None)
    assert bundle.describer == "V" and calls[-1] == ("vlm", "Qwen/Qwen3-VL-4B-Instruct", False)


def test_missing_easyocr_disables_ocr(monkeypatch, calls):
    def no_ocr(device):
        raise ImportError("No module named 'easyocr'")

    monkeypatch.setattr(models, "_make_ocr", no_ocr)
    messages = []
    bundle = models.load_models(PROFILES["lite"], log=messages.append)
    assert bundle.describer is None and any("easyocr" in m for m in messages)
