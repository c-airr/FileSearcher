"""Ładowanie modeli dla wybranego profilu."""
from dataclasses import dataclass

from ..hardware import OCR_ONLY
from .base import Describer, Embedder, TextReader


@dataclass
class ModelBundle:
    embedder: Embedder
    describer: Describer | None
    reader: TextReader | None = None


def _make_embedder(model_id: str, device: str):
    from .st_embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_id, device)


def _make_vlm(model_id: str, device: str, four_bit: bool, with_text: bool = True):
    from .qwen_vl import QwenVLDescriber

    return QwenVLDescriber(model_id, device, four_bit=four_bit, with_text=with_text)


def load_reader(log=print) -> TextReader | None:
    """Windows OCR, gdy dostepny; inaczej tekst przepisuje model opisow (wolniej)."""
    try:
        from .win_ocr import WindowsOCR

        return WindowsOCR()
    except ImportError:
        return None
    except Exception as e:
        log(f"Uwaga: Windows OCR niedostępny ({e}) – tekst przepisze model opisów (wolniej).")
        return None


def _make_ocr(device: str):
    from .easyocr_only import EasyOCRDescriber

    return EasyOCRDescriber(device)


def load_models(profile, log=print, reader=None) -> ModelBundle:
    embedder = _make_embedder(profile.embedder_id, profile.device)
    describer = None
    with_text = reader is None
    if profile.describer_id == OCR_ONLY:
        if reader is not None:  # Windows OCR robi to samo szybciej niz EasyOCR
            return ModelBundle(embedder, None, reader)
        try:
            describer = _make_ocr(profile.device)
        except ImportError:
            log("Uwaga: brak pakietu easyocr – OCR wyłączony (pip install easyocr).")
    elif profile.describer_id:
        try:
            describer = _make_vlm(profile.describer_id, profile.device, profile.describer_4bit, with_text)
        except Exception as e:
            if not (profile.describer_4bit and profile.fallback_describer_id):
                raise
            log(
                f"Uwaga: nie udało się załadować {profile.describer_id} w 4-bit ({e}). "
                f"Używam {profile.fallback_describer_id} w bf16."
            )
            describer = _make_vlm(profile.fallback_describer_id, profile.device, False, with_text)
    return ModelBundle(embedder, describer, reader)
