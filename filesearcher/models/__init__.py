"""Ładowanie modeli dla wybranego profilu."""
from dataclasses import dataclass

from ..hardware import OCR_ONLY
from .base import Describer, Embedder


@dataclass
class ModelBundle:
    embedder: Embedder
    describer: Describer | None


def _make_embedder(model_id: str, device: str):
    from .st_embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_id, device)


def _make_vlm(model_id: str, device: str, four_bit: bool):
    from .qwen_vl import QwenVLDescriber

    return QwenVLDescriber(model_id, device, four_bit=four_bit)


def _make_ocr(device: str):
    from .easyocr_only import EasyOCRDescriber

    return EasyOCRDescriber(device)


def load_models(profile, log=print) -> ModelBundle:
    embedder = _make_embedder(profile.embedder_id, profile.device)
    describer = None
    if profile.describer_id == OCR_ONLY:
        try:
            describer = _make_ocr(profile.device)
        except ImportError:
            log("Uwaga: brak pakietu easyocr – OCR wyłączony (pip install easyocr).")
    elif profile.describer_id:
        try:
            describer = _make_vlm(profile.describer_id, profile.device, profile.describer_4bit)
        except Exception as e:
            if not (profile.describer_4bit and profile.fallback_describer_id):
                raise
            log(
                f"Uwaga: nie udało się załadować {profile.describer_id} w 4-bit ({e}). "
                f"Używam {profile.fallback_describer_id} w bf16."
            )
            describer = _make_vlm(profile.fallback_describer_id, profile.device, False)
    return ModelBundle(embedder, describer)
