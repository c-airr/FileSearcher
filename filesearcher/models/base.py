"""Wspólne interfejsy modeli. Reszta programu zna tylko te typy."""
import json
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Description:
    opis: str
    tekst: str


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """(n, dim), wiersze L2-znormalizowane."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """(n, dim), wiersze L2-znormalizowane."""


class Describer(Protocol):
    model_id: str

    def describe(self, image: Image.Image) -> Description: ...

    def describe_batch(self, images: list[Image.Image]) -> list[Description]: ...


class TextReader(Protocol):
    """Szybki OCR (np. Windows) – gdy jest, opisujacy model nie przepisuje tekstu."""

    model_id: str

    def read(self, image: Image.Image) -> str: ...


def l2_normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(norms, 1e-12)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_LABELED = re.compile(r"^\s*OPIS:\s*(?P<opis>.*?)(?:^\s*TEKST:\s*(?P<tekst>.*))?\Z", re.DOTALL | re.MULTILINE)


def parse_description(raw: str) -> Description:
    """Odpowiedź VLM → Description. Format „OPIS: … TEKST: …” (odporny na ucięcie na limicie
    tokenów) albo JSON; gdy żaden, całość idzie do opisu."""
    labeled = _LABELED.match(raw)
    if labeled:
        return Description(labeled["opis"].strip(), (labeled["tekst"] or "").strip())
    match = _JSON_OBJECT.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "opis" in data:
                return Description(str(data.get("opis") or "").strip(), str(data.get("tekst") or "").strip())
        except json.JSONDecodeError:
            pass
    return Description(raw.strip(), "")
