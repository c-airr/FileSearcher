"""Profil lite: sam OCR (EasyOCR), bez opisów scen."""
import numpy as np

from .base import Description


class EasyOCRDescriber:
    model_id = "easyocr"

    def __init__(self, device: str):
        import easyocr

        self.reader = easyocr.Reader(["pl", "en"], gpu=device == "cuda", verbose=False)

    def describe(self, image) -> Description:
        lines = self.reader.readtext(np.asarray(image.convert("RGB")), detail=0, paragraph=True)
        return Description("", " ".join(lines).strip())
