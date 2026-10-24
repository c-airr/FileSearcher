"""Deterministyczne modele do testów – bez GPU i bez sieci."""
import time
import zlib

import numpy as np
from PIL import Image

from .base import Description, l2_normalize

COLOR_WORDS = {
    "czerwony": (1.0, 0.0, 0.0),
    "zielony": (0.0, 1.0, 0.0),
    "niebieski": (0.0, 0.0, 1.0),
    "biały": (1.0, 1.0, 1.0),
}


def _color_name(rgb) -> str:
    r, g, b = rgb
    if min(r, g, b) > 200:
        return "biały"
    return ("czerwony", "zielony", "niebieski")[int(np.argmax(rgb))]


class FakeEmbedder:
    def __init__(self, dim: int = 16, model_id: str = "fake/embedder"):
        self.dim = dim
        self.model_id = model_id
        self.image_calls: list[int] = []

    def embed_images(self, images):
        self.image_calls.append(len(images))
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, :3] = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) / 255.0
        return l2_normalize(out)

    def embed_texts(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                if word in COLOR_WORDS:
                    out[i, :3] += COLOR_WORDS[word]
                else:
                    out[i, 3 + zlib.crc32(word.encode()) % (self.dim - 3)] += 0.5
        return l2_normalize(out)


class FakeDescriber:
    def __init__(self, fn=None, delay: float = 0.0, model_id: str = "fake/describer"):
        self.fn = fn
        self.delay = delay
        self.model_id = model_id

    def describe(self, image):
        if self.delay:
            time.sleep(self.delay)
        if self.fn:
            return self.fn(image)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
        return Description(f"{_color_name(rgb)} obraz", "")
