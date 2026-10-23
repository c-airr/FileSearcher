"""Model wektorowy przez sentence-transformers (Qwen3-VL-Embedding, jina-clip-v2)."""
import numpy as np

from .base import l2_normalize

EMBED_DIM = 1024  # Matryoshka: obie rodziny modeli dobrze znoszą przycięcie do 1024
QWEN_MAX_PIXELS = 600 * 600


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str, device: str):
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}}
        if model_id.startswith("Qwen/"):
            kwargs["processor_kwargs"] = {"min_pixels": 28 * 28, "max_pixels": QWEN_MAX_PIXELS}
        else:
            kwargs["trust_remote_code"] = True
        self.model = SentenceTransformer(model_id, device=device, truncate_dim=EMBED_DIM, **kwargs)
        self.model_id = model_id
        self.dim = EMBED_DIM

    def _encode(self, inputs) -> np.ndarray:
        out = self.model.encode(inputs, batch_size=len(inputs), convert_to_numpy=True, normalize_embeddings=True)
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_images(self, images):
        return self._encode(list(images))

    def embed_texts(self, texts):
        return self._encode(list(texts))
