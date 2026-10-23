"""Wszystkie wektory w jednej macierzy w pamięci: na GPU (torch) albo CPU (numpy)."""
import threading
from typing import Literal, Sequence

import numpy as np

KIND_CODES = {"full": 0, "tile0": 1, "tile1": 2, "tile2": 3, "tile3": 4, "text": 5}
KIND_NAMES = {v: k for k, v in KIND_CODES.items()}
TEXT_CODE = KIND_CODES["text"]
ROWS_PER_IMAGE = 5  # pełny kadr + 4 wycinki


class VectorIndex:
    def __init__(self, dim: int, device: str = "cpu", capacity: int = 1024):
        self.dim = dim
        self._lock = threading.Lock()
        self._n = 0
        self._ids = np.zeros(capacity, dtype=np.int64)
        self._codes = np.zeros(capacity, dtype=np.int8)
        self._torch = None
        if device == "cuda":
            import torch

            self._torch = torch
            self._mat = torch.zeros((capacity, dim), dtype=torch.float16, device="cuda")
        else:
            self._mat = np.zeros((capacity, dim), dtype=np.float32)

    def __len__(self):
        return self._n

    def _grow(self, needed: int):
        cap = len(self._ids)
        if needed <= cap:
            return
        while cap < needed:
            cap *= 2
        self._ids = np.resize(self._ids, cap)
        self._codes = np.resize(self._codes, cap)
        if self._torch is not None:
            new = self._torch.zeros((cap, self.dim), dtype=self._mat.dtype, device=self._mat.device)
            new[: self._n] = self._mat[: self._n]
            self._mat = new
        else:
            new = np.zeros((cap, self.dim), dtype=np.float32)
            new[: self._n] = self._mat[: self._n]
            self._mat = new

    def add(self, image_ids: Sequence[int], kinds: Sequence[str], vecs: np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32).reshape(len(image_ids), self.dim)
        with self._lock:
            start, end = self._n, self._n + len(image_ids)
            self._grow(end)
            self._ids[start:end] = image_ids
            self._codes[start:end] = [KIND_CODES[k] for k in kinds]
            if self._torch is not None:
                self._mat[start:end] = self._torch.from_numpy(vecs).to(self._mat.device, self._mat.dtype)
            else:
                self._mat[start:end] = vecs
            self._n = end

    def _similarities(self, query: np.ndarray) -> np.ndarray:
        if self._torch is not None:
            q = self._torch.from_numpy(np.asarray(query, dtype=np.float32)).to(self._mat.device, self._mat.dtype)
            return (self._mat[: self._n] @ q).float().cpu().numpy()
        return self._mat[: self._n] @ np.asarray(query, dtype=np.float32)

    def search(self, query: np.ndarray, group: Literal["image", "text"], k: int) -> list[tuple[int, float, str]]:
        """Najlepszy wiersz na zdjęcie, malejąco: [(image_id, podobieństwo, rodzaj)]."""
        with self._lock:
            if self._n == 0 or k <= 0:
                return []
            sims = self._similarities(query)
            ids = self._ids[: self._n].copy()
            codes = self._codes[: self._n].copy()
        in_group = codes == TEXT_CODE if group == "text" else codes != TEXT_CODE
        sims = np.where(in_group, sims, -np.inf)
        m = min(len(sims), k * (1 if group == "text" else ROWS_PER_IMAGE))
        top = np.argpartition(-sims, m - 1)[:m]
        top = top[np.argsort(-sims[top])]
        hits, seen = [], set()
        for row in top:
            if not np.isfinite(sims[row]) or ids[row] in seen:
                continue
            seen.add(ids[row])
            hits.append((int(ids[row]), float(sims[row]), KIND_NAMES[int(codes[row])]))
            if len(hits) == k:
                break
        return hits
