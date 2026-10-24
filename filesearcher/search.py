"""Wyszukiwanie w trzech źródłach (obraz, opis, tekst) i łączenie przez RRF."""
import re
from dataclasses import dataclass, field

from .store import fold

RRF_K = 60
SNIPPET_LEN = 100
KIND_LABELS = {
    "full": "cały kadr",
    "tile0": "wycinek lewa góra",
    "tile1": "wycinek prawa góra",
    "tile2": "wycinek lewy dół",
    "tile3": "wycinek prawy dół",
}

__all__ = ["Hit", "fold", "rrf", "search"]


@dataclass
class Hit:
    image_id: int
    path: str
    score: float | None
    sources: dict = field(default_factory=dict)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _snippet(text: str, words: list[str]) -> str:
    folded = fold(text)
    positions = [folded.find(w) for w in words if w in folded] if len(folded) == len(text) else []
    pos = min(positions) if positions else 0
    start = max(0, pos - 30)
    out = text[start : start + SNIPPET_LEN].strip()
    if start > 0:
        out = "…" + out
    if start + SNIPPET_LEN < len(text):
        out += "…"
    return out


def search(query, store, index, embedder, arbiter, limit, candidates=200) -> list[Hit]:
    with arbiter.priority():
        q = embedder.embed_texts([query])[0]
    image_hits = index.search(q, "image", candidates)
    text_hits = index.search(q, "text", candidates)
    fts_ids = store.fts_search(query, candidates)

    fused = rrf([[h[0] for h in image_hits], [h[0] for h in text_hits], fts_ids])
    # Źródło pokazujemy przy wyniku tylko, gdy zdjęcie jest w czołówce tego źródła.
    shown = max(limit * 2, 10)
    image_top = {h[0]: h for h in image_hits[:shown]}
    text_top = {h[0]: h for h in text_hits[:shown]}
    fts_top = set(fts_ids[:shown])
    cosines = {}
    for image_id, sim, _ in image_hits + text_hits:
        cosines[image_id] = max(sim, cosines.get(image_id, -1.0))

    top_ids = [i for i, _ in fused[: limit * 2]]
    info = store.get_info(top_ids)
    words = re.findall(r"\w+", fold(query))
    hits = []
    for image_id in top_ids:
        if image_id not in info:  # usunięte w międzyczasie
            continue
        data = info[image_id]
        hit = Hit(image_id, data.path, cosines.get(image_id))
        if image_id in image_top:
            hit.sources["obraz"] = KIND_LABELS[image_top[image_id][2]]
        if image_id in fts_top and data.tekst and any(w in fold(data.tekst) for w in words):
            hit.sources["OCR"] = _snippet(data.tekst, words)
        elif (image_id in fts_top or image_id in text_top) and data.opis:
            hit.sources["opis"] = _snippet(data.opis, words)
        hits.append(hit)
        if len(hits) == limit:
            break
    return hits
