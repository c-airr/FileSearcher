"""Faza 2 w tle, z postępem i wznawianiem: najpierw szybki OCR (gdy jest), potem opisy w paczkach."""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

MAX_PIXELS = 1_000_000
OCR_MAX_PIXELS = 8_000_000  # OCR potrzebuje pelnej rozdzielczosci drobnego tekstu na zrzutach
DESCRIBE_BATCH = 8
OCR_CHUNK = 32  # ile tekstow naraz idzie do modelu wektorowego


@dataclass(frozen=True)
class ProgressState:
    done: int
    total: int
    failed: int
    rate: float | None  # zdjęcia na sekundę
    eta_s: float | None
    finished: bool
    stage: str = "Opisy i OCR"


def _num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "<1 min"
    if minutes < 60:
        return f"~{minutes} min"
    return f"~{minutes // 60} h {minutes % 60} min"


def format_progress(s: ProgressState) -> str:
    if s.finished:
        text = f"[{s.stage}: gotowe, {_num(s.total)} zdjęć"
    else:
        pct = s.done * 100 // s.total if s.total else 100
        text = f"[{s.stage}: {_num(s.done)} / {_num(s.total)} · {pct}% · "
        if s.rate is None or s.eta_s is None:
            text += "szacuję…"
        else:
            text += f"{s.rate:.1f}".replace(".", ",") + f" zdj/s · {_eta(s.eta_s)}"
    if s.failed:
        text += f" · błędy: {s.failed}"
    return text + "]"


def load_for_description(path: str, max_pixels: int = MAX_PIXELS) -> Image.Image:
    with Image.open(path) as img:
        if img.format == "JPEG":
            side = int(math.sqrt(max_pixels))
            img.draft("RGB", (side, side))
        img = ImageOps.exif_transpose(img).convert("RGB")
    if img.width * img.height > max_pixels:
        scale = math.sqrt(max_pixels / (img.width * img.height))
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    return img


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


class BackgroundDescriber:
    def __init__(
        self,
        store,
        index,
        embedder,
        describer,
        arbiter,
        log_path: Path,
        reader=None,
        batch: int = DESCRIBE_BATCH,
        window: int = 64,
        clock=time.monotonic,
    ):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.describer = describer
        self.reader = reader
        self.arbiter = arbiter
        self.log_path = Path(log_path)
        self.batch = batch
        self.clock = clock
        self._times = deque(maxlen=window)
        self._stop = threading.Event()
        self._thread = None
        self._remaining = 0
        self._stage = "OCR" if reader else "Opisy i OCR"
        self._finished = False

    def start(self):
        ocr_todo = self.store.pending_ocr() if self.reader else []
        todo = self.store.pending_descriptions(include_failed=True) if self.describer else []
        self._thread = threading.Thread(target=self._run, args=(ocr_todo, todo), daemon=True, name="opisy")
        self._thread.start()

    def stop(self, timeout: float | None = None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> ProgressState:
        stage = self._stage
        if stage == "OCR":
            done, total = self.store.ocr_counts()
            failed = 0
        else:
            c = self.store.counts()
            done, total, failed = c.done, c.total, c.failed
        rate = eta = None
        times = list(self._times)
        if len(times) >= 2:
            span = times[-1] - times[0]
            if span > 0:
                rate = (len(times) - 1) / span
                eta = self._remaining / rate
        return ProgressState(done, total, failed, rate, eta, self._finished, stage)

    def _log(self, message: str):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")

    def _tick(self, n: int):
        self._remaining -= n
        now = self.clock()
        self._times.extend([now] * n)

    def _enter_stage(self, stage: str, remaining: int):
        self._stage = stage
        self._remaining = remaining
        self._times.clear()

    def _save_texts(self, image_ids: list[int], texts: list[str], save):
        """Wektory tekstu w jednej paczce, potem zapis kazdego zdjecia funkcja save(id, vec)."""
        idx = [i for i, t in enumerate(texts) if t]
        vecs = {}
        if idx:
            with self.arbiter.background():
                emb = self.embedder.embed_texts([texts[i] for i in idx])
            vecs = {i: emb[n] for n, i in enumerate(idx)}
        for i, image_id in enumerate(image_ids):
            vec = vecs.get(i)
            save(image_id, vec)
            self.index.remove_text([image_id])
            if vec is not None:
                self.index.add([image_id], ["text"], vec[None])

    # ── OCR ──
    def _ocr_chunk(self, chunk):
        ids, texts = [], []
        for image_id, path in chunk:
            try:
                tekst = self.reader.read(load_for_description(path, OCR_MAX_PIXELS))
            except Exception as e:
                self.store.save_ocr(image_id, None, None)
                self._log(f"Błąd OCR {path}: {e!r}")
                continue
            ids.append(image_id)
            texts.append(tekst)
        by_id = dict(zip(ids, texts))
        self._save_texts(ids, texts, lambda image_id, vec: self.store.save_ocr(image_id, by_id[image_id], vec))

    # ── opisy ──
    def _describe_items(self, items):
        """items: [(image_id, path, obraz)]. Przy braku pamieci GPU dzieli paczke na pol."""
        try:
            with self.arbiter.background():
                descs = self.describer.describe_batch([img for _, _, img in items])
        except Exception as e:
            if _is_oom(e) and len(items) > 1:
                half = len(items) // 2
                self._log(f"Brak pamięci GPU przy paczce {len(items)} – dzielę na pół")
                self._describe_items(items[:half])
                self._describe_items(items[half:])
                return
            for image_id, path, _ in items:
                self.store.mark_failed(image_id, str(e))
                self._log(f"Błąd opisu {path}: {e!r}")
            return
        ids = [image_id for image_id, _, _ in items]
        by_id = dict(zip(ids, descs))
        texts = []
        for image_id, desc in zip(ids, descs):
            tekst = desc.tekst or self.store.get_text(image_id)[1]
            texts.append("\n".join(t for t in (desc.opis, tekst) if t).strip())
        self._save_texts(ids, texts, lambda image_id, vec: self.store.save_description(image_id, by_id[image_id], vec))

    def _describe_chunk(self, chunk):
        items = []
        for image_id, path in chunk:
            try:
                items.append((image_id, path, load_for_description(path)))
            except Exception as e:
                self.store.mark_failed(image_id, str(e))
                self._log(f"Błąd opisu {path}: {e!r}")
        if items:
            self._describe_items(items)

    def _guard(self, fn, chunk):
        """Blad jednej paczki nie moze ubic watku tla: zostaje w logu, zdjecia czekaja na kolejny start."""
        try:
            fn(chunk)
        except Exception as e:
            self._log(f"Błąd paczki ({len(chunk)} zdjęć, od {chunk[0][1]}): {e!r}")

    def _run(self, ocr_todo, todo):
        if ocr_todo:
            self._enter_stage("OCR", len(ocr_todo))
            for start in range(0, len(ocr_todo), OCR_CHUNK):
                if self._stop.is_set():
                    return
                chunk = ocr_todo[start : start + OCR_CHUNK]
                self._guard(self._ocr_chunk, chunk)
                self._tick(len(chunk))
        if self.describer is None:  # sam OCR (profil lite z Windows OCR)
            self._finished = True
            return
        self._enter_stage("Opisy" if self.reader else "Opisy i OCR", len(todo))
        for start in range(0, len(todo), self.batch):
            if self._stop.is_set():
                return
            chunk = todo[start : start + self.batch]
            self._guard(self._describe_chunk, chunk)
            self._tick(len(chunk))
        self._finished = True
