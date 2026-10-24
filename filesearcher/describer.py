"""Faza 2: opisy scen i OCR w tle, z postępem i wznawianiem."""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

MAX_PIXELS = 1_000_000


@dataclass(frozen=True)
class ProgressState:
    done: int
    total: int
    failed: int
    rate: float | None  # zdjęcia na sekundę
    eta_s: float | None
    finished: bool


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
        text = f"[Opisy i OCR: gotowe, {_num(s.total)} zdjęć"
    else:
        pct = s.done * 100 // s.total if s.total else 100
        text = f"[Opisy i OCR: {_num(s.done)} / {_num(s.total)} · {pct}% · "
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


class BackgroundDescriber:
    def __init__(self, store, index, embedder, describer, arbiter, log_path: Path, window: int = 20, clock=time.monotonic):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.describer = describer
        self.arbiter = arbiter
        self.log_path = Path(log_path)
        self.clock = clock
        self._times = deque(maxlen=window)
        self._stop = threading.Event()
        self._thread = None
        self._remaining = 0
        self._finished = False

    def start(self):
        todo = self.store.pending_descriptions(include_failed=True)
        self._remaining = len(todo)
        self._thread = threading.Thread(target=self._run, args=(todo,), daemon=True, name="opisy")
        self._thread.start()

    def stop(self, timeout: float | None = None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> ProgressState:
        c = self.store.counts()
        rate = eta = None
        if len(self._times) >= 2:
            span = self._times[-1] - self._times[0]
            if span > 0:
                rate = (len(self._times) - 1) / span
                eta = self._remaining / rate
        return ProgressState(c.done, c.total, c.failed, rate, eta, self._finished)

    def _log(self, message: str):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")

    def _describe_one(self, image_id: int, path: str):
        img = load_for_description(path)
        with self.arbiter.background():
            desc = self.describer.describe(img)
            text = "\n".join(t for t in (desc.opis, desc.tekst) if t).strip()
            vec = self.embedder.embed_texts([text])[0] if text else None
        self.store.save_description(image_id, desc, vec)
        if vec is not None:
            self.index.add([image_id], ["text"], vec[None])

    def _run(self, todo):
        for image_id, path in todo:
            if self._stop.is_set():
                return
            try:
                self._describe_one(image_id, path)
            except Exception as e:
                self.store.mark_failed(image_id, str(e))
                self._log(f"Błąd opisu {path}: {e!r}")
            self._remaining -= 1
            self._times.append(self.clock())
        self._finished = True
