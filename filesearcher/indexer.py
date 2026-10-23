"""Faza 1: skan folderu, dekodowanie w wątkach, wektory obrazu w batchach."""
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from .store import NewImage, plan_sync

MAX_SIDE = 1024
MIN_TILE_SIDE = 448
TILE_KINDS = ("tile0", "tile1", "tile2", "tile3")
_DONE = object()


@dataclass
class IndexResult:
    added: int = 0
    removed: int = 0
    failed: list = field(default_factory=list)


def scan_folder(folder: Path, extensions: frozenset) -> dict:
    """{ścieżka bezwzględna: (mtime, rozmiar)} dla plików o podanych rozszerzeniach."""
    found = {}
    stack = [os.path.realpath(folder)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif os.path.splitext(entry.name)[1].lower() in extensions:
                    st = entry.stat()
                    found[entry.path] = (st.st_mtime, st.st_size)
            except OSError:
                continue
    return found


def load_image(path: str, max_side: int = MAX_SIDE) -> Image.Image:
    with Image.open(path) as img:
        if img.format == "JPEG":
            img.draft("RGB", (max_side, max_side))  # dekodowanie od razu w mniejszej skali
        img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def make_tiles(img: Image.Image) -> list:
    """Wycinki 2×2: lewa góra, prawa góra, lewy dół, prawy dół. Puste dla małych obrazów."""
    w, h = img.size
    if min(w, h) < MIN_TILE_SIDE:
        return []
    mx, my = w // 2, h // 2
    boxes = [(0, 0, mx, my), (mx, 0, w, my), (0, my, mx, h), (mx, my, w, h)]
    return [img.crop(b) for b in boxes]


def _decode_worker(paths: queue.Queue, out: queue.Queue):
    while True:
        item = paths.get()
        if item is None:
            out.put(_DONE)
            return
        path, state = item
        try:
            img = load_image(path)
            out.put((path, state, [img] + make_tiles(img)))
        except Exception as e:  # uszkodzony lub nieobsługiwany plik
            out.put((path, state, e))


def run_phase1(store, index, embedder, config, arbiter, show_progress: bool = True) -> IndexResult:
    found = scan_folder(config.folder, config.rozszerzenia)
    plan = plan_sync(store.file_states(), found)
    result = IndexResult(removed=len(set(plan.removed) - set(plan.new)))
    index.remove(store.remove_paths(plan.removed))
    if not plan.new:
        return result

    paths = queue.Queue()
    out = queue.Queue(maxsize=config.queue_maxsize * config.batch_size)
    workers = [threading.Thread(target=_decode_worker, args=(paths, out), daemon=True) for _ in range(config.io_workers)]
    for w in workers:
        w.start()
    for path in plan.new:
        paths.put((path, found[path]))
    for _ in workers:
        paths.put(None)

    pbar = tqdm(total=len(plan.new), desc="Wektory obrazu", unit="zdj", disable=not show_progress)
    batch = []

    def flush():
        crops = [img for _, _, imgs in batch for img in imgs]
        vecs = []
        with arbiter.priority():
            for i in range(0, len(crops), config.batch_size):
                vecs.extend(embedder.embed_images(crops[i : i + config.batch_size]))
        items, pos = [], 0
        for path, (mtime, size), imgs in batch:
            kinds = ["full"] + list(TILE_KINDS[: len(imgs) - 1])
            items.append(NewImage(path, mtime, size, list(zip(kinds, vecs[pos : pos + len(imgs)]))))
            pos += len(imgs)
        ids = store.add_images(items)
        index.add(
            [i for i, item in zip(ids, items) for _ in item.vectors],
            [k for item in items for k, _ in item.vectors],
            [v for item in items for _, v in item.vectors],
        )
        result.added += len(items)
        batch.clear()

    finished = 0
    try:
        while finished < len(workers):
            item = out.get()
            if item is _DONE:
                finished += 1
                continue
            path, state, payload = item
            if isinstance(payload, Exception):
                result.failed.append(path)
                if show_progress:
                    tqdm.write(f"Pominięto {path}: {payload}")
            else:
                batch.append((path, state, payload))
                if len(batch) >= config.batch_size:
                    flush()
            pbar.update(1)
        if batch:
            flush()
    finally:
        pbar.close()
    return result
