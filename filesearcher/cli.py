"""Terminal: indeksowanie, a potem pętla zapytań z postępem fazy 2."""
import argparse
import sys
from pathlib import Path

from . import hardware, models
from .app import open_app
from .config import ConfigError, load_config
from .describer import format_progress
from .indexer import run_phase1
from .search import search

PROMPT = "Szukaj (q = wyjście, s = status): "
SOURCE_ORDER = ("obraz", "opis", "OCR")


def format_hit(n: int, hit) -> str:
    score = f"{hit.score:.2f}" if hit.score is not None else "   –"
    line = f"[{n}] {score}  {hit.path}"
    parts = []
    for name in SOURCE_ORDER:
        if name in hit.sources:
            value = hit.sources[name]
            parts.append(f"{name}: {value}" if name == "obraz" else f"{name}: „{value}”")
    return line + ("\n    " + " · ".join(parts) if parts else "")


def run_cli(app, input_fn=None, print_fn=print):
    input_fn = input_fn or input
    if app.store.counts().total == 0:
        print_fn("Baza jest pusta – w folderze nie ma obsługiwanych zdjęć.")
        return
    while True:
        if app.background is not None:
            print_fn(format_progress(app.background.status()))
        try:
            query = input_fn(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print_fn("")
            return
        if not query or query.lower() == "q":
            return
        if query.lower() == "s":
            if app.background is None:
                print_fn("Opisy i OCR są wyłączone.")
            continue
        hits = search(query, app.store, app.index, app.embedder, app.arbiter, app.config.wyniki)
        print_fn("")
        if not hits:
            print_fn("Nic nie znaleziono.")
        for n, hit in enumerate(hits, 1):
            print_fn(format_hit(n, hit))
            print_fn("")


def _ask(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("t", "tak", "y", "yes")
    except EOFError:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="filesearcher", description="Wyszukiwanie zdjęć po opisie.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Błąd konfiguracji: {e}", file=sys.stderr)
        return 1
    if not config.folder.is_dir():
        print(f"Błąd: folder {config.folder} nie istnieje.", file=sys.stderr)
        return 1

    hw = hardware.detect()
    problem = hardware.torch_problem(hw)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    profile = hardware.resolve_profile(config.profil, hw)
    where = f"{hw.name}, {hw.vram_gb:.1f} GB VRAM" if hw.device == "cuda" else "CPU"
    print(f"Profil: {profile.name} ({where})")
    print(f"  wektory: {profile.embedder_id}")
    print(f"  opisy i OCR: {profile.describer_id if config.opisy else 'wyłączone'}")

    print("Ładowanie modeli…")
    bundle = models.load_models(profile)
    app = open_app(config, bundle.embedder, bundle.describer if config.opisy else None, _ask, device=profile.device)
    if app is None:
        print("Bez przebudowy bazy nie mogę użyć nowego modelu. Kończę.")
        return 1

    try:
        result = run_phase1(app.store, app.index, app.embedder, config, app.arbiter)
    except KeyboardInterrupt:
        print("\nPrzerwano – zdjęcia zapisane do tej pory zostają w bazie.")
        return 130
    print(f"Wektory obrazu: dodano {result.added}, usunięto {result.removed}, pominięto {len(result.failed)}.")

    if app.background is not None:
        app.background.start()
        print("Opisy i OCR liczą się w tle – możesz już szukać.")
    try:
        run_cli(app)
    finally:
        if app.background is not None:
            print("Zatrzymuję opisy (kończę bieżące zdjęcie)…")
            app.background.stop(timeout=120)
        app.store.close()
    return 0
