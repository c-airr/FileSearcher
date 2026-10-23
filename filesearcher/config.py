"""Wczytywanie i walidacja config.json."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROFILE_NAMES = ("auto", "max", "high", "mid", "lite")
DEFAULT_BAZA_DIR = Path("./filesearcher_db")
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    folder: Path
    baza_dir: Path = DEFAULT_BAZA_DIR
    rozszerzenia: frozenset = frozenset(DEFAULT_EXTENSIONS)
    io_workers: int = 4
    batch_size: int = 16
    queue_maxsize: int = 8
    profil: str = "auto"
    opisy: bool = True
    wyniki: int = 5


def _positive_int(data, key, default):
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{key} musi być dodatnią liczbą całkowitą (jest: {value!r})")
    return value


def load_config(path: Path) -> Config:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Nie znaleziono pliku konfiguracji: {path}") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"Błędny JSON w {path}: {e}") from None

    if not data.get("FOLDER_ZE_ZDJECIAMI"):
        raise ConfigError("Brak klucza FOLDER_ZE_ZDJECIAMI w config.json")

    baza = data.get("BAZA_DIR")
    if baza is None or Path(baza) == Path("./chroma_db"):
        if baza is not None:
            print(
                "Uwaga: BAZA_DIR wskazuje na starą bazę chroma_db – nowa baza trafi do "
                f"{DEFAULT_BAZA_DIR}. Stary katalog można usunąć.",
                file=sys.stderr,
            )
        baza = DEFAULT_BAZA_DIR

    profil = data.get("PROFIL", "auto")
    if profil not in PROFILE_NAMES:
        raise ConfigError(f"PROFIL musi być jednym z: {', '.join(PROFILE_NAMES)} (jest: {profil!r})")

    exts = frozenset(
        (e if e.startswith(".") else "." + e).lower() for e in data.get("ROZSZERZENIA", DEFAULT_EXTENSIONS)
    )

    return Config(
        folder=Path(data["FOLDER_ZE_ZDJECIAMI"]),
        baza_dir=Path(baza),
        rozszerzenia=exts,
        io_workers=_positive_int(data, "IO_WORKERS", 4),
        batch_size=_positive_int(data, "BATCH_SIZE", 16),
        queue_maxsize=_positive_int(data, "QUEUE_MAXSIZE", 8),
        profil=profil,
        opisy=bool(data.get("OPISY", True)),
        wyniki=_positive_int(data, "WYNIKI", 5),
    )
