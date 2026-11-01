"""Baza SQLite: zdjęcia, wektory, wyszukiwanie pełnotekstowe i stan fazy 2."""
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .models.base import Description

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    opis_status INTEGER NOT NULL DEFAULT 0,  -- 0 czeka, 1 gotowe, -1 błąd
    opis_error TEXT,
    opis TEXT,
    tekst TEXT
);
CREATE TABLE IF NOT EXISTS vectors (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS vectors_image ON vectors(image_id);
CREATE VIRTUAL TABLE IF NOT EXISTS teksty USING fts5(opis, tekst, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class NewImage:
    path: str
    mtime: float
    size: int
    vectors: list  # [(kind, np.ndarray)]


@dataclass(frozen=True)
class SyncPlan:
    new: list
    removed: list


@dataclass(frozen=True)
class Counts:
    total: int
    done: int
    failed: int


@dataclass(frozen=True)
class ImageInfo:
    path: str
    opis: str
    tekst: str


def fold(text: str) -> str:
    """Małe litery bez polskich znaków: „Łódź” → „lodz”."""
    text = text.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def fts_query(text: str) -> str | None:
    words = re.findall(r"\w+", fold(text))
    return " OR ".join(f'"{w}"' for w in words) if words else None


def plan_sync(db: dict, found: dict) -> SyncPlan:
    """Zmienione pliki są w obu listach: najpierw usunięcie, potem ponowne dodanie."""
    changed = [p for p, state in found.items() if p in db and db[p] != state]
    new = [p for p in found if p not in db] + changed
    removed = [p for p in db if p not in found] + changed
    return SyncPlan(new=new, removed=removed)


class Store:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self):
        columns = {r[1] for r in self._db.execute("PRAGMA table_info(images)")}
        if "ocr_status" not in columns:
            # Stare opisy z JSON-a bywaly uciete na limicie tokenow – liczymy je od nowa nowym formatem.
            self._db.execute("ALTER TABLE images ADD COLUMN ocr_status INTEGER NOT NULL DEFAULT 0")
            self._db.execute("UPDATE images SET opis_status = 0, opis_error = NULL")

    def close(self):
        with self._lock:
            self._db.close()

    def _read(self, sql, params=()):
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # ── meta ──
    def get_meta(self, key: str) -> str | None:
        rows = self._read("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0][0] if rows else None

    def set_meta(self, key: str, value: str):
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ── faza 1 ──
    def file_states(self) -> dict:
        return {p: (m, s) for p, m, s in self._read("SELECT path, mtime, size FROM images")}

    def add_images(self, items: list[NewImage]) -> list[int]:
        ids = []
        with self._lock, self._db:
            for item in items:
                cur = self._db.execute(
                    "INSERT INTO images(path, mtime, size) VALUES (?, ?, ?)", (item.path, item.mtime, item.size)
                )
                ids.append(cur.lastrowid)
                self._db.executemany(
                    "INSERT INTO vectors(image_id, kind, vec) VALUES (?, ?, ?)",
                    [(cur.lastrowid, kind, np.asarray(v, dtype=np.float16).tobytes()) for kind, v in item.vectors],
                )
        return ids

    def remove_paths(self, paths: Iterable[str]) -> list[int]:
        removed = []
        with self._lock, self._db:
            for path in paths:
                row = self._db.execute("SELECT id FROM images WHERE path = ?", (path,)).fetchone()
                if row:
                    self._db.execute("DELETE FROM teksty WHERE rowid = ?", row)
                    self._db.execute("DELETE FROM images WHERE id = ?", row)
                    removed.append(row[0])
        return removed

    def clear_all(self):
        with self._lock, self._db:
            self._db.execute("DELETE FROM teksty")
            self._db.execute("DELETE FROM vectors")
            self._db.execute("DELETE FROM images")

    def all_vectors(self):
        rows = self._read("SELECT image_id, kind, vec FROM vectors ORDER BY rowid")
        if not rows:
            return np.zeros(0, dtype=np.int64), [], np.zeros((0, 0), dtype=np.float32)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = np.stack([np.frombuffer(r[2], dtype=np.float16) for r in rows]).astype(np.float32)
        return ids, [r[1] for r in rows], mat

    # ── faza 2 ──
    def pending_descriptions(self, include_failed: bool) -> list[tuple[int, str]]:
        statuses = "(0, -1)" if include_failed else "(0)"
        return [tuple(r) for r in self._read(f"SELECT id, path FROM images WHERE opis_status IN {statuses} ORDER BY id")]

    def counts(self) -> Counts:
        total, done, failed = self._read(
            "SELECT COUNT(*), COALESCE(SUM(opis_status = 1), 0), COALESCE(SUM(opis_status = -1), 0) FROM images"
        )[0]
        return Counts(total, done, failed)

    def pending_ocr(self) -> list[tuple[int, str]]:
        return [tuple(r) for r in self._read("SELECT id, path FROM images WHERE ocr_status = 0 ORDER BY id")]

    def ocr_counts(self) -> tuple[int, int]:
        """(zrobione, wszystkie) – błąd OCR też liczy się jako zrobione."""
        return tuple(self._read("SELECT COALESCE(SUM(ocr_status != 0), 0), COUNT(*) FROM images")[0])

    def save_ocr(self, image_id: int, tekst: str | None, text_vec: np.ndarray | None):
        """tekst=None oznacza błąd OCR: zdjęcie nie wraca do kolejki, opis i tak je obejmie."""
        with self._lock, self._db:
            if tekst is None:
                self._db.execute("UPDATE images SET ocr_status = -1 WHERE id = ?", (image_id,))
                return
            self._db.execute("UPDATE images SET ocr_status = 1, tekst = ? WHERE id = ?", (tekst, image_id))
            self._write_text_index(image_id, text_vec)

    def save_description(self, image_id: int, desc: Description, text_vec: np.ndarray | None):
        """Pusty desc.tekst zostawia tekst z OCR."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE images SET opis_status = 1, opis_error = NULL, opis = ?, "
                "tekst = CASE WHEN ? = '' THEN tekst ELSE ? END WHERE id = ?",
                (desc.opis, desc.tekst, desc.tekst, image_id),
            )
            self._write_text_index(image_id, text_vec)

    def get_text(self, image_id: int) -> tuple[str, str]:
        row = self._read("SELECT opis, tekst FROM images WHERE id = ?", (image_id,))
        return (row[0][0] or "", row[0][1] or "") if row else ("", "")

    def _write_text_index(self, image_id: int, text_vec: np.ndarray | None):
        """FTS i wektor tekstu z tego, co jest w wierszu; wolac pod blokada i w transakcji."""
        row = self._db.execute("SELECT opis, tekst FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:  # plik usuniety w miedzyczasie
            return
        self._db.execute("DELETE FROM teksty WHERE rowid = ?", (image_id,))
        self._db.execute(
            "INSERT INTO teksty(rowid, opis, tekst) VALUES (?, ?, ?)", (image_id, fold(row[0] or ""), fold(row[1] or ""))
        )
        self._db.execute("DELETE FROM vectors WHERE image_id = ? AND kind = 'text'", (image_id,))
        if text_vec is not None:
            self._db.execute(
                "INSERT INTO vectors(image_id, kind, vec) VALUES (?, 'text', ?)",
                (image_id, np.asarray(text_vec, dtype=np.float16).tobytes()),
            )

    def mark_failed(self, image_id: int, error: str):
        with self._lock, self._db:
            self._db.execute("UPDATE images SET opis_status = -1, opis_error = ? WHERE id = ?", (error, image_id))

    # ── wyszukiwanie ──
    def fts_search(self, query: str, limit: int) -> list[int]:
        q = fts_query(query)
        if q is None:
            return []
        rows = self._read("SELECT rowid FROM teksty WHERE teksty MATCH ? ORDER BY bm25(teksty) LIMIT ?", (q, limit))
        return [r[0] for r in rows]

    def get_info(self, ids: list[int]) -> dict:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._read(f"SELECT id, path, opis, tekst FROM images WHERE id IN ({marks})", tuple(ids))
        return {r[0]: ImageInfo(r[1], r[2] or "", r[3] or "") for r in rows}
