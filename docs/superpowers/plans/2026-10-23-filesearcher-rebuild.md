# FileSearcher – przebudowa: plan wdrożenia

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zastąpić `search.py` pakietem `filesearcher/`: szybka faza wektorów obrazu (z wycinkami), druga faza opisów i OCR w tle z postępem, wyszukiwanie hybrydowe w terminalu, profile modeli dobierane do sprzętu.

**Architecture:** Moduły rozmawiają przez interfejsy `Embedder` / `Describer`. Testy używają modeli sztucznych, prawdziwe adaptery są zamknięte w `filesearcher/models/`. Stan trzyma SQLite (WAL + FTS5), wektory w pamięci leżą w macierzy torch na GPU lub CPU.

**Tech Stack:** Python ≥3.10, torch, transformers ≥4.57, accelerate, Pillow, numpy, tqdm, SQLite FTS5, pytest. Opcjonalnie bitsandbytes i easyocr.

**Spec:** `docs/superpowers/specs/2026-10-23-filesearcher-rebuild-design.md`

## Global Constraints

- Uruchamianie: `python -m filesearcher` (opcja `--config`, domyślnie `config.json`).
- Klucze configu i wartości domyślne dokładnie jak w tabeli w specyfikacji; `BAZA_DIR` = `./chroma_db` traktowane jak brak klucza (komunikat).
- Identyfikatory modeli: `Qwen/Qwen3-VL-Embedding-2B`, `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, `jinaai/jina-clip-v2`.
- Wektory zapisane w bazie: float16, L2-znormalizowane; rodzaje `full`, `tile0`..`tile3`, `text`.
- RRF k = 60, 200 kandydatów na źródło, `max_new_tokens` = 384, opis przy ~1 MP, pełny obraz zmniejszony do dłuższego boku 1024 px, wycinki tylko gdy krótszy bok ≥ 448 px.
- Teksty dla użytkownika po polsku, liczby z przecinkiem dziesiętnym i spacją jako separatorem tysięcy.
- Żaden moduł poza `filesearcher/models/` (bez `fake.py`) nie importuje `transformers`, `bitsandbytes` ani `easyocr`.

## Review Focus

1. Ścieżki z polskimi znakami i spacjami (Windows): `scan_folder` i wyszukiwanie mają je zwracać bez zmian (test w zadaniu 5).
2. Pusty folder albo folder bez zdjęć: faza 1 kończy się bez błędu, CLI działa i mówi, że baza jest pusta (test w zadaniu 8).
3. Plik usunięty z dysku między fazą 1 a fazą 2: zdjęcie dostaje `opis_status = -1`, wątek pracuje dalej (test w zadaniu 6).
4. Zapytanie z samymi znakami interpunkcyjnymi albo samymi spacjami: FTS nie rzuca błędu składni (test w zadaniu 3).
5. Zapytanie z „ł” zamiast „l” (i odwrotnie): wyszukiwanie pełnotekstowe nadal trafia (test w zadaniu 3).

Progi profili w GiB mają margines (karty raportują trochę mniej niż nominalnie, np. 16 GB → 15,9 GiB): max ≥ 15,5, high ≥ 9,5, mid ≥ 5,5.

---

### Task 1: Szkielet, config, interfejsy modeli, modele sztuczne

**Files:**
- Create: `filesearcher/__init__.py`, `filesearcher/config.py`, `filesearcher/models/__init__.py` (na razie pusty), `filesearcher/models/base.py`, `filesearcher/models/fake.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_models_base.py`, `requirements-dev.txt`

**Interfaces:**
- Produces:
  - `Config` (frozen dataclass): `folder: Path`, `baza_dir: Path`, `rozszerzenia: frozenset[str]`, `io_workers: int`, `batch_size: int`, `queue_maxsize: int`, `profil: str`, `opisy: bool`, `wyniki: int`; `load_config(path: Path) -> Config`, `ConfigError(Exception)`.
  - `Description(opis: str, tekst: str)`; protokoły `Embedder` (`model_id`, `dim`, `embed_images`, `embed_texts`) i `Describer` (`model_id`, `describe`), jak w specyfikacji; `parse_description(raw: str) -> Description`; `l2_normalize(a: np.ndarray) -> np.ndarray`.
  - `FakeEmbedder(dim=16)`: obraz → wymiary 0–2 = średni kolor RGB; tekst → wymiary 0–2 ze słów kolorów (`czerwony`, `zielony`, `niebieski`, `biały`), reszta wymiarów z haszu słów; wynik znormalizowany. `FakeDescriber(fn=None, delay=0.0)`: domyślnie opis z koloru dominującego.
  - `tests/conftest.py`: fixture `make_image(path, size, color, corner=None)` zapisująca JPEG (opcjonalny kwadrat w innym kolorze w prawym dolnym rogu).

- [ ] Testy `test_config.py`: `test_defaults_applied` (minimalny config tylko z `FOLDER_ZE_ZDJECIAMI` → `baza_dir == Path("./filesearcher_db")`, `wyniki == 5`, `profil == "auto"`, `opisy is True`); `test_old_chroma_dir_replaced` (`BAZA_DIR="./chroma_db"` → domyślny katalog i komunikat w stderr); `test_missing_folder_key_raises` (`ConfigError`); `test_bad_profile_raises`; `test_extensions_lowercased`.
- [ ] Testy `test_models_base.py`: `parse_description` dla poprawnego JSON, JSON w bloku ```json```, niepoprawnego JSON (całość → `opis`, `tekst == ""`), JSON bez pola `tekst`; `FakeEmbedder`: wynik czerwonego obrazu ma najwyższe podobieństwo z tekstem „czerwony”.
- [ ] Implementacja, `pytest -q` zielone, commit.

### Task 2: Wykrywanie sprzętu i profile

**Files:** Create `filesearcher/hardware.py`, `tests/test_hardware.py`

**Interfaces:**
- Produces: `HardwareInfo(device: str, name: str | None, vram_gb: float | None, capability: tuple[int, int] | None, arch_list: list[str])`; `detect() -> HardwareInfo`; `Profile(name, embedder_id, describer_id: str | None, describer_4bit: bool, fallback_describer_id: str | None, device)`; `PROFILES: dict[str, Profile]` (bez `device`, uzupełniane w `resolve_profile`); `pick_profile_name(vram_gb: float | None) -> str`; `torch_problem(hw: HardwareInfo) -> str | None`; `resolve_profile(config_profil: str, hw: HardwareInfo) -> Profile`.
- [ ] Testy: `pick_profile_name` dla `None`, 4.0, 7.9, 11.9, 15.9, 24 → `lite`, `lite`, `mid`, `high`, `max`, `max`; `torch_problem` dla capability (12, 0) bez `sm_120` zwraca tekst zawierający `cu128`; z `sm_120` → `None`; `resolve_profile("lite", gpu_hw)` wymusza lite; na CPU profil inny niż lite przy wymuszeniu działa, ale `device == "cpu"`.
- [ ] Implementacja, testy, commit.

### Task 3: Baza SQLite

**Files:** Create `filesearcher/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Description`.
- Produces: `class Store(db_path: Path)` z metodami: `get_meta(key) -> str | None`, `set_meta(key, value)`, `file_states() -> dict[str, tuple[float, int]]`, `add_images(items: list[NewImage]) -> list[int]` (gdzie `NewImage(path, mtime, size, vectors: list[tuple[str, np.ndarray]])`), `remove_paths(paths: Iterable[str])`, `clear_all()`, `pending_descriptions(include_failed: bool) -> list[tuple[int, str]]`, `counts() -> Counts(total, done, failed)`, `save_description(image_id, desc: Description, text_vec: np.ndarray | None)`, `mark_failed(image_id, error: str)`, `all_vectors() -> tuple[np.ndarray, list[str], np.ndarray]` (ids, kinds, macierz float32), `fts_search(query: str, limit: int) -> list[int]`, `get_info(ids: list[int]) -> dict[int, ImageInfo(path, opis, tekst)]`; `plan_sync(db: dict, found: dict) -> SyncPlan(new: list[str], removed: list[str])` (zmienione pliki są w obu listach); `fts_query(text: str) -> str | None`.
- `fts_query`: słowa z `re.findall(r"\w+", ...)`, każde w cudzysłowie, połączone `OR`; brak słów → `None`.
- [ ] Testy: `plan_sync` dla nowych, zmienionych i usuniętych; zapis i odczyt wektorów (float16 → float32, kształt); `remove_paths` usuwa wektory i tekst; `save_description` → `counts().done == 1`, `fts_search("zolty")` trafia w „żółty”; `fts_search("łódź")` i `fts_search("lodz")` trafiają w „Łódź”; `fts_query("?!  ")` → `None`, a `fts_search("?!")` → `[]`; `pending_descriptions(include_failed=False)` pomija `-1`, `True` je zwraca; `clear_all` zeruje `counts()`; plik bazy na dysku działa po ponownym otwarciu.
- [ ] Implementacja (WAL, `check_same_thread=False`, `threading.Lock` na zapisach, jedna transakcja na wywołanie), testy, commit.

### Task 4: Indeks wektorów w pamięci i arbiter GPU

**Files:** Create `filesearcher/vectors.py`, `filesearcher/gpu.py`, `tests/test_vectors.py`, `tests/test_gpu.py`

**Interfaces:**
- Produces: `class VectorIndex(dim: int, device: str = "cpu")`: `add(image_ids: Sequence[int], kinds: Sequence[str], vecs: np.ndarray)`, `search(query: np.ndarray, group: Literal["image", "text"], k: int) -> list[tuple[int, float, str]]` (najlepszy wiersz na zdjęcie, malejąco), `__len__`; `class GpuArbiter` z kontekstami `priority()` i `background()`.
- `search` dla grupy `image`: `topk(min(n, k * 5))` po zamaskowanych podobieństwach, potem deduplikacja po zdjęciu.
- [ ] Testy: wynik z wycinka (`tile3`) wygrywa, gdy pasuje lepiej niż `full`, i zwraca rodzaj `tile3`; grupa `text` nie widzi wektorów obrazu; dopisanie ponad pojemność początkową zachowuje dane; pusty indeks → `[]`; arbiter: gdy `background()` trzyma blokadę, a w kolejce czekają zapytanie i drugi `background()`, zapytanie wchodzi pierwsze (kolejność zapisana w liście, wątki + `threading.Event`).
- [ ] Implementacja, testy, commit.

### Task 5: Faza 1

**Files:** Create `filesearcher/indexer.py`, `tests/test_indexer.py`

**Interfaces:**
- Consumes: `Store`, `NewImage`, `plan_sync`, `VectorIndex`, `GpuArbiter`, `Embedder`, `Config`.
- Produces: `scan_folder(folder: Path, extensions: frozenset[str]) -> dict[str, tuple[float, int]]` (ścieżki bezwzględne jako `str`); `load_image(path: str, max_side: int = 1024) -> PIL.Image.Image`; `make_tiles(img) -> list[Image]` (kolejność: lewa góra, prawa góra, lewy dół, prawy dół); `run_phase1(store, index, embedder, config, arbiter, show_progress=True) -> IndexResult(added: int, removed: int, failed: list[str])`.
- Wątki dekodujące i kolejka wyników ograniczona do `queue_maxsize * batch_size`. Główny wątek zbiera `batch_size` zdjęć, wywołuje `embed_images` w porcjach po `batch_size` obrazów, zapisuje po każdym batchu przez `add_images` i dopisuje do `index`.
- [ ] Testy: 7 zdjęć, `batch_size=2` → wszystkie w bazie i w indeksie; uszkodzony plik → w `failed`, reszta dodana; obraz 300×300 bez wycinków, 1000×800 z 4; drugie uruchomienie bez zmian → `added == 0`; zmiana pliku (nowa treść) → plik zaindeksowany ponownie, `opis_status` wraca do 0; usunięcie pliku → `removed == 1`; ścieżka `zdjęcia/łódź nad morzem.jpg` wraca w `file_states()` bez zmian; `queue_maxsize=1` przy 20 zdjęciach kończy się (timeout testu 30 s).
- [ ] Implementacja, testy, commit.

### Task 6: Faza 2 w tle i postęp

**Files:** Create `filesearcher/describer.py`, `tests/test_describer.py`

**Interfaces:**
- Consumes: `Store`, `VectorIndex`, `GpuArbiter`, `Embedder`, `Describer`.
- Produces: `ProgressState(done, total, failed, rate: float | None, eta_s: float | None, finished: bool)`; `format_progress(state) -> str`; `class BackgroundDescriber(store, index, embedder, describer, arbiter, log_path: Path, window: int = 20, clock=time.monotonic)` z `start()`, `stop(timeout: float | None = None)`, `status() -> ProgressState`; `load_for_description(path: str, max_pixels: int = 1_000_000) -> Image`.
- Lista do zrobienia jest brana raz, przy `start()`, z `include_failed=True`. Każde zdjęcie: `describe` i `embed_texts` w jednym `arbiter.background()`; wyjątek → `mark_failed` + wpis do logu.
- [ ] Testy `format_progress`: `(1240, 10000, 0, 0.9, 9720)` → `"[Opisy i OCR: 1 240 / 10 000 · 12% · 0,9 zdj/s · ~2 h 42 min]"`; bez szybkości → `"… · szacuję…]"`; `failed=3` → kończy się `" · błędy: 3]"`; `finished` → `"[Opisy i OCR: gotowe, 10 000 zdjęć]"`; ETA 240 s → `"~4 min"`, 30 s → `"<1 min"`.
- [ ] Testy działania: 5 zdjęć → `done == 5`, w indeksie 5 wektorów `text`, `fts_search` znajduje opis; `stop()` po 2 zdjęciach (opóźnienie w `FakeDescriber`) i nowy `BackgroundDescriber` kończy pozostałe 3; zdjęcie usunięte z dysku przed fazą 2 → `failed == 1`, reszta gotowa, błąd w logu; opis z pustym tekstem i opisem → brak wektora `text`.
- [ ] Implementacja, testy, commit.

### Task 7: Wyszukiwanie hybrydowe

**Files:** Create `filesearcher/search.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `Store.fts_search`, `Store.get_info`, `VectorIndex.search`, `Embedder.embed_texts`, `GpuArbiter.priority`.
- Produces: `Hit(image_id, path, score: float | None, sources: dict[str, str])` (klucze `obraz`, `opis`, `OCR`); `rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]`; `fold(text: str) -> str` (małe litery, bez diakrytyków, `ł` → `l`); `search(query, store, index, embedder, arbiter, limit, candidates=200) -> list[Hit]`.
- Etykiety `obraz`: `full` → `cały kadr`, `tile0..3` → `wycinek lewa góra` / `wycinek prawa góra` / `wycinek lewy dół` / `wycinek prawy dół`. Trafienie FTS dostaje etykietę `OCR`, gdy słowo zapytania (po `fold`) występuje w `tekst`, w przeciwnym razie `opis`. Fragment to ok. 100 znaków wokół pierwszego trafienia.
- [ ] Testy: `rrf([[1,2],[2,3]])` → 2 pierwsze; zdjęcie z napisem w OCR `„ROWER MIEJSKI”` znalezione zapytaniem „rower” ze źródłem `OCR`; białe zdjęcie z czerwonym rogiem → `sources["obraz"] == "wycinek prawy dół"`; zdjęcie bez opisu nadal w wynikach; `limit` respektowany; pusta baza → `[]`.
- [ ] Implementacja, testy, commit.

### Task 8: Aplikacja i terminal

**Files:** Create `filesearcher/app.py`, `filesearcher/cli.py`, `filesearcher/__main__.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: wszystko powyżej, `load_models` z zadania 9 (w testach podmieniane).
- Produces: `ensure_embedder(store, embedder, ask: Callable[[str], bool]) -> bool`; `App` (config, store, index, embedder, arbiter, background: BackgroundDescriber | None); `open_app(config, embedder, describer, ask) -> App | None` (ładuje wektory z bazy do indeksu); `run_cli(app, input_fn=input, print_fn=print)`; `format_hit(n: int, hit: Hit) -> str`; `main(argv=None) -> int`.
- `main`: config → `detect()` → `torch_problem` (błąd = kod 1) → `resolve_profile` → wypisanie profilu i modeli → `load_models` → `open_app` → `run_phase1` → start fazy 2 (gdy `opisy` i jest describer) → `run_cli` → `stop()`. `KeyboardInterrupt` w fazie 1 kończy program (zapisane batche zostają).
- [ ] Testy: `ensure_embedder`: pusta baza zapisuje meta; inny model + `ask → False` → `False`; `ask → True` → baza wyczyszczona; `run_cli` z wejściem `["czerwony", "s", "q"]` wypisuje linijkę postępu, wynik `[1]` i kończy; pusta linia kończy; `EOFError` kończy bez wyjątku; pusta baza → komunikat `Baza jest pusta`; `format_hit` zgodny z przykładem w specyfikacji.
- [ ] Implementacja, testy, commit.

### Task 9: Prawdziwe adaptery modeli

**Files:** Create `filesearcher/models/qwen_embed.py`, `filesearcher/models/jina_clip.py`, `filesearcher/models/qwen_vl.py`, `filesearcher/models/easyocr_only.py`; Modify `filesearcher/models/__init__.py`; Test `tests/test_load_models.py`

**Interfaces:**
- Produces: `ModelBundle(embedder: Embedder, describer: Describer | None)`; `load_models(profile: Profile, log=print) -> ModelBundle`. Importy bibliotek wewnątrz funkcji ładujących.
- Kod ładowania i wywołania według kart modeli na Hugging Face. `qwen_vl.QwenVLDescriber` używa promptu z `DESCRIBE_PROMPT` (stała w module) i `parse_description`.
- Powrót z 4-bit: wyjątek przy ładowaniu z `BitsAndBytesConfig` → komunikat i `fallback_describer_id` w bf16.
- [ ] Test `test_load_models.py`: z podmienionymi funkcjami ładującymi (monkeypatch) awaria 4-bit uruchamia ładowanie `Qwen/Qwen3-VL-2B-Instruct` i wypisuje ostrzeżenie; profil `lite` daje describer EasyOCR. Prawdziwych modeli nie da się tu uruchomić (brak GPU i dostępu do HF): sprawdzenie `python -c "import filesearcher.models.qwen_vl"` i ręczny przegląd z kartą modelu.
- [ ] Commit.

### Task 10: Sprzątanie i dokumentacja

**Files:** Delete `search.py`; Modify `requirements.txt`, `README.md`, `config.json` (bez `BAZA_DIR` = `./chroma_db`), `.gitignore` (baza, `__pycache__`)

- [ ] README: instalacja (osobne polecenie torch z `cu128` dla RTX 50xx, sprawdzenie `torch.cuda.get_arch_list()`), profile i ich modele, opis dwóch faz, komendy `q` / `s`, uwaga o starej bazie `chroma_db`.
- [ ] `pytest -q` całość zielona, commit, push.
