# FileSearcher – przebudowa (projekt)

Data: 2026-10-23 · Status: do akceptacji

## Cel

Lokalne narzędzie w terminalu do wyszukiwania własnych zdjęć po opisie w języku
naturalnym (głównie po polsku). W stosunku do obecnego `search.py` ma:

- działać poprawnie (obecny kod woła `CLIPModel` przez pipeline `feature-extraction`,
  co się wysypuje, a nawet gdyby przeszło, wektory obrazu i tekstu miałyby różne wymiary),
- używać lepszych, wielojęzycznych modeli,
- znajdować drobne obiekty, tekst widoczny na zdjęciach (OCR) i złożone sceny,
- indeksować szybciej i dobierać modele do sprzętu użytkownika.

Sprzęt docelowy autora: RTX 5070, 12 GB VRAM, Windows. Program ma działać także na
słabszych kartach i na samym CPU.

## Poza zakresem

Interfejs w przeglądarce, wideo, rozpoznawanie twarzy, obserwowanie folderu na
bieżąco (zmiany wykrywane są przy uruchomieniu), migracja starej bazy `chroma_db`.

## Struktura

Pakiet `filesearcher/`, uruchamiany przez `python -m filesearcher`.

| Moduł | Odpowiedzialność |
|---|---|
| `config.py` | Wczytuje `config.json`, uzupełnia wartości domyślne, waliduje. |
| `hardware.py` | Wykrywa urządzenie i VRAM, wybiera profil, sprawdza zgodność PyTorcha z GPU. |
| `models/base.py` | Interfejsy `Embedder` i `Describer` (protokoły). |
| `models/qwen_embed.py`, `models/jina_clip.py` | Implementacje `Embedder`. |
| `models/qwen_vl.py`, `models/easyocr_only.py` | Implementacje `Describer`. |
| `models/fake.py` | Deterministyczne modele do testów. |
| `store.py` | SQLite: zdjęcia, wektory, FTS5, stan drugiej fazy, metadane bazy. |
| `vectors.py` | Macierz wektorów w pamięci (GPU lub CPU) + wyszukiwanie top-k. |
| `gpu.py` | Arbiter dostępu do GPU (pierwszeństwo dla zapytań). |
| `indexer.py` | Faza 1: skan, wykrywanie zmian, dekodowanie, wektory obrazu. |
| `describer.py` | Faza 2 w tle: opisy + OCR, postęp, wznawianie. |
| `search.py` | Wyszukiwanie w trzech źródłach i łączenie wyników (RRF). |
| `cli.py`, `__main__.py` | Terminal. |

Interfejsy modeli:

```python
class Embedder(Protocol):
    model_id: str
    dim: int
    def embed_images(self, images: list[PIL.Image.Image]) -> np.ndarray: ...  # (n, dim), L2-znormalizowane
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...                # (n, dim), L2-znormalizowane

class Describer(Protocol):
    model_id: str
    def describe(self, image: PIL.Image.Image) -> Description: ...  # Description(opis: str, tekst: str)
```

Reszta programu zna tylko te interfejsy. Dzięki temu testy działają na `models/fake.py`,
a zmiana modelu dotyczy jednego pliku.

## Konfiguracja

`config.json` zachowuje obecne klucze. Nowe klucze są opcjonalne:

| Klucz | Domyślnie | Znaczenie |
|---|---|---|
| `FOLDER_ZE_ZDJECIAMI` | – (wymagany) | folder do indeksowania |
| `BAZA_DIR` | `./filesearcher_db` | katalog z `index.sqlite` |
| `ROZSZERZENIA` | jak teraz | obsługiwane rozszerzenia |
| `IO_WORKERS` | 4 | wątki dekodujące zdjęcia |
| `BATCH_SIZE` | 16 | batch dla modelu wektorowego |
| `QUEUE_MAXSIZE` | 8 | limit batchy czekających w pamięci |
| `PROFIL` | `"auto"` | `auto` / `max` / `high` / `mid` / `lite` |
| `OPISY` | `true` | czy uruchamiać fazę 2 |
| `WYNIKI` | 5 | liczba wyników na zapytanie |

Domyślna wartość `BAZA_DIR` zmienia się celowo, żeby nie mieszać nowej bazy ze starym
katalogiem `chroma_db`. Wartość `./chroma_db` w istniejącym configu jest traktowana jak
brak klucza, z komunikatem.

## Profile i modele

`hardware.py` wybiera profil według całkowitego VRAM pierwszego GPU CUDA:

| Profil | VRAM | Embedder | Describer |
|---|---|---|---|
| max | ≥16 GB | Qwen3-VL-Embedding-2B, bf16 | Qwen3-VL-4B-Instruct, bf16 |
| high | 10–16 GB | Qwen3-VL-Embedding-2B, bf16 | Qwen3-VL-4B-Instruct, 4-bit NF4 |
| mid | 6–10 GB | Qwen3-VL-Embedding-2B, bf16 | Qwen3-VL-2B-Instruct, 4-bit NF4 |
| lite | <6 GB lub CPU | jina-clip-v2 | EasyOCR (`pl`, `en`), bez opisów |

- Ten sam embedder zamienia na wektory zdjęcia, wycinki, opisy, tekst z OCR i zapytania,
  więc wszystko leży w jednej przestrzeni i nie ma osobnego modelu tekstowego.
- Profile high i mid: jeśli nie da się załadować 4-bit (brak lub błąd `bitsandbytes`),
  describer przechodzi na Qwen3-VL-2B-Instruct w bf16 i wypisuje ostrzeżenie.
- Jeśli GPU ma compute capability 12.x (RTX 50xx), a `torch.cuda.get_arch_list()` nie
  zawiera `sm_120`, program kończy się komunikatem z poleceniem instalacji PyTorcha z
  indeksu `cu128`.
- Dokładny sposób ładowania i wywołania każdego modelu (klasy, `trust_remote_code`,
  format promptu embeddera) wynika z karty modelu na Hugging Face. Ten kod jest
  zamknięty w module danego adaptera.

W tabeli `meta` baza zapisuje `embedder_id` i `dim`. Jeśli przy starcie wybrany embedder
jest inny, program pyta, czy przebudować wektory (tak: czyści wektory i stan fazy 2,
nie: kończy się). Zmiana samego describera nie unieważnia bazy. Nowe opisy powstają
nowym modelem, a już istniejące zostają.

## Baza danych (SQLite, tryb WAL)

```sql
images(id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL, size INTEGER,
       opis_status INTEGER DEFAULT 0,   -- 0 czeka, 1 gotowe, -1 błąd
       opis_error TEXT)
vectors(image_id INTEGER, kind TEXT,    -- 'full', 'tile0'..'tile3', 'text'
        vec BLOB)                       -- float16, L2-znormalizowany
teksty USING fts5(opis, tekst, tokenize='unicode61 remove_diacritics 2')
       -- rowid = images.id; tabela przechowuje treść, więc działa snippet()
meta(key TEXT PRIMARY KEY, value TEXT)
```

`remove_diacritics` sprawia, że zapytanie „zolty” trafia w „żółty”.

Zapisy idą z jednego wątku naraz (blokada w `store.py`), a każde zdjęcie w fazie 2 to
osobna transakcja.

## Faza 1 – szybka

1. Skan folderu (`os.scandir` rekurencyjnie) i porównanie z bazą po `(path, mtime, size)`:
   - nowe pliki: do zaindeksowania,
   - zmienione: usunięcie starych wektorów i tekstu, `opis_status = 0`, ponowne indeksowanie,
   - usunięte z dysku: usunięcie z bazy.
2. `IO_WORKERS` wątków dekoduje obrazy: `Image.draft` dla JPEG, `ImageOps.exif_transpose`,
   konwersja do RGB, zmniejszenie do dłuższego boku 1024 px. Z każdego obrazu powstaje
   pełny obraz i 4 wycinki 2×2. Wycinki pomijam, gdy krótszy bok ma mniej niż 448 px.
3. Kolejka ograniczona do `QUEUE_MAXSIZE` batchy. Główny wątek odbiera batche **w trakcie**
   dekodowania, liczy wektory (`torch.inference_mode`, bf16 na GPU) i zapisuje je po
   każdym batchu. Przerwanie nie traci tego, co już zapisano.
4. Pasek tqdm z liczbą zdjęć i szybkością. Błędne pliki: jedna linia ostrzeżenia, plik
   pominięty, ponowna próba przy następnym uruchomieniu.

## Faza 2 – opisy i OCR w tle

- Startuje w osobnym wątku po fazie 1 (gdy `OPISY` = true) i bierze zdjęcia z
  `opis_status = 0`, a przy nowym uruchomieniu także `-1` (jedna ponowna próba na sesję).
- Qwen3-VL dostaje zdjęcie ograniczone do ~1 MP i jedno polecenie po polsku. Ma zwrócić
  JSON `{"opis": "...", "tekst": "..."}` (opis sceny ze szczegółami, cały widoczny tekst
  dosłownie), przy `max_new_tokens` = 384. Jeśli JSON się nie sparsuje, cała odpowiedź
  idzie do `opis`, a `tekst` zostaje pusty.
- Po każdym zdjęciu, w jednej transakcji: wpis do `teksty`, wektor `text` z embeddera dla
  `opis + "\n" + tekst` (jeśli niepusty), `opis_status = 1`. Wektor trafia też od razu do
  macierzy w pamięci.
- Postęp: `zrobione / wszystkie`, szybkość jako średnia krocząca z ostatnich 20 zdjęć,
  szacowany czas do końca.
- Ctrl+C lub `q`: wątek kończy bieżące zdjęcie (do 1 zdjęcia opóźnienia) i się zatrzymuje.
  Twarde wyłączenie gubi co najwyżej bieżące zdjęcie.

## Arbiter GPU

`gpu.py` udostępnia dwa konteksty na jednej blokadzie:

- `priority()` dla zapytań: ustawia flagę „czeka zapytanie” i bierze blokadę,
- `background()` dla fazy 2: przed wzięciem blokady czeka, aż flaga zniknie.

Faza 2 bierze blokadę na jedno zdjęcie. Zapytanie czeka więc najwyżej na dokończenie
bieżącego opisu, a potem idzie przed kolejnym.

## Wyszukiwanie

1. Zapytanie → jeden wektor (`embed_texts`, w `priority()`).
2. Trzy listy kandydatów, po 200 każda:
   - **obraz:** podobieństwo kosinusowe do wektorów `full` i `tile*`; dla zdjęcia liczy
     się najlepszy z nich, a w wyniku zapamiętuję, który to był,
   - **opis:** podobieństwo do wektorów `text`,
   - **tekst:** FTS5 `MATCH` na słowach zapytania połączonych przez `OR`, ranking `bm25`,
     fragment z `snippet()`.
3. Łączenie przez Reciprocal Rank Fusion (k = 60), top `WYNIKI`.
4. Pokazywana liczba to najwyższe podobieństwo kosinusowe ze źródeł wektorowych (dla
   orientacji). O kolejności decyduje RRF.

Macierz wektorów (`vectors.py`) leży na GPU w float16 (na CPU w float32) razem z tablicą
`wiersz → (image_id, kind)`. Dopisywanie wierszy z fazy 2 idzie pod blokadą, z
podwajaniem pojemności jak w liście.

## Terminal

```
[Opisy i OCR: 1 240 / 10 000 · 12% · 0,9 zdj/s · ~2 h 42 min]
Szukaj (q = wyjście, s = status): czerwony rower przy płocie

[1] 0.87  F:/POZOSTALOSCI/2019/IMG_0412.jpg
    obraz: wycinek prawy dół · opis: „czerwony rower oparty o drewniany płot…”
[2] 0.81  F:/POZOSTALOSCI/2021/paragon.jpg
    OCR: „…ROWER MIEJSKI…”
```

- Linijka postępu nad każdym promptem. `s` wypisuje ją od razu. Po zakończeniu fazy 2:
  `[Opisy i OCR: gotowe, 10 000 zdjęć]`.
- Przy wyniku tylko źródła, które go znalazły; opis i OCR skrócone do ~100 znaków.
- `q`, pusta linia lub Ctrl+C: zatrzymanie fazy 2 (jak wyżej) i wyjście.
- Podczas fazy 2 wątek w tle nic nie wypisuje, żeby nie psuć wpisywania. Błędy idą do
  `BAZA_DIR/filesearcher.log`, a ich liczba pojawia się w linijce postępu.

## Wydajność – podsumowanie zmian

Prawdziwe batche na GPU zamiast pojedynczych obrazów; bf16 i `inference_mode`; faza 1
przetwarza w trakcie wczytywania, z ograniczoną pamięcią; `Image.draft` dla JPEG;
wyszukiwanie przez jedno mnożenie macierzy na GPU; najwolniejszy etap (VLM) w tle, z
możliwością szukania od razu.

## Testy

pytest, tylko na `models/fake.py` (bez GPU i bez sieci), z obrazami generowanymi w
`tmp_path`:

- config: wartości domyślne, obsługa starego `./chroma_db`,
- wybór profilu dla symulowanych wartości VRAM i komunikat dla sm_120,
- store: wykrywanie nowych, zmienionych i usuniętych plików; zmiana embeddera,
- faza 1: wszystkie pliki zaindeksowane, uszkodzony plik pominięty, wycinki tylko dla
  dużych obrazów, ograniczona kolejka (brak zakleszczeń),
- faza 2: wznawianie po przerwaniu, błędny JSON, licznik postępu i szacowany czas,
- arbiter: zapytanie wchodzi przed kolejnym zadaniem w tle,
- wyszukiwanie: RRF, trafienie przez FTS bez polskich znaków, wynik z wycinka,
- CLI: przebieg z wejściem z pliku (wyszukanie + `s` + `q`).

Prawdziwe modele i szybkość autor sprawdza na swoim komputerze. Środowisko, w którym
powstaje kod, nie ma GPU ani dostępu do Hugging Face.

## Zależności

`requirements.txt`: `torch` (w README osobne polecenie z indeksem `cu128` dla RTX 50xx),
`transformers>=4.57`, `accelerate`, `pillow`, `numpy`, `tqdm`. Opcjonalnie:
`bitsandbytes` (profile high/mid), `easyocr` (profil lite). Do testów: `pytest`.

## Ryzyka

- Benchmarki Qwen3-VL-Embedding pochodzą od autorów modelu.
- `bitsandbytes` na Windows i RTX 50xx: obsłużone przez automatyczny powrót do 2B w bf16.
- Szybkość fazy 2 (szacunkowo 1–2 s/zdjęcie na RTX 5070) jest niezweryfikowana.
- Kod adapterów modeli powstaje według kart modeli, bez uruchomienia. Mogą być potrzebne
  poprawki po pierwszym teście u autora.

## Do usunięcia

`search.py` (zastąpiony pakietem). README zaktualizowany: instalacja, profile,
PyTorch dla RTX 50xx, nowa baza.
