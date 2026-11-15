# 🔍 FileSearcher (wyszukiwarka zdjęć AI)

**FileSearcher** indeksuje zdjęcia na Twoim dysku i pozwala szukać ich zwykłym opisem po polsku, np. „pies na plaży”, „paragon ze stacji benzynowej” albo „czerwony rower przy płocie”. Wszystko działa lokalnie, na Twoim komputerze.

## ✨ Jak to działa

Indeksowanie ma dwie fazy:

1. **Szybka faza – wektory obrazu.** Każde zdjęcie (i dodatkowo jego 4 wycinki, żeby łapać drobne obiekty) trafia do wielojęzycznego modelu wektorowego. Po tej fazie można już szukać.
2. **Druga faza w tle – OCR i zwięzłe opisy.** Składa się z dwóch etapów:
   - **Szybki OCR (Windows OCR):** Wbudowany silnik systemowy (`Windows.Media.Ocr`) natychmiast przepisuje widoczny tekst (zrzuty ekranu, komunikatory, kody, dokumenty) z pełnej rozdzielczości z prędkością ok. **20 zdjęć/s** na CPU. Po tym etapie cały tekst jest już wyszukiwalny.
   - **Opisy scen (Qwen3-VL w paczkach):** Model widzenia generuje zwięzłe opisy (1–3 zdania w formacie `OPIS:`) w paczkach (batch 8). Jeśli OCR wyciągnął tekst wcześniej, model skupia się wyłącznie na opisie sceny, co znacząco przyspiesza generowanie i eliminuje ucinanie tokenów. Przy braku pamięci VRAM paczka jest automatycznie dzielona na pół.

   Postęp widać nad każdym zapytaniem w terminalu (najpierw etap OCR, a potem Opisy):

   ```
   [OCR: 2 100 / 5 302 · 40% · 21,4 zdj/s · ~2 min]
   [Opisy: 1 240 / 5 302 · 23% · 0,9 zdj/s · ~1 h 15 min]
   ```

   Faza 2 zapisuje postęp na bieżąco w bazie SQLite, więc po przerwaniu program wznawia się dokładnie od miejsca, w którym skończył.

Zapytanie przeszukuje trzy źródła naraz (obraz, opisy, tekst z OCR) i łączy wyniki. Przy każdym wyniku widać, skąd się wziął:

```
[1] 0.87  F:/POZOSTALOSCI/2019/IMG_0412.jpg
    obraz: wycinek prawy dół · opis: „Czerwony rower oparty o drewniany płot…”
[2] 0.81  F:/POZOSTALOSCI/2021/paragon.jpg
    OCR: „…ROWER MIEJSKI…”
```

Kolejne uruchomienia przetwarzają tylko nowe i zmienione pliki, a usunięte z dysku znikają z bazy.

## 🧠 Profile modeli

Program sam wybiera profil według pamięci karty graficznej (można go wymusić kluczem `PROFIL` w `config.json`):

| Profil | VRAM | Wektory obrazu | Opisy scen | Silnik OCR |
|---|---|---|---|---|
| `max` | ≥ 16 GB | Qwen3-VL-Embedding-2B | Qwen3-VL-4B | Windows OCR (lub model) |
| `high` | 10–16 GB (np. RTX 5070) | Qwen3-VL-Embedding-2B | Qwen3-VL-4B w 4-bit* | Windows OCR (lub model) |
| `mid` | 6–10 GB | Qwen3-VL-Embedding-2B | Qwen3-VL-2B w 4-bit* | Windows OCR (lub model) |
| `lite` | < 6 GB lub CPU | jina-clip-v2 | – (sam OCR) | Windows OCR (fallback: EasyOCR) |

\* Gdy 4-bit nie zadziała (brak `bitsandbytes`), program sam przełączy się na Qwen3-VL-2B w pełnej precyzji (bfloat16).
Na Windowsie silnik OCR korzysta z wbudowanego API systemowego (szybko i bez pobierania modeli). Gdy Windows OCR jest niedostępny (np. na Linuksie), tekst przepisuje model opisów (lub EasyOCR w profilu `lite`).

Modele pobierają się z Hugging Face przy pierwszym uruchomieniu (kilka GB).

## 💾 Wersja dla Windows (bez instalowania Pythona)

1. Pobierz `FileSearcher-windows-x64.zip` z [najnowszego wydania](https://github.com/c-airr/FileSearcher/releases/latest) i rozpakuj w dowolnym miejscu (potrzeba ok. 10 GB wolnego miejsca na biblioteki i modele).
2. Uruchom `FileSearcher.exe`. Przy pierwszym starcie program:
   - wykrywa kartę graficzną i instaluje pasującego PyTorcha (ok. 3 GB, jednorazowo),
   - pyta o folder ze zdjęciami i zapisuje go w `config.json`,
   - pobiera modele do folderu `modele` obok programu.
3. Kolejne uruchomienia startują od razu.

| Karta | Co się instaluje |
|---|---|
| NVIDIA (np. RTX 5070) | PyTorch z CUDA 12.8 |
| Radeon RX 7700/7800/7900, RX 9060/9070, AI PRO R9700 | PyTorch ROCm od AMD (wymaga sterownika Adrenalin 26.2.2 lub nowszego) |
| starszy Radeon (np. RX 6000, RX 7600), Intel, brak karty | PyTorch na CPU – profil `lite` |

Jeśli instalacja wersji z GPU się nie uda, program sam zainstaluje wersję na CPU. Na Radeonach nie ma kwantyzacji 4-bit, więc opisy robi mniejszy model Qwen3-VL-2B.

## 🚀 Instalacja ze źródeł

Wymagany Python 3.10 lub nowszy.

1. **Zainstaluj PyTorch z obsługą GPU.** Dla kart RTX 50xx (np. RTX 5070) potrzebna jest wersja z CUDA 12.8:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   Sprawdzenie: `python -c "import torch; print(torch.cuda.get_arch_list())"` powinno pokazać `sm_120`. Jeśli wersja się nie zgadza, FileSearcher przy starcie powie, co zrobić.

2. **Zainstaluj resztę pakietów:**
   ```bash
   pip install -r requirements.txt
   ```
   Na Windowsie biblioteki systemowego OCR (`winrt-*`) instalują się automatycznie z `requirements.txt`.
   Dla profilu `lite` na systemach bez Windows OCR (np. Linux) opcjonalnie: `pip install easyocr`.

3. **Skonfiguruj `config.json`:**

   | Klucz | Domyślnie | Znaczenie |
   |---|---|---|
   | `FOLDER_ZE_ZDJECIAMI` | – | folder ze zdjęciami (wymagany) |
   | `BAZA_DIR` | `./filesearcher_db` | gdzie trzymać bazę |
   | `ROZSZERZENIA` | `.jpg .jpeg .png .webp .bmp` | obsługiwane pliki |
   | `IO_WORKERS` | 4 | wątki wczytujące zdjęcia (HDD: 4, SSD: 8–16) |
   | `BATCH_SIZE` | 16 | ile obrazów naraz idzie do modelu |
   | `QUEUE_MAXSIZE` | 8 | ile batchy czeka w pamięci |
   | `PROFIL` | `auto` | `auto` / `max` / `high` / `mid` / `lite` |
   | `OPISY` | `true` | czy liczyć opisy i OCR w tle |
   | `WYNIKI` | 5 | liczba wyników na zapytanie |

4. **Uruchom:**
   ```bash
   python -m filesearcher
   ```
   W okienku zapytań: wpisz opis, `s` pokazuje postęp drugiej fazy, `q` (albo pusta linia) kończy program.

## ℹ️ Uwagi

- Stara baza `chroma_db` z poprzedniej wersji nie jest już używana – możesz ją usunąć.
- Błędy drugiej fazy (np. uszkodzone pliki) trafiają do `filesearcher_db/filesearcher.log`.
- Zmiana modelu wektorowego (np. przejście z `lite` na `high`) wymaga przebudowania bazy – program zapyta o to przy starcie.

## 🧪 Testy

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Testy używają sztucznych modeli, więc nie potrzebują GPU ani internetu.

---

<sub>*projekt stworzony przy użyciu AI (claude opus 4.7, gemini 3.5 flash, gemini 3.1 pro (antigravity))*</sub>
