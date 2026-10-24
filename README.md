# 🔍 FileSearcher (wyszukiwarka zdjęć AI)

**FileSearcher** indeksuje zdjęcia na Twoim dysku i pozwala szukać ich zwykłym opisem po polsku, np. „pies na plaży”, „paragon ze stacji benzynowej” albo „czerwony rower przy płocie”. Wszystko działa lokalnie, na Twoim komputerze.

## ✨ Jak to działa

Indeksowanie ma dwie fazy:

1. **Szybka faza – wektory obrazu.** Każde zdjęcie (i dodatkowo jego 4 wycinki, żeby łapać drobne obiekty) trafia do wielojęzycznego modelu wektorowego. Po tej fazie można już szukać.
2. **Druga faza w tle – opisy i OCR.** Model językowy z obrazem pisze szczegółowy opis sceny i przepisuje cały widoczny tekst (szyldy, dokumenty, zrzuty ekranu). Postęp widać nad każdym zapytaniem:

   ```
   [Opisy i OCR: 1 240 / 10 000 · 12% · 0,9 zdj/s · ~2 h 42 min]
   ```

   Faza 2 zapisuje każde zdjęcie od razu, więc po przerwaniu wznawia się od miejsca, w którym skończyła.

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

| Profil | VRAM | Wektory obrazu | Opisy i OCR |
|---|---|---|---|
| `max` | ≥ 16 GB | Qwen3-VL-Embedding-2B | Qwen3-VL-4B |
| `high` | 10–16 GB (np. RTX 5070) | Qwen3-VL-Embedding-2B | Qwen3-VL-4B w 4-bit* |
| `mid` | 6–10 GB | Qwen3-VL-Embedding-2B | Qwen3-VL-2B w 4-bit* |
| `lite` | < 6 GB lub CPU | jina-clip-v2 | sam OCR (EasyOCR, opcjonalnie) |

\* Gdy 4-bit nie zadziała (brak `bitsandbytes`), program sam przełączy się na Qwen3-VL-2B w pełnej precyzji.

Modele pobierają się z Hugging Face przy pierwszym uruchomieniu (kilka GB).

## 🚀 Instalacja

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
   Dla profilu `lite` z OCR dodatkowo: `pip install easyocr`.

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
