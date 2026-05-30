# 🔍 FileSearcher (AI Image Search)

*[🇬🇧 English version below](#-english-version)*

**FileSearcher** to program, który indeksuje zdjęcia na Twoim dysku przy użyciu sztucznej inteligencji (OpenCLIP + ChromaDB), pozwalając na późniejsze wyszukiwanie obrazów za pomocą naturalnych opisów tekstowych (zrozumiałych dla człowieka).

## ✨ Główne funkcje
- Szybkie indeksowanie tysięcy obrazów przy użyciu wielowątkowości.
- Przetwarzanie i kodowanie zdjęć w lokalnej bazie danych wektorowych (ChromaDB).
- Semantic search: wyszukiwanie zdjęć na podstawie opisów (np. "pies na plaży", "czerwony samochód").

## 🚀 Instalacja i użycie

1. **Zainstaluj wymagane pakiety:**
   W terminalu lub wierszu poleceń wpisz:
   ```bash
   pip install -r requirements.txt
   ```

2. **Skonfiguruj ustawienia:**
   Otwórz plik `config.json` i zdefiniuj ścieżkę folderu, który skrypt ma przeszukać (`FOLDER_ZE_ZDJECIAMI`), oraz dostosuj resztę parametrów.

3. **Uruchom program:**
   Rozpocznij indeksowanie obrazów wpisując:
   ```bash
   python search.py
   ```
   Przy pierwszym uruchomieniu poczekaj na zakończenie analizy zdjęć przez skrypt.

4. **Szukaj i baw się dobrze!**
   Gdy baza zostanie zbudowana, w wierszu poleceń pojawi się miejsce na Twoje zapytania. Wpisz to, czego szukasz, i ciesz się wynikami!

---

# 🇬🇧 English Version

**FileSearcher** is an application that indexes photos on your drive using AI (OpenCLIP + ChromaDB), allowing you to search for images later using natural text descriptions.

## ✨ Key Features
- Fast indexing of thousands of images using multi-threading.
- Processing and encoding photos directly into a local vector database (ChromaDB).
- Semantic search: find photos based on text descriptions (e.g., "dog on the beach", "red car").

## 🚀 Installation & Usage

1. **Install required packages:**
   In your terminal or command prompt, type:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure settings:**
   Open the `config.json` file and define the path to the folder the script should scan (`FOLDER_ZE_ZDJECIAMI`), along with any other parameters according to your preferences.

3. **Run the program:**
   Start the image indexing process by typing:
   ```bash
   python search.py
   ```
   On the first run, wait for the script to finish analyzing your photos.

4. **Search and have fun!**
   Once the database is built, a prompt will appear in the command line. Type what you are looking for and enjoy the results!
