import os
from pathlib import Path
from PIL import Image
import numpy as np
import chromadb
from transformers import pipeline
import logging
from tqdm import tqdm
import threading
import queue
import json

# ──────────────────────────────────────────────
#  KONFIGURACJA
# ──────────────────────────────────────────────
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

FOLDER_ZE_ZDJECIAMI = config["FOLDER_ZE_ZDJECIAMI"]
BAZA_DIR             = config["BAZA_DIR"]
ROZSZERZENIA         = set(config["ROZSZERZENIA"])

IO_WORKERS    = config["IO_WORKERS"]     # wątki do wczytywania z dysku (HDD: 4, SSD: 8-16)
BATCH_SIZE    = config["BATCH_SIZE"]    # ile zdjęć trafia jednocześnie do modelu
QUEUE_MAXSIZE = config["QUEUE_MAXSIZE"]     # ile batchów czeka w kolejce (pilnuje RAM)

SENTINEL = None  # sygnał "koniec pracy" dla wątków

# ──────────────────────────────────────────────
#  INICJALIZACJA BAZY
# ──────────────────────────────────────────────
def zainicjalizuj_baze():
    logging.getLogger("transformers").setLevel(logging.ERROR)
    print("Ładowanie modeli (pipeline)...")
    pipe_image = pipeline("image-feature-extraction", model="openai/clip-vit-base-patch32")
    pipe_text = pipeline("feature-extraction", model="openai/clip-vit-base-patch32")
    
    client = chromadb.PersistentClient(path=BAZA_DIR)
    collection = client.get_or_create_collection(
        name="image_search"
    )
    return collection, pipe_image, pipe_text


# ──────────────────────────────────────────────
#  ETAP 1 – WCZYTYWANIE Z DYSKU (wiele wątków)
# ──────────────────────────────────────────────
def worker_wczytaj(pliki_queue: queue.Queue, obrazy_queue: queue.Queue, pbar):
    """
    Pobiera ścieżki z pliki_queue, wczytuje obraz i wrzuca do obrazy_queue.
    Działa jako jeden z wielu równoległych wątków I/O.
    """
    while True:
        plik = pliki_queue.get()
        if plik is SENTINEL:
            pliki_queue.task_done()
            break

        try:
            sciezka_str = str(plik.resolve())
            with Image.open(plik) as img:
                img = img.convert("RGB")
                img = img.resize((224, 224), Image.BILINEAR)
                arr = np.array(img)
            obrazy_queue.put((sciezka_str, arr, {"nazwa_pliku": plik.name}))
        except Exception as e:
            tqdm.write(f"Pominięto {plik.name}: {e}")
        finally:
            pbar.update(1)
            pliki_queue.task_done()


# ──────────────────────────────────────────────
#  ETAP 2 – BATCHER
# ──────────────────────────────────────────────
def worker_batcher(obrazy_queue: queue.Queue, batche_queue: queue.Queue,
                   io_workers: int):
    """
    Zbiera pojedyncze obrazy z obrazy_queue i pakuje je w batche BATCH_SIZE.
    """
    batch_ids, batch_images, batch_meta = [], [], []
    sentinels_seen = 0

    while True:
        item = obrazy_queue.get()

        if item is SENTINEL:
            sentinels_seen += 1
            if sentinels_seen == io_workers:
                if batch_ids:
                    batche_queue.put((batch_ids, batch_images, batch_meta))
                batche_queue.put(SENTINEL)
                break
            continue

        sciezka, arr, meta = item
        batch_ids.append(sciezka)
        batch_images.append(arr)
        batch_meta.append(meta)

        if len(batch_ids) >= BATCH_SIZE:
            batche_queue.put((batch_ids, batch_images, batch_meta))
            batch_ids, batch_images, batch_meta = [], [], []


# ──────────────────────────────────────────────
#  ETAP 3 – EMBEDDINGI + ZAPIS DO CHROMADB
# ──────────────────────────────────────────────
def zapisuj_batche(collection, batche_queue: queue.Queue, pipe_image):
    """
    Pobiera batche, generuje embeddingi przez HF Pipeline i zapisuje do ChromaDB.
    """
    while True:
        item = batche_queue.get()
        if item is SENTINEL:
            break
        ids, images, metas = item
        try:
            embeddings = []
            for arr in images:
                out = pipe_image(Image.fromarray(arr))
                embeddings.append(np.array(out).flatten().tolist())
            collection.add(ids=ids, embeddings=embeddings, metadatas=metas)
        except Exception as e:
            tqdm.write(f"Błąd zapisu batcha: {e}")


# ──────────────────────────────────────────────
#  GŁÓWNA FUNKCJA INDEKSOWANIA
# ──────────────────────────────────────────────
def indeksuj_folder(collection, folder_path, pipe_image, io_workers=IO_WORKERS):
    path = Path(folder_path)
    if not path.exists():
        print(f"Błąd: Folder '{folder_path}' nie istnieje!")
        return False

    istniejace_pliki = set(collection.get(include=[])["ids"])

    print("Przeszukuję folder w poszukiwaniu zdjęć...")
    wszystkie_pliki = [p for p in path.rglob("*") if p.suffix.lower() in ROZSZERZENIA]
    pliki_do_dodania = [p for p in wszystkie_pliki if str(p.resolve()) not in istniejace_pliki]

    if not pliki_do_dodania:
        print(f"Wszystkie zdjęcia ({len(wszystkie_pliki)}) są już w bazie.")
        return True

    print(f"Znaleziono {len(pliki_do_dodania)} nowych zdjęć.")
    print(f"Pipeline: {io_workers}x I/O → batcher ({BATCH_SIZE}/batch) → HF Pipeline → ChromaDB\n")

    pliki_queue  = queue.Queue()
    obrazy_queue = queue.Queue()
    batche_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    pbar = tqdm(total=len(pliki_do_dodania), desc="Wczytywanie", unit="zdjęć")

    # Uruchom wątki I/O
    io_threads = []
    for _ in range(io_workers):
        t = threading.Thread(
            target=worker_wczytaj,
            args=(pliki_queue, obrazy_queue, pbar),
            daemon=True
        )
        t.start()
        io_threads.append(t)

    # Uruchom batcher
    batcher_thread = threading.Thread(
        target=worker_batcher,
        args=(obrazy_queue, batche_queue, io_workers),
        daemon=True
    )
    batcher_thread.start()

    # Załaduj pliki do kolejki
    for plik in pliki_do_dodania:
        pliki_queue.put(plik)

    # Poczekaj aż I/O wątki skończą, następnie wyślij SENTINELe do batchera
    for _ in range(io_workers):
        pliki_queue.put(SENTINEL)
    for t in io_threads:
        t.join()
    for _ in range(io_workers):
        obrazy_queue.put(SENTINEL)

    pbar.close()
    print("\nGeneruję embeddingi i zapisuję do bazy...")

    # Główny wątek: HF Pipeline + ChromaDB
    zapisuj_batche(collection, batche_queue, pipe_image)
    batcher_thread.join()

    print("Indeksowanie zakończone!")
    return True


# ──────────────────────────────────────────────
#  WYSZUKIWANIE
# ──────────────────────────────────────────────
def wyszukaj_zdjecie(collection, zapytanie, pipe_text, limit=3):
    print(f"\nSzukam: '{zapytanie}'...")
    out = pipe_text(zapytanie)
    query_emb = np.array(out).flatten().tolist()
    
    wyniki = collection.query(
        query_embeddings=[query_emb],
        n_results=limit
    )

    if not wyniki["ids"] or not wyniki["ids"][0]:
        print("Nie znaleziono pasujących zdjęć.")
        return

    print("\nNajbardziej pasujące zdjęcia:")
    for i in range(len(wyniki["ids"][0])):
        sciezka   = wyniki["ids"][0][i]
        odleglosc = wyniki["distances"][0][i]
        print(f"[{i+1}] Trafienie: {1 - odleglosc:.2%}")
        print(f"    Ścieżka: {sciezka}\n")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    baza, pipe_img, pipe_txt = zainicjalizuj_baze()
    if indeksuj_folder(baza, FOLDER_ZE_ZDJECIAMI, pipe_img, io_workers=IO_WORKERS):
        while True:
            zapytanie = input("Wpisz opis zdjęcia (lub 'q' aby wyjść): ").strip()
            if zapytanie.lower() == "q" or not zapytanie:
                break
            wyszukaj_zdjecie(baza, zapytanie, pipe_txt, limit=3)
