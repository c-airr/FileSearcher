import os
from pathlib import Path
from PIL import Image
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from tqdm import tqdm

FOLDER_ZE_ZDJECIAMI = r"C:\Users\cairr\Documents\ShareX\Screenshots"
BAZA_DIR = "./chroma_db"
ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def zainicjalizuj_baze():
    embedding_function = OpenCLIPEmbeddingFunction()
    client = chromadb.PersistentClient(path=BAZA_DIR)
    collection = client.get_or_create_collection(
        name="image_search",
        embedding_function=embedding_function
    )
    return collection, embedding_function

def indeksuj_folder(collection, embedding_function, folder_path):
    path = Path(folder_path)
    if not path.exists():
        print(f"Błąd: Folder '{folder_path}' nie istnieje!")
        return False

    istniejace_pliki = set(collection.get()["ids"])
    print("Przeszukuję folder w poszukiwaniu zdjęć...")
    wszystkie_pliki = [p for p in path.rglob("*") if p.suffix.lower() in ROZSZERZENIA]
    pliki_do_dodania = [p for p in wszystkie_pliki if str(p.resolve()) not in istniejace_pliki]

    if not pliki_do_dodania:
        print(f"Wszystkie zdjęcia ({len(wszystkie_pliki)}) są już w bazie.")
        return True

    print(f"Znaleziono {len(pliki_do_dodania)} nowych zdjęć — generuję embeddingi (to zajmie chwilę)...")

    BATCH_SIZE = 32
    batch_ids, batch_images, batch_meta = [], [], []

    for plik in tqdm(pliki_do_dodania):
        try:
            sciezka_str = str(plik.resolve())
            with Image.open(plik) as img:
                img.verify()
            # Otwieramy ponownie po verify (verify zamyka/psuje obiekt)
            with Image.open(plik) as img:
                img = img.convert("RGB")
                img = img.resize((224, 224))
                arr = np.array(img)

            batch_ids.append(sciezka_str)
            batch_images.append(arr)
            batch_meta.append({"nazwa_pliku": plik.name})

            if len(batch_ids) >= BATCH_SIZE:
                collection.add(
                    ids=batch_ids,
                    images=batch_images,
                    metadatas=batch_meta
                )
                batch_ids, batch_images, batch_meta = [], [], []

        except Exception as e:
            print(f"\nPominięto {plik.name}: {e}")

    # Ostatni niepełny batch
    if batch_ids:
        collection.add(
            ids=batch_ids,
            images=batch_images,
            metadatas=batch_meta
        )

    print("Indeksowanie zakończone!")
    return True

def wyszukaj_zdjecie(collection, zapytanie, limit=3):
    print(f"\nSzukam: '{zapytanie}'...")
    wyniki = collection.query(
        query_texts=[zapytanie],
        n_results=limit
    )

    if not wyniki["ids"] or not wyniki["ids"][0]:
        print("Nie znaleziono pasujących zdjęć.")
        return

    print("\nNajbardziej pasujące zdjęcia:")
    for i in range(len(wyniki["ids"][0])):
        sciezka = wyniki["ids"][0][i]
        odleglosc = wyniki["distances"][0][i]
        print(f"[{i+1}] Trafienie: {1 - odleglosc:.2%}")
        print(f"    Ścieżka: {sciezka}\n")

if __name__ == "__main__":
    baza, ef = zainicjalizuj_baze()
    if indeksuj_folder(baza, ef, FOLDER_ZE_ZDJECIAMI):
        while True:
            zapytanie = input("Wpisz opis zdjęcia (lub 'q' aby wyjść): ").strip()
            if zapytanie.lower() == 'q' or not zapytanie:
                break
            wyszukaj_zdjecie(baza, zapytanie, limit=3)