import pytest

from filesearcher.config import Config
from filesearcher.gpu import GpuArbiter
from filesearcher.indexer import run_phase1
from filesearcher.models.base import Description
from filesearcher.models.fake import FakeEmbedder
from filesearcher.search import fold, rrf, search
from filesearcher.store import Store
from filesearcher.vectors import VectorIndex


def test_rrf_rewards_agreement():
    assert [i for i, _ in rrf([[1, 2], [2, 3]])][:1] == [2]


def test_fold_reexported():
    assert fold("Łąka") == "laka"


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "zdjecia"
    store = Store(tmp_path / "db.sqlite")
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    arb = GpuArbiter()

    def index_all():
        run_phase1(store, index, emb, Config(folder=folder, batch_size=4), arb, show_progress=False)
        return {p: i for i, p in store.pending_descriptions(True)}

    def find(query, limit=5):
        return search(query, store, index, emb, arb, limit)

    return folder, store, emb, index, index_all, find


def describe(store, index, emb, image_id, opis, tekst):
    vec = emb.embed_texts([f"{opis}\n{tekst}"])[0]
    store.save_description(image_id, Description(opis, tekst), vec)
    index.add([image_id], ["text"], vec[None])


def test_ocr_hit_labelled(env, make_image):
    folder, store, emb, index, index_all, find = env
    a = make_image(folder / "paragon.jpg", color="white")
    make_image(folder / "inne.jpg", color="blue")
    ids = index_all()
    describe(store, index, emb, ids[str(a.resolve())], "paragon ze sklepu", "ROWER MIEJSKI 12,99")
    hits = find("rower")
    assert hits[0].path == str(a.resolve())
    assert "ROWER MIEJSKI" in hits[0].sources["OCR"]


def test_tile_hit_labelled(env, make_image):
    folder, _, _, _, index_all, find = env
    make_image(folder / "rog.jpg", size=(1000, 1000), color="white", corner="red")
    make_image(folder / "zielony.jpg", color="green")
    index_all()
    hits = find("czerwony")
    assert hits[0].path.endswith("rog.jpg")
    assert hits[0].sources["obraz"] == "wycinek prawy dół"
    assert hits[0].score is not None


def test_undescribed_images_still_found(env, make_image):
    folder, _, _, _, index_all, find = env
    make_image(folder / "a.jpg", color="blue")
    index_all()
    assert [h.path.endswith("a.jpg") for h in find("niebieski")] == [True]


def test_limit_respected(env, make_image):
    folder, _, _, _, index_all, find = env
    for i in range(8):
        make_image(folder / f"{i}.jpg")
    index_all()
    assert len(find("czerwony", limit=3)) == 3


def test_empty_database(env):
    *_, find = env
    assert find("cokolwiek") == []


def test_opis_hit_without_ocr_word(env, make_image):
    folder, store, emb, index, index_all, find = env
    a = make_image(folder / "a.jpg", color="white")
    ids = index_all()
    describe(store, index, emb, ids[str(a.resolve())], "Żółty rower przy płocie", "")
    hit = find("zolty")[0]
    assert "OCR" not in hit.sources
    assert "Żółty rower" in hit.sources["opis"]
