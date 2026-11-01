import time

import pytest

from filesearcher.describer import BackgroundDescriber, ProgressState, format_progress, load_for_description
from filesearcher.gpu import GpuArbiter
from filesearcher.models.base import Description
from filesearcher.models.fake import FakeDescriber, FakeEmbedder, FakeReader
from filesearcher.store import NewImage, Store
from filesearcher.vectors import TEXT_CODE, VectorIndex


def test_format_progress_full():
    s = ProgressState(done=1240, total=10000, failed=0, rate=0.9, eta_s=9720, finished=False)
    assert format_progress(s) == "[Opisy i OCR: 1 240 / 10 000 · 12% · 0,9 zdj/s · ~2 h 42 min]"


def test_format_progress_without_rate():
    s = ProgressState(0, 10, 0, None, None, False)
    assert format_progress(s) == "[Opisy i OCR: 0 / 10 · 0% · szacuję…]"


def test_format_progress_with_errors():
    s = ProgressState(5, 10, 3, 1.0, 5, False)
    assert format_progress(s).endswith(" · błędy: 3]")


def test_format_progress_finished():
    assert format_progress(ProgressState(10000, 10000, 0, None, None, True)) == "[Opisy i OCR: gotowe, 10 000 zdjęć]"


@pytest.mark.parametrize("eta,text", [(240, "~4 min"), (30, "<1 min")])
def test_format_eta(eta, text):
    assert text in format_progress(ProgressState(1, 10, 0, 1.0, eta, False))


@pytest.fixture
def env(tmp_path, make_image):
    store = Store(tmp_path / "db" / "index.sqlite")
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    paths = [make_image(tmp_path / f"{i}.jpg", color=["red", "green", "blue"][i % 3]) for i in range(5)]
    store.add_images([NewImage(str(p), 1.0, 1, [("full", emb.embed_images([load_for_description(str(p))])[0])]) for p in paths])

    def make(describer, reader=None, batch=2):
        return BackgroundDescriber(
            store, index, emb, describer, GpuArbiter(), tmp_path / "db" / "filesearcher.log", reader=reader, batch=batch
        )

    return store, index, paths, make, tmp_path


def wait(bg, timeout=10):
    end = time.monotonic() + timeout
    while not bg.status().finished and time.monotonic() < end:
        time.sleep(0.01)


def test_describes_all(env):
    store, index, _, make, _ = env
    bg = make(FakeDescriber())
    bg.start()
    wait(bg)
    assert store.counts().done == 5 and bg.status().finished
    assert sum(1 for kind in store.all_vectors()[1] if kind == "text") == 5
    assert len(store.fts_search("zielony", 10)) == 2
    assert len(index.search(FakeEmbedder().embed_texts(["czerwony obraz"])[0], "text", 10)) == 5


def test_resume_after_stop(env):
    store, _, _, make, _ = env
    bg = make(FakeDescriber(delay=0.05))
    bg.start()
    end = time.monotonic() + 10
    while store.counts().done < 2 and time.monotonic() < end:
        time.sleep(0.005)
    bg.stop(timeout=5)
    assert 2 <= store.counts().done < 5
    bg2 = make(FakeDescriber())
    bg2.start()
    wait(bg2)
    assert store.counts().done == 5


def test_missing_file_marked_failed(env):
    store, _, paths, make, tmp_path = env
    paths[0].unlink()
    bg = make(FakeDescriber())
    bg.start()
    wait(bg)
    assert store.counts().failed == 1 and store.counts().done == 4
    assert bg.status().failed == 1
    assert "0.jpg" in (tmp_path / "db" / "filesearcher.log").read_text(encoding="utf-8")


def test_empty_description_has_no_text_vector(env):
    store, _, _, make, _ = env
    bg = make(FakeDescriber(fn=lambda img: Description("", "")))
    bg.start()
    wait(bg)
    assert store.counts().done == 5
    assert "text" not in store.all_vectors()[1]


def test_rate_and_eta_reported(env):
    store, _, _, make, _ = env
    bg = make(FakeDescriber(delay=0.05))
    bg.start()
    end = time.monotonic() + 10
    while store.counts().done < 3 and time.monotonic() < end:
        time.sleep(0.005)
    s = bg.status()
    bg.stop(timeout=5)
    assert s.rate is not None and s.rate > 0 and s.eta_s is not None


def test_load_for_description_limits_pixels(tmp_path, make_image):
    img = load_for_description(str(make_image(tmp_path / "x.jpg", size=(4000, 3000))))
    assert img.width * img.height <= 1_000_000


def test_describes_in_batches(env):
    store, _, _, make, _ = env
    d = FakeDescriber()
    bg = make(d, batch=2)
    bg.start()
    wait(bg)
    assert d.batch_sizes == [2, 2, 1] and store.counts().done == 5


def test_ocr_stage_then_descriptions_keep_ocr_text(env):
    store, index, _, make, _ = env
    bg = make(FakeDescriber(), reader=FakeReader())
    bg.start()
    wait(bg)
    assert store.ocr_counts() == (5, 5) and store.counts().done == 5
    assert bg.status().stage == "Opisy"
    # tekst z OCR zostaje po opisie i jest w FTS
    assert len(store.fts_search("ZIELONY", 10)) == 2
    assert all(store.get_text(i)[1] for i in range(1, 6))
    # jeden aktywny wektor tekstu na zdjecie (stary z OCR usuniety z pamieci)
    assert int((index._codes[: len(index)] == TEXT_CODE).sum()) == 5
    assert sum(1 for kind in store.all_vectors()[1] if kind == "text") == 5


def test_ocr_only_without_describer(env):
    store, _, _, make, _ = env
    bg = make(None, reader=FakeReader())
    bg.start()
    wait(bg)
    s = bg.status()
    assert s.finished and s.stage == "OCR" and store.ocr_counts() == (5, 5)
    assert format_progress(s) == "[OCR: gotowe, 5 zdjęć]"


def test_ocr_error_does_not_block(env):
    store, _, paths, make, _ = env
    paths[0].unlink()
    bg = make(FakeDescriber(), reader=FakeReader())
    bg.start()
    wait(bg)
    assert store.ocr_counts() == (5, 5) and store.pending_ocr() == []
    assert store.counts().failed == 1 and store.counts().done == 4


def test_chunk_error_does_not_kill_thread(env, monkeypatch):
    store, _, _, make, tmp_path = env
    real = store.save_description
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("zapis padl")
        return real(*a, **kw)

    monkeypatch.setattr(store, "save_description", flaky)
    bg = make(FakeDescriber(), batch=2)
    bg.start()
    wait(bg)
    assert bg.status().finished and store.counts().done >= 3
    assert "zapis padl" in (tmp_path / "db" / "filesearcher.log").read_text(encoding="utf-8")


def test_oom_splits_batch(env):
    store, _, _, make, _ = env

    class Picky(FakeDescriber):
        def describe_batch(self, images):
            if len(images) > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            return super().describe_batch(images)

    bg = make(Picky(), batch=4)
    bg.start()
    wait(bg)
    assert store.counts().done == 5 and store.counts().failed == 0
