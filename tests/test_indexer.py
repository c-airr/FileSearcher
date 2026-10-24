import os

import pytest

from filesearcher.config import Config
from filesearcher.gpu import GpuArbiter
from filesearcher.indexer import load_image, make_tiles, run_phase1, scan_folder
from filesearcher.models.base import Description
from filesearcher.models.fake import FakeEmbedder
from filesearcher.store import Store
from filesearcher.vectors import VectorIndex


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "zdjecia"
    folder.mkdir()
    store = Store(tmp_path / "db" / "index.sqlite")
    emb = FakeEmbedder()

    def run(**overrides):
        cfg = Config(folder=folder, baza_dir=tmp_path / "db", batch_size=2, io_workers=2, **overrides)
        index = VectorIndex(emb.dim)
        return run_phase1(store, index, emb, cfg, GpuArbiter(), show_progress=False), index

    return folder, store, emb, run


def test_indexes_all_files(env, make_image):
    folder, store, _, run = env
    for i in range(7):
        make_image(folder / f"sub{i % 2}" / f"{i}.jpg")
    (folder / "notatka.txt").write_text("x")
    result, index = run()
    assert result.added == 7 and result.failed == []
    assert store.counts().total == 7
    assert len(index) == 7


def test_broken_file_skipped(env, make_image):
    folder, store, _, run = env
    make_image(folder / "ok.jpg")
    (folder / "zepsuty.jpg").write_bytes(b"to nie jest jpeg")
    result, _ = run()
    assert result.added == 1
    assert len(result.failed) == 1 and result.failed[0].endswith("zepsuty.jpg")


def test_tiles_only_for_large_images(env, make_image):
    folder, store, _, run = env
    make_image(folder / "maly.jpg", size=(300, 300))
    make_image(folder / "duzy.jpg", size=(1000, 800))
    run()
    _, kinds, _ = store.all_vectors()
    assert kinds.count("full") == 2
    assert sorted(k for k in kinds if k.startswith("tile")) == ["tile0", "tile1", "tile2", "tile3"]


def test_second_run_adds_nothing(env, make_image):
    folder, _, _, run = env
    make_image(folder / "a.jpg")
    run()
    result, _ = run()
    assert result.added == 0 and result.removed == 0


def test_changed_file_reindexed(env, make_image):
    folder, store, emb, run = env
    path = make_image(folder / "a.jpg", color="red")
    run()
    image_id = store.pending_descriptions(False)[0][0]
    store.save_description(image_id, Description("czerwony", ""), None)
    assert store.counts().done == 1
    make_image(path, color="blue", size=(80, 80))
    os.utime(path, (1_900_000_000, 1_900_000_000))
    result, _ = run()
    assert result.added == 1
    assert store.counts().done == 0 and store.counts().total == 1


def test_deleted_file_removed(env, make_image):
    folder, store, _, run = env
    make_image(folder / "a.jpg")
    path = make_image(folder / "b.jpg")
    run()
    path.unlink()
    result, _ = run()
    assert result.removed == 1 and store.counts().total == 1


def test_polish_path_preserved(env, make_image):
    folder, store, _, run = env
    path = make_image(folder / "zdjęcia" / "łódź nad morzem.jpg")
    run()
    assert list(store.file_states()) == [str(path.resolve())]


def test_small_queue_does_not_deadlock(env, make_image):
    folder, store, _, run = env
    for i in range(20):
        make_image(folder / f"{i}.jpg")
    result, _ = run(queue_maxsize=1)
    assert result.added == 20


def test_empty_folder(env):
    _, store, _, run = env
    result, index = run()
    assert result.added == 0 and len(index) == 0


def test_scan_folder_filters_extensions(tmp_path, make_image):
    make_image(tmp_path / "A.JPG")
    (tmp_path / "b.txt").write_text("x")
    found = scan_folder(tmp_path, frozenset({".jpg"}))
    assert [os.path.basename(p) for p in found] == ["A.JPG"]


def test_load_image_limits_size(make_image, tmp_path):
    img = load_image(str(make_image(tmp_path / "x.jpg", size=(3000, 1500))))
    assert max(img.size) == 1024


def test_make_tiles_order(tmp_path, make_image):
    img = load_image(str(make_image(tmp_path / "x.jpg", size=(1000, 1000), color="white", corner="red")))
    tiles = make_tiles(img)
    assert len(tiles) == 4
    r, g, b = tiles[3].resize((1, 1)).getpixel((0, 0))
    assert r > 200 and g < 150
