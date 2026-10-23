import numpy as np

from filesearcher.models.base import Description
from filesearcher.store import NewImage, Store, fold, fts_query, plan_sync


def vec(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def add(store, path, kinds=("full",)):
    return store.add_images([NewImage(path, 1.0, 10, [(k, vec(1, 2, 3)) for k in kinds])])[0]


def test_plan_sync_new_changed_removed():
    db = {"a": (1.0, 10), "b": (1.0, 10), "c": (1.0, 10)}
    found = {"a": (1.0, 10), "b": (2.0, 10), "d": (1.0, 5)}
    plan = plan_sync(db, found)
    assert sorted(plan.new) == ["b", "d"]
    assert sorted(plan.removed) == ["b", "c"]


def test_vectors_roundtrip(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    image_id = add(s, "x.jpg", kinds=("full", "tile0"))
    ids, kinds, mat = s.all_vectors()
    assert list(ids) == [image_id, image_id] and kinds == ["full", "tile0"]
    assert mat.dtype == np.float32 and mat.shape == (2, 3)
    assert np.allclose(mat[0], vec(1, 2, 3), atol=1e-3)


def test_remove_paths_removes_everything(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    image_id = add(s, "x.jpg")
    s.save_description(image_id, Description("żółty dom", "SKLEP"), vec(0, 1, 0))
    s.remove_paths(["x.jpg"])
    assert s.all_vectors()[0].size == 0
    assert s.fts_search("sklep", 10) == []
    assert s.counts().total == 0


def test_save_description_and_fold(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    image_id = add(s, "x.jpg")
    s.save_description(image_id, Description("Żółty rower w Łodzi", "Łódź Kaliska"), vec(0, 1, 0))
    assert s.counts().done == 1
    assert s.fts_search("zolty", 10) == [image_id]
    assert s.fts_search("łódź", 10) == [image_id]
    assert s.fts_search("lodz", 10) == [image_id]
    info = s.get_info([image_id])[image_id]
    assert info.opis == "Żółty rower w Łodzi" and info.tekst == "Łódź Kaliska" and info.path == "x.jpg"
    assert "text" in s.all_vectors()[1]


def test_fold():
    assert fold("Łódź ŻÓŁTY") == "lodz zolty"


def test_fts_query_punctuation_only():
    assert fts_query("?!  ") is None
    assert fts_query('rower "miejski"') == '"rower" OR "miejski"'


def test_fts_search_punctuation_returns_empty(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    assert s.fts_search("?!", 10) == []


def test_pending_and_failed(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    a, b = add(s, "a.jpg"), add(s, "b.jpg")
    s.mark_failed(b, "boom")
    assert s.pending_descriptions(include_failed=False) == [(a, "a.jpg")]
    assert s.pending_descriptions(include_failed=True) == [(a, "a.jpg"), (b, "b.jpg")]
    assert s.counts().failed == 1


def test_clear_all(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    add(s, "a.jpg")
    s.set_meta("embedder_id", "x")
    s.clear_all()
    assert s.counts().total == 0 and s.all_vectors()[0].size == 0


def test_reopen_persists(tmp_path):
    s = Store(tmp_path / "i.sqlite")
    add(s, "a.jpg")
    s.set_meta("dim", "3")
    s.close()
    s2 = Store(tmp_path / "i.sqlite")
    assert s2.get_meta("dim") == "3"
    assert s2.file_states() == {"a.jpg": (1.0, 10)}
