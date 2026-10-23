import numpy as np

from filesearcher.vectors import VectorIndex


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_tile_beats_full_and_reports_kind():
    idx = VectorIndex(3)
    idx.add([1, 1, 2], ["full", "tile3", "full"], np.stack([unit(1, 1, 1), unit(1, 0, 0), unit(0, 1, 0)]))
    hits = idx.search(unit(1, 0, 0), "image", 2)
    assert hits[0][0] == 1 and hits[0][2] == "tile3"
    assert abs(hits[0][1] - 1.0) < 1e-5
    assert [h[0] for h in hits] == [1, 2]


def test_text_group_ignores_image_vectors():
    idx = VectorIndex(3)
    idx.add([1, 2], ["full", "text"], np.stack([unit(1, 0, 0), unit(0, 1, 0)]))
    assert [h[0] for h in idx.search(unit(1, 0, 0), "text", 5)] == [2]


def test_growth_keeps_data():
    idx = VectorIndex(3, capacity=2)
    for i in range(10):
        idx.add([i], ["full"], unit(1, i, 0)[None])
    assert len(idx) == 10
    assert idx.search(unit(1, 9, 0), "image", 1)[0][0] == 9


def test_one_result_per_image():
    idx = VectorIndex(3)
    kinds = ["full", "tile0", "tile1", "tile2", "tile3"]
    idx.add([1] * 5 + [2], kinds + ["full"], np.stack([unit(1, 0, 0)] * 5 + [unit(1, 0.1, 0)]))
    assert sorted(h[0] for h in idx.search(unit(1, 0, 0), "image", 2)) == [1, 2]


def test_empty_index():
    assert VectorIndex(3).search(unit(1, 0, 0), "image", 5) == []
