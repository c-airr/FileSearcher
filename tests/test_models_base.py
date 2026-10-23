import numpy as np
from PIL import Image

from filesearcher.models.base import Description, parse_description
from filesearcher.models.fake import FakeDescriber, FakeEmbedder


def test_parse_valid_json():
    assert parse_description('{"opis": "pies", "tekst": "STOP"}') == Description("pies", "STOP")


def test_parse_fenced_json():
    raw = 'Oto wynik:\n```json\n{"opis": "kot na kanapie", "tekst": ""}\n```'
    assert parse_description(raw) == Description("kot na kanapie", "")


def test_parse_invalid_json_goes_to_opis():
    assert parse_description("po prostu tekst") == Description("po prostu tekst", "")


def test_parse_missing_tekst():
    assert parse_description('{"opis": "las"}') == Description("las", "")


def test_fake_embedder_matches_color_word():
    emb = FakeEmbedder()
    imgs = emb.embed_images([Image.new("RGB", (8, 8), (220, 20, 20)), Image.new("RGB", (8, 8), (20, 20, 220))])
    q = emb.embed_texts(["czerwony"])[0]
    sims = imgs @ q
    assert sims[0] > sims[1]
    assert np.allclose(np.linalg.norm(imgs, axis=1), 1.0)


def test_fake_describer_default_uses_color():
    d = FakeDescriber().describe(Image.new("RGB", (8, 8), (20, 200, 20)))
    assert "zielony" in d.opis
