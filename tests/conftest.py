import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLORS = {
    "red": (220, 20, 20),
    "green": (20, 200, 20),
    "blue": (20, 20, 220),
    "white": (250, 250, 250),
}


@pytest.fixture
def make_image():
    """Zapisuje JPEG o danym rozmiarze i kolorze; opcjonalnie kwadrat w prawym dolnym rogu."""

    def _make(path, size=(64, 64), color="red", corner=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", size, COLORS[color])
        if corner:
            w, h = size
            img.paste(COLORS[corner], (w // 2 + 10, h // 2 + 10, w, h))
        img.save(path, "JPEG", quality=95)
        return path

    return _make
