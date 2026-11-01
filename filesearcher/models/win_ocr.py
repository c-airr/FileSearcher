"""OCR wbudowany w Windows (Windows.Media.Ocr): ~20 zdj/s na CPU, bez pobierania modeli."""
import asyncio

from PIL import Image

# Windows OCR odrzuca wieksze obrazy (OcrEngine.MaxImageDimension)
MAX_SIDE = 10_000


class WindowsOCR:
    model_id = "windows-ocr"

    def __init__(self, languages=("pl", "en-US")):
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter

        self._engine = None
        for tag in languages:
            if OcrEngine.is_language_supported(Language(tag)):
                self._engine = OcrEngine.try_create_from_language(Language(tag))
                self.language = tag
                break
        if self._engine is None:
            raise RuntimeError("Windows OCR: brak zainstalowanego pakietu jezyka (pl/en) w Ustawieniach → Język")
        self._format = BitmapPixelFormat.RGBA8
        self._bitmap = SoftwareBitmap
        self._writer = DataWriter
        self._loop = asyncio.new_event_loop()

    def read(self, image: Image.Image) -> str:
        img = image.convert("RGBA")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        w = self._writer()
        w.write_bytes(img.tobytes())
        bmp = self._bitmap.create_copy_from_buffer(w.detach_buffer(), self._format, img.width, img.height)
        result = self._loop.run_until_complete(self._engine.recognize_async(bmp))
        return "\n".join(line.text for line in result.lines).strip()
