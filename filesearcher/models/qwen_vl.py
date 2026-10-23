"""Opisy scen i OCR przez Qwen3-VL (transformers)."""
from .base import Description, parse_description

MAX_NEW_TOKENS = 384
DESCRIBE_PROMPT = (
    "Opisz to zdjęcie po polsku i odczytaj widoczny na nim tekst. "
    'Odpowiedz wyłącznie obiektem JSON: {"opis": "...", "tekst": "..."}. '
    "W polu opis podaj szczegółowy opis sceny w 2–4 zdaniach: kto lub co jest na zdjęciu, "
    "także drobne obiekty w tle, kolory, miejsce i porę dnia. "
    "W polu tekst przepisz dosłownie cały widoczny tekst (napisy, szyldy, dokumenty, ekrany); "
    "jeśli tekstu nie ma, wpisz pusty ciąg."
)


class QwenVLDescriber:
    def __init__(self, model_id: str, device: str, four_bit: bool = False):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        kwargs = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32}
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            kwargs["device_map"] = "cuda"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if not four_bit:
            self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self._torch = torch

    def describe(self, image) -> Description:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": DESCRIBE_PROMPT}]}
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return parse_description(raw)
