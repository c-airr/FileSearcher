"""Opisy scen (i OCR, gdy nie ma szybszego) przez Qwen3-VL (transformers), w paczkach."""
from .base import Description, parse_description

# Sam opis: 1–3 zdania. Z przepisywaniem tekstu odpowiedz jest kilka razy dluzsza.
MAX_NEW_TOKENS = 128
MAX_NEW_TOKENS_WITH_TEXT = 512
DESCRIBE_PROMPT = (
    "Opisz po polsku w 1–3 zdaniach, co jest na tym obrazie: jaki to program, strona albo miejsce, "
    "kto lub co jest widoczne (osoby, ich ubrania, zdjecia, wykresy, przedmioty, kolory). "
    "Nie przepisuj tekstu. Zacznij od slowa OPIS:"
)
DESCRIBE_WITH_TEXT_PROMPT = (
    "Odpowiedz po polsku w dokladnie tym formacie:\n"
    "OPIS: 1–3 zdania – jaki to program, strona albo miejsce, kto lub co jest widoczne "
    "(osoby, ich ubrania, zdjecia, wykresy, przedmioty, kolory).\n"
    "TEKST: doslownie przepisany caly widoczny tekst, linia po linii (pusto, jesli tekstu nie ma)."
)


class QwenVLDescriber:
    def __init__(self, model_id: str, device: str, four_bit: bool = False, with_text: bool = True):
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
        self.processor.tokenizer.padding_side = "left"  # wymagane przy generowaniu w paczce
        self.model_id = model_id
        self.with_text = with_text
        self._prompt = DESCRIBE_WITH_TEXT_PROMPT if with_text else DESCRIBE_PROMPT
        self._max_tokens = MAX_NEW_TOKENS_WITH_TEXT if with_text else MAX_NEW_TOKENS
        self._torch = torch

    def describe(self, image) -> Description:
        return self.describe_batch([image])[0]

    def describe_batch(self, images) -> list[Description]:
        conversations = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": self._prompt}]}]
            for img in images
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self._max_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        out = [parse_description(r) for r in raw]
        # Bez prosby o tekst model czasem i tak go dopisuje – OCR i tak przyjdzie z osobnego silnika.
        return out if self.with_text else [Description(d.opis, "") for d in out]
