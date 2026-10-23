"""Wykrywanie GPU i dobór profilu modeli."""
from dataclasses import dataclass, replace

EMBED_QWEN = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_JINA = "jinaai/jina-clip-v2"
VLM_4B = "Qwen/Qwen3-VL-4B-Instruct"
VLM_2B = "Qwen/Qwen3-VL-2B-Instruct"
OCR_ONLY = "easyocr"

# Progi w GiB z marginesem: karta „16 GB” raportuje ok. 15,9 GiB.
THRESHOLDS = (("max", 15.5), ("high", 9.5), ("mid", 5.5))


@dataclass(frozen=True)
class HardwareInfo:
    device: str
    name: str | None
    vram_gb: float | None
    capability: tuple[int, int] | None
    arch_list: list[str]


@dataclass(frozen=True)
class Profile:
    name: str
    embedder_id: str
    describer_id: str | None
    describer_4bit: bool
    fallback_describer_id: str | None
    device: str = "cpu"


PROFILES = {
    "max": Profile("max", EMBED_QWEN, VLM_4B, False, None),
    "high": Profile("high", EMBED_QWEN, VLM_4B, True, VLM_2B),
    "mid": Profile("mid", EMBED_QWEN, VLM_2B, True, VLM_2B),
    "lite": Profile("lite", EMBED_JINA, OCR_ONLY, False, None),
}


def detect() -> HardwareInfo:
    try:
        import torch
    except ImportError:
        return HardwareInfo("cpu", None, None, None, [])
    if not torch.cuda.is_available():
        return HardwareInfo("cpu", None, None, None, [])
    props = torch.cuda.get_device_properties(0)
    return HardwareInfo(
        device="cuda",
        name=props.name,
        vram_gb=props.total_memory / 1024**3,
        capability=(props.major, props.minor),
        arch_list=list(torch.cuda.get_arch_list()),
    )


def pick_profile_name(vram_gb: float | None) -> str:
    if vram_gb is not None:
        for name, minimum in THRESHOLDS:
            if vram_gb >= minimum:
                return name
    return "lite"


def torch_problem(hw: HardwareInfo) -> str | None:
    """Komunikat, gdy zainstalowany PyTorch nie obsługuje tej karty."""
    if hw.device != "cuda" or hw.capability is None:
        return None
    arch = f"sm_{hw.capability[0]}{hw.capability[1]}"
    if hw.capability[0] >= 12 and arch not in hw.arch_list:
        return (
            f"Zainstalowany PyTorch nie obsługuje karty {hw.name} ({arch}).\n"
            "Zainstaluj wersję z CUDA 12.8:\n"
            "  pip uninstall -y torch torchvision\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
    return None


def resolve_profile(config_profil: str, hw: HardwareInfo) -> Profile:
    name = pick_profile_name(hw.vram_gb) if config_profil == "auto" else config_profil
    profile = replace(PROFILES[name], device=hw.device)
    if hw.device != "cuda" and profile.describer_4bit:
        # 4-bit (bitsandbytes) wymaga CUDA – na CPU od razu model zapasowy.
        profile = replace(profile, describer_id=profile.fallback_describer_id, describer_4bit=False)
    return profile
