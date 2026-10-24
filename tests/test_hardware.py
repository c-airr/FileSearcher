import pytest

from filesearcher.hardware import HardwareInfo, pick_profile_name, resolve_profile, torch_problem

GPU = HardwareInfo(device="cuda", name="RTX 5070", vram_gb=11.9, capability=(12, 0), arch_list=["sm_90", "sm_120"])
CPU = HardwareInfo(device="cpu", name=None, vram_gb=None, capability=None, arch_list=[])


@pytest.mark.parametrize(
    "vram,expected",
    [(None, "lite"), (4.0, "lite"), (7.9, "mid"), (11.9, "high"), (15.9, "max"), (24.0, "max")],
)
def test_pick_profile_name(vram, expected):
    assert pick_profile_name(vram) == expected


def test_blackwell_without_sm120_reports_cu128():
    hw = HardwareInfo("cuda", "RTX 5070", 11.9, (12, 0), ["sm_80", "sm_90"])
    assert "cu128" in torch_problem(hw)


def test_blackwell_with_sm120_ok():
    assert torch_problem(GPU) is None


def test_cpu_has_no_torch_problem():
    assert torch_problem(CPU) is None


def test_auto_on_rtx5070_is_high_with_4bit():
    p = resolve_profile("auto", GPU)
    assert p.name == "high"
    assert p.embedder_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert p.describer_id == "Qwen/Qwen3-VL-4B-Instruct" and p.describer_4bit
    assert p.fallback_describer_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert p.device == "cuda"


def test_forced_lite_on_gpu():
    p = resolve_profile("lite", GPU)
    assert p.name == "lite" and p.embedder_id == "jinaai/jina-clip-v2" and p.describer_id == "easyocr"


def test_forced_profile_on_cpu_uses_cpu():
    p = resolve_profile("high", CPU)
    assert p.name == "high" and p.device == "cpu"
