import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launcher"))
import launcher  # noqa: E402


@pytest.mark.parametrize(
    "names,backend",
    [
        (["NVIDIA GeForce RTX 5070"], "cuda"),
        (["AMD Radeon RX 7900 XTX"], "rocm"),
        (["AMD Radeon RX 7800 XT"], "rocm"),
        (["AMD Radeon RX 9070 XT"], "rocm"),
        (["AMD Radeon RX 9060 XT"], "rocm"),
        (["AMD Radeon AI PRO R9700"], "rocm"),
        (["AMD Radeon RX 7600"], "cpu"),
        (["AMD Radeon RX 6800 XT"], "cpu"),
        (["Intel(R) UHD Graphics 770"], "cpu"),
        ([], "cpu"),
        (["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"], "cuda"),
    ],
)
def test_choose_backend(names, backend):
    assert launcher.choose_backend(names) == backend


def test_torch_commands():
    assert "https://download.pytorch.org/whl/cu128" in launcher.torch_install_commands("cuda")[0]
    assert "https://download.pytorch.org/whl/cpu" in launcher.torch_install_commands("cpu")[0]
    rocm = launcher.torch_install_commands("rocm")
    assert len(rocm) == 2
    assert any("rocm_sdk_core-7.2.1" in a for a in rocm[0])
    assert any("torch-2.9.1%2Brocm7.2.1-cp312" in a for a in rocm[1])


def test_needs_install(tmp_path):
    marker = tmp_path / "deps.json"
    assert launcher.needs_install(marker, "cuda", "abc")
    launcher.write_marker(marker, detected="cuda", installed="cuda", req_hash="abc")
    assert not launcher.needs_install(marker, "cuda", "abc")
    assert launcher.needs_install(marker, "cuda", "zmiana")
    assert launcher.needs_install(marker, "rocm", "abc")


def test_failed_gpu_install_not_retried_every_start(tmp_path):
    marker = tmp_path / "deps.json"
    launcher.write_marker(marker, detected="rocm", installed="cpu", req_hash="abc")
    assert not launcher.needs_install(marker, "rocm", "abc")


def make_root(tmp_path):
    (tmp_path / "python").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    return tmp_path


def test_install_falls_back_to_cpu(tmp_path):
    root = make_root(tmp_path)
    calls = []

    def run(cmd):
        calls.append(cmd)
        return 1 if any("cu128" in a for a in cmd) else 0

    installed = launcher.install_dependencies(root, "cuda", run=run, log=lambda m: None)
    assert installed == "cpu"
    assert any("whl/cpu" in a for cmd in calls for a in cmd)
    assert calls[-1][-2:] == ["-r", str(root / "app" / "requirements.txt")]
    data = json.loads((root / "python" / launcher.MARKER).read_text())
    assert data["detected"] == "cuda" and data["installed"] == "cpu"


def test_install_failure_of_requirements_raises(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(launcher.InstallError):
        launcher.install_dependencies(root, "cpu", run=lambda cmd: 1, log=lambda m: None)
    assert not (root / "python" / launcher.MARKER).exists()


def test_backend_change_uninstalls_old_torch(tmp_path):
    root = make_root(tmp_path)
    launcher.write_marker(root / "python" / launcher.MARKER, detected="cpu", installed="cpu", req_hash="x")
    calls = []
    launcher.install_dependencies(root, "cuda", run=lambda c: calls.append(c) or 0, log=lambda m: None)
    assert calls[0][-4:] == ["uninstall", "-y", "torch", "torchvision"]


def test_ensure_folder_prompts_and_saves(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"FOLDER_ZE_ZDJECIAMI": ""}), encoding="utf-8")
    photos = tmp_path / "zdjęcia"
    photos.mkdir()
    answers = iter([str(tmp_path / "brak"), f'"{photos}"'])
    launcher.ensure_folder(cfg, input_fn=lambda p: next(answers), log=lambda m: None)
    assert json.loads(cfg.read_text(encoding="utf-8"))["FOLDER_ZE_ZDJECIAMI"] == str(photos)


def test_ensure_folder_keeps_existing(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"FOLDER_ZE_ZDJECIAMI": str(tmp_path)}), encoding="utf-8")
    launcher.ensure_folder(cfg, input_fn=lambda p: pytest.fail("nie powinno pytać"), log=lambda m: None)


def test_app_command_and_env(tmp_path):
    cmd = launcher.app_command(tmp_path, ["--x"])
    assert cmd[1:4] == ["-m", "filesearcher", "--config"] and cmd[-1] == "--x"
    env = launcher.app_env(tmp_path, {})
    assert env["HF_HOME"] == str(tmp_path / "modele") and env["PYTHONUTF8"] == "1"


def test_check_survives_cp1252_console(monkeypatch, tmp_path):
    import io

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    monkeypatch.setattr(launcher, "app_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "detect_gpus", lambda: ["AMD Radeon RX 9070 XT"])
    launcher.main(["--check"])
    sys.stdout.flush()
    assert "AMD ROCm" in raw.getvalue().decode("utf-8")
