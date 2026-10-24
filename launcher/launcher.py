"""FileSearcher.exe: przy pierwszym starcie instaluje biblioteki do wbudowanego Pythona, potem uruchamia program.

Budowany PyInstallerem (packaging/build_windows.ps1). Nie importuje niczego spoza biblioteki standardowej.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MARKER = ".filesearcher-deps.json"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
ROCM_SDK = [
    f"{ROCM_BASE}/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_BASE}/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_BASE}/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_BASE}/rocm-7.2.1.tar.gz",
]
ROCM_TORCH = [
    f"{ROCM_BASE}/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    f"{ROCM_BASE}/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
]
# Karty z oficjalnym PyTorchem ROCm na Windows (gfx1100, gfx1101, gfx1200, gfx1201).
ROCM_CARDS = re.compile(r"RX\s*(79\d\d|78\d\d|77\d\d|90[67]\d)|R9700|W7900|W7800", re.IGNORECASE)
BACKEND_NAMES = {"cuda": "NVIDIA CUDA", "rocm": "AMD ROCm", "cpu": "CPU (bez karty graficznej)"}


class InstallError(Exception):
    pass


def detect_gpus() -> list[str]:
    """Nazwy kart graficznych z Windowsa."""
    commands = [
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
        ["wmic", "path", "win32_VideoController", "get", "name"],
    ]
    for cmd in commands:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        names = [line.strip() for line in out.splitlines() if line.strip() and line.strip() != "Name"]
        if names:
            return names
    return []


def choose_backend(gpu_names: list[str]) -> str:
    if any("nvidia" in n.lower() for n in gpu_names):
        return "cuda"
    if any(ROCM_CARDS.search(n) for n in gpu_names):
        return "rocm"
    return "cpu"


def torch_install_commands(backend: str) -> list[list[str]]:
    """Argumenty dla `python -m pip`, w kolejności wykonania."""
    if backend == "cuda":
        return [["install", "torch", "torchvision", "--index-url", CUDA_INDEX]]
    if backend == "rocm":
        return [["install", "--no-cache-dir", *ROCM_SDK], ["install", "--no-cache-dir", *ROCM_TORCH]]
    return [["install", "torch", "torchvision", "--index-url", CPU_INDEX]]


def requirements_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def read_marker(marker: Path) -> dict | None:
    try:
        return json.loads(Path(marker).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_marker(marker: Path, detected: str, installed: str, req_hash: str):
    data = {"detected": detected, "installed": installed, "requirements": req_hash}
    Path(marker).write_text(json.dumps(data), encoding="utf-8")


def needs_install(marker: Path, detected: str, req_hash: str) -> bool:
    data = read_marker(marker)
    # Porównujemy wykrytą kartę, nie zainstalowaną wersję: nieudana instalacja GPU
    # (przejście na CPU) nie powtarza się przy każdym starcie.
    return data is None or data.get("detected") != detected or data.get("requirements") != req_hash


def python_exe(root: Path) -> Path:
    return Path(root) / "python" / ("python.exe" if os.name == "nt" else "python")


def install_dependencies(root: Path, detected: str, run=None, log=print) -> str:
    """Instaluje PyTorcha dla wykrytej karty (z przejściem na CPU) i resztę bibliotek. Zwraca zainstalowany backend."""
    root = Path(root)
    run = run or (lambda cmd: subprocess.run(cmd).returncode)
    pip = [str(python_exe(root)), "-m", "pip"]
    marker = root / "python" / MARKER
    requirements = root / "app" / "requirements.txt"

    previous = read_marker(marker)
    if previous and previous.get("detected") != detected:
        run([*pip, "uninstall", "-y", "torch", "torchvision"])

    installed = detected
    log(f"Instaluję PyTorcha: {BACKEND_NAMES[detected]}. To jednorazowe i może potrwać kilkanaście minut (ok. 3 GB).")
    if not all(run([*pip, *args]) == 0 for args in torch_install_commands(detected)):
        if detected == "cpu":
            raise InstallError("Nie udało się zainstalować PyTorcha. Sprawdź połączenie z internetem.")
        log(f"Nie udało się zainstalować wersji {BACKEND_NAMES[detected]}. Instaluję wersję na CPU.")
        installed = "cpu"
        if not all(run([*pip, *args]) == 0 for args in torch_install_commands("cpu")):
            raise InstallError("Nie udało się zainstalować PyTorcha. Sprawdź połączenie z internetem.")

    log("Instaluję pozostałe biblioteki…")
    if run([*pip, "install", "-r", str(requirements)]) != 0:
        raise InstallError("Nie udało się zainstalować bibliotek. Sprawdź połączenie z internetem.")
    write_marker(marker, detected=detected, installed=installed, req_hash=requirements_hash(requirements))
    return installed


def ensure_folder(config_path: Path, input_fn=input, log=print):
    """Pyta o folder ze zdjęciami, gdy w config.json go brak albo nie istnieje."""
    config_path = Path(config_path)
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    folder = data.get("FOLDER_ZE_ZDJECIAMI") or ""
    if folder and Path(folder).is_dir():
        return
    if folder:
        log(f"Folder {folder} z config.json nie istnieje.")
    while True:
        answer = input_fn("Podaj folder ze zdjęciami (np. D:\\Zdjecia): ").strip().strip('"')
        if answer and Path(answer).is_dir():
            break
        log("Taki folder nie istnieje, spróbuj jeszcze raz.")
    data["FOLDER_ZE_ZDJECIAMI"] = answer
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def app_command(root: Path, args: list[str]) -> list[str]:
    return [str(python_exe(root)), "-m", "filesearcher", "--config", str(Path(root) / "config.json"), *args]


def app_env(root: Path, base: dict) -> dict:
    env = dict(base)
    env.setdefault("HF_HOME", str(Path(root) / "modele"))  # modele leżą obok programu
    env["PYTHONUTF8"] = "1"
    return env


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent / "dist" / "FileSearcher"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = app_root()
    gpus = detect_gpus()
    detected = choose_backend(gpus)
    if "--check" in argv:
        print(f"Folder programu: {root}")
        print(f"Karty: {', '.join(gpus) or 'brak'} → {BACKEND_NAMES[detected]}")
        return 0 if python_exe(root).exists() else 1

    try:
        if needs_install(root / "python" / MARKER, detected, requirements_hash(root / "app" / "requirements.txt")):
            print(f"Wykryte karty: {', '.join(gpus) or 'brak'}")
            install_dependencies(root, detected)
        ensure_folder(root / "config.json")
        code = subprocess.run(app_command(root, argv), cwd=root, env=app_env(root, os.environ)).returncode
    except InstallError as e:
        print(f"Błąd: {e}")
        code = 1
    except KeyboardInterrupt:
        code = 130
    if code not in (0, 130):
        input("\nNaciśnij Enter, aby zamknąć okno…")
    return code


if __name__ == "__main__":
    sys.exit(main())
