# Składa przenośną paczkę dist\FileSearcher-windows-x64.zip:
#   FileSearcher.exe (program startowy, PyInstaller), python\ (wbudowany Python 3.12 z pip), app\ (kod).
# Uruchamiany przez .github/workflows/release.yml; lokalnie: pwsh packaging/build_windows.ps1
$ErrorActionPreference = "Stop"
$PythonVersion = "3.12.10"  # Python 3.12 – tego wymagają paczki PyTorch ROCm dla Windows

$repo = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $repo "dist\FileSearcher"
$work = Join-Path $repo "build"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force -Path $dist, $work | Out-Null

# 1. Wbudowany Python
$embedZip = Join-Path $work "python-embed.zip"
Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $embedZip
Expand-Archive $embedZip -DestinationPath "$dist\python"
$pth = Get-ChildItem "$dist\python\python*._pth" | Select-Object -First 1
Set-Content -Path $pth.FullName -Encoding ascii -Value @(
    ($pth.BaseName + ".zip"), ".", "Lib\site-packages", "..\app", "import site"
)
$py = "$dist\python\python.exe"

# 2. pip (+ setuptools do instalacji paczek źródłowych, np. rocm-7.2.1.tar.gz)
Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile "$work\get-pip.py"
& $py "$work\get-pip.py" --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip nie powiodło się" }
& $py -m pip install --no-warn-script-location setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "instalacja setuptools nie powiodła się" }

# 3. Kod programu
New-Item -ItemType Directory -Force -Path "$dist\app" | Out-Null
Copy-Item -Recurse "$repo\filesearcher" "$dist\app\filesearcher"
Get-ChildItem "$dist\app" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Copy-Item "$repo\requirements.txt" "$dist\app\requirements.txt"
Copy-Item "$repo\packaging\config.json" "$dist\config.json"
Copy-Item "$repo\README.md", "$repo\LICENSE" $dist

# 4. Program startowy
python -m PyInstaller --onefile --console --name FileSearcher --distpath $dist `
    --workpath "$work\pyinstaller" --specpath $work "$repo\launcher\launcher.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller nie powiódł się" }

# 5. Szybkie sprawdzenie paczki
& "$dist\FileSearcher.exe" --check
if ($LASTEXITCODE -ne 0) { throw "FileSearcher.exe --check nie powiódł się" }
& $py -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(a)'); import filesearcher.config; print('sqlite FTS5 i import programu: OK')"
if ($LASTEXITCODE -ne 0) { throw "wbudowany Python nie ma FTS5 albo nie widzi kodu programu" }

# 6. Zip
$zip = Join-Path $repo "dist\FileSearcher-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $dist -DestinationPath $zip
Write-Host "Gotowe: $zip"
