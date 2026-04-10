$ErrorActionPreference = "Stop"

# Ensure we run from the script's own directory
Push-Location $PSScriptRoot

Write-Host "Starting setup for CTranslate2 + Whisper + Demucs (ROCm) in translator/asr..."

# 1. Ensure uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
  Write-Error "uv is not found in PATH. Please install it globally."
  exit 1
}

# 2. Download and extract CTranslate2 ROCm wheels (Windows + Linux).
#    uv lock needs both local path sources in pyproject.toml even on one platform.
$ZipUrlWin      = "https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Windows.zip"
$ZipUrlLinux    = "https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip"
$ZipFileWin     = Join-Path $PSScriptRoot "rocm-python-wheels-Windows.zip"
$ZipFileLinux   = Join-Path $PSScriptRoot "rocm-python-wheels-Linux.zip"
$WheelDir       = Join-Path $PSScriptRoot "rocm_wheels"
$WheelNameWin   = "ctranslate2-4.7.1-cp312-cp312-win_amd64.whl"
$WheelNameLinux = "ctranslate2-4.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

function Ensure-CTranslate2WheelFromZip {
  param(
    [string]$ZipUrl,
    [string]$ZipFile,
    [string]$WheelName,
    [string]$TempDir
  )
  $WheelDest = Join-Path $WheelDir $WheelName
  if (Test-Path $WheelDest) {
    return
  }
  if (-not (Test-Path $ZipFile)) {
    Write-Host "Downloading $(Split-Path $ZipUrl -Leaf)..."
    (New-Object Net.WebClient).DownloadFile($ZipUrl, $ZipFile)
  }
  if (Test-Path $TempDir) {
    Remove-Item -Recurse -Force $TempDir
  }
  Expand-Archive -Path $ZipFile -DestinationPath $TempDir -Force
  $WheelFile = Get-ChildItem -Path $TempDir -Filter $WheelName -Recurse | Select-Object -First 1
  if (-not $WheelFile) {
    Write-Error "No matching CTranslate2 wheel found in zip: $WheelName"
    exit 1
  }
  New-Item -ItemType Directory -Path $WheelDir -Force | Out-Null
  Copy-Item $WheelFile.FullName $WheelDest
  Write-Host "Prepared CTranslate2 ROCm wheel: $WheelName"
  Remove-Item -Recurse -Force $TempDir
}

New-Item -ItemType Directory -Path $WheelDir -Force | Out-Null
$TempWin = Join-Path $PSScriptRoot "temp-ct2-win"
$TempLinux = Join-Path $PSScriptRoot "temp-ct2-linux"
Ensure-CTranslate2WheelFromZip -ZipUrl $ZipUrlWin -ZipFile $ZipFileWin -WheelName $WheelNameWin -TempDir $TempWin
Ensure-CTranslate2WheelFromZip -ZipUrl $ZipUrlLinux -ZipFile $ZipFileLinux -WheelName $WheelNameLinux -TempDir $TempLinux

# 3. Sync main dependencies
Write-Host "Syncing main ASR dependencies (Transformers 5.x)..."
uv sync
if ($LASTEXITCODE -ne 0) {
  Write-Error "Main uv sync failed."
  exit $LASTEXITCODE
}

# 4. Sync Qwen dependencies (Transformers 4.57.6)
Write-Host "`nSyncing isolated Qwen ASR dependencies..."
Push-Location qwen_env
uv sync
if ($LASTEXITCODE -ne 0) {
  Write-Error "Qwen uv sync failed."
  Pop-Location
  exit $LASTEXITCODE
}
Pop-Location

# 4. Verification
Write-Host "`nVerifying installation..."
$VerifyScript = @"
import sys, torch, ctranslate2, demucs
from faster_whisper import WhisperModel
print(f'Python:       {sys.version.split()[0]}')
print(f'Torch:        {torch.__version__}')
avail = torch.cuda.is_available()
print(f'ROCm/HIP:     {avail}')
if avail:
    print(f'Device:       {torch.cuda.get_device_name(0)}')
print(f'CTranslate2:  {ctranslate2.__version__}')
print(f'Demucs:       imported OK')
print(f'faster-whisper imported OK')
"@

uv run python -c $VerifyScript

Pop-Location
Write-Host "`nSetup Complete!"
