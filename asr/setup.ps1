$ErrorActionPreference = "Stop"

# Ensure we run from the script's own directory
Push-Location $PSScriptRoot

Write-Host "Starting setup for CTranslate2 + Whisper + Demucs (ROCm) in translator/asr..."

# 1. Ensure uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
  Write-Error "uv is not found in PATH. Please install it globally."
  exit 1
}

# 2. Download and extract CTranslate2 ROCm wheel
$ZipUrl      = "https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Windows.zip"
$ZipFile     = Join-Path $PSScriptRoot "rocm-python-wheels-Windows.zip"
$ExtractPath = Join-Path $PSScriptRoot "rocm_wheels_tmp"
$WheelName   = "ctranslate2-4.7.1-cp312-cp312-win_amd64.whl"
$WheelDir    = Join-Path $PSScriptRoot "rocm_wheels"
$WheelDest   = Join-Path $WheelDir $WheelName

if (-not (Test-Path $WheelDest)) {
  if (-not (Test-Path $ZipFile)) {
    Write-Host "Downloading CTranslate2 ROCm wheels..."
    (New-Object Net.WebClient).DownloadFile($ZipUrl, $ZipFile)
  }

  if (Test-Path $ExtractPath) {
    Remove-Item -Recurse -Force $ExtractPath
  }
  Expand-Archive -Path $ZipFile -DestinationPath $ExtractPath -Force

  $WheelFile = Get-ChildItem -Path $ExtractPath -Filter $WheelName -Recurse | Select-Object -First 1
  if (-not $WheelFile) {
    Write-Error "No matching CTranslate2 wheel found: $WheelName"
    exit 1
  }

  New-Item -ItemType Directory -Path $WheelDir -Force | Out-Null
  Copy-Item $WheelFile.FullName $WheelDest
  Write-Host "Prepared CTranslate2 ROCm wheel: $WheelName"

  Remove-Item -Recurse -Force $ExtractPath
}

# 3. Sync dependencies
Write-Host "Syncing dependencies..."
uv sync
if ($LASTEXITCODE -ne 0) {
  Write-Error "uv sync failed."
  exit $LASTEXITCODE
}

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
