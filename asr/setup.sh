#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the script's own directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting setup for CTranslate2 + Whisper + Demucs (ROCm) in translator/asr..."

# 1. Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not found in PATH. Please install it globally." >&2
  exit 1
fi

# 2. Download and extract CTranslate2 ROCm wheels (Linux + Windows).
#    uv lock needs both local path sources in pyproject.toml even on one platform.
ZIP_URL_LINUX="https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip"
ZIP_URL_WIN="https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Windows.zip"
ZIP_FILE_LINUX="$SCRIPT_DIR/rocm-python-wheels-Linux.zip"
ZIP_FILE_WIN="$SCRIPT_DIR/rocm-python-wheels-Windows.zip"
WHEEL_DIR="$SCRIPT_DIR/rocm_wheels"
WHEEL_LINUX="ctranslate2-4.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
WHEEL_WIN="ctranslate2-4.7.1-cp312-cp312-win_amd64.whl"

ensure_wheel_from_zip() {
  local zip_url="$1" zip_file="$2" wheel_name="$3" temp_dir="$4"
  local dest="$WHEEL_DIR/$wheel_name"
  if [[ -f "$dest" ]]; then
    return 0
  fi
  if [[ ! -f "$zip_file" ]]; then
    echo "Downloading $(basename "$zip_url")..."
    curl -fL -o "$zip_file" "$zip_url"
  fi
  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"
  unzip -q "$zip_file" -d "$temp_dir"
  local wheel_src
  wheel_src="$(find "$temp_dir" -name "$wheel_name" -type f -print -quit)"
  if [[ -z "$wheel_src" ]]; then
    echo "No matching CTranslate2 wheel found in zip: $wheel_name" >&2
    exit 1
  fi
  mkdir -p "$WHEEL_DIR"
  cp "$wheel_src" "$dest"
  echo "Prepared CTranslate2 ROCm wheel: $wheel_name"
  rm -rf "$temp_dir"
}

mkdir -p "$WHEEL_DIR"
ensure_wheel_from_zip "$ZIP_URL_LINUX" "$ZIP_FILE_LINUX" "$WHEEL_LINUX" "$SCRIPT_DIR/temp-ct2-linux"
ensure_wheel_from_zip "$ZIP_URL_WIN" "$ZIP_FILE_WIN" "$WHEEL_WIN" "$SCRIPT_DIR/temp-ct2-win"

# 3. Sync main dependencies
echo "Syncing main ASR dependencies (Transformers 5.x)..."
uv sync

# 4. Sync Qwen dependencies (Transformers 4.57.6)
echo ""
echo "Syncing isolated Qwen ASR dependencies..."
( cd qwen_env && uv sync )

# 5. Verification
echo ""
echo "Verifying installation..."
uv run python <<'PY'
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
PY

echo ""
echo "Setup Complete!"
