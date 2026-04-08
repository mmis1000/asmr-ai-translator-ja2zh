# asmr-translator

A CLI pipeline for transcribing and translating Japanese ASMR audio to Traditional or Simplified Chinese. It chains GPU-accelerated ASR (Whisper / SenseVoice / MMS / Qwen / Gemma) with a GGUF LLM for context-aware translation, optionally using DLSite metadata for a per-work glossary.

## Requirements

- **Node.js** 20+, **npm** (for the TypeScript CLI)
- **Python 3.12** + [uv](https://docs.astral.sh/uv/) (for the ASR subsystem)
- **llama-server** from [llama.cpp](https://github.com/ggerganov/llama.cpp) in `PATH` (or supply `--llama-server <path>`)
- A fine-tuned translation GGUF model (see `--model` / `--hf-repo`)
- **AMD GPU with ROCm 7.2** (Windows) — the ASR environment installs ROCm wheels automatically

## Setup

### 1. Install Node dependencies

```bash
cd translator
npm install
npm run build
```

### 2. Set up the Python ASR environment (AMD ROCm)

```powershell
cd translator/asr
./setup.ps1
```

This downloads the CTranslate2 ROCm wheel, syncs the main venv (`asr/.venv`), and syncs the isolated Qwen venv (`asr/qwen_env/.venv`).

## Usage

**Basic Command**
```bash
npm run start -- --input <dir> --output <dir> --model <path> [options]
```

### Suggested Settings (Full Pipeline)

For a complete run using Qwen for echo translation, Unsloth Qwen for metadata gathering, DLSite parsing, and Python Whisper with vocal repair, use the following:

```powershell
npm run start -- `
  --input "<dir>" `
  --output "<dir>" `
  --hf-repo "mmis1000/asmr-qwen3.5-9b-zh-tw-echo-gguf-v0.1:Q8_0" `
  --meta-hf-repo "unsloth/Qwen3.5-9B-GGUF:UD-Q6_K_XL" `
  --dlsite "RJxxxxxx" `
  --llama-server "<path-to-llama-server.exe>" `
  --asr python `
  --lang zh-tw `
  --repair-with-vocal `
  --mode echo
```

### Suggested Settings (No DLSite metadata)

If you do not have a DLSite ID for your work, you can create a generic metadata JSON file (e.g. `generic-meta.json`):

```json
{
  "asr": {
    "prompt": "シチュエーション音声"
  },
  "translate": {
    "title": "",
    "summary": "",
    "cv_mapping": [],
    "character_mapping": [],
    "term_mapping": []
  }
}
```

### Suggested Settings (Manual Metadata Entry)

If you know the work's details (such as character names, glossary terms, and titles) but do not have a DLSite ID to scrape them automatically, you can manually construct a populated `metadata.json` file. Please refer to the [Metadata JSON format](#metadata-json-format) section at the bottom of this page for the exact structure and examples on how to fill it out.

Whether using an empty generic file or a manually filled one, run the translator with the `--metadata` flag instead of `--dlsite`:

```powershell
npm run start -- `
  --input "<dir>" `
  --output "<dir>" `
  --hf-repo "mmis1000/asmr-qwen3.5-9b-zh-tw-echo-gguf-v0.1:Q8_0" `
  --metadata "generic-meta.json" `
  --llama-server "<path-to-llama-server.exe>" `
  --asr python `
  --lang zh-tw `
  --repair-with-vocal `
  --mode echo
```

### Required

| Flag | Description |
|------|-------------|
| `--input <dir>` | Directory of audio files (mp3, wav, flac, m4a, ogg, aac — searched recursively) |
| `--output <dir>` | Output directory (separate from input) |
| `--model <path>` | Fine-tuned translation GGUF model |
| `--hf-repo <repo>` | HuggingFace repo as alternative to `--model` (`user/model[:quant]` or full HF URL) |

One of `--model`, `--hf-repo`, or `--server-url` is required.

### Metadata (optional)

Metadata provides a per-work glossary of character names, CV names, and terms that improves translation accuracy.

| Flag | Description |
|------|-------------|
| `--dlsite <id\|url>` | DLSite work ID or URL — scrapes metadata for context |
| `--metadata <file>` | User-supplied metadata JSON |
| `--meta-model <path>` | GGUF model for metadata extraction (used with `--dlsite`) |
| `--meta-hf-repo <repo>` | HuggingFace repo for metadata model |
| `--meta-server-url <url>` | External server URL for metadata extraction |
| `--meta-ctx-size <n>` | Context size for metadata model (default: 16384) |

Without `--meta-model` / `--meta-server-url`, `--dlsite` still scrapes DLSite for basic metadata (title, VA, description) but skips LLM glossary extraction.

### Translation server

| Flag | Description |
|------|-------------|
| `--server-url <url>` | Use an already-running llama-server (skips internal management) |
| `--llama-server <path>` | llama-server executable (default: `llama-server` in PATH) |
| `--port <n>` | Translation server port (default: 8181) |
| `--gpu-layers <n\|all\|auto>` | GPU layers to offload (default: `all`) |
| `--ctx-size <n>` | Context size (default: 8192) |
| `--parallel <n>` | Parallel inference slots (default: 1) |

### Translation options

| Flag | Description |
|------|-------------|
| `--lang <zh-tw\|zh-cn>` | Target language (default: `zh-tw`) |
| `--mode <base\|echo>` | Translation mode (default: `echo`) |
| `--seed <n>` | Fixed RNG seed for reproducible output |
| `--debug-log` | Write LLM prompts and responses to `debug_logs/` |

### ASR options

| Flag | Description |
|------|-------------|
| `--asr <python\|skip>` | ASR mode (default: `skip` — expects pre-existing transcription JSON) |
| `--asr-engine <engine>` | Engine: `whisper`, `mms`, `qwen`, `sensevoice`, `gemma` (default: `whisper`) |
| `--python-exe <path>` | Python executable (default: `asr/.venv/Scripts/python.exe`) |
| `--repair-with-vocal` | Use Demucs vocal stem for the repair pass |
| `--repair-engine <engine>` | Surgical repair engine: `whisper`, `mms`, `qwen`, `sensevoice`, `gemma` (default: `qwen`) |
| `--save-repair-audio` | Save audio fragments used for surgical repairs |
| `--vocal-threshold <n>` | Vocal energy threshold (default: 0.001) |
| `--snr-threshold <n>` | SNR dB threshold (default: 2.0) |

## Pipeline overview

1. **Audio discovery** — walks `--input` recursively for audio files
2. **Metadata resolution** — DLSite scrape + optional LLM extraction, or user JSON
3. **ASR transcription** — Whisper (or alternative engine) via Python subprocess
4. **Transcript cleaning** — filters hallucinations, low-SNR segments, repetition loops
5. **Surgical repair** — re-transcribes flagged windows with stricter settings; cached between runs
6. **Translation** — windowed LLM translation with glossary injection
7. **Output** — JSON transcription, JSON translation, SRT subtitles per track

## Output structure

```
<output>/
  metadata.json            # resolved work metadata + glossary
  <track-dir>/
    <stem>.transcription.json
    <stem>.translation.json
    <stem>.srt
    <stem>.surgical.json   # repair log (for incremental re-runs)
    demucs_output/         # vocal stems (if --save-audio)
```

## Utilities

**Pipeline visualizer** — renders a timeline PNG showing ASR segments, repair windows, and SNR per track:

```bash
uv run visualize_pipeline.py <output-dir> [track-stem-filter]
```

## Repair engine comparison (MMS vs Qwen)

Tested on 8 tracks from the same work.

| Metric | `--repair-engine mms` | `--repair-engine qwen` |
|--------|---------------|----------------|
| Kanji / vocabulary accuracy | Worse — outputs phonetic kana where kanji expected | Better — correctly produces kanji (e.g. 監禁 vs かんきん) |
| Avg segment duration | Consistent ~3 s | More variable; collapses to ~5 s on long tracks |
| Translation tone | Missing `~` / formal feel | Preserves playful speech register with `~` |
| Translation on garbled input | Over-generates (long fabricated runs) | Stays shorter / safer |
| Noise section handling | Small garbage segments | Long merged hallucinations (20–43 s) |
| Long-track stability | Good | Degrades badly toward the end of long tracks |

**MMS-specific issues observed:**
- Can produce a long garbled segment over non-speech/noise sections due to CTC misalignment (e.g. 21-second segment at track start)
- Occasional foreign-script leakage (Turkish `ş`, Polish `ę`, German words)
- Phonetic kana instead of kanji causes cascading translation errors (e.g. `かんきん` → `金金` instead of `監禁`)
- Translator over-generates wildly when fed garbled input (a short phrase expands to a paragraph of repeated text)

**Qwen-specific issues observed:**
- Degrades severely toward the end of long tracks — produces repetitive filler sounds and English nonsense tokens
- Creates very long segments (20–43 s) that merge multiple utterances
- One mistranslation from ambiguous kanji: `自己室` → `自慰室` (masturbation room) instead of "own room"

**Bottom line:** Qwen repair produces better vocabulary and more natural-sounding translations for clear speech. MMS repair is more stable for difficult/noisy sections and long tracks. Neither is dominant — Qwen wins on clean speech, MMS wins on robustness.

## Metadata JSON format

User-supplied metadata (`--metadata`) should follow this shape:

```json
{
  "asr": {
    "prompt": "囁き声、耳舐め、シチュエーション音声など、ASRエンジンへのヒントとなるキーワード"
  },
  "translate": {
    "title": "作品のタイトル / 作品標題",
    "summary": "シナリオの要約や状況設定など / 情境設定摘要",
    "cv_mapping": [
      { "ja": "声優の名前", "zh": "配音員中文名" }
    ],
    "character_mapping": [
      { "ja": "お姉ちゃん", "zh": "姊姊" }
    ],
    "term_mapping": [
      { "ja": "耳舐め", "zh": "舔耳" }
    ]
  }
}
```
