# Plan: Add yt-dlp Video Site Support to Translator

## Context

Add `--ytdlp <url>` to the translator:

1. **Downloads audio** into `--input <dir>` via `uv run --project <translator/asr> yt-dlp` (version pinned in [`translator/asr/pyproject.toml`](../asr/pyproject.toml) / `uv.lock`)
2. **Extracts metadata** (title, description, creator/uploader, tags) from the same page

`--input` remains required — it is the download destination and where the pipeline discovers audio tracks. The CLI spawns `uv` with `run --project <ASR_PROJECT_ROOT> yt-dlp <args>` (see [`ytdlp-runner.ts`](../src/pipeline/ytdlp-runner.ts)); no separate Python wrapper script.

`ASR_PROJECT_ROOT` is `path.resolve(__dirname, "../asr")` from the compiled CLI (same base as the default ASR Python path in [`config.ts`](../src/config.ts)).

## Files to Modify

| File | Change |
|------|--------|
| [`translator/src/pipeline/ytdlp-runner.ts`](../src/pipeline/ytdlp-runner.ts) | Subprocess: `uv run --project … yt-dlp`, parse JSON / `.info.json` |
| [`translator/src/config.ts`](../src/config.ts) | `ytdlpUrl?`, `uvExe`, `ytdlpAudioFormat`, export `ASR_PROJECT_ROOT` |
| [`translator/src/cli.ts`](../src/cli.ts) | Flags, mutual exclusion of metadata sources, Step 0 download before track discovery, yt-dlp metadata branch |
| [`translator/asr/pyproject.toml`](../asr/pyproject.toml) | `yt-dlp` dependency |
| [`translator/README.md`](../README.md) | Document flags + usage |

## Implementation

### 1. [`src/pipeline/ytdlp-runner.ts`](../src/pipeline/ytdlp-runner.ts)

- **`ytdlpUvArgs(asrProjectRoot, innerArgs)`** → `["run", "--project", asrProjectRoot, "yt-dlp", ...innerArgs]`
- **`downloadYtdlpAudio(url, downloadDir, uvExe, asrProjectRoot, audioFormat)`** — `-x`, optional `--audio-format`, `--write-info-json`, `--no-playlist`, `--paths`, `-o`, then `url`; read first `*.info.json` in `downloadDir`
- **`fetchYtdlpMetadata(url, uvExe, asrProjectRoot)`** — `--dump-json`, `--no-playlist`
- **`parseYtdlpInfo`** — builds `metadataMd` (Title / Creator / Tags / Description); optional fields respect `exactOptionalPropertyTypes`

### 2. [`src/config.ts`](../src/config.ts)

- `ytdlpUrl?`, `uvExe` (default `"uv"`), `ytdlpAudioFormat` (default `"best"`)
- `ASR_PROJECT_ROOT = path.resolve(__dirname, "../asr")`

### 3. [`src/cli.ts`](../src/cli.ts)

**Flags:** `ytdlp`, `uv-exe`, `ytdlp-audio-format`

**Mutual exclusion:** at most one of `--dlsite`, `--ytdlp`, `--metadata`

**Step 0** (before discover audio):

```typescript
let ytdlpMeta: YtdlpMetadata | undefined;
if (config.ytdlpUrl) {
  const dirEntries = await fs.readdir(config.inputDir).catch(() => []);
  const infoFiles = dirEntries.filter((e) => e.endsWith(".info.json"));
  if (infoFiles.length > 0) {
    ytdlpMeta = await fetchYtdlpMetadata(config.ytdlpUrl, config.uvExe, ASR_PROJECT_ROOT);
  } else {
    ytdlpMeta = await downloadYtdlpAudio(
      config.ytdlpUrl,
      config.inputDir,
      config.uvExe,
      ASR_PROJECT_ROOT,
      config.ytdlpAudioFormat,
    );
  }
}
```

**Metadata branch** — `else if (config.ytdlpUrl && ytdlpMeta)` after the DLSite block: same pattern as DLSite (`tryLoadCachedMetadata`, optional `MetadataExtractor` + `LlamaServerManager`, else scraped-only `UserMetadata`).

**Import:**

```typescript
import {
  downloadYtdlpAudio,
  fetchYtdlpMetadata,
  type YtdlpMetadata,
} from "./pipeline/ytdlp-runner.js";
```

## Existing Components Reused Unchanged

- `MetadataExtractor`, `LlamaServerManager`, `LlmClient` — same as DLSite path
- `tryLoadCachedMetadata()` — unchanged
- `metadataMd` markdown shape — aligned with DLSite for downstream LLM use

## Output Structure

```
<input>/
  <id>.<ext>       # downloaded audio
  <id>.info.json   # yt-dlp metadata JSON (cache marker on re-run)
<output>/
  metadata.json
  <track-dir>/
    <stem>.transcription.json
    <stem>.translation.json
    <stem>.srt
```

## Verification

1. `cd translator/asr && uv run yt-dlp --dump-json --no-playlist <url>` — JSON includes `title`, `description`, `uploader` (or equivalent)
2. `--ytdlp <url> --input <dir> --output <out>` — audio + `.info.json` under `<dir>`, `metadata.json` under `<out>`
3. Re-run — if `.info.json` exists in `<dir>`, skips download, metadata-only fetch
4. With `--meta-hf-repo <repo>` — full LLM glossary extraction

## Caveats

- Multiple `*.info.json` in one `--input`: implementation picks the first filename match; prefer one work per input directory.
