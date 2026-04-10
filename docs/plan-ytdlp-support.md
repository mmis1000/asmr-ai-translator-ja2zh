# Plan: Add yt-dlp Video Site Support to Translator

## Context

Add `--ytdlp <url>` to the translator:
1. **Downloads audio** into `--input <dir>` via `uvx yt-dlp`
2. **Extracts metadata** (title, description, creator, tags) from the same page

`--input` remains required — it is the download destination and where the pipeline discovers audio tracks. yt-dlp is invoked directly as `uvx yt-dlp <args>` — no Python script wrapper needed.

## Files to Modify

| File | Change |
|------|--------|
| `translator/src/pipeline/fetch-ytdlp-metadata.ts` | **New file** — spawns `uvx yt-dlp`, parses JSON output |
| `translator/src/config.ts` | Add `ytdlpUrl?`, `uvxExe`, `ytdlpAudioFormat` |
| `translator/src/cli.ts` | Add flags, Step 0 download before track discovery, metadata branch |
| `translator/README.md` | Document new flags + usage example |

## Implementation

### 1. New: `src/pipeline/fetch-ytdlp-metadata.ts`

```typescript
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

export interface YtdlpMetadata {
  id: string;
  title?: string;
  uploader?: string;
  description?: string;
  tags?: string[];
  webpageUrl?: string;
  /** Formatted markdown for LLM extraction — same format as DlsiteMetadata.metadataMd. */
  metadataMd: string;
}

/**
 * Download audio from url into downloadDir.
 * Uses --write-info-json to write <id>.info.json alongside the audio.
 * Returns parsed metadata.
 */
export async function downloadYtdlpAudio(
  url: string,
  downloadDir: string,
  uvxExe = "uvx",
  audioFormat = "best",
): Promise<YtdlpMetadata> {
  console.log(`[yt-dlp] Downloading audio from ${url} → ${downloadDir}`);

  const args = [
    "yt-dlp",
    "-x",
    ...(audioFormat !== "best" ? ["--audio-format", audioFormat] : []),
    "--write-info-json",
    "--no-playlist",
    "--paths", downloadDir,
    "-o", "%(id)s.%(ext)s",
    url,
  ];

  await execFileAsync(uvxExe, args, { maxBuffer: 10 * 1024 * 1024 });

  // Read the .info.json written by yt-dlp
  const { readdir, readFile } = await import("fs/promises");
  const { join } = await import("path");
  const entries = await readdir(downloadDir);
  const infoFile = entries.find(e => e.endsWith(".info.json"));
  if (!infoFile) throw new Error("[yt-dlp] No .info.json found after download");

  const info = JSON.parse(await readFile(join(downloadDir, infoFile), "utf-8"));
  return parseYtdlpInfo(info);
}

/**
 * Fetch metadata only (no download) — used when audio is already cached.
 */
export async function fetchYtdlpMetadata(
  url: string,
  uvxExe = "uvx",
): Promise<YtdlpMetadata> {
  console.log(`[yt-dlp] Fetching metadata from ${url}`);
  const { stdout } = await execFileAsync(uvxExe, [
    "yt-dlp", "--dump-json", "--no-playlist", url,
  ], { maxBuffer: 50 * 1024 * 1024 });
  return parseYtdlpInfo(JSON.parse(stdout));
}

function parseYtdlpInfo(info: Record<string, unknown>): YtdlpMetadata {
  const id = (info.id as string) ?? "unknown";
  const title = info.title as string | undefined;
  const uploader = (info.uploader ?? info.channel ?? info.creator) as string | undefined;
  const description = info.description as string | undefined;
  const tags = info.tags as string[] | undefined;
  const webpageUrl = info.webpage_url as string | undefined;

  let metadataMd = "";
  if (title)        metadataMd += `# Title\n\n${title}\n\n`;
  if (uploader)     metadataMd += `# Creator\n\n${uploader}\n\n`;
  if (tags?.length) metadataMd += `# Tags\n\n${tags.join(", ")}\n\n`;
  if (description)  metadataMd += `# Description\n\n${description}\n\n`;

  console.log(`[yt-dlp] ${id}: title="${title}", uploader="${uploader}"`);
  return { id, title, uploader, description, tags, webpageUrl, metadataMd };
}
```

Notes:
- No new npm deps, no Python script — just `execFile` calling `uvx yt-dlp`
- `downloadYtdlpAudio`: reads the `.info.json` yt-dlp writes alongside the audio for metadata (no second subprocess)
- `fetchYtdlpMetadata`: `--dump-json` stdout → parse JSON directly

### 2. `src/config.ts`

Add to `TranslatorConfig`:
```typescript
ytdlpUrl?: string | undefined;
uvxExe: string;            // uvx executable (default: "uvx")
ytdlpAudioFormat: string;  // audio format (default: "best" = native format)
```

Add to `DEFAULT_CONFIG`:
```typescript
uvxExe: "uvx",
ytdlpAudioFormat: "best",
```

### 3. `src/cli.ts`

**New flags** in `parseArgs`:
```typescript
"ytdlp":              { type: "string" },
"uvx-exe":            { type: "string" },
"ytdlp-audio-format": { type: "string" },
```

**Config assignment** (`--input` stays required, no change):
```typescript
ytdlpUrl:         values.ytdlp as string | undefined,
uvxExe:           (values["uvx-exe"] as string) ?? DEFAULT_CONFIG.uvxExe,
ytdlpAudioFormat: (values["ytdlp-audio-format"] as string) ?? DEFAULT_CONFIG.ytdlpAudioFormat,
```

**New Step 0** (insert before the existing "Step 1: Discover audio tracks"):
```typescript
// ── Step 0: yt-dlp audio download ────────────────────────────────────────

let ytdlpMeta: YtdlpMetadata | undefined;

if (config.ytdlpUrl) {
  // Cache: if .info.json already exists in --input, skip re-download
  const infoFiles = (await fs.readdir(config.inputDir).catch(() => [])).filter(e => e.endsWith(".info.json"));
  if (infoFiles.length > 0) {
    console.log(`[yt-dlp] Audio already present in ${config.inputDir}, fetching metadata only`);
    ytdlpMeta = await fetchYtdlpMetadata(config.ytdlpUrl, config.uvxExe);
  } else {
    ytdlpMeta = await downloadYtdlpAudio(config.ytdlpUrl, config.inputDir, config.uvxExe, config.ytdlpAudioFormat);
  }
}
```

**Metadata branch** — add as `else if` after the existing DLSite block (after line ~380, before `else if (config.metadataFile)`):
```typescript
} else if (config.ytdlpUrl && ytdlpMeta) {
  if (await tryLoadCachedMetadata()) {
    // use cached metadata.json
  } else {
    const hasMetaModel = config.metaModelPath || config.metaServerUrl || config.metaHfRepo;
    if (ytdlpMeta.metadataMd && hasMetaModel) {
      // Full LLM extraction — exact same code as DLSite block
      const fileList = tracks.map(t => `- ${t.relativePath}`).join("\n");
      const fullMd = ytdlpMeta.metadataMd + `\n# File List\n\n${fileList}\n`;
      const metaServer = new LlamaServerManager({
        llamaServerExe: config.llamaServerExe,
        modelPath: config.metaModelPath ?? "",
        hfRepo: config.metaHfRepo,
        serverPort: config.metaServerPort,
        gpuLayers: config.gpuLayers,
        contextSize: config.metaContextSize,
        parallel: 1,
        serverUrl: config.metaServerUrl,
      }, "MetaServer");
      try {
        await metaServer.start();
        const metaClient = new LlmClient(metaServer.baseUrl, { ...config, temperature: config.metaTemperature, repeatPenalty: 1.0 });
        const extractor = new MetadataExtractor(metaClient, config.locale, config.seed);
        const result = await extractor.extract(fullMd);
        glossary = result.glossary;
        outputMetadata = result.metadata;
      } finally {
        await metaServer.stop();
      }
    } else if (ytdlpMeta.metadataMd) {
      console.log(`[Metadata] No --meta-model. Using yt-dlp scraped metadata only.`);
      outputMetadata = {
        title: ytdlpMeta.title,
        summary: ytdlpMeta.description,
        glossary: {
          cvs: ytdlpMeta.uploader ? [{ ja: ytdlpMeta.uploader, zh: ytdlpMeta.uploader }] : [],
          characters: [],
          terms: [],
        },
      } satisfies UserMetadata;
      glossary = {
        cvs: (outputMetadata as UserMetadata).glossary?.cvs ?? [],
        characters: [],
        terms: [],
        summary: ytdlpMeta.description ?? "",
      };
    }
  }
}
```

Add imports at top of `cli.ts`:
```typescript
import { fetchYtdlpMetadata, downloadYtdlpAudio, type YtdlpMetadata } from "./pipeline/fetch-ytdlp-metadata.js";
```

**Update `printUsage`** (Metadata section):
```
--ytdlp <url>                  Download audio into --input dir and extract metadata via yt-dlp
--uvx-exe <path>               uvx executable (default: uvx in PATH)
--ytdlp-audio-format <format>  Audio format for yt-dlp extraction (default: best — native format)
```

### 4. `README.md`

- Add `--ytdlp`, `--uvx-exe`, `--ytdlp-audio-format` rows to the Metadata flags table
- Note: requires `uv` installed (provides `uvx`); yt-dlp installed automatically on first run
- Add a "Suggested Settings (yt-dlp Source)" usage example

## Existing Components Reused Unchanged

- `MetadataExtractor`, `LlamaServerManager`, `LlmClient` — exact same calls as DLSite path
- `tryLoadCachedMetadata()` — unchanged
- `metadataMd` markdown format — identical shape to DLSite; LLM/glossary downstream logic untouched

## Output Structure

```
<input>/
  <id>.<ext>       # downloaded audio (native best-audio format)
  <id>.info.json   # yt-dlp metadata JSON (used as cache marker on re-run)
<output>/
  metadata.json
  <track-dir>/
    <stem>.transcription.json
    <stem>.translation.json
    <stem>.srt
```

## Verification

1. `uvx yt-dlp --dump-json --no-playlist <url>` manually → confirm JSON has `title`, `description`, `uploader`
2. `--ytdlp <url> --input <dir> --output <out>` → audio + `.info.json` appear in `<dir>`, `metadata.json` written with description as summary
3. Re-run same command → detects `.info.json` in `<dir>`, skips download, fetches metadata only
4. With `--meta-hf-repo <repo>` → full LLM glossary extraction from description
