import { execFile } from "child_process";
import { promisify } from "util";
import { readFile, readdir } from "fs/promises";
import { join } from "path";

const execFileAsync = promisify(execFile);

export interface YtdlpMetadata {
  id: string;
  title?: string;
  uploader?: string;
  description?: string;
  tags?: string[];
  webpageUrl?: string;
  /** Formatted markdown for LLM extraction — same shape as DlsiteMetadata.metadataMd. */
  metadataMd: string;
}

function ytdlpUvArgs(asrProjectRoot: string, ytDlpArgs: string[]): string[] {
  return ["run", "--project", asrProjectRoot, "yt-dlp", ...ytDlpArgs];
}

/**
 * Download audio from url into downloadDir.
 * Uses --write-info-json to write id.info.json alongside the audio.
 * Returns parsed metadata.
 */
export async function downloadYtdlpAudio(
  url: string,
  downloadDir: string,
  uvExe: string,
  asrProjectRoot: string,
  audioFormat: string,
): Promise<YtdlpMetadata> {
  console.log(`[yt-dlp] Downloading audio from ${url} → ${downloadDir}`);

  const ytArgs = [
    "-x",
    ...(audioFormat !== "best" ? ["--audio-format", audioFormat] : []),
    "--write-info-json",
    "--no-playlist",
    "--paths",
    downloadDir,
    "-o",
    "%(id)s.%(ext)s",
    url,
  ];

  await execFileAsync(uvExe, ytdlpUvArgs(asrProjectRoot, ytArgs), {
    maxBuffer: 10 * 1024 * 1024,
  });

  const entries = await readdir(downloadDir);
  const infoFile = entries.find((e) => e.endsWith(".info.json"));
  if (!infoFile) {
    throw new Error("[yt-dlp] No .info.json found after download");
  }

  const text = await readFile(join(downloadDir, infoFile), "utf-8");
  const raw: Record<string, unknown> = JSON.parse(text);
  return parseYtdlpInfo(raw);
}

/**
 * Fetch metadata only (no download) — used when audio is already cached.
 */
export async function fetchYtdlpMetadata(
  url: string,
  uvExe: string,
  asrProjectRoot: string,
): Promise<YtdlpMetadata> {
  console.log(`[yt-dlp] Fetching metadata from ${url}`);
  const { stdout } = await execFileAsync(
    uvExe,
    ytdlpUvArgs(asrProjectRoot, ["--dump-json", "--no-playlist", url]),
    { maxBuffer: 50 * 1024 * 1024 },
  );
  const raw: Record<string, unknown> = JSON.parse(stdout);
  return parseYtdlpInfo(raw);
}

function strField(obj: Record<string, unknown>, key: string): string | undefined {
  const v = obj[key];
  return typeof v === "string" ? v : undefined;
}

function strArrayField(obj: Record<string, unknown>, key: string): string[] | undefined {
  const v = obj[key];
  if (!Array.isArray(v)) return undefined;
  const out: string[] = [];
  for (const x of v) {
    if (typeof x === "string") out.push(x);
  }
  return out.length ? out : undefined;
}

export function parseYtdlpInfo(info: Record<string, unknown>): YtdlpMetadata {
  const id = strField(info, "id") ?? "unknown";
  const title = strField(info, "title");
  const uploader =
    strField(info, "uploader") ?? strField(info, "channel") ?? strField(info, "creator");
  const description = strField(info, "description");
  const tags = strArrayField(info, "tags");
  const webpageUrl = strField(info, "webpage_url");

  let metadataMd = "";
  if (title) metadataMd += `# Title\n\n${title}\n\n`;
  if (uploader) metadataMd += `# Creator\n\n${uploader}\n\n`;
  if (tags?.length) metadataMd += `# Tags\n\n${tags.join(", ")}\n\n`;
  if (description) metadataMd += `# Description\n\n${description}\n\n`;

  console.log(`[yt-dlp] ${id}: title="${title}", uploader="${uploader}"`);
  const result: YtdlpMetadata = { id, metadataMd };
  if (title !== undefined) result.title = title;
  if (uploader !== undefined) result.uploader = uploader;
  if (description !== undefined) result.description = description;
  if (tags !== undefined) result.tags = tags;
  if (webpageUrl !== undefined) result.webpageUrl = webpageUrl;
  return result;
}
