import fs from "fs/promises";
import path from "path";
import type { TranscriptSegment, TranslationEntry, FinalMetadata, UserMetadata } from "../util/types.js";
import type { WindowResult } from "./translator.js";
import { toLrc, toVtt } from "./subtitle-exporter.js";

/** Ensure a directory exists, creating it recursively if needed. */
async function ensureDir(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

/** Write the cleaned transcription for a single track. */
export async function writeTranscription(
  outputDir: string,
  relativeDir: string,
  stem: string,
  segments: TranscriptSegment[],
): Promise<void> {
  const dir = path.join(outputDir, relativeDir);
  await ensureDir(dir);
  await fs.writeFile(
    path.join(dir, `${stem}.transcription.json`),
    JSON.stringify({ segments }, null, 2),
    "utf-8",
  );
}

/** Write the translation output for a single track (JSON + LRC + VTT). */
export async function writeTranslation(
  outputDir: string,
  relativeDir: string,
  stem: string,
  entries: TranslationEntry[],
): Promise<void> {
  const dir = path.join(outputDir, relativeDir);
  await ensureDir(dir);

  await Promise.all([
    fs.writeFile(
      path.join(dir, `${stem}.translation.json`),
      JSON.stringify(entries, null, 2),
      "utf-8",
    ),
    fs.writeFile(
      path.join(dir, `${stem}.lrc`),
      toLrc(entries),
      "utf-8",
    ),
    fs.writeFile(
      path.join(dir, `${stem}.vtt`),
      toVtt(entries),
      "utf-8",
    ),
  ]);
}

/** Write per-window LLM intermediates for debugging (raw output + parsed entries). */
export async function writeWindowResults(
  outputDir: string,
  relativeDir: string,
  stem: string,
  windowResults: WindowResult[],
): Promise<void> {
  const dir = path.join(outputDir, relativeDir);
  await ensureDir(dir);
  await fs.writeFile(
    path.join(dir, `${stem}.windows.json`),
    JSON.stringify(windowResults, null, 2),
    "utf-8",
  );
}

/** Write metadata.json at the output root. */
export async function writeMetadata(
  outputDir: string,
  metadata: FinalMetadata | UserMetadata,
): Promise<void> {
  await ensureDir(outputDir);
  await fs.writeFile(
    path.join(outputDir, "metadata.json"),
    JSON.stringify(metadata, null, 2),
    "utf-8",
  );
}
