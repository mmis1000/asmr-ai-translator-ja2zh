import fs from "fs/promises";
import path from "path";
import type { TranslatorConfig } from "../config.js";
import type { LlmClient } from "../server/llm-client.js";
import { buildChatPrompt, extractJsonArray } from "../server/llm-client.js";
import type { Segment, TranscriptSegment, GlossaryLang, TranslationEntry, TranslationEchoEntry } from "../util/types.js";
import {
  getPromptBuilder,
  formatTranscriptionJson,
  filterGlossary,
  buildGlossaryJson,
} from "./prompt-builder.js";
import { makeInferenceWindows } from "./windowing.js";

const MAX_RETRIES = 2;

/**
 * Convert cleaned ASR segments into the flat Segment[] format used by prompts.
 * Timestamps are converted from seconds to milliseconds.
 */
export function asrToSegments(cleaned: TranscriptSegment[]): Segment[] {
  return cleaned.map((seg, i) => ({
    id: i + 1,
    text: seg.text,
    start: Math.round(seg.start_time * 1000),
    end: Math.round(seg.end_time * 1000),
  }));
}

/**
 * Translate a single track's segments using windowed LLM inference.
 * Returns an array of translation entries with global (not window-local) IDs.
 */
export async function translateTrack(
  segments: Segment[],
  glossary: GlossaryLang,
  trackName: string,
  config: TranslatorConfig,
  client: LlmClient,
  grammar: string,
): Promise<TranslationEntry[]> {
  const promptBuilder = getPromptBuilder(config.locale, config.mode);

  const windows = makeInferenceWindows(segments, (segs) => {
    const jaText = segs.map(s => s.text).join("");
    const filtered = filterGlossary(glossary, jaText);
    return promptBuilder(trackName, glossary.summary, buildGlossaryJson(filtered), "[]").length;
  });

  if (windows.length === 0) {
    console.log(`  [translate] No windows generated (too few segments)`);
    return [];
  }

  console.log(`  [translate] ${windows.length} window(s) for "${trackName}"`);

  const allEntries: TranslationEntry[] = [];

  for (let wi = 0; wi < windows.length; wi++) {
    const win = windows[wi]!;
    const windowJaText = win.segments.map(s => s.text).join("");
    const filtered = filterGlossary(glossary, windowJaText);
    const glossaryJson = buildGlossaryJson(filtered);
    const transcriptionJson = formatTranscriptionJson(win.segments);
    const userPrompt = promptBuilder(trackName, glossary.summary, glossaryJson, transcriptionJson);
    const prompt = buildChatPrompt(userPrompt);

    let parsed: TranslationEntry[] | null = null;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const raw = await client.complete(prompt, { grammar });
        const jsonStr = extractJsonArray(raw);
        parsed = JSON.parse(jsonStr) as TranslationEntry[];
        break;
      } catch (err: any) {
        console.error(`  [translate] Window ${wi + 1}/${windows.length} attempt ${attempt + 1} failed: ${err.message}`);
        if (attempt === MAX_RETRIES) {
          console.error(`  [translate] Giving up on window ${wi + 1} after ${MAX_RETRIES + 1} attempts`);
        }
      }
    }

    if (!parsed) continue;

    // Restore global IDs from window-local IDs
    for (const entry of parsed) {
      entry.ids = entry.ids.map(localId => win.idMap.get(localId) ?? localId);
      allEntries.push(entry);
    }
  }

  return allEntries;
}
