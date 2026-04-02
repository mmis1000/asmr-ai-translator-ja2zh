import type { TranslatorConfig } from "../config.js";
import type { LlmClient } from "../server/llm-client.js";
import { buildChatPrompt, extractJsonArray } from "../server/llm-client.js";
import type { Segment, TranscriptSegment, GlossaryLang, TranslationEntry } from "../util/types.js";
import {
  getPromptBuilder,
  formatTranscriptionJson,
  filterGlossary,
  buildGlossaryJson,
} from "./prompt-builder.js";
import { makeInferenceWindows } from "./windowing.js";
import { generateTranslationGrammar } from "./grammar-generator.js";

const MAX_RETRIES = 2;

export interface WindowResult {
  /** 1-based window index */
  index: number;
  segmentCount: number;
  /** Number of attempts made (1 = succeeded first try) */
  attempts: number;
  /** Raw LLM output string (last successful attempt, or last failed attempt) */
  rawLlm: string | null;
  /** Parsed translation entries with global IDs restored; null if all attempts failed */
  parsed: TranslationEntry[] | null;
  /** Error message from last attempt if all failed */
  error?: string | undefined;
}

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
 * Returns translation entries (global IDs) and per-window intermediates.
 */
export async function translateTrack(
  segments: Segment[],
  glossary: GlossaryLang,
  trackName: string,
  config: TranslatorConfig,
  client: LlmClient,
): Promise<{ entries: TranslationEntry[]; windowResults: WindowResult[] }> {
  const promptBuilder = getPromptBuilder(config.locale, config.mode);

  const windows = makeInferenceWindows(segments, (segs) => {
    const jaText = segs.map(s => s.text).join("");
    const filtered = filterGlossary(glossary, jaText);
    return promptBuilder(trackName, glossary.summary, buildGlossaryJson(filtered), "[]").length;
  });

  if (windows.length === 0) {
    console.log(`  [translate] No windows generated (too few segments)`);
    return { entries: [], windowResults: [] };
  }

  console.log(`  [translate] ${windows.length} window(s) for "${trackName}"`);

  const allEntries: TranslationEntry[] = [];
  const windowResults: WindowResult[] = [];

  for (let wi = 0; wi < windows.length; wi++) {
    const win = windows[wi]!;
    const windowJaText = win.segments.map(s => s.text).join("");
    const filtered = filterGlossary(glossary, windowJaText);
    const glossaryJson = buildGlossaryJson(filtered);
    const transcriptionJson = formatTranscriptionJson(win.segments);
    const userPrompt = promptBuilder(trackName, glossary.summary, glossaryJson, transcriptionJson);
    const prompt = buildChatPrompt(userPrompt);
    const grammar = generateTranslationGrammar(win.segments, config.mode);

    let parsed: TranslationEntry[] | null = null;
    let lastRaw: string | null = null;
    let lastError: string | undefined;
    let attempts = 0;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      attempts = attempt + 1;
      try {
        const raw = await client.complete(prompt, { grammar });
        lastRaw = raw;
        const jsonStr = extractJsonArray(raw);
        parsed = JSON.parse(jsonStr) as TranslationEntry[];
        break;
      } catch (err: any) {
        lastError = err.message;
        console.error(`  [translate] Window ${wi + 1}/${windows.length} attempt ${attempts} failed: ${err.message}`);
        if (attempt === MAX_RETRIES) {
          console.error(`  [translate] Giving up on window ${wi + 1} after ${attempts} attempts`);
        }
      }
    }

    const winResult: WindowResult = {
      index: wi + 1,
      segmentCount: win.segments.length,
      attempts,
      rawLlm: lastRaw,
      parsed: null,
      ...(parsed ? {} : { error: lastError }),
    };

    if (parsed) {
      // Restore global IDs from window-local IDs
      const globalParsed = parsed.map(entry => ({
        ...entry,
        ids: entry.ids.map(localId => win.idMap.get(localId) ?? localId),
      }));
      winResult.parsed = globalParsed;
      for (const entry of globalParsed) {
        allEntries.push(entry);
      }
    }

    windowResults.push(winResult);
  }

  return { entries: allEntries, windowResults };
}
