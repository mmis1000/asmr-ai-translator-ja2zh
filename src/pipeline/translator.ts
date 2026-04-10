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
import { makeInferenceWindows, MAX_CHARS_BASE, MAX_CHARS_ECHO } from "./windowing.js";
import { generateTranslationGrammar } from "./grammar-generator.js";

const MAX_RETRIES = 3;

/** Minimum text length to be considered a repeated collapse pattern. */
const COLLAPSE_MIN_LEN = 5;
/** Number of consecutive entries with identical text before declaring collapse. */
const COLLAPSE_THRESHOLD = 3;
/**
 * Above this length a sentence is considered "long and complex". Even 2
 * distinct inputs mapping to the same output is almost certainly a collapse.
 */
const COLLAPSE_LONG_LEN = 15;
/** Stricter threshold applied to long/complex outputs. */
const COLLAPSE_LONG_THRESHOLD = 2;

/**
 * CJK-only name (1–10 chars) followed by a full-width or half-width colon.
 * Matches at the very start of a translated string.
 * The ASR source never contains speaker labels, so any match is a hallucination.
 */
const SPEAKER_PREFIX_RE = /^([\u3040-\u9FFF\uF900-\uFAFF]{1,10}[：:])\s*/u;

/**
 * Strip spurious "Name：" prefixes from translated entries.
 *
 * The source ASR has no speaker labels, so a CJK-name colon at the start of
 * any translated text is always a model hallucination. We collect every
 * distinct prefix that appears at least once and strip it from all entries.
 */
function stripSpeakerPrefixes(entries: TranslationEntry[]): TranslationEntry[] {
  const toStrip = new Set<string>();
  for (const e of entries) {
    if (e.text == null) continue;
    const m = e.text.match(SPEAKER_PREFIX_RE);
    if (m) toStrip.add(m[1]!);
  }

  if (toStrip.size === 0) return entries;

  console.log(`  [translate] Stripping speaker prefixes: ${[...toStrip].join(", ")}`);

  return entries.map(e => {
    if (e.text == null) return e;
    const m = e.text.match(SPEAKER_PREFIX_RE);
    if (m && toStrip.has(m[1]!)) {
      return { ...e, text: e.text.slice(m[0].length) };
    }
    return e;
  });
}

/** Bump temperature by this amount on the first retry. */
const TEMPERATURE_BUMP = 0.6;

/** Tokens budgeted per segment for output generation. */
const N_PREDICT_PER_SEGMENT = 512;

/**
 * Rough chars per token for the built prompt string. `length/1.6` over-counts tokens vs
 * llama.cpp’s tokenizer on JP+JSON (e.g. ~1763 tok for ~3.5k chars → ~2.0 chars/tok),
 * which made `roomTrain` / `byRatio` too tight.
 */
const CHARS_PER_TOKEN_EST = 2.0;

/**
 * Max new tokens vs estimated prompt size — stops runaway loops without capping sane
 * echo JSON (long `input` + translation). ~3.5× keeps headroom above typical completion.
 */
const MAX_GEN_TO_PROMPT_RATIO = 3.5;

const TRAINING_CONTEXT_TOKENS = 4096;

function estimatePromptTokens(prompt: string): number {
  return Math.max(1, Math.ceil(prompt.length / CHARS_PER_TOKEN_EST));
}

function computeNPredict(
  segments: Segment[],
  prompt: string,
  config: TranslatorConfig,
): number {
  const promptTok = estimatePromptTokens(prompt);
  const bySegments = segments.length * N_PREDICT_PER_SEGMENT;
  const byRatio = Math.ceil(promptTok * MAX_GEN_TO_PROMPT_RATIO);
  const roomCtx = Math.max(0, config.contextSize - promptTok);
  const roomTrain = Math.max(0, TRAINING_CONTEXT_TOKENS - promptTok);
  const n = Math.min(bySegments, byRatio, roomCtx, roomTrain);
  return Math.max(1, n);
}

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
    ...(seg.vocal_energy !== undefined ? { vocal_energy: seg.vocal_energy } : {}),
    ...(seg.other_energy !== undefined ? { other_energy: seg.other_energy } : {}),
    ...(seg.snr !== undefined ? { snr: seg.snr } : {}),
  } as Segment));
}

/**
 * Detect attention collapse: when the LLM hallucinates by outputting the same
 * text for many distinct source inputs, whether consecutive or interleaved.
 *
 * Builds a map of outputText → Set<inputText>. If any output text (longer than
 * `COLLAPSE_MIN_LEN`) is generated for `COLLAPSE_THRESHOLD` or more distinct
 * inputs, it is considered a collapse.
 *
 * Legitimate repeated translations (same input → same output, e.g. refrain
 * lines) are excluded by only counting *distinct* inputs per output.
 */
export function detectAttentionCollapse(entries: TranslationEntry[], segments: Segment[]): void {
  // Build id → source text map for fast lookup
  const inputByLocalId = new Map<number, string>();
  for (const seg of segments) inputByLocalId.set(seg.id, seg.text);

  // outputText → set of distinct source inputs that produced it
  const distinctInputsPerOutput = new Map<string, Set<string>>();

  for (const entry of entries) {
    const outputText = entry.text;
    if (outputText === null || outputText.length <= COLLAPSE_MIN_LEN) continue;

    const inputText = inputByLocalId.get(entry.ids[0]!) ?? "";

    let inputSet = distinctInputsPerOutput.get(outputText);
    if (inputSet === undefined) {
      inputSet = new Set<string>();
      distinctInputsPerOutput.set(outputText, inputSet);
    }
    inputSet.add(inputText);

    const threshold = outputText.length > COLLAPSE_LONG_LEN ? COLLAPSE_LONG_THRESHOLD : COLLAPSE_THRESHOLD;
    if (inputSet.size >= threshold) {
      throw new Error(
        `Attention collapse detected: "${outputText.slice(0, 40)}…" produced for ${inputSet.size} distinct inputs`,
      );
    }
  }
}

/**
 * Build the prompt, call the LLM, parse JSON, and run collapse detection.
 * Returns parsed entries with **window-local** IDs (callers restore global IDs).
 */
async function runLLMCore(
  segments: Segment[],
  trackName: string,
  glossary: GlossaryLang,
  promptBuilder: ReturnType<typeof getPromptBuilder>,
  grammar: string,
  client: LlmClient,
  config: TranslatorConfig,
  temperature?: number | undefined,
  seed?: number | undefined,
  label?: string | undefined,
): Promise<TranslationEntry[]> {
  const filtered = filterGlossary(glossary, segments.map(s => s.text).join(""));
  const glossaryJson = buildGlossaryJson(filtered);
  const transcriptionJson = formatTranscriptionJson(segments);
  const userPrompt = promptBuilder(trackName, glossary.summary, glossaryJson, transcriptionJson);
  const prompt = buildChatPrompt(userPrompt);
  const nPredict = computeNPredict(segments, prompt, config);

  const raw = await client.complete(prompt, {
    grammar,
    nPredict,
    ...(temperature !== undefined ? { temperature } : {}),
    ...(seed !== undefined ? { seed } : {}),
    ...(label !== undefined ? { label } : {}),
  });
  const jsonStr = extractJsonArray(raw);
  const parsed = JSON.parse(jsonStr) as TranslationEntry[];
  detectAttentionCollapse(parsed, segments);
  return parsed;
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

  const windowCharBudget = config.mode === "echo" ? MAX_CHARS_ECHO : MAX_CHARS_BASE;
  const windows = makeInferenceWindows(segments, (segs) => {
    const jaText = segs.map(s => s.text).join("");
    const filtered = filterGlossary(glossary, jaText);
    return promptBuilder(trackName, glossary.summary, buildGlossaryJson(filtered), "[]").length;
  }, windowCharBudget);

  if (windows.length === 0) {
    console.log(`  [translate] No windows generated (too few segments)`);
    return { entries: [], windowResults: [] };
  }

  console.log(`  [translate] ${windows.length} window(s) for "${trackName}"`);

  const allEntries: TranslationEntry[] = [];
  const windowResults: WindowResult[] = [];

  for (let wi = 0; wi < windows.length; wi++) {
    const win = windows[wi]!;
    const grammar = generateTranslationGrammar(win.segments, config.mode);

    let parsed: TranslationEntry[] | null = null;
    let lastRaw: string | null = null;
    let lastError: string | undefined;
    let attempts = 0;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      attempts = attempt + 1;
      try {
        if (attempt === 0) {
          // ── Attempt 1: standard run (seed for reproducibility if configured)
          const localParsed = await runLLMCore(
            win.segments, trackName, glossary, promptBuilder, grammar, client,
            config, undefined, config.seed, `translation-${trackName}-window${wi + 1}-attempt1`
          );
          parsed = localParsed.map(entry => ({
            ...entry,
            ids: entry.ids.map(localId => win.idMap.get(localId) ?? localId),
          }));

        } else if (attempt === 1) {
          // ── Attempt 2: temperature bump fallback ──────────────────────────
          const bumpedTemp = (config.temperature ?? 0.3) + TEMPERATURE_BUMP;
          console.warn(`  [translate] Window ${wi + 1} retry with temperature ${bumpedTemp.toFixed(2)}`);
          const localParsed = await runLLMCore(
            win.segments, trackName, glossary, promptBuilder, grammar, client,
            config, bumpedTemp, config.seed, `translation-${trackName}-window${wi + 1}-attempt2`
          );
          parsed = localParsed.map(entry => ({
            ...entry,
            ids: entry.ids.map(localId => win.idMap.get(localId) ?? localId),
          }));

        } else if (attempt === 2) {
          // ── Attempt 3: sub-chunking fallback ──────────────────
          console.warn(`  [translate] Window ${wi + 1} retry with sub-chunking`);
          const rawSegs = win.segments;
          const mid = Math.ceil(rawSegs.length / 2);
          const chunkA = rawSegs.slice(0, mid);
          const chunkB = rawSegs.slice(mid);

          const chunkAGrammar = generateTranslationGrammar(chunkA, config.mode);
          const chunkBGrammar = generateTranslationGrammar(chunkB, config.mode);

          const [resultA, resultB] = await Promise.all([
            runLLMCore(chunkA, trackName, glossary, promptBuilder, chunkAGrammar, client, config, undefined, config.seed, `translation-${trackName}-window${wi + 1}-attempt3-sub1`),
            runLLMCore(chunkB, trackName, glossary, promptBuilder, chunkBGrammar, client, config, undefined, config.seed, `translation-${trackName}-window${wi + 1}-attempt3-sub2`),
          ]);

          const restoreIds = (entries: TranslationEntry[]) =>
            entries.map(entry => ({
              ...entry,
              ids: entry.ids.map(localId => win.idMap.get(localId) ?? localId),
            }));

          parsed = [...restoreIds(resultA), ...restoreIds(resultB)];

        } else {
          // ── Attempt 4: micro-chunking (1–2 segments at a time) ────────
          // Best-effort: translate each micro-chunk individually and collect
          // whatever succeeds. Segments that still fail are omitted.
          console.warn(`  [translate] Window ${wi + 1} retry with micro-chunking (1-2 segments)`);
          const MICRO_SIZE = 2;
          const collected: TranslationEntry[] = [];
          let microFailed = 0;

          for (let mi = 0; mi < win.segments.length; mi += MICRO_SIZE) {
            const micro = win.segments.slice(mi, mi + MICRO_SIZE);
            const microGrammar = generateTranslationGrammar(micro, config.mode);
            try {
              const microResult = await runLLMCore(
                micro, trackName, glossary, promptBuilder, microGrammar, client, config, undefined, config.seed, `translation-${trackName}-window${wi + 1}-attempt4-sub${Math.floor(mi / MICRO_SIZE) + 1}`
              );
              for (const entry of microResult) {
                collected.push({
                  ...entry,
                  ids: entry.ids.map(localId => win.idMap.get(localId) ?? localId),
                });
              }
            } catch (microErr: unknown) {
              microFailed++;
              const microMsg = microErr instanceof Error ? microErr.message : String(microErr);
              console.warn(`  [translate] Micro-chunk [${mi + 1}–${mi + micro.length}] failed, falling back to original text: ${microMsg}`);
              
              for (const m of micro) {
                const entry = {
                  ids: [win.idMap.get(m.id) ?? m.id],
                  text: m.text,
                  start: m.start,
                  end: m.end,
                } as TranslationEntry & { input?: string };
                
                if (config.mode === "echo") {
                  entry.input = m.text;
                }
                collected.push(entry);
              }
            }
          }

          if (microFailed > 0) {
            console.warn(`  [translate] Micro-chunking: ${microFailed} chunk(s) defaulted to original text in window ${wi + 1}`);
          }
          // Even if some micro-chunks failed, treat this attempt as successful
          // with whatever we collected (may be empty if everything failed).
          parsed = collected;
        }

        break; // success — exit retry loop
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        lastError = msg;
        console.error(`  [translate] Window ${wi + 1}/${windows.length} attempt ${attempts} failed: ${msg}`);
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
      winResult.parsed = parsed;
      for (const entry of parsed) {
        allEntries.push(entry);
      }
    }

    windowResults.push(winResult);
  }

  return { entries: stripSpeakerPrefixes(allEntries), windowResults };
}
