import type { Segment } from "../util/types.js";
import { formatTranscriptionJson } from "./prompt-builder.js";

// Character-budget windowing: grow each window until the estimated total
// character count (fixed prompt overhead + transcription JSON) exceeds maxChars.
//
// The training pipeline (instruct-dataset-pipeline.ts) uses MAX_CHARS=5000 for the
// COMBINED input+output budget per window (~3500 tokens at ~1.4 chars/token).
// At inference we only count input here. The fine-tune target is total seq len 4096
// (prompt + completion), so budgets here stay conservative even if llama-server
// uses n_ctx=8192.
//
// Base: ~1.6 chars/token → rough input-side budget; completion stays under the 4096
// training cap via translator.ts maxNPredict.
//
// Echo: real runs often land ~2–2.3k tokens total (prompt + completion) vs 4096 train
// length — under-filled. Use a **larger** char budget than base so each window carries
// more segments and moves closer to the training distribution (fewer HTTP round-trips).
export const MAX_CHARS_BASE = 2500;
export const MAX_CHARS_ECHO = 3500;
const MIN_WINDOW = 3;

export interface InferenceWindow {
  /** Segments with IDs renumbered from 1 within this window. */
  segments: Segment[];
  /** Map from window-local ID → global segment ID. */
  idMap: Map<number, number>;
}

/**
 * Slice segments into character-budget windows for inference.
 * IDs are renumbered from 1 within each window (matching training format).
 *
 * @param allSegments  All cleaned transcript segments with global IDs.
 * @param getOverheadChars  Returns the prompt header length for a given set of segments
 *                          (glossary + fixed rules). Called per candidate slice so that
 *                          the filtered glossary is used for sizing.
 */
export function makeInferenceWindows(
  allSegments: Segment[],
  getOverheadChars: (segs: Segment[]) => number,
  maxChars = MAX_CHARS_BASE,
  minWindow = MIN_WINDOW,
): InferenceWindow[] {
  const windows: InferenceWindow[] = [];

  function sliceChars(start: number, end: number): number {
    const slice = allSegments.slice(start, end);
    return getOverheadChars(slice) + formatTranscriptionJson(slice).length;
  }

  let wiStart = 0;
  while (wiStart < allSegments.length) {
    // Must have at least minWindow segments left
    if (wiStart + minWindow > allSegments.length) break;

    // Start with minWindow, then greedily grow
    let wiEnd = wiStart + minWindow;
    while (wiEnd < allSegments.length) {
      if (sliceChars(wiStart, wiEnd + 1) > maxChars) break;
      wiEnd++;
    }

    const rawSegs = allSegments.slice(wiStart, wiEnd);

    // Renumber IDs from 1 within window, store reverse mapping
    const idMap = new Map<number, number>(); // local → global
    const segments: Segment[] = rawSegs.map((s, i) => {
      const localId = i + 1;
      idMap.set(localId, s.id);
      return { ...s, id: localId };
    });

    windows.push({ segments, idMap });
    wiStart = wiEnd;
  }

  return windows;
}
