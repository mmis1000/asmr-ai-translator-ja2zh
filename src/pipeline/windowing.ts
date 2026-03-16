import type { Segment } from "../util/types.js";
import { formatTranscriptionJson } from "./prompt-builder.js";

// Character-budget windowing: grow each window until the estimated total
// character count (fixed prompt overhead + transcription JSON) exceeds MAX_CHARS.
//
// Qwen3 tokenizer: CJK ≈ 2 chars/token → MAX_CHARS=5000 ≈ 2500 tokens,
// leaving headroom inside max_seq_len=4096.
const MAX_CHARS = 5000;
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
  maxChars = MAX_CHARS,
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
