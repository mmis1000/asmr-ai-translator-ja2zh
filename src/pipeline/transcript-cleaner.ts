import type { TranscriptSegment, TranscriptFile } from "../util/types.js";

/**
 * Re-split a single ASR segment at internal word-level gaps >= threshold.
 * Whisper often groups multiple utterances into one ~30s chunk when the audio
 * has long silences mid-segment. Word timestamps let us recover the boundaries.
 *
 * Quality metrics (avg_logprob, compression_ratio, no_speech_prob) are
 * inherited from the parent segment since we cannot recompute them.
 */
function resplitSegment(seg: TranscriptSegment, gapThresholdSec: number): TranscriptSegment[] {
  const words = seg.words;
  if (!words || words.length === 0) return [seg];

  const groups: typeof words[] = [];
  let current: typeof words = [words[0]!];

  for (let i = 1; i < words.length; i++) {
    const gap = words[i]!.start_time - words[i - 1]!.end_time;
    if (gap >= gapThresholdSec) {
      groups.push(current);
      current = [words[i]!];
    } else {
      current.push(words[i]!);
    }
  }
  groups.push(current);

  if (groups.length === 1) return [seg];

  return groups.map(grp => ({
    text: grp.map(w => w.text).join(""),
    start_time: grp[0]!.start_time,
    end_time: grp[grp.length - 1]!.end_time,
    words: grp,
    avg_logprob: seg.avg_logprob,
    compression_ratio: seg.compression_ratio,
    no_speech_prob: seg.no_speech_prob,
  }));
}

/**
 * Detect garbled/hallucinated ASR segments using Whisper quality metrics
 * and content heuristics. Ported from the parent project's transcript-match-pipeline.
 */
export function isGarbled(seg: TranscriptSegment): boolean {
  // Quality metrics from Whisper segments[]
  if (seg.no_speech_prob > 0.5) return true;
  if (seg.avg_logprob < -1.0) return true;
  if (seg.compression_ratio > 2.4) return true;

  const text = seg.text;

  // First-segment hallucination: very long at time 0
  if (seg.start_time === 0 && seg.end_time > 15 && text.length > 150) return true;

  // Excessive bigram repetition (sound-effect loops like ピュッ×30)
  const bigrams = new Map<string, number>();
  for (let i = 0; i < text.length - 1; i++) {
    const bg = text.slice(i, i + 2);
    bigrams.set(bg, (bigrams.get(bg) ?? 0) + 1);
  }
  if (bigrams.size > 0 && Math.max(...bigrams.values()) > 10) return true;

  // High ASCII symbol ratio (garbled encodings)
  const symbols = (text.match(/[!-/:-@[-`{-~]/g) ?? []).length;
  if (text.length > 0 && symbols / text.length > 0.25) return true;

  // Too short after stripping punctuation and whitespace
  const effective = text.replace(/[\s\p{P}\p{S}]/gu, "");
  if (effective.length < 3) return true;

  return false;
}

/**
 * Filter garbled segments and re-split at internal gaps.
 * Order: resplit first (exposes boundaries), then filter garbled sub-segments.
 */
export function cleanTranscript(
  transcript: TranscriptFile,
  resplitGapSec = 1.0,
): TranscriptSegment[] {
  if (!transcript.segments || transcript.segments.length === 0) return [];

  return transcript.segments
    .flatMap(s => resplitSegment(s, resplitGapSec))
    .filter(s => !isGarbled(s));
}
