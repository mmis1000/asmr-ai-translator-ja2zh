import type { TranscriptSegment, TranscriptFile } from "../util/types.js";

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

/** Filter garbled segments from a transcript, returning only clean ones. */
export function cleanTranscript(transcript: TranscriptFile): TranscriptSegment[] {
  if (transcript.segments && transcript.segments.length > 0) {
    return transcript.segments.filter(s => !isGarbled(s));
  }
  return [];
}
