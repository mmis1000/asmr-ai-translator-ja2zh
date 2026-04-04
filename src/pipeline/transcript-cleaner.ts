import type { TranscriptSegment, TranscriptFile, DemucsWindow } from "../util/types.js";

export interface TimeRange {
  start: number;
  end: number;
  reason: string;
}

/**
 * Re-split a single ASR segment at internal word-level gaps >= threshold.
 * Whisper often groups multiple utterances into one ~30s chunk when the audio
 * has long silences mid-segment. Word timestamps let us recover the boundaries.
 */
function resplitSegment(seg: TranscriptSegment, gapThresholdSec: number): TranscriptSegment[] {
  const words = seg.words;
  if (!words || words.length === 0) return [seg];

  const groups: typeof words[] = [];
  let current: typeof words = [words[0]!];

  for (let i = 1; i < words.length; i++) {
    const gap = words[i]!.start_time - words[i - 1]!.end_time;
    const prevWord = words[i - 1]!;
    
    const isSilenceToken = /^[\s…・。、！？!?]+$/.test(prevWord.text) && prevWord.text.includes("…");
    const prevIsLongSilence =
      isSilenceToken &&
      prevWord.end_time - prevWord.start_time >= gapThresholdSec;

    if (gap >= gapThresholdSec || prevIsLongSilence) {
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
    // Note: Energy metrics for sub-segments would ideally be re-merged here,
    // but for now they inherit the parent's (they will be close enough).
    ...(seg.vocal_energy !== undefined ? { vocal_energy: seg.vocal_energy } : {}),
    ...(seg.other_energy !== undefined ? { other_energy: seg.other_energy } : {}),
    ...(seg.snr !== undefined ? { snr: seg.snr } : {}),
  } as TranscriptSegment));
}

/**
 * Detect garbled/hallucinated ASR segments using Whisper metrics + Demucs energy.
 */
export function isGarbled(
  seg: TranscriptSegment,
  options: {
    vocalThreshold: number;
    vocalSilenceThreshold: number;
    snrThreshold: number;
    minHallLength: number;
    rejectNegativeSnr: boolean;
  },
): boolean {
  // Capture signal status from Demucs
  const hasSignalInfo = seg.vocal_energy !== undefined && seg.snr !== undefined;
  const vocalEnergy = seg.vocal_energy ?? 0;
  const snr = seg.snr ?? 0;
  const duration = seg.end_time - seg.start_time;

  const isSilent = hasSignalInfo && vocalEnergy < options.vocalThreshold;
  const isAbsoluteSilence = hasSignalInfo && vocalEnergy < options.vocalSilenceThreshold;
  const isNoisy = hasSignalInfo && snr < options.snrThreshold;
  const signalOk = hasSignalInfo && !isSilent && !isNoisy;

  // Whisper Quality Metrics (Standard Flags)
  const whisperRejected = seg.no_speech_prob > 0.8 || seg.avg_logprob < -1.5 || seg.compression_ratio > 2.5;

  // 1. Scenario A: Whisper thinks it's non-speech/garbage, but signal IS high
  if (whisperRejected && signalOk) {
    seg.mismatch = {
      type: "uncertain",
      reason: `Whisper flagged low quality (no_speech_prob: ${seg.no_speech_prob.toFixed(2)}), but Demucs saw signal (SNR: ${snr.toFixed(1)}dB)`,
    };
    return false; // KEEP it for now — LLM can decide
  }

  // 2. Scenario B Extensions (Hallucinations)
  if (!whisperRejected && hasSignalInfo) {
    let hallReason: string | null = null;

    if (isAbsoluteSilence) {
      hallReason = `Absolute silence (RMS: ${vocalEnergy.toFixed(6)})`;
    } else if (isSilent && duration < 0.2) {
      hallReason = `Unrealistic duration (${duration.toFixed(2)}s) with low signal`;
    } else if (isSilent && seg.text.length > options.minHallLength) {
      hallReason = `No vocal signal for long text (RMS: ${vocalEnergy.toFixed(5)})`;
    } else if (options.rejectNegativeSnr && snr < 0 && isSilent) {
      hallReason = `Background noise dominated (SNR: ${snr.toFixed(1)}dB)`;
    }

    if (hallReason) {
      seg.mismatch = {
        type: "hallucinated",
        reason: hallReason,
      };
      return true;
    }
  }

  // 3. Early return for clear Whisper rejects
  if (whisperRejected) return true;

  // 4. Content Heuristics
  const text = seg.text;
  if (seg.start_time === 0 && seg.end_time > 15 && text.length > 200) return true;

  const effective = text.replace(/[\s\p{P}\p{S}]/gu, "");
  if (effective.length < 2) return true;

  return false;
}

/**
 * Detect Scenario C: Demucs see speech but Whisper was completely silent.
 */
function detectMissedSpeech(
  segments: TranscriptSegment[],
  windows: DemucsWindow[],
  vocalThreshold: number,
  snrThreshold: number,
): TranscriptSegment[] {
  const missed: TranscriptSegment[] = [];
  const LEEWAY = 1.0; // seconds

  // 1. Group continuous active windows into single SignalBlocks
  const blocks: DemucsWindow[][] = [];
  let currentBlock: DemucsWindow[] = [];
  
  for (const win of windows) {
    const isActive = win.vocal_rms > vocalThreshold && win.snr_db > snrThreshold;
    if (isActive) {
      currentBlock.push(win);
    } else if (currentBlock.length > 0) {
      blocks.push(currentBlock);
      currentBlock = [];
    }
  }
  if (currentBlock.length > 0) blocks.push(currentBlock);

  // 2. Filter blocks that are "continuations" of or very close to existing segments
  for (const block of blocks) {
    const bStart = block[0]!.start;
    const bEnd = block[block.length - 1]!.end;

    const isIsolated = !segments.some(s => {
      // Direct temporal overlap
      const overlapStart = Math.max(s.start_time, bStart);
      const overlapEnd = Math.min(s.end_time, bEnd);
      if (overlapEnd > overlapStart) return true;

      // Proximity (check if it "continues from" or "leads into" a segment)
      const gap = Math.min(
        Math.abs(bStart - s.end_time),
        Math.abs(bEnd - s.start_time)
      );
      return gap < LEEWAY;
    });

    if (isIsolated) {
      missed.push(createMissedSegment(block));
    }
  }
  
  return missed;
}

function createMissedSegment(windows: DemucsWindow[]): TranscriptSegment {
  const start = windows[0]!.start;
  const end = windows[windows.length - 1]!.end;
  const avgVocal = windows.reduce((acc, w) => acc + w.vocal_rms, 0) / windows.length;
  const avgSnr = windows.reduce((acc, w) => acc + w.snr_db, 0) / windows.length;

  return {
    text: "[未検知の音声]", // [Undetected Speech]
    start_time: start,
    end_time: end,
    words: [],
    avg_logprob: 0,
    compression_ratio: 0,
    no_speech_prob: 0,
    vocal_energy: avgVocal,
    snr: avgSnr,
    mismatch: {
      type: "missed",
      reason: `Demucs detected signal (SNR: ${avgSnr.toFixed(1)}dB) but Whisper generated no segments`,
    },
  };
}

export function cleanTranscript(
  transcript: TranscriptFile,
  options: {
    vocalThreshold: number;
    vocalSilenceThreshold: number;
    snrThreshold: number;
    minHallLength: number;
    rejectNegativeSnr: boolean;
    resplitGapSec: number;
    maxRepetitions: number;
  },
): { segments: TranscriptSegment[]; mismatches: TranscriptSegment[]; repairRanges: TimeRange[] } {
  if (!transcript.segments) return { segments: [], mismatches: [], repairRanges: [] };

  const mismatches: TranscriptSegment[] = [];

  // 1. Initial cleaning (resplit + filter garbled)
  const expanded = transcript.segments.flatMap(s => resplitSegment(s, options.resplitGapSec));
  
  // 2. Repetition Detection
  const repeatedIndices = detectRepetitions(expanded, options.maxRepetitions);

  const cleaned = expanded.filter((s, i) => {
    const isRepeated = repeatedIndices.has(i);
    const garbled = isGarbled(s, options);
    
    if (isRepeated) {
      s.mismatch = {
        type: "repeated",
        reason: `Text repeated ${options.maxRepetitions}+ times consecutively`,
      };
      mismatches.push({ ...s });
      return false;
    }

    // Capture Scenario A (Uncertain) and B (Hallucinated)
    if (s.mismatch) {
      mismatches.push({ ...s });
    }
    
    return !garbled;
  });

  // 3. Scenario C Detection (Missed Speech)
  if (transcript.demucs_windows) {
    const missed = detectMissedSpeech(cleaned, transcript.demucs_windows, options.vocalThreshold, options.snrThreshold);
    mismatches.push(...missed);
  }

  // 4. Identify Surgical Repair Ranges
  const repairRanges = identifyRepairRanges(cleaned, mismatches, {
    vocalThreshold: options.vocalThreshold,
    minMissedDuration: 1.5, // Only repair meaningful chunks
  });

  // Sort mismatches by time for investigation
  const sortedMismatches = [...mismatches].sort((a, b) => a.start_time - b.start_time);

  return {
    segments: cleaned,
    mismatches: sortedMismatches,
    repairRanges,
  };
}

/**
 * Detects consecutive segments with identical text.
 */
function detectRepetitions(segments: TranscriptSegment[], maxRepetitions: number): Set<number> {
  const repeated = new Set<number>();
  if (segments.length < maxRepetitions) return repeated;

  let currentText = "";
  let currentCount = 0;
  let startIdx = 0;

  const normalize = (t: string) => t.trim().replace(/[\s\p{P}\p{S}]/gu, "").toLowerCase();
  
  /** Returns true if two strings are similar enough to be considered a loop. */
  const isLoopMatch = (a: string, b: string) => {
    const na = normalize(a);
    const nb = normalize(b);
    if (!na || !nb) return false;
    if (na === nb) return true;
    // Basic fuzzy: if one contains the other and they are long enough
    if (na.length > 10 && nb.length > 10) {
      if (na.includes(nb) || nb.includes(na)) return true;
    }
    return false;
  };

  for (let i = 0; i < segments.length; i++) {
    const text = segments[i]!.text;
    if (!text.trim()) continue;

    if (isLoopMatch(text, currentText)) {
      currentCount++;
    } else {
      if (currentCount >= maxRepetitions) {
        for (let j = startIdx; j < i; j++) repeated.add(j);
      }
      currentText = text;
      currentCount = 1;
      startIdx = i;
    }
  }

  if (currentCount >= maxRepetitions) {
    for (let j = startIdx; j < segments.length; j++) repeated.add(j);
  }

  return repeated;
}

/**
 * Groups various failures (repeated text, certain uncertains, long missed speech) 
 * into time windows for surgical re-processing.
 */
function identifyRepairRanges(
  segments: TranscriptSegment[],
  mismatches: TranscriptSegment[],
  options: { vocalThreshold: number; minMissedDuration: number }
): TimeRange[] {
  const rawRanges: TimeRange[] = [];

  // Range 1: Repetition Loops (Check signal)
  mismatches.filter(m => m.mismatch?.type === "repeated" && (m.vocal_energy ?? 0) > options.vocalThreshold)
    .forEach(m => rawRanges.push({ start: m.start_time, end: m.end_time, reason: "repetition_loop" }));

  // Range 2: Missed Speech (Scenario C) long enough to fix
  mismatches.filter(m => m.mismatch?.type === "missed" && (m.end_time - m.start_time) >= options.minMissedDuration)
    .forEach(m => rawRanges.push({ start: m.start_time, end: m.end_time, reason: "missed_speech" }));

  // Range 3: Uncertain segments (Scenario A) with high energy but very bad logprob
  // These are often "hallucinated garbage" even if signal is present.
  segments.filter(s => s.mismatch?.type === "uncertain" && s.avg_logprob < -2.0)
    .forEach(s => rawRanges.push({ start: s.start_time, end: s.end_time, reason: "low_quality_signal" }));

  if (rawRanges.length === 0) return [];

  // Merge nearby/overlapping ranges
  rawRanges.sort((a, b) => a.start - b.start);
  
  const merged: TimeRange[] = [];
  let current = { ...rawRanges[0]! };
  const PADDING = 2.0; // 2s padding for context

  for (let i = 1; i < rawRanges.length; i++) {
    const next = rawRanges[i]!;
    // If gaps are small (< 10s), merge them into one re-run window to avoid overhead
    if (next.start <= current.end + 10.0) {
      current.end = Math.max(current.end, next.end);
      if (!current.reason.includes(next.reason)) current.reason += `+${next.reason}`;
    } else {
      merged.push({ 
        start: Math.max(0, current.start - PADDING), 
        end: current.end + PADDING, 
        reason: current.reason 
      });
      current = { ...next };
    }
  }
  merged.push({ 
    start: Math.max(0, current.start - PADDING), 
    end: current.end + PADDING, 
    reason: current.reason 
  });

  return merged;
}
