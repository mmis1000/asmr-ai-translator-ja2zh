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
    
    // Check for terminal punctuation at end of previous word (legit break)
    const endsWithTerminalPunc = /[。！？!?]$/.test(prevWord.text.trim());
    
    const isSilenceToken = /^[\s…・。、！？!?]+$/.test(prevWord.text) && prevWord.text.includes("…");
    const prevIsLongSilence =
      isSilenceToken &&
      prevWord.end_time - prevWord.start_time >= gapThresholdSec;

    if (gap >= gapThresholdSec || prevIsLongSilence || endsWithTerminalPunc) {
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
  isRepair: boolean = false
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

  // Hard reject: high compression_ratio means Whisper looped on repetitive output
  // (e.g. echo/reverb artifacts). This is unambiguously a hallucination — Demucs
  // signal presence is irrelevant and must NOT override this check.
  if (seg.compression_ratio > 2.5) {
    seg.mismatch = {
      type: "hallucinated",
      reason: `Repetitive ASR loop detected (compression_ratio: ${seg.compression_ratio.toFixed(2)})`,
    };
    return true;
  }

  // 1. Scenario A: Whisper thinks it's non-speech/garbage, but signal IS high
  // (Only reaches here when rejected by no_speech_prob or avg_logprob, not CR)
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

  // 5. Impossible Speed Check (CPS)
  // Human speech is typically 5-15 CPS. 25+ is almost certainly hallucinated garbage.
  const cps = effective.length / Math.max(0.1, duration);
  if (cps > 25 && effective.length > 10) {
    seg.mismatch = {
      type: "hallucinated",
      reason: `Impossible speech speed detected (${cps.toFixed(1)} CPS). Whisper 'vomited' text.`,
    };
    return true;
  }

  // 6. Multi-character loop detection (e.g., "うぅうぅ" or "らららら")
  if (effective.length > 10) {
    const counts: Record<string, number> = {};
    for (const char of effective) {
      counts[char] = (counts[char] || 0) + 1;
    }
    
    const sortedCounts = Object.values(counts).sort((a, b) => b - a);
    const top1 = sortedCounts[0] || 0;
    const top2 = sortedCounts[1] || 0;
    const total = effective.length;

    // Single character loop (>60%) or dual-character loop (>85%)
    if ((top1 / total > 0.6) || ((top1 + top2) / total > 0.85 && total > 20)) {
      seg.mismatch = {
        type: "hallucinated",
        reason: `Repetitive character loop detected (${((top1 + top2) / total * 100).toFixed(0)}% of text consists of top 2 characters)`,
      };
      return true;
    }
  }

  // 8. Hallucination "Heat" Filter (Co-occurrence of quality issues)
  // If Whisper rejected this (Scenario A) but signal is present, we are more strict.
  const isWhisperHeated = seg.avg_logprob < -2.5;
  const isWhisperTerminal = seg.avg_logprob < -3.5;

  // Mixed Script Check: Japanese text mixed with random Latin words
  const latinCount = (text.match(/[a-zA-Z]/g) || []).length;
  const cjkCount = (text.match(/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]/g) || []).length;
  const isMixedScript = cjkCount > 5 && latinCount / (cjkCount + latinCount) > 0.3;

  if (isWhisperTerminal) {
    seg.mismatch = {
      type: "hallucinated",
      reason: `Extreme low quality (logprob: ${seg.avg_logprob.toFixed(2)}). Impossible recovery.`,
    };
    return true;
  }

  if (isWhisperHeated && (isMixedScript || cps > 20)) {
    seg.mismatch = {
      type: "hallucinated",
      reason: `Moderate low quality (${seg.avg_logprob.toFixed(2)}) combined with junk signal (MixedScripts: ${isMixedScript}, CPS: ${cps.toFixed(1)}).`,
    };
    return true;
  }

  // 9. Common 'Outro' hallucinations (Repair-only)
  // These are common when Whisper forces transcription in silent blocks.
  // We only filter them in repair mode to avoid deleting real "Goodnight" lines.
  if (isRepair) {
    const OUTRO_PATTERNS = [
      /^[おオ]やすみなさい[。！]?$/i,
      /^ご視聴ありがとうございました[。！]?$/i,
      /^チャンネル登録[。、]?よろしくお願いします[。！]?$/i,
      /^チャンネル登録よろしくお願いします[。！]?$/i,
      /^次回の動画でお会いしましょう[。！]?$/i,
      /^動画を聞いてくれてありがとう[。！]?$/i,
      /^動画を見てくれてありがとう[。！]?$/i,
      /^高評価[、。]?よろしくお願いします[。！]?$/i,
      /^[バパ]イ[バパ]イ[。！]?$/i,
      /^またね[。！]?$/i,
    ];

    if (OUTRO_PATTERNS.some(re => re.test(text.trim()))) {
      seg.mismatch = {
        type: "hallucinated",
        reason: "Detected common Whisper 'outro' hallucination in repair pass.",
      };
      return true;
    }
  }

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
  const MAX_BLOCK_DURATION = 30.0; // Split long noisy blocks
  
  for (const win of windows) {
    const isActive = win.vocal_rms > vocalThreshold && win.snr_db > snrThreshold;
    const currentDuration = currentBlock.length > 0 ? win.end - currentBlock[0]!.start : 0;

    if (isActive && currentDuration < MAX_BLOCK_DURATION) {
      currentBlock.push(win);
    } else if (currentBlock.length > 0) {
      blocks.push(currentBlock);
      currentBlock = (isActive) ? [win] : [];
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

/**
 * Merge consecutive segments that share the same start timestamp (after ms
 * rounding). ASR hallucination loops can produce many segments at near-identical
 * timestamps; without merging, the grammar emits multiple translation entries
 * sharing the same LRC timestamp.
 */
function mergeSameStartSegments(segments: TranscriptSegment[]): TranscriptSegment[] {
  if (segments.length === 0) return segments;

  const result: TranscriptSegment[] = [];
  let current: TranscriptSegment = { ...segments[0]!, words: [...segments[0]!.words] };

  for (let i = 1; i < segments.length; i++) {
    const seg = segments[i]!;
    if (Math.round(current.start_time * 1000) === Math.round(seg.start_time * 1000)) {
      current.text += seg.text;
      current.end_time = Math.max(current.end_time, seg.end_time);
      current.words = current.words.concat(seg.words);
      // Keep worst-case quality metrics so downstream filters stay conservative
      current.avg_logprob = Math.min(current.avg_logprob, seg.avg_logprob);
      current.compression_ratio = Math.max(current.compression_ratio, seg.compression_ratio);
      current.no_speech_prob = Math.max(current.no_speech_prob, seg.no_speech_prob);
      if (seg.vocal_energy !== undefined)
        current.vocal_energy = Math.max(current.vocal_energy ?? 0, seg.vocal_energy);
      if (seg.snr !== undefined)
        current.snr = Math.min(current.snr ?? Infinity, seg.snr);
    } else {
      result.push(current);
      current = { ...seg, words: [...seg.words] };
    }
  }
  result.push(current);
  return result;
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
    const garbled = isGarbled(s, options, false);

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

  // 2b. Merge consecutive segments that collapsed to the same start timestamp
  //     after ms rounding (e.g. ASR hallucination loops producing many segments
  //     at near-identical timestamps). Without this the grammar emits multiple
  //     translation entries sharing the same LRC timestamp.
  const deduped = mergeSameStartSegments(cleaned);

  // 3. Scenario C Detection (Missed Speech)
  if (transcript.demucs_windows) {
    const missed = detectMissedSpeech(deduped, transcript.demucs_windows, options.vocalThreshold, options.snrThreshold);
    mismatches.push(...missed);
  }

  // 4. Identify Surgical Repair Ranges
  const repairRanges = identifyRepairRanges(deduped, mismatches, {
    vocalThreshold: options.vocalThreshold,
    minMissedDuration: 1.5, // Only repair meaningful chunks
  });

  // Sort mismatches by time for investigation
  const sortedMismatches = [...mismatches].sort((a, b) => a.start_time - b.start_time);

  return {
    segments: deduped,
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
  const MAX_RANGE_DURATION = 90.0; // Increased to be less likely to split awkwardly
  const MAX_MERGE_GAP = 8.0; // Be slightly more inclusive with merging

  for (let i = 1; i < rawRanges.length; i++) {
    const next = rawRanges[i]!;
    const wouldBeDuration = next.end - current.start;

    // If gaps are small (< 8s) AND the range doesn't get too long, merge them
    if (next.start <= current.end + MAX_MERGE_GAP && wouldBeDuration <= MAX_RANGE_DURATION) {
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
