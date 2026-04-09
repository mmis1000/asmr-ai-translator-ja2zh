import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import type { TranscriptSegment, TranscriptFile, DemucsWindow } from "../util/types.js";
import type { TimeRange } from "./transcript-cleaner.js";
import { isGarbled } from "./transcript-cleaner.js";
import { mergeEnergyToSegment } from "./asr-runner.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ASR_SCRIPT = path.resolve(__dirname, "../../asr/asr_cli.py");

interface RepairOptions {
  pythonExe: string;
  asrScript: string;
  model: string;
  device: string;
  vocalThreshold: number;
  snrThreshold: number;
  vocalSilenceThreshold: number;
  minHallLength: number;
  rejectNegativeSnr: boolean;
  repairTemperature: number | null;
  repairBeamSize: number;
  repairWithVocal: boolean;
  mixWeight: number; // Weight for mixing original audio back into vocals
  asrPrompt?: string | undefined;
  repairEngine: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma";
  saveRepairAudio: boolean;
}

/**
 * Surgically repairs faulty regions of a transcription by re-running Whisper 
 * with different settings (e.g. non-greedy sampling to break loops).
 */
export interface SurgicalRepairEntry {
  range: TimeRange;
  originalSegments: TranscriptSegment[];
  newSegments: TranscriptSegment[];
  status: "success" | "failed" | "no_change";
}

/**
 * Surgically repairs faulty regions of a transcription by re-running Whisper 
 * with different settings (e.g. non-greedy sampling to break loops).
 */
export async function repairTranscription(
  audioPath: string,
  initialSegments: TranscriptSegment[],
  initialMismatches: TranscriptSegment[],
  repairRanges: TimeRange[],
  windows: DemucsWindow[],
  options: RepairOptions,
  existingLog: SurgicalRepairEntry[] = [],
  vocalPath?: string
): Promise<{ 
  segments: TranscriptSegment[]; 
  mismatches: TranscriptSegment[];
  surgicalLog: SurgicalRepairEntry[];
}> {
  // Merge overlapping/adjacent repair ranges before sending to the ASR engine
  // so we never dispatch duplicate or overlapping windows in the same pass.
  const windowsToRepair = mergeRanges(repairRanges);

  if (windowsToRepair.length === 0) {
    return { 
      segments: initialSegments, 
      mismatches: initialMismatches,
      surgicalLog: existingLog 
    };
  }

  console.log(`\n[REPAIR] Starting surgical repair for ${windowsToRepair.length} new windows...`);

  let currentSegments = [...initialSegments];
  let currentMismatches = [...initialMismatches];
  const surgicalLog = [...existingLog];

  // Derive the output directory for temp files (if vocalPath is provided, it's inside demucs_output)
  // Otherwise default to the dirname of the audioPath (original behavior, but usually we have vocalPath now)
  const tempDir = vocalPath ? path.dirname(path.dirname(vocalPath)) : path.dirname(audioPath);

  // Determine which engine to use (one engine per repair pass)
  const engineType: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma" = options.repairEngine;

  console.log(`  -> Engine: ${engineType} | Batching ${windowsToRepair.length} window(s) into one call...`);

  // Choose input audio once (same for all windows in this pass)
  let inputAudio = audioPath;
  let mixAudio: string | undefined = undefined;
  if (options.repairWithVocal && vocalPath) {
    if (fs.existsSync(vocalPath)) {
      inputAudio = vocalPath;
      mixAudio = audioPath;
      console.log(`  -> Using mixed vocal stem: ${path.basename(vocalPath)} (+ ${Math.round(options.mixWeight * 100)}% original)`);
    } else {
      console.warn(`  -> Warning: Vocal stem NOT found at ${vocalPath}. Falling back to original audio.`);
    }
  }

  // Build per-window jobs
  type WindowJob = {
    range: TimeRange;
    rangeArgs: TimeRange[];     // non-MMS: [range]; MMS: speech clusters within range
    originalInRange: TranscriptSegment[];
    mismatchesInRange: TranscriptSegment[];
  };

  const jobs: WindowJob[] = windowsToRepair.map(range => {
    const originalInRange = initialSegments.filter(s =>
      s.start_time >= range.start && s.end_time <= range.end
    );
    const mismatchesInRange = initialMismatches.filter(m =>
      m.start_time >= range.start && m.end_time <= range.end
    );
    const rangeArgs = options.repairEngine === "mms"
      ? getSpeechClusters(windows, range, options.vocalThreshold, options.snrThreshold)
      : [range];
    return { range, rangeArgs, originalInRange, mismatchesInRange };
  });

  // Log per-window mismatches before the batch call
  for (const job of jobs) {
    console.log(`  -> [${job.range.start.toFixed(1)}s - ${job.range.end.toFixed(1)}s] (${job.range.reason})`);
    job.mismatchesInRange.forEach(m => {
      console.log(`     - [${m.mismatch?.type}] ${m.mismatch?.reason}`);
    });
    if (options.repairEngine === "mms") {
      console.log(`     - ${job.rangeArgs.length} MMS cluster(s)`);
    }
  }

  // Build repair audio base path (Python derives per-window paths as <base>_<start>s<ext>)
  let repairAudioBasePath: string | undefined = undefined;
  if (options.saveRepairAudio) {
    const repairAudioDir = path.join(tempDir, "repair_audio_fragments");
    if (!fs.existsSync(repairAudioDir)) fs.mkdirSync(repairAudioDir, { recursive: true });
    const stem = path.basename(audioPath, path.extname(audioPath));
    repairAudioBasePath = path.join(repairAudioDir, `${stem}_repair.wav`);
  }

  // Active jobs: skip MMS windows with no clusters (nothing to process)
  const activeJobs = jobs.filter(j => j.rangeArgs.length > 0);
  const allRangeArgs = activeJobs.flatMap(j => j.rangeArgs);

  const batchRepairFile = path.join(tempDir, `.tmp_repair_batch.json`);
  let windowResults: WindowResult[] = [];

  try {
    if (activeJobs.length > 0) {
      windowResults = await runSurgicalASR(
        inputAudio, allRangeArgs, batchRepairFile, options, mixAudio, engineType, repairAudioBasePath
      );
    }
  } catch (err) {
    console.error(`  [REPAIR] Batch ASR call failed:`, err);
  } finally {
    if (fs.existsSync(batchRepairFile)) fs.unlinkSync(batchRepairFile);
  }

  // Apply results per-window
  // Non-MMS: windowResults are in the same order as activeJobs (one result per window)
  // MMS: one combined result; filter by time range
  let activeJobIdx = 0;

  for (const job of jobs) {
    const { range } = job;
    const repairEntry: SurgicalRepairEntry = {
      range,
      originalSegments: job.originalInRange,
      newSegments: [],
      status: "no_change",
    };

    if (job.rangeArgs.length === 0) {
      // MMS found no speech clusters in this window
      repairEntry.status = "failed";
      console.log(`  -> [${range.start.toFixed(1)}s] No MMS clusters found, skipping.`);
      surgicalLog.push(repairEntry);
      continue;
    }

    let newSegments: TranscriptSegment[];
    if (options.repairEngine === "mms") {
      // MMS: one combined result for all clusters; filter to this window's range
      const mmsResult = windowResults[0];
      newSegments = (mmsResult?.segments ?? []).filter(
        s => s.start_time >= range.start - 0.5 && s.end_time <= range.end + 0.5
      );
    } else {
      // Non-MMS: one result element per active job, in order
      const result = windowResults[activeJobIdx++];
      newSegments = result?.segments ?? [];
    }

    if (newSegments.length > 0) {
      repairEntry.newSegments = newSegments;

      // Only remove BAD (mismatch) segments in this range. Good segments
      // within the padded region are preserved — the 2s padding is for ASR
      // context only, not a replacement zone. Removing all segments caused
      // repair fragments landing in the padding to silently drop valid content.
      const goodInRange = currentSegments.filter(s =>
        !(s.end_time <= range.start || s.start_time >= range.end) && !s.mismatch
      );

      // Filter the new segments (Re-apply Demucs check)
      const passed: TranscriptSegment[] = [];
      for (const s of newSegments) {
        // KEY: Merge original energy data into the new segments
        mergeEnergyToSegment(s, windows);

        const isMms = s.engine === "mms";
        
        if (isMms) {
          const originalText = s.text;
          const trimmed = s.text.trim();

          // Pure short Latin noise (e.g. "a", "bc") → blank it
          const isLatinNoise = /^[a-z]{1,2}$/i.test(trimmed);
          if (isLatinNoise) s.text = "";

          // Sub-word CTC artifact: single sound/phoneme rather than a word.
          // In MMS, a segment this short is a stray frame alignment, not speech content.
          const cjkCharCount = (trimmed.match(/[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f]/g) || []).length;
          if (cjkCharCount < 2 && trimmed.replace(/\s|[^\w\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f]/g, "").length < 2) {
            console.log(`     [DEBUG] MMS segment rejected (sub-word): "${originalText}" at ${s.start_time.toFixed(2)}s`);
            continue;
          }

          const isSilent = (s.vocal_energy || 0) < options.vocalSilenceThreshold;
          if (isSilent) {
            console.log(`     [DEBUG] MMS segment rejected (silent): "${originalText}" at ${s.start_time.toFixed(2)}s (Energy: ${s.vocal_energy?.toFixed(6)})`);
            continue;
          }
        }

        const garbled = isMms ? false : isGarbled(s, {
          vocalThreshold: options.vocalThreshold,
          vocalSilenceThreshold: options.vocalSilenceThreshold,
          snrThreshold: options.snrThreshold,
          minHallLength: options.minHallLength,
          rejectNegativeSnr: options.rejectNegativeSnr,
        }, true);

        if (!garbled) {
          // Only insert repair segments that don't conflict with good originals
          // that were kept from the padding zone.
          const blockingGood = goodInRange.find(g =>
            g.start_time < s.end_time && g.end_time > s.start_time
          );
          if (!blockingGood) {
            passed.push(s);
          } else {
            console.log(`     [DEBUG] MMS segment rejected (conflict): "${s.text}" overlaps "${blockingGood.text}" at ${blockingGood.start_time.toFixed(1)}s`);
          }
        }
      }

      if (passed.length > 0) {
        // IDEMPOTENCY FIX: Remove EVERYTHING in the range that isn't a "Good Original".
        // A "Good Original" has NO engine property and NO mismatch.
        // This ensures the new repair replaces ANY previous attempts (failed or successful).
        currentSegments = currentSegments.filter(s =>
          (s.end_time <= range.start || s.start_time >= range.end) || (!s.mismatch && !s.engine)
        );
        currentMismatches = currentMismatches.filter(m =>
          m.end_time <= range.start || m.start_time >= range.end
        );

        currentSegments.push(...passed);
        repairEntry.status = "success";
        repairEntry.newSegments = passed;
        console.log(`     [${range.start.toFixed(1)}s] Repair successful: ${passed.length} new valid segments added.`);
      } else {
        repairEntry.status = "failed";
        console.log(`     [${range.start.toFixed(1)}s] Repair produced no non-conflicting valid content.`);
      }
    } else {
      console.log(`     [${range.start.toFixed(1)}s] Repair produced no new content.`);
    }

    if (repairEntry.status !== "no_change") {
      surgicalLog.push(repairEntry);
    }
  }

  // Final sort to ensure chronological order before returning
  currentSegments.sort((a, b) => a.start_time - b.start_time);
  currentMismatches.sort((a, b) => a.start_time - b.start_time);

  return { 
    segments: currentSegments, 
    mismatches: currentMismatches,
    surgicalLog
  };
}

/**
 * Merges successful repairs from a surgical log into a set of segments.
 * This is used to hydrate the cache at the start of a run.
 */
export function applySurgicalRepair(
  segments: TranscriptSegment[],
  mismatches: TranscriptSegment[],
  log: SurgicalRepairEntry[]
): { segments: TranscriptSegment[]; mismatches: TranscriptSegment[] } {
  let currentSegments = [...segments];
  let currentMismatches = [...mismatches];

  for (const entry of log) {
    if (entry.status !== "success" || entry.newSegments.length === 0) continue;

    const range = entry.range;

    // Mirror the live repair logic: only remove BAD (mismatch) segments.
    // Good segments in the padded zone must be preserved.
    const goodInRange = currentSegments.filter(s =>
      !(s.end_time <= range.start || s.start_time >= range.end) && !s.mismatch
    );

    // Only insert repair segments that don't conflict with kept good segments.
    const toAdd: TranscriptSegment[] = [];
    for (const s of entry.newSegments) {
      const conflictsWithGood = goodInRange.some(g =>
        g.start_time < s.end_time && g.end_time > s.start_time
      );
      if (!conflictsWithGood) {
        toAdd.push(s);
      }
    }

    if (toAdd.length > 0) {
      // Filter out mismatches AND previous repairs (any segment with engine property)
      // to ensure the newly applied cached repair is the only thing for this range.
      currentSegments = currentSegments.filter(s =>
        (s.end_time <= range.start || s.start_time >= range.end) || (!s.mismatch && !s.engine)
      );
      currentMismatches = currentMismatches.filter(m =>
        m.end_time <= range.start || m.start_time >= range.end
      );
      currentSegments.push(...toAdd);
    }
  }

  // Final sort
  currentSegments.sort((a, b) => a.start_time - b.start_time);
  currentMismatches.sort((a, b) => a.start_time - b.start_time);

  return { segments: currentSegments, mismatches: currentMismatches };
}

/**
 * Groups contiguous Demucs windows that exceed vocal energy and SNR thresholds
 * to identify speech regions for MMS pre-slicing.
 */
function getSpeechClusters(
  windows: DemucsWindow[],
  range: TimeRange,
  vocalThreshold: number,
  snrThreshold: number
): TimeRange[] {
  const MAX_GAP = 1.2;   // Merge gaps <= 1.2s
  const PADDING = 0.2;   // Adding padding to avoid cut-offs
  const MIN_DURATION = 1.5; // Ensure at least 1.5s per cluster

  // Find windows that overlap with this range and meet thresholds
  const relevant = windows.filter(w => 
    w.start < range.end && w.end > range.start &&
    w.vocal_rms >= vocalThreshold && 
    w.snr_db >= snrThreshold
  );

  if (relevant.length === 0) return [];

  const initialClusters: TimeRange[] = [];
  let current: TimeRange | null = null;

  for (const w of relevant) {
    if (!current) {
      current = { start: Math.max(w.start, range.start), end: Math.min(w.end, range.end), reason: "mms_cluster" };
    } else {
      // Merge if the gap between windows is within MAX_GAP
      if (w.start <= current.end + MAX_GAP) {
        current.end = Math.min(w.end, range.end);
      } else {
        initialClusters.push(current);
        current = { start: Math.max(w.start, range.start), end: Math.min(w.end, range.end), reason: "mms_cluster" };
      }
    }
  }
  if (current) initialClusters.push(current);

  // Post-process: Add padding and enforce minimum duration
  return initialClusters.map(c => {
    let start = Math.max(range.start, c.start - PADDING);
    let end = Math.min(range.end, c.end + PADDING);
    
    const dur = end - start;
    if (dur < MIN_DURATION) {
      const needed = MIN_DURATION - dur;
      // Expand around the center
      const center = (start + end) / 2;
      start = Math.max(range.start, center - MIN_DURATION / 2);
      end = Math.min(range.end, start + MIN_DURATION);
      // Re-boundary check in case end overflowed range.end
      if (end > range.end) {
        end = range.end;
        start = Math.max(range.start, end - MIN_DURATION);
      }
    }
    
    return { start, end, reason: "mms_cluster" };
  });
}

/**
 * Merges overlapping or adjacent TimeRanges into the minimal set of
 * non-overlapping ranges, preserving a combined reason string.
 */
function mergeRanges(ranges: TimeRange[]): TimeRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const merged: TimeRange[] = [{ ...sorted[0]! }];
  for (let i = 1; i < sorted.length; i++) {
    const cur = merged[merged.length - 1]!;
    const next = sorted[i]!;
    if (next.start <= cur.end) {
      // Overlapping — extend current range
      cur.end = Math.max(cur.end, next.end);
      if (!cur.reason.includes(next.reason)) cur.reason += `+${next.reason}`;
    } else {
      merged.push({ ...next });
    }
  }
  return merged;
}

type WindowResult = {
  window: [number, number] | number[][];
  full_text: string;
  sentences: unknown[];
  segments: TranscriptSegment[];
};

async function runSurgicalASR(
  audioPath: string,
  ranges: TimeRange[],
  outputJson: string,
  options: RepairOptions,
  mixAudioPath: string | undefined,
  engine: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma",
  repairAudioBasePath?: string
): Promise<WindowResult[]> {
  return new Promise((resolve, reject) => {
    const asrScript = options.asrScript || DEFAULT_ASR_SCRIPT;

    // Select the correct Python virtual environment
    // Qwen requires Transformers 4.x, so it has its own isolated env.
    let enginePythonExe = options.pythonExe;
    if (engine === "qwen") {
      const asrDir = path.dirname(asrScript);
      const qwenVenvExe = path.join(asrDir, "qwen_env", ".venv", "Scripts", "python.exe");
      if (fs.existsSync(qwenVenvExe)) {
        enginePythonExe = qwenVenvExe;
      } else {
        console.warn(`  [ASR] Isolated Qwen environment not found at ${qwenVenvExe}. Falling back to default.`);
      }
    }

    const args = [
      asrScript,
      "--audio", audioPath,
      "--output", outputJson,
      "--model", options.model,
      "--device", options.device,
      "--engine", engine,
      "--windows", JSON.stringify(ranges.map(r => [r.start, r.end])),
    ];

    if (repairAudioBasePath) {
      // Python derives per-window paths as <base>_<start>s<ext>
      args.push("--save-audio-slice", repairAudioBasePath);
    }

    if (engine === "whisper") {
      args.push("--beam-size", options.repairBeamSize.toString());
      if (options.repairTemperature !== null) {
        args.push("--temperature", options.repairTemperature.toString());
      }
    }

    if (options.asrPrompt && engine !== "mms") {
      args.push("--prompt", options.asrPrompt);
    }

    if (mixAudioPath) {
      args.push("--mix-audio", mixAudioPath);
      args.push("--mix-weight", options.mixWeight.toString());
    }

    const child = spawn(enginePythonExe, ["-u", ...args]);
    let output = "";
    let done = false;

    child.stdout.on("data", (data) => {
      const chunk = data.toString();
      process.stdout.write(chunk);
      output += chunk;
      // Handle the ROCm hang sentinel
      if (!done && output.includes("[ASR_DONE]")) {
        done = true;
        child.kill("SIGKILL");
      }
    });

    child.stderr?.on("data", (data) => process.stderr.write(data));

    child.on("close", (code) => {
      if (done) {
        // Sentinel reached, success
      } else if (code !== 0 && code !== null) {
        reject(new Error(`Surgical ASR process (${engine}) failed with code ${code}`));
        return;
      }

      if (!fs.existsSync(outputJson)) {
        resolve([]);
        return;
      }

      try {
        const raw = JSON.parse(fs.readFileSync(outputJson, "utf-8"));
        if (Array.isArray(raw)) {
          // Multi-window output: tag segments with engine and return as-is
          const results = (raw as WindowResult[]).map(w => ({
            ...w,
            segments: (w.segments || []).map(s => ({ ...s, engine })),
          }));
          resolve(results);
        } else {
          // Single full-file result (no --windows): wrap in array
          const data = raw as TranscriptFile;
          resolve([{
            window: [0, Infinity],
            full_text: data.full_text || "",
            sentences: data.sentences || [],
            segments: (data.segments || []).map(s => ({ ...s, engine })),
          }]);
        }
      } catch (err) {
        reject(err);
      }
    });

    child.on("error", reject);
  });
}
