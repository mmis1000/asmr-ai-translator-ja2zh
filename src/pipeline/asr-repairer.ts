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
  useMmsRepair?: boolean;
  useQwenRepair?: boolean;
  useSenseVoiceRepair?: boolean;
  useGemmaRepair?: boolean;
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
  // Identify windows that actually need a NEW repair (not already in existingLog)
  const windowsToRepair = repairRanges.filter(range => {
    const cached = existingLog.find(e => 
      Math.abs(e.range.start - range.start) < 1.0 && 
      Math.abs(e.range.end - range.end) < 1.0 &&
      e.status === "success"
    );
    if (cached) {
      console.log(`  -> Skipping already repaired window [${range.start.toFixed(1)}s - ${range.end.toFixed(1)}s]`);
      return false;
    }
    return true;
  });

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

  for (const range of windowsToRepair) {
    console.log(`  -> Repairing [${range.start.toFixed(1)}s - ${range.end.toFixed(1)}s] (${range.reason})...`);

    // Capture original segments in this range for logging
    const originalInRange = initialSegments.filter(s => 
      s.start_time >= range.start && s.end_time <= range.end
    );
    const mismatchesInRange = initialMismatches.filter(m => 
      m.start_time >= range.start && m.end_time <= range.end
    );
    mismatchesInRange.forEach(m => {
      console.log(`     - [${m.mismatch?.type}] ${m.mismatch?.reason}`);
    });

    const repairFile = path.join(tempDir, `.tmp_repair_${range.start.toFixed(0)}.json`);
    
    try {
      // Choose input audio: vocal stem if requested and available, else original
      let inputAudio = audioPath;
      let mixAudio: string | undefined = undefined;

      if (options.repairWithVocal && vocalPath) {
        if (fs.existsSync(vocalPath)) {
          inputAudio = vocalPath;
          mixAudio = audioPath; // Mix-back the original audio
          console.log(`     - Using mixed vocal stem for repair: ${path.basename(vocalPath)} (+ ${Math.round(options.mixWeight * 100)}% original)`);
        } else {
          console.warn(`     - Warning: Vocal stem requested but NOT found at ${vocalPath}. Falling back to original audio.`);
        }
      }

      let newSegments: TranscriptSegment[] = [];

      if (options.useQwenRepair) {
        console.log(`     - Using Qwen engine for repair (Full Switch)...`);
        newSegments = await runSurgicalASR(inputAudio, range, repairFile, options, mixAudio, "qwen");
      } else if (options.useMmsRepair) {
        console.log(`     - Using MMS engine for repair (Full Switch)...`);
        const clusters = getSpeechClusters(windows, range, options.vocalThreshold, options.snrThreshold);
        console.log(`     - Batching ${clusters.length} speech cluster(s) for MMS engine.`);
        
        if (clusters.length > 0) {
           newSegments = await runSurgicalASR(inputAudio, clusters, repairFile, options, mixAudio, "mms");
        }
      } else if (options.useSenseVoiceRepair) {
        console.log(`     - Using SenseVoice engine for repair (Full Switch)...`);
        newSegments = await runSurgicalASR(inputAudio, range, repairFile, options, mixAudio, "sensevoice");
      } else if (options.useGemmaRepair) {
        console.log(`     - Using Gemma engine for repair (Full Switch)...`);
        newSegments = await runSurgicalASR(inputAudio, range, repairFile, options, mixAudio, "gemma");
      } else {
        newSegments = await runSurgicalASR(inputAudio, range, repairFile, options, mixAudio, "whisper");
      }
      
      const repairEntry: SurgicalRepairEntry = {
        range,
        originalSegments: originalInRange,
        newSegments: [], // To be populated
        status: "no_change"
      };

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

          const garbled = isGarbled(s, {
            vocalThreshold: options.vocalThreshold,
            vocalSilenceThreshold: options.vocalSilenceThreshold,
            snrThreshold: options.snrThreshold,
            minHallLength: options.minHallLength,
            rejectNegativeSnr: options.rejectNegativeSnr,
          }, true);

          if (!garbled) {
            // Only insert repair segments that don't conflict with good originals
            // that were kept from the padding zone.
            const conflictsWithGood = goodInRange.some(g =>
              g.start_time < s.end_time && g.end_time > s.start_time
            );
            if (!conflictsWithGood) {
              passed.push(s);
            }
          } else if (s.mismatch) {
            currentMismatches.push(s);
          }
        }

        if (passed.length > 0) {
          // Atomic Replacement: Only remove originals if we have valid ones to add
          currentSegments = currentSegments.filter(s =>
            (s.end_time <= range.start || s.start_time >= range.end) || !s.mismatch
          );
          currentMismatches = currentMismatches.filter(m =>
            m.end_time <= range.start || m.start_time >= range.end
          );

          currentSegments.push(...passed);
          repairEntry.status = "success";
          console.log(`     Repair successful: ${passed.length} new valid segments added.`);
        } else {
          repairEntry.status = "failed";
          console.log(`     Repair produced no non-conflicting valid content. Original segments preserved.`);
        }
      } else {
        console.log(`     Repair produced no new content.`);
      }

      if (repairEntry.status !== "no_change") {
        surgicalLog.push(repairEntry);
      }
    } catch (err) {
      console.error(`     Repair failed for range ${range.start}-${range.end}:`, err);
    } finally {
      if (fs.existsSync(repairFile)) fs.unlinkSync(repairFile);
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
      currentSegments = currentSegments.filter(s =>
        (s.end_time <= range.start || s.start_time >= range.end) || !s.mismatch
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

async function runSurgicalASR(
  audioPath: string,
  range: TimeRange | TimeRange[],
  outputJson: string,
  options: RepairOptions,
  mixAudioPath: string | undefined,
  engine: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma"
): Promise<TranscriptSegment[]> {
  return new Promise((resolve, reject) => {
    const asrScript = options.asrScript || DEFAULT_ASR_SCRIPT;

    const args = [
      asrScript,
      "--audio", audioPath,
      "--output", outputJson,
      "--model", options.model,
      "--device", options.device,
      "--engine", engine,
    ];

    if ((engine === "whisper" || engine === "qwen" || engine === "sensevoice" || engine === "gemma") && !Array.isArray(range)) {
      args.push("--start", range.start.toFixed(3));
      args.push("--end", range.end.toFixed(3));
      if (engine === "whisper") {
        args.push("--beam-size", options.repairBeamSize.toString());
        if (options.repairTemperature !== null) {
          args.push("--temperature", options.repairTemperature.toString());
        }
      }
      if (options.asrPrompt) {
        args.push("--prompt", options.asrPrompt);
      }
    } else if (engine === "mms") {
      const ranges = Array.isArray(range) ? range : [range];
      const intervals = ranges.map(r => `${r.start.toFixed(3)},${r.end.toFixed(3)}`).join("|");
      args.push("--intervals", intervals);
    }

    if (mixAudioPath) {
      args.push("--mix-audio", mixAudioPath);
      args.push("--mix-weight", options.mixWeight.toString());
    }

    const child = spawn(options.pythonExe, ["-u", ...args]);
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
        const data = JSON.parse(fs.readFileSync(outputJson, "utf-8")) as TranscriptFile;
        // Tag the segments with the engine used
        const segments = (data.segments || []).map(s => ({ ...s, engine }));
        resolve(segments);
      } catch (err) {
        reject(err);
      }
    });

    child.on("error", reject);
  });
}
