import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import type { TranscriptSegment, TranscriptFile, DemucsWindow } from "../util/types.js";
import type { TimeRange } from "./transcript-cleaner.js";
import { isGarbled } from "./transcript-cleaner.js";
import { mergeEnergyToSegment } from "./asr-runner.js";

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
  repairTemperature: number;
  repairBeamSize: number;
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
  existingLog: SurgicalRepairEntry[] = []
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

  for (const range of windowsToRepair) {
    console.log(`  -> Repairing [${range.start.toFixed(1)}s - ${range.end.toFixed(1)}s] (${range.reason})...`);

    // Capture original segments in this range for logging
    const originalInRange = initialSegments.filter(s => 
      s.start_time >= range.start && s.end_time <= range.end
    );

    const repairFile = path.join(path.dirname(audioPath), `.tmp_repair_${range.start.toFixed(0)}.json`);
    
    try {
      const newSegments = await runSurgicalWhisper(audioPath, range, repairFile, options);
      
      const repairEntry: SurgicalRepairEntry = {
        range,
        originalSegments: originalInRange,
        newSegments: [], // To be populated
        status: "no_change"
      };

      if (newSegments.length > 0) {
        repairEntry.newSegments = newSegments;
        
        // Remove old segments that fall within or overlap significantly with this range
        currentSegments = currentSegments.filter(s => 
          s.end_time <= range.start || s.start_time >= range.end
        );
        // Also remove from mismatches in this range
        currentMismatches = currentMismatches.filter(m => 
          m.end_time <= range.start || m.start_time >= range.end
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
          });

          if (!garbled) {
            passed.push(s);
          } else if (s.mismatch) {
            currentMismatches.push(s);
          }
        }

        if (passed.length > 0) {
          currentSegments.push(...passed);
          repairEntry.status = "success";
          console.log(`     Repair successful: ${passed.length} new valid segments added.`);
        } else {
          repairEntry.status = "failed";
          console.log(`     Repair produced no valid content (filtered by Demucs).`);
        }
      } else {
        console.log(`     Repair produced no new content.`);
      }

      surgicalLog.push(repairEntry);
    } catch (err) {
      console.error(`     Repair failed for range ${range.start}-${range.end}:`, err);
    } finally {
      if (fs.existsSync(repairFile)) fs.unlinkSync(repairFile);
    }
  }

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

    // Remove original segments in this range
    currentSegments = currentSegments.filter(s => 
      s.end_time <= range.start || s.start_time >= range.end
    );
    currentMismatches = currentMismatches.filter(m => 
      m.end_time <= range.start || m.start_time >= range.end
    );

    // Add back the repaired ones
    currentSegments.push(...entry.newSegments);
    
    // Note: Mismatches in the repairs themselves are handled during the initial LOG generation,
    // so we assume entry.newSegments are already the "cleaned" products.
  }

  // Final sort
  currentSegments.sort((a, b) => a.start_time - b.start_time);
  currentMismatches.sort((a, b) => a.start_time - b.start_time);

  return { segments: currentSegments, mismatches: currentMismatches };
}

async function runSurgicalWhisper(
  audioPath: string,
  range: TimeRange,
  outputJson: string,
  options: RepairOptions
): Promise<TranscriptSegment[]> {
  return new Promise((resolve, reject) => {
    // Non-greedy settings to break loops: configurable temp/beam, no condition on previous
    const args = [
      options.asrScript,
      "--audio", audioPath,
      "--output", outputJson,
      "--model", options.model,
      "--device", options.device,
      "--start", range.start.toString(),
      "--end", range.end.toString(),
      "--temperature", options.repairTemperature.toString(),
      "--beam-size", options.repairBeamSize.toString(),
      "--no-condition"
    ];

    const child = spawn(options.pythonExe, args);
    let output = "";

    child.stdout.on("data", (data) => {
      output += data.toString();
      // Handle the ROCm hang sentinel
      if (output.includes("[ASR_DONE]")) {
        child.kill("SIGKILL");
      }
    });

    child.on("close", (code) => {
      if (!fs.existsSync(outputJson)) {
        resolve([]);
        return;
      }

      try {
        const data = JSON.parse(fs.readFileSync(outputJson, "utf-8")) as TranscriptFile;
        resolve(data.segments || []);
      } catch (err) {
        reject(err);
      }
    });

    child.on("error", reject);
  });
}
