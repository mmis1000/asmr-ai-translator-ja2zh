import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { spawn, type ChildProcess } from "child_process";
import type { TranscriptFile, AudioTrack } from "../util/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ASR_SCRIPT = path.resolve(__dirname, "../../asr/asr_cli.py");

/**
 * Run Whisper ASR on a single audio file via Python CLI.
 * Uses sentinel-based termination to handle CTranslate2/ROCm hang on Windows.
 */
function runWhisperAsr(
  pythonExe: string,
  asrScript: string,
  audioPath: string,
  asrPrompt: string,
  outputPath: string,
  timeoutMs = 30 * 60 * 1000,
): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`  [ASR] Running Whisper on ${path.basename(audioPath)}`);

    const child: ChildProcess = spawn(pythonExe, [
      "-u",
      asrScript,
      "--audio", audioPath,
      "--prompt", asrPrompt,
      "--output", outputPath,
    ]);

    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      setTimeout(() => {
        reject(new Error(`ASR process timed out after ${timeoutMs / 1000}s`));
      }, 1000);
    }, timeoutMs);

    // CTranslate2+ROCm cannot exit cleanly on Windows.
    // Python prints [ASR_DONE] after flushing JSON, then blocks.
    const SENTINEL = "[ASR_DONE]";
    let stdoutBuf = "";
    let done = false;

    child.stdout?.on("data", (data: Buffer) => {
      const chunk = data.toString();
      process.stdout.write(chunk);
      stdoutBuf += chunk;
      if (!done && stdoutBuf.includes(SENTINEL)) {
        done = true;
        clearTimeout(timeout);
        child.kill("SIGKILL");
      }
    });
    child.stderr?.on("data", (data: Buffer) => process.stderr.write(data));

    child.on("close", (code: number | null) => {
      if (done) { resolve(); return; }
      reject(new Error(`ASR process exited with code ${code}`));
    });
  });
}

/**
 * Get transcription for an audio track.
 *
 * - In "skip" mode: looks for a pre-existing .json file next to the audio file.
 * - In "python" mode: runs Whisper ASR and writes output to the output directory.
 */
export async function getTranscription(
  track: AudioTrack,
  outputDir: string,
  options: {
    asrMode: "python" | "skip";
    pythonExe: string;
    asrScript?: string | undefined;
    asrPrompt?: string | undefined;
  },
): Promise<TranscriptFile | null> {
  if (options.asrMode === "skip") {
    // Look for pre-existing JSON next to audio file
    const jsonPath = track.absolutePath.replace(/\.[^.]+$/, ".json");
    try {
      const content = await fs.readFile(jsonPath, "utf-8");
      return JSON.parse(content) as TranscriptFile;
    } catch {
      console.warn(`  [ASR] No pre-transcribed JSON found for ${track.relativePath}`);
      return null;
    }
  }

  // Python ASR mode
  const outDir = path.join(outputDir, track.relativeDir);
  await fs.mkdir(outDir, { recursive: true });
  const outputPath = path.join(outDir, `${track.stem}.transcription.json`);

  // Check if already transcribed
  try {
    const content = await fs.readFile(outputPath, "utf-8");
    console.log(`  [ASR] Using cached transcription for ${track.relativePath}`);
    return JSON.parse(content) as TranscriptFile;
  } catch {
    // Not cached, run ASR
  }

  const asrScript = options.asrScript ?? DEFAULT_ASR_SCRIPT;
  await runWhisperAsr(
    options.pythonExe,
    asrScript,
    track.absolutePath,
    options.asrPrompt ?? "",
    outputPath,
  );

  try {
    const content = await fs.readFile(outputPath, "utf-8");
    return JSON.parse(content) as TranscriptFile;
  } catch {
    console.error(`  [ASR] Failed to read ASR output for ${track.relativePath}`);
    return null;
  }
}
