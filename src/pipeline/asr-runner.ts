import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { spawn, type ChildProcess } from "child_process";
import type { TranscriptFile, AudioTrack, TranscriptSegment, DemucsWindow } from "../util/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ASR_SCRIPT = path.resolve(__dirname, "../../asr/asr_cli.py");
const DEFAULT_DEMUCS_SCRIPT = path.resolve(__dirname, "../../asr/demucs_cli.py");

interface DemucsResult {
  windows: DemucsWindow[];
}

/**
 * Generic runner for Python CLI tools with sentinel-based termination.
 * Used for both ASR (Whisper) and Demucs separation.
 */
function runPythonCommand(
  pythonExe: string,
  scriptPath: string,
  args: string[],
  sentinel: string,
  label: string,
  timeoutMs: number,
): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`  [${label}] Running ${path.basename(scriptPath)}`);

    const child: ChildProcess = spawn(pythonExe, ["-u", scriptPath, ...args]);

    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      setTimeout(() => {
        reject(new Error(`${label} process timed out after ${timeoutMs / 1000}s`));
      }, 1000);
    }, timeoutMs);

    let stdoutBuf = "";
    let done = false;

    child.stdout?.on("data", (data: Buffer) => {
      const chunk = data.toString();
      process.stdout.write(chunk);
      stdoutBuf += chunk;
      if (!done && stdoutBuf.includes(sentinel)) {
        done = true;
        clearTimeout(timeout);
        child.kill("SIGKILL");
      }
    });

    child.stderr?.on("data", (data: Buffer) => process.stderr.write(data));

    child.on("close", (code: number | null) => {
      if (done) { resolve(); return; }
      reject(new Error(`${label} process exited with code ${code}`));
    });
  });
}

/**
 * Average Demucs 1s window data into a Whisper segment.
 */
export function mergeEnergyToSegment(seg: TranscriptSegment, windows: DemucsWindow[]) {
  const start = seg.start_time;
  const end = seg.end_time;
  
  const overlaps = windows.filter(w => w.start < end && w.end > start);
  if (overlaps.length === 0) return;
  
  const sumVocal = overlaps.reduce((acc, w) => acc + w.vocal_rms, 0);
  const sumOther = overlaps.reduce((acc, w) => acc + w.other_rms, 0);
  const sumSnr = overlaps.reduce((acc, w) => acc + w.snr_db, 0);
  
  seg.vocal_energy = sumVocal / overlaps.length;
  seg.other_energy = sumOther / overlaps.length;
  seg.snr = sumSnr / overlaps.length;
}

/**
 * Get transcription for an audio track.
 *
 * - In "skip" mode: looks for a pre-existing .json file next to the audio file.
 * - In "python" mode: runs Demucs separation and Whisper ASR.
 * - Results for both phases are cached in the output directory.
 */
export async function getTranscription(
  track: AudioTrack,
  outputDir: string,
  options: {
    asrMode: "python" | "skip";
    asrEngine: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma";
    pythonExe: string;
    asrScript?: string | undefined;
    demucsScript?: string | undefined;
    asrPrompt?: string | undefined;
    saveAudioStems?: boolean;
    useMmsRepair?: boolean;
    useQwenRepair?: boolean;
    useSenseVoiceRepair?: boolean;
    useGemmaRepair?: boolean;
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

  // Python mode
  const outDir = path.join(outputDir, track.relativeDir);
  await fs.mkdir(outDir, { recursive: true });

  const demucsPath = path.join(outDir, `${track.stem}.demucs.json`);
  const asrPath = path.join(outDir, `${track.stem}.raw-transcription.json`);

  // 1. Demucs Phase (Always run/cache)
  let demucsData: DemucsResult | null = null;
  try {
    const raw = await fs.readFile(demucsPath, "utf-8");
    demucsData = JSON.parse(raw);
    console.log(`  [ASR] Using cached Demucs results for ${track.relativePath}`);
  } catch {
    const demucsScript = options.demucsScript ?? DEFAULT_DEMUCS_SCRIPT;
    const demucsOutputDir = path.join(outDir, "demucs_output");
    const demucsArgs = ["--audio", track.absolutePath, "--output", demucsPath, "--audio-output-dir", demucsOutputDir];
    
    if (options.saveAudioStems) {
      await fs.mkdir(demucsOutputDir, { recursive: true });
      demucsArgs.push("--save-audio");
    }

    await runPythonCommand(
      options.pythonExe,
      demucsScript,
      demucsArgs,
      "[DEMUCS_DONE]",
      "Demucs",
      60 * 60 * 1000, // 1h timeout for separation
    );

    try {
      const raw = await fs.readFile(demucsPath, "utf-8");
      demucsData = JSON.parse(raw);
    } catch {
      console.error(`  [ASR] Failed to read Demucs output for ${track.relativePath}`);
    }
  }

  // 2. Whisper Phase
  let asrData: TranscriptFile | null = null;
  try {
    const raw = await fs.readFile(asrPath, "utf-8");
    asrData = JSON.parse(raw);
    console.log(`  [ASR] Using cached raw transcription for ${track.relativePath}`);
  } catch {
    const asrScript = options.asrScript ?? DEFAULT_ASR_SCRIPT;
    const asrArgs = [
      "--audio", track.absolutePath, 
      "--prompt", options.asrPrompt ?? "", 
      "--engine", options.asrEngine ?? "openai/whisper",
      "--output", asrPath,
    ];

    // Select the correct Python virtual environment
    // Qwen requires Transformers 4.x, so it has its own isolated env.
    // Others (Gemma, Whisper, SenseVoice) use the main env (Transformers 5.x).
    let enginePythonExe = options.pythonExe;
    if (options.asrEngine === "qwen") {
      const asrDir = path.dirname(asrScript);
      const qwenVenvExe = path.join(asrDir, "qwen_env", ".venv", "Scripts", "python.exe");
      try {
        await fs.access(qwenVenvExe);
        enginePythonExe = qwenVenvExe;
      } catch {
        console.warn(`  [ASR] Isolated Qwen environment not found at ${qwenVenvExe}. Falling back to default.`);
      }
    }

    await runPythonCommand(
      enginePythonExe,
      asrScript,
      asrArgs,
      "[ASR_DONE]",
      "ASR",
      45 * 60 * 1000,
    );

    try {
      const raw = await fs.readFile(asrPath, "utf-8");
      asrData = JSON.parse(raw);
    } catch {
      console.error(`  [ASR] Failed to read ASR output for ${track.relativePath}`);
    }
  }

  // 3. Post-processing: Merge Energy Data
  if (asrData && demucsData) {
    asrData.demucs_windows = demucsData.windows;
    for (const seg of asrData.segments) {
      mergeEnergyToSegment(seg, demucsData.windows);
    }
  }

  return asrData;
}
