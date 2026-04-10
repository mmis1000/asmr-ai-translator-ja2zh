import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pythonSubPath = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
/** Root of `translator/asr` (contains `pyproject.toml`, `uv.lock`, `.venv`). */
export const ASR_PROJECT_ROOT = path.resolve(__dirname, "../asr");
const DEFAULT_ASR_PYTHON = path.resolve(ASR_PROJECT_ROOT, pythonSubPath);

/**
 * Python for the isolated Qwen3-ASR venv (`qwen_env/.venv`, see `translator/asr/qwen_env/`).
 * `asrProjectDir` is usually the folder containing `asr_cli.py` (defaults to `ASR_PROJECT_ROOT`).
 */
export function qwenVenvPythonPath(asrProjectDir: string): string {
  const segments =
    process.platform === "win32"
      ? ["qwen_env", ".venv", "Scripts", "python.exe"]
      : ["qwen_env", ".venv", "bin", "python"];
  return path.join(asrProjectDir, ...segments);
}

export interface TranslatorConfig {
  // llama-server (translation model — fine-tuned)
  llamaServerExe: string;
  modelPath: string;
  hfRepo?: string | undefined;          // HuggingFace repo "user/model[:quant]" — overrides modelPath
  serverPort: number;
  gpuLayers: number | "auto" | "all";   // "all" forces every layer to GPU; "auto" lets llama-server decide
  contextSize: number;
  parallel: number;
  serverUrl?: string | undefined;  // external server mode — skip internal management

  // Metadata extraction model (general-purpose, NOT the fine-tuned one)
  metaModelPath?: string | undefined;    // separate GGUF for metadata extraction
  metaHfRepo?: string | undefined;       // HuggingFace repo for metadata model — overrides metaModelPath
  metaServerUrl?: string | undefined;    // or point to an external server
  metaServerPort: number;                // port for the meta model server (default: 8182)
  metaContextSize: number;               // ctx-size for the meta model (default: 16384 — extraction output can be long)

  // Sampling — translation model (fine-tuned, low temperature)
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
  /** Fixed RNG seed for reproducible outputs; omit (undefined) for random. */
  seed?: number | undefined;
  // Sampling — metadata extraction model (general-purpose / thinking, higher temperature)
  metaTemperature: number;

  // Translation
  locale: "zh-tw" | "zh-cn";
  mode: "base" | "echo";

  // ASR
  asrMode: "python" | "skip";
  asrEngine: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma";
  pythonExe: string;
  asrScript?: string | undefined;
  demucsScript?: string | undefined;

  // Demucs Filtering
  vocalEnergyThreshold: number;
  vocalSilenceThreshold: number;
  snrThreshold: number;
  minHallucinationLength: number;
  rejectNegativeSnr: boolean;
  resplitGapSec: number;
  maxRepetitions: number;
  repairTemperature: number | null;
  repairBeamSize: number;
  repairWithVocal: boolean;
  mixWeight: number;
  saveAudio: boolean;
  repairLargeV3: boolean;
  saveAudioStems: boolean;
  repairEngine: "whisper" | "mms" | "qwen" | "sensevoice" | "gemma";
  saveRepairAudio: boolean;

  // Paths
  inputDir: string;
  outputDir: string;
  metadataFile?: string | undefined;
  dlsiteId?: string | undefined;
  /** Video/audio URL for yt-dlp — downloads into `inputDir` and supplies metadata. */
  ytdlpUrl?: string | undefined;
  /** `uv` executable used to run `uv run --project … yt-dlp`. */
  uvExe: string;
  /** Passed to yt-dlp `-x` / `--audio-format` when not `"best"`. */
  ytdlpAudioFormat: string;

  // Debug
  debugLog: boolean;
}

export const DEFAULT_CONFIG: Omit<TranslatorConfig,
  "modelPath" | "inputDir" | "outputDir"
> = {
  llamaServerExe: "llama-server",
  serverPort:     8181,
  gpuLayers:      "all",
  contextSize:    8192,
  parallel:       1,
  metaServerPort:    8182,
  metaContextSize:   16384,
  temperature:    0,   // fine-tuned translation model — low, deterministic
  repeatPenalty:  1.1,   // penalize repetition to prevent attention collapse loops
  metaTemperature: 0.7,  // general-purpose model (Qwen3 non-thinking default)
  topP:           0.95,
  topK:           20,
  minP:           0.0,
  locale:         "zh-tw",
  mode:           "echo",
  asrMode:        "skip",
  asrEngine:      "whisper",
  pythonExe:      DEFAULT_ASR_PYTHON,
  vocalEnergyThreshold: 0.001,
  vocalSilenceThreshold: 0.00005,
  snrThreshold:   2.0,
  minHallucinationLength: 20,
  rejectNegativeSnr: true,
  resplitGapSec: 1.0,
  maxRepetitions: 3,
  repairTemperature: null,
  repairBeamSize: 10,
  saveAudio: false,
  repairWithVocal: false,
  mixWeight: 0.07,
  repairLargeV3: false,
  saveAudioStems: false,
  repairEngine: "qwen",
  saveRepairAudio: false,
  uvExe:          "uv",
  ytdlpAudioFormat: "best",
  debugLog:       false,
};
