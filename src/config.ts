export interface TranslatorConfig {
  // llama-server (translation model — fine-tuned)
  llamaServerExe: string;
  modelPath: string;
  serverPort: number;
  gpuLayers: number;
  contextSize: number;
  parallel: number;
  serverUrl?: string | undefined;  // external server mode — skip internal management

  // Metadata extraction model (general-purpose, NOT the fine-tuned one)
  metaModelPath?: string | undefined;    // separate GGUF for metadata extraction
  metaServerUrl?: string | undefined;    // or point to an external server
  metaServerPort: number;                // port for the meta model server (default: 8182)

  // Sampling
  temperature: number;
  topP: number;
  topK: number;
  minP: number;

  // Translation
  locale: "zh-tw" | "zh-cn";
  mode: "base" | "echo";

  // ASR
  asrMode: "python" | "skip";
  pythonExe: string;
  asrScript?: string | undefined;

  // Paths
  inputDir: string;
  outputDir: string;
  metadataFile?: string | undefined;
  dlsiteId?: string | undefined;
}

export const DEFAULT_CONFIG: Omit<TranslatorConfig,
  "modelPath" | "inputDir" | "outputDir"
> = {
  llamaServerExe: "llama-server",
  serverPort:     8181,
  gpuLayers:      99,
  contextSize:    8192,
  parallel:       1,
  metaServerPort: 8182,
  temperature:    0.6,
  topP:           0.95,
  topK:           20,
  minP:           0.0,
  locale:         "zh-tw",
  mode:           "echo",
  asrMode:        "skip",
  pythonExe:      "python",
};
