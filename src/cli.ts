#!/usr/bin/env node

import fs from "fs/promises";
import path from "path";
import { parseArgs } from "node:util";
import { type TranslatorConfig, DEFAULT_CONFIG } from "./config.js";
import { LlamaServerManager } from "./server/llama-server-manager.js";
import { LlmClient } from "./server/llm-client.js";
import { cleanTranscript } from "./pipeline/transcript-cleaner.js";
import { getTranscription } from "./pipeline/asr-runner.js";
import { fetchDlsiteMetadata, parseDlsiteId } from "./pipeline/fetch-metadata.js";
import { MetadataExtractor } from "./pipeline/metadata-extractor.js";
import {  repairTranscription,
  applySurgicalRepair,
} from "./pipeline/asr-repairer.js";
import { 
  writeTranscription, 
  writeTranslation, 
  writeMetadata, 
  writeWindowResults, 
  writeSurgicalLog, 
  readSurgicalLog,
} from "./pipeline/output-writer.js";
import { asrToSegments, translateTrack } from "./pipeline/translator.js";
import type { AudioTrack, GlossaryLang, UserMetadata, FinalMetadata } from "./util/types.js";


// ── CLI argument parsing ─────────────────────────────────────────────────────

function printUsage(): void {
  console.log(`
asmr-translator — Translate Japanese ASMR audio using a GGUF model

Usage:
  npx asmr-translator --input <dir> --output <dir> --model <path> [options]

Required:
  --input <dir>            Local directory of audio files
  --output <dir>           Output directory (separate from input)
  --model <path>           Fine-tuned translation GGUF model (required unless --hf-repo or --server-url)
  --hf-repo <repo>         HuggingFace repo for translation model ("user/model[:quant]" or full HF URL)

Metadata (optional, pick one):
  --dlsite <id|url>        DLSite work ID or URL — scrapes metadata for context
  --metadata <file>        User-supplied metadata JSON

Metadata extraction (requires a general-purpose model, NOT the translation model):
  --meta-model <path>      GGUF model for metadata extraction (used with --dlsite)
  --meta-hf-repo <repo>    HuggingFace repo for metadata model ("user/model[:quant]" or full HF URL)
  --meta-server-url <url>  External server URL for metadata extraction
  --meta-ctx-size <number> Context size for metadata model (default: 16384)

Translation server:
  --llama-server <path>    llama-server executable (default: in PATH)
  --server-url <url>       External llama-server URL for translation
  --port <number>          Translation server port (default: 8181)
  --gpu-layers <n|all|auto>  GPU layers to offload (default: all — forces every layer to GPU)
  --ctx-size <number>      Context size (default: 8192)
  --parallel <number>      Parallel inference slots (default: 1)

Translation:
  --lang <zh-tw|zh-cn>     Target language (default: zh-tw)
  --mode <base|echo>       Translation mode (default: echo)
  --seed <number>          Fixed RNG seed for reproducible output (default: random)
  --debug-log              Write LLM prompt and responses to debug_logs folder

ASR:
  --asr <python|skip>      ASR mode (default: skip)
  --python-exe <path>      Python executable (default: asr/.venv/Scripts/python.exe)
  --asr-script <path>      Path to asr_cli.py
  --demucs-script <path>   Path to demucs_cli.py
  --vocal-threshold <n>    Vocal energy threshold (default: 0.001)
  --snr-threshold <n>      SNR dB threshold (default: 2.0)
  --min-hallucination <n>  Min text length to filter (default: 20)
  --repair-with-vocal      Use Demucs vocal stem for repair pass (forces --save-audio)
  --save-audio             Save separated audio stems for debugging
  --mms-repair             Enable MMS ASR as a fallback during surgical repair
  --qwen-repair            Enable Qwen-ASR as a fallback during surgical repair
  --sensevoice-repair      Enable SenseVoice-ASR as a fallback during surgical repair
  --gemma-repair           Enable Gemma-ASR as a fallback during surgical repair
  --asr-engine <engine>    ASR engine to use: whisper, mms, qwen, sensevoice, gemma (default: whisper)
  --save-repair-audio      Save audio fragments used for surgical repairs to a dedicated directory

Note: --dlsite with LLM extraction requires --meta-model or --meta-server-url.
      Without these, --dlsite still scrapes DLSite for basic metadata (title, VA,
      description) but won't produce a structured glossary.
  `);
}

function parseCliArgs(): TranslatorConfig {
  const { values } = parseArgs({
    options: {
      input:             { type: "string" },
      output:            { type: "string" },
      model:             { type: "string" },
      "hf-repo":         { type: "string" },
      dlsite:            { type: "string" },
      metadata:          { type: "string" },
      "meta-model":      { type: "string" },
      "meta-hf-repo":    { type: "string" },
      "meta-server-url": { type: "string" },
      "meta-ctx-size":   { type: "string" },
      "llama-server":    { type: "string" },
      "server-url":      { type: "string" },
      port:              { type: "string" },
      "gpu-layers":      { type: "string" },
      "ctx-size":        { type: "string" },
      parallel:          { type: "string" },
      lang:              { type: "string" },
      mode:              { type: "string" },
      seed:              { type: "string" },
      "debug-log":       { type: "boolean" },
      asr:               { type: "string" },
      "python-exe":      { type: "string" },
      "asr-script":      { type: "string" },
      "demucs-script":   { type: "string" },
      "vocal-threshold": { type: "string" },
      "snr-threshold": { type: "string" },
      "vocal-silence-threshold": { type: "string" },
      "min-hallucination-length": { type: "string" },
      "reject-negative-snr": { type: "boolean" },
      "repair-temp": { type: "string" },
      "repair-beam": { type: "string" },
      "mix-weight": { type: "string" },
      "force-repair": { type: "boolean" },
      "force-asr": { type: "boolean" },
      "repair-with-vocal": { type: "boolean" },
      "save-audio": { type: "boolean" },
      "repair-large-v3": { type: "boolean" },
      "mms-repair":      { type: "boolean" },
      "qwen-repair":     { type: "boolean" },
      "sensevoice-repair": { type: "boolean" },
      "gemma-repair":      { type: "boolean" },
      "asr-engine":      { type: "string" },
      "save-repair-audio": { type: "boolean" },
      help:              { type: "boolean", short: "h" },
    },
    strict: true,
  });

  if (values.help) {
    printUsage();
    process.exit(0);
  }

  if (!values.input) { console.error("Error: --input is required"); printUsage(); process.exit(1); }
  if (!values.output) { console.error("Error: --output is required"); printUsage(); process.exit(1); }
  if (!values.model && !values["hf-repo"] && !values["server-url"]) {
    console.error("Error: --model or --hf-repo is required (unless --server-url is provided)");
    printUsage();
    process.exit(1);
  }

  const lang = values.lang as "zh-tw" | "zh-cn" | undefined;
  if (lang && lang !== "zh-tw" && lang !== "zh-cn") {
    console.error("Error: --lang must be 'zh-tw' or 'zh-cn'"); process.exit(1);
  }

  const mode = values.mode as "base" | "echo" | undefined;
  if (mode && mode !== "base" && mode !== "echo") {
    console.error("Error: --mode must be 'base' or 'echo'"); process.exit(1);
  }

  const asrMode = values.asr as "python" | "skip" | undefined;
  if (asrMode && asrMode !== "python" && asrMode !== "skip") {
    console.error("Error: --asr must be 'python' or 'skip'"); process.exit(1);
  }

  const asrEngine = values["asr-engine"] as "whisper" | "mms" | "qwen" | "sensevoice" | "gemma" | undefined;
  if (asrEngine && !["whisper", "mms", "qwen", "sensevoice", "gemma"].includes(asrEngine)) {
    console.error("Error: --asr-engine must be 'whisper', 'mms', 'qwen', 'sensevoice', or 'gemma'"); process.exit(1);
  }

  const gpuLayersRaw = values["gpu-layers"];
  const gpuLayers: number | "auto" | "all" =
    gpuLayersRaw === "all" || gpuLayersRaw === "auto"
      ? gpuLayersRaw
      : gpuLayersRaw
        ? parseInt(gpuLayersRaw, 10)
        : DEFAULT_CONFIG.gpuLayers;

  return {
    ...DEFAULT_CONFIG,
    inputDir: path.resolve(values.input as string),
    outputDir: path.resolve(values.output as string),
    modelPath: values.model ? path.resolve(values.model as string) : "",
    hfRepo: values["hf-repo"] ? parseHfRepo(values["hf-repo"] as string) : undefined,
    dlsiteId: values.dlsite as string | undefined,
    metadataFile: values.metadata ? path.resolve(values.metadata as string) : undefined,
    metaModelPath: values["meta-model"] ? path.resolve(values["meta-model"] as string) : undefined,
    metaHfRepo: values["meta-hf-repo"] ? parseHfRepo(values["meta-hf-repo"] as string) : undefined,
    metaServerUrl: values["meta-server-url"] as string | undefined,
    metaContextSize: values["meta-ctx-size"] ? parseInt(values["meta-ctx-size"] as string, 10) : DEFAULT_CONFIG.metaContextSize,
    llamaServerExe: (values["llama-server"] as string) ?? DEFAULT_CONFIG.llamaServerExe,
    serverUrl: values["server-url"] as string | undefined,
    serverPort: values.port ? parseInt(values.port as string, 10) : DEFAULT_CONFIG.serverPort,
    gpuLayers,
    contextSize: values["ctx-size"] ? parseInt(values["ctx-size"] as string, 10) : DEFAULT_CONFIG.contextSize,
    parallel: values.parallel ? parseInt(values.parallel as string, 10) : DEFAULT_CONFIG.parallel,
    locale: lang ?? DEFAULT_CONFIG.locale,
    mode: mode ?? DEFAULT_CONFIG.mode,
    seed: values.seed !== undefined ? parseInt(values.seed as string, 10) : undefined,
    debugLog: (values["debug-log"] as boolean) ?? DEFAULT_CONFIG.debugLog,
    asrMode: asrMode ?? DEFAULT_CONFIG.asrMode,
    pythonExe: (values["python-exe"] as string) ?? DEFAULT_CONFIG.pythonExe,
    asrScript: values["asr-script"] as string | undefined,
    demucsScript: values["demucs-script"] as string | undefined,
    vocalEnergyThreshold: values["vocal-threshold"] ? parseFloat(values["vocal-threshold"] as string) : DEFAULT_CONFIG.vocalEnergyThreshold,
    vocalSilenceThreshold: values["vocal-silence-threshold"] ? parseFloat(values["vocal-silence-threshold"] as string) : DEFAULT_CONFIG.vocalSilenceThreshold,
    snrThreshold: values["snr-threshold"] ? parseFloat(values["snr-threshold"] as string) : DEFAULT_CONFIG.snrThreshold,
    minHallucinationLength: values["min-hallucination-length"] ? parseInt(values["min-hallucination-length"] as string) : DEFAULT_CONFIG.minHallucinationLength,
    rejectNegativeSnr: (values["reject-negative-snr"] as boolean) ?? DEFAULT_CONFIG.rejectNegativeSnr,
    repairTemperature: values["repair-temp"] ? parseFloat(values["repair-temp"] as string) : DEFAULT_CONFIG.repairTemperature,
    repairBeamSize: values["repair-beam"] ? parseInt(values["repair-beam"] as string) : DEFAULT_CONFIG.repairBeamSize,
    repairWithVocal:   (values["repair-with-vocal"] as boolean) ?? DEFAULT_CONFIG.repairWithVocal,
    mixWeight:         values["mix-weight"] ? parseFloat(values["mix-weight"] as string) : DEFAULT_CONFIG.mixWeight,
    saveAudioStems:    ((values["repair-with-vocal"] as boolean) ?? DEFAULT_CONFIG.repairWithVocal) ? true : ((values["save-audio"] as boolean) ?? DEFAULT_CONFIG.saveAudioStems),
    repairLargeV3:     (values["repair-large-v3"] as boolean)   ?? DEFAULT_CONFIG.repairLargeV3,
    useMmsRepair:      (values["mms-repair"] as boolean)        ?? DEFAULT_CONFIG.useMmsRepair,
    useQwenRepair:     (values["qwen-repair"] as boolean)       ?? DEFAULT_CONFIG.useQwenRepair,
    useSenseVoiceRepair: (values["sensevoice-repair"] as boolean) ?? DEFAULT_CONFIG.useSenseVoiceRepair,
    useGemmaRepair:      (values["gemma-repair"] as boolean)      ?? DEFAULT_CONFIG.useGemmaRepair,
    asrEngine:         asrEngine                   ?? DEFAULT_CONFIG.asrEngine,
    saveRepairAudio:   (values["save-repair-audio"] as boolean) ?? DEFAULT_CONFIG.saveRepairAudio,
  };
}

/** Strip the HuggingFace base URL prefix if present, leaving just "user/model[:quant]". */
function parseHfRepo(input: string): string {
  const base = "https://huggingface.co/";
  return input.startsWith(base) ? input.slice(base.length) : input;
}

// ── Audio discovery ──────────────────────────────────────────────────────────

const AUDIO_EXTS = new Set([".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]);

async function discoverAudioTracks(inputDir: string): Promise<AudioTrack[]> {
  const tracks: AudioTrack[] = [];

  async function walk(dir: string, relDir: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const absPath = path.join(dir, entry.name);
      const rel = relDir ? `${relDir}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        await walk(absPath, rel);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (AUDIO_EXTS.has(ext)) {
          tracks.push({
            absolutePath: absPath,
            relativePath: rel,
            stem: path.basename(entry.name, ext),
            relativeDir: relDir,
          });
        }
      }
    }
  }

  await walk(inputDir, "");
  return tracks;
}

// ── Main pipeline ────────────────────────────────────────────────────────────

async function main() {
  const config = parseCliArgs();

  console.log(`=== asmr-translator ===`);
  console.log(`Input:    ${config.inputDir}`);
  console.log(`Output:   ${config.outputDir}`);
  console.log(`Language: ${config.locale}`);
  console.log(`Mode:     ${config.mode}`);
  console.log(`ASR:      ${config.asrMode}`);
  console.log();

  // ── Step 1: Discover audio tracks ────────────────────────────────────────

  const tracks = await discoverAudioTracks(config.inputDir);
  if (tracks.length === 0) {
    console.error("No audio files found in input directory.");
    process.exit(1);
  }
  console.log(`Found ${tracks.length} audio track(s)`);

  // ── Step 2: Resolve metadata ─────────────────────────────────────────────

  let glossary: GlossaryLang = { cvs: [], characters: [], terms: [], summary: "" };
  let outputMetadata: FinalMetadata | UserMetadata = {};

  /** Load an existing metadata.json from the output dir if present, returning true on success. */
  async function tryLoadCachedMetadata(): Promise<boolean> {
    const cachedPath = `${config.outputDir}/metadata.json`;
    try {
      const raw = await fs.readFile(cachedPath, "utf-8");
      const parsed = JSON.parse(raw) as FinalMetadata | UserMetadata;
      if ("translate" in parsed) {
        // FinalMetadata
        const m = parsed as FinalMetadata;
        outputMetadata = m;
        glossary = {
          cvs: m.translate.cv_mapping,
          characters: m.translate.character_mapping,
          terms: m.translate.term_mapping,
          summary: m.translate.summary,
        };
      } else {
        // UserMetadata
        const m = parsed as UserMetadata;
        outputMetadata = m;
        glossary = {
          cvs: m.glossary?.cvs ?? [],
          characters: m.glossary?.characters ?? [],
          terms: m.glossary?.terms ?? [],
          summary: m.summary ?? "",
        };
      }
      console.log(`[Metadata] Loaded cached metadata from output dir`);
      return true;
    } catch {
      return false;
    }
  }

  if (config.dlsiteId) {
    if (await tryLoadCachedMetadata()) {
      // Already extracted — skip re-running LLM
    } else {
      const dlsiteId = parseDlsiteId(config.dlsiteId);
      const dlsite = await fetchDlsiteMetadata(dlsiteId);

      const hasMetaModel = config.metaModelPath || config.metaServerUrl || config.metaHfRepo;

      if (dlsite.metadataMd && hasMetaModel) {
        // Full LLM extraction using a separate general-purpose model
        const fileList = tracks.map(t => `- ${t.relativePath}`).join("\n");
        const fullMd = dlsite.metadataMd + `\n# File List\n\n${fileList}\n`;

        const metaServer = new LlamaServerManager({
          llamaServerExe: config.llamaServerExe,
          modelPath: config.metaModelPath ?? "",
          hfRepo: config.metaHfRepo,
          serverPort: config.metaServerPort,
          gpuLayers: config.gpuLayers,
          contextSize: config.metaContextSize,
          parallel: 1,
          serverUrl: config.metaServerUrl,
        }, "MetaServer");

        try {
          await metaServer.start();
          const metaClient = new LlmClient(metaServer.baseUrl, { ...config, temperature: config.metaTemperature, repeatPenalty: 1.0 });
          const extractor = new MetadataExtractor(metaClient, config.locale, config.seed);
          const result = await extractor.extract(fullMd);
          glossary = result.glossary;
          outputMetadata = result.metadata;
        } finally {
          await metaServer.stop();
        }
      } else if (dlsite.metadataMd) {
        // Scrape-only fallback: build basic metadata without LLM
        console.log(`[Metadata] No --meta-model provided. Using scraped metadata only (no glossary extraction).`);
        outputMetadata = {
          title: dlsite.title,
          summary: dlsite.description,
          glossary: {
            // Use VA names as basic CV entries (Japanese only, no translation)
            cvs: dlsite.va
              ? dlsite.va.split(/[,、／/]/).map(v => v.trim()).filter(Boolean).map(v => ({ ja: v, zh: v }))
              : [],
            characters: [],
            terms: [],
          },
        } satisfies UserMetadata;
        glossary = {
          cvs: (outputMetadata as UserMetadata).glossary?.cvs ?? [],
          characters: [],
          terms: [],
          summary: dlsite.description ?? "",
        };
      }
    }
  } else if (config.metadataFile) {
    const raw = await fs.readFile(config.metadataFile, "utf-8");
    const parsed = JSON.parse(raw) as FinalMetadata | UserMetadata;
    outputMetadata = parsed;
    if ("translate" in parsed) {
      const m = parsed as FinalMetadata;
      glossary = {
        cvs: m.translate.cv_mapping,
        characters: m.translate.character_mapping,
        terms: m.translate.term_mapping,
        summary: m.translate.summary,
      };
    } else {
      const m = parsed as UserMetadata;
      glossary = {
        cvs: m.glossary?.cvs ?? [],
        characters: m.glossary?.characters ?? [],
        terms: m.glossary?.terms ?? [],
        summary: m.summary ?? "",
      };
    }
    console.log(`[Metadata] Loaded user-supplied metadata`);
  } else {
    console.log(`[Metadata] No metadata source — translating without glossary/context`);
  }

  await writeMetadata(config.outputDir, outputMetadata);

  // ── Step 3: ASR transcription (all tracks, before starting LLM server) ──

  interface TrackWithTranscript {
    track: AudioTrack;
    cleaned: import("./util/types.js").TranscriptSegment[];
  }
  const readyTracks: TrackWithTranscript[] = [];
  let skipped = 0;

  const asrPrompt = ("asr" in outputMetadata)
    ? (outputMetadata as FinalMetadata).asr.prompt
    : undefined;

  for (const track of tracks) {
    console.log(`\n[ASR ${readyTracks.length + skipped + 1}/${tracks.length}] ${track.relativePath}`);

    const transcript = await getTranscription(track, config.outputDir, {
      asrMode: config.asrMode,
      asrEngine: config.asrEngine,
      pythonExe: config.pythonExe,
      asrScript: config.asrScript,
      demucsScript: config.demucsScript,
      asrPrompt,
      saveAudioStems: config.saveAudioStems,
    });

    if (!transcript) {
      console.log(`  Skipped (no transcription)`);
      skipped++;
      continue;
    }

    let { segments: cleaned, mismatches, repairRanges } = cleanTranscript(transcript, {
      vocalThreshold: config.vocalEnergyThreshold,
      vocalSilenceThreshold: config.vocalSilenceThreshold,
      snrThreshold: config.snrThreshold,
      minHallLength: config.minHallucinationLength,
      rejectNegativeSnr: config.rejectNegativeSnr,
      resplitGapSec: config.resplitGapSec,
      maxRepetitions: config.maxRepetitions,
    });

    // --- NEW: Surgical ASR Repair ---
    // 1. Load existing repairs if available
    const cachedLog = await readSurgicalLog(config.outputDir, track.relativeDir, track.stem);
    if (cachedLog) {
      const merged = applySurgicalRepair(cleaned, mismatches, cachedLog);
      cleaned = merged.segments;
      mismatches = merged.mismatches;
    }

    // 2. Re-detect missed speech on the post-repair segments (gaps may have changed).
    //    The first-pass repairRanges already cover repetition loops and uncertain
    //    segments; this second pass only adds newly visible missed-speech windows.
    const { repairRanges: postRepairRanges } = cleanTranscript({ ...transcript, segments: cleaned }, {
      vocalThreshold: config.vocalEnergyThreshold,
      vocalSilenceThreshold: config.vocalSilenceThreshold,
      snrThreshold: config.snrThreshold,
      minHallLength: config.minHallucinationLength,
      rejectNegativeSnr: config.rejectNegativeSnr,
      resplitGapSec: config.resplitGapSec,
      maxRepetitions: config.maxRepetitions,
    });

    // Merge: first-pass ranges (repetition/uncertain/missed) + any new missed-speech
    // ranges uncovered after cached repairs were applied. repairTranscription will
    // skip ranges already present in the surgical log, so duplicates are harmless.
    const rangeKey = (r: { start: number; end: number }) => `${r.start.toFixed(1)}-${r.end.toFixed(1)}`;
    const seenKeys = new Set(repairRanges.map(rangeKey));
    const finalRepairRanges = [...repairRanges];
    for (const r of postRepairRanges) {
      if (!seenKeys.has(rangeKey(r))) finalRepairRanges.push(r);
    }

    let surgicalLog: any[] = cachedLog || [];

    if (finalRepairRanges.length > 0) {
      const vocalPath = path.join(config.outputDir, track.relativeDir, "demucs_output", `${track.stem}.vocals.wav`);

      const repaired = await repairTranscription(
        track.absolutePath,
        cleaned,
        mismatches,
        finalRepairRanges,
        transcript.demucs_windows || [],
        {
          pythonExe: config.pythonExe,
          asrScript: config.asrScript || "",
          model: config.repairLargeV3 ? "large-v3" : "large-v3-turbo",
          device: "cuda", // default
          vocalThreshold: config.vocalEnergyThreshold,
          vocalSilenceThreshold: config.vocalSilenceThreshold,
          snrThreshold: config.snrThreshold,
          minHallLength: config.minHallucinationLength,
          rejectNegativeSnr: config.rejectNegativeSnr,
          repairTemperature: config.repairTemperature,
          repairBeamSize: config.repairBeamSize,
          repairWithVocal: config.repairWithVocal,
          mixWeight: config.mixWeight,
          asrPrompt: asrPrompt || undefined,
          useMmsRepair: config.useMmsRepair,
          useQwenRepair: config.useQwenRepair,
          useSenseVoiceRepair: config.useSenseVoiceRepair,
          useGemmaRepair: config.useGemmaRepair,
          saveRepairAudio: config.saveRepairAudio,
        },
        surgicalLog,
        vocalPath
      );
      cleaned = repaired.segments;
      mismatches = repaired.mismatches;
      surgicalLog = repaired.surgicalLog;
      
      // Persist the combined log
      await writeSurgicalLog(config.outputDir, track.relativeDir, track.stem, surgicalLog);
    }

    if (cleaned.length === 0) {
      console.log(`  Skipped (all segments garbled)`);
      // Still write an empty transcription so the user knows it was processed
      await writeTranscription(config.outputDir, track.relativeDir, track.stem, [], mismatches);
      skipped++;
      continue;
    }
    console.log(`  ${cleaned.length} clean segments (${transcript.segments.length - cleaned.length} garbled removed)`);

    await writeTranscription(config.outputDir, track.relativeDir, track.stem, cleaned, mismatches);
    readyTracks.push({ track, cleaned });
  }

  if (readyTracks.length === 0) {
    console.log(`\nNo tracks to translate.`);
    return;
  }

  // ── Step 4: Start translation server (ASR is done, GPU is free) ─────────

  const server = new LlamaServerManager({
    llamaServerExe: config.llamaServerExe,
    modelPath: config.modelPath,
    hfRepo: config.hfRepo,
    serverPort: config.serverPort,
    gpuLayers: config.gpuLayers,
    contextSize: config.contextSize,
    parallel: config.parallel,
    serverUrl: config.serverUrl,
  }, "TranslateServer");

  await server.start();
  const client = new LlmClient(server.baseUrl, config);


  // ── Step 5: Translate each track ────────────────────────────────────────

  let processed = 0;

  try {
    for (const { track, cleaned } of readyTracks) {
      console.log(`\n[Translate ${processed + 1}/${readyTracks.length}] ${track.relativePath}`);

      const segments = asrToSegments(cleaned);
      const trackName = track.relativePath;

      try {
        const { entries, windowResults } = await translateTrack(segments, glossary, trackName, config, client);

        await writeWindowResults(config.outputDir, track.relativeDir, track.stem, windowResults);

        if (entries.length === 0) {
          console.log(`  Skipped (translation produced no output)`);
          skipped++;
          continue;
        }

        await writeTranslation(config.outputDir, track.relativeDir, track.stem, entries);
        console.log(`  Done: ${entries.length} translated entries`);
        processed++;
      } catch (err: any) {
        // Track-level error: log and continue with next track
        console.error(`  ERROR: ${err.message}`);

        // Check if server is still alive, force restart if needed
        const healthy = await client.healthCheck();
        if (!healthy) {
          console.error(`  Server unresponsive, force restarting...`);
          await server.forceRestart();
        }
        skipped++;
      }
    }

    // ── Summary ──────────────────────────────────────────────────────────

    console.log(`\n=== Complete ===`);
    console.log(`Processed: ${processed}`);
    console.log(`Skipped:   ${skipped}`);
    console.log(`Output:    ${config.outputDir}`);

  } finally {
    await server.stop();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
