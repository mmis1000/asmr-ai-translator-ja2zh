// ── ASR transcript types (matching Whisper output schema) ─────────────────────

export interface TranscriptWord {
  text: string;
  start_time: number;
  end_time: number;
}

export interface TranscriptSegment {
  text: string;
  start_time: number;
  end_time: number;
  words: TranscriptWord[];
  avg_logprob: number;
  compression_ratio: number;
  no_speech_prob: number;
  vocal_energy?: number;
  other_energy?: number;
  snr?: number;
  mismatch?: {
    type: "missed" | "hallucinated" | "uncertain" | "repeated";
    reason: string;
  };
  engine?: "whisper" | "mms" | "qwen";
}

export interface DemucsWindow {
  start: number;
  end: number;
  vocal_rms: number;
  other_rms: number;
  snr_db: number;
}

export interface TranscriptFile {
  full_text: string;
  sentences?: { text: string; start_time: number; end_time: number }[];
  segments: TranscriptSegment[];
  mismatches?: TranscriptSegment[];
  demucs_windows?: DemucsWindow[];
}

// ── Translation I/O types ────────────────────────────────────────────────────

export interface Segment {
  id: number;
  text: string;
  start: number;  // ms
  end: number;    // ms
  vocal_energy?: number;
  other_energy?: number;
  snr?: number;
}

export interface TranslationEntry {
  ids: number[];
  text: string | null;
  start: number;  // ms
  end: number;    // ms
}

export interface TranslationEchoEntry extends TranslationEntry {
  input: string;
}

// ── Glossary / metadata types ────────────────────────────────────────────────

export interface GlossaryEntry {
  ja: string;
  zh: string;
}

export interface GlossaryLang {
  cvs: GlossaryEntry[];
  characters: GlossaryEntry[];
  terms: GlossaryEntry[];
  summary: string;
}

export interface UserMetadata {
  title?: string | undefined;
  summary?: string | undefined;
  glossary?: {
    cvs?: GlossaryEntry[];
    characters?: GlossaryEntry[];
    terms?: GlossaryEntry[];
  };
  track_list?: { filename: string; title?: string }[];
}

/** Maximum character length for the ASR initial prompt. */
export const MAX_ASR_PROMPT_LENGTH = 250;

// ── Metadata extraction types (LLM Phase 1 output) ──────────────────────────

export interface Pass1Output {
  asr: { prompt: string };
  ja: {
    title: string;
    circle: string;
    cv_list: Array<{ name: string; role?: string; note?: string }>;
    character_list: Array<{ name: string; note?: string }>;
    track_list: Array<{ number: number; title: string }>;
    summary: string;
    term_list: Array<{ ja: string; note?: string }>;
  };
}

export interface GlossaryTranslatedEntry { zh: string; note?: string }
export interface GlossaryTranslated {
  cv:         Record<string, GlossaryTranslatedEntry>;
  characters: Record<string, GlossaryTranslatedEntry>;
  circles:    Record<string, GlossaryTranslatedEntry>;
  terms:      Record<string, GlossaryTranslatedEntry>;
}

export interface Pass4bOutput {
  title: string;
  track_list: Array<{ number: number; zh: string }>;
  summary: string;
}

export interface FinalMetadata {
  asr: { prompt: string };
  translate: {
    title: string;
    circle: string;
    cv_mapping:        GlossaryEntry[];
    character_mapping: GlossaryEntry[];
    track_list:        Array<{ number: number; ja: string; zh: string }>;
    summary: string;
    term_mapping:      GlossaryEntry[];
  };
}

// ── Audio file discovery ─────────────────────────────────────────────────────

export const AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"];

export interface AudioTrack {
  /** Absolute path to the audio file */
  absolutePath: string;
  /** Relative path from input root (forward-slash normalized) */
  relativePath: string;
  /** Filename stem (no extension) */
  stem: string;
  /** Relative directory from input root */
  relativeDir: string;
}
