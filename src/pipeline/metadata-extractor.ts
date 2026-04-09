import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import type { LlmClient } from "../server/llm-client.js";
import { buildChatPromptWithSystem, extractJsonObject } from "../server/llm-client.js";
import { MAX_ASR_PROMPT_LENGTH } from "../util/types.js";
import type {
  Pass1Output,
  GlossaryTranslated,
  GlossaryTranslatedEntry,
  Pass4bOutput,
  FinalMetadata,
  GlossaryLang,
} from "../util/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.resolve(__dirname, "../../docs");

function parseTemplate(full: string): string {
  const match = full.match(/`{4}[\r\n]+([\s\S]*?)`{4}/);
  let sys = match ? match[1]! : full;
  const split = sys.indexOf("---");
  if (split !== -1) sys = sys.slice(0, split).trim();
  return sys;
}

/**
 * Full LLM-based metadata extraction pipeline for a single work.
 * Phase 1: Extract Japanese entities from metadata.md
 * Phase 3: Translate glossary to target locale
 * Phase 4: Translate prose (title, track list, summary)
 * Assembly: Combine into FinalMetadata
 */
export class MetadataExtractor {
  private pass1System: string | null = null;
  private pass1Grammar: string | null = null;
  private pass3System: string | null = null;
  private pass3Grammar: string | null = null;
  private pass4System: string | null = null;
  private pass4Grammar: string | null = null;
  private asrCleanupSystem: string | null = null;
  private asrCleanupGrammar: string | null = null;

  constructor(
    private client: LlmClient,
    private locale: "zh-tw" | "zh-cn",
    private seed?: number,
  ) {}

  // ── Phase 1: Japanese extraction ─────────────────────────────────────────

  async extractJapanese(metadataMd: string): Promise<Pass1Output> {
    if (!this.pass1System) {
      const full = await fs.readFile(path.join(DOCS_DIR, "metadata-extraction-prompt-template.md"), "utf-8");
      this.pass1System = parseTemplate(full);
    }
    if (!this.pass1Grammar) {
      this.pass1Grammar = await fs.readFile(path.join(DOCS_DIR, "metadata-extraction-pass1.gbnf"), "utf-8");
    }

    const user = `---\nMETADATA:\n${metadataMd}\n---`;
    const prompt = buildChatPromptWithSystem(this.pass1System, user, false);
    // Thinking is enabled for Phase 1 — allow up to 20 minutes for reasoning + output
    // Phase 1 uses thinking — lower temperature (0.6) keeps thinking chains short and stable.
    // Higher temperatures cause the model to think indefinitely, filling nPredict.
    const raw = await this.client.complete(prompt, { grammar: this.pass1Grammar, temperature: 0.6, nPredict: 14000, seed: this.seed, label: "metadata-phase-1" });
    return JSON.parse(extractJsonObject(raw)) as Pass1Output;
  }

  // ── ASR prompt cleanup ───────────────────────────────────────────────────

  async cleanAsrPrompt(asrPrompt: string, originalSummary: string): Promise<string> {
    if (!this.asrCleanupSystem) {
      const full = await fs.readFile(path.join(DOCS_DIR, "asr-cleanup-prompt-template.md"), "utf-8");
      this.asrCleanupSystem = parseTemplate(full);
    }
    if (!this.asrCleanupGrammar) {
      this.asrCleanupGrammar = await fs.readFile(path.join(DOCS_DIR, "asr-cleanup.gbnf"), "utf-8");
    }

    const user = `INPUT ASR PROMPT:\n${asrPrompt}\n\nRAW SUMMARY (for context):\n${originalSummary}`;
    // ASR cleanup uses thinking to prioritize names and core scenarios.
    const prompt = buildChatPromptWithSystem(this.asrCleanupSystem, user, false);
    const raw = await this.client.complete(prompt, { grammar: this.asrCleanupGrammar, temperature: 0.6, seed: this.seed, label: "asr-cleanup" });
    
    const parsed = JSON.parse(extractJsonObject(raw)) as { cleaned_prompt: string };
    let cleaned = parsed.cleaned_prompt.trim();

    if (cleaned.length > MAX_ASR_PROMPT_LENGTH) {
      console.warn(`  [Metadata] ASR prompt exceeded ${MAX_ASR_PROMPT_LENGTH} chars (${cleaned.length}), truncating...`);
      cleaned = cleaned.slice(0, MAX_ASR_PROMPT_LENGTH);
    }
    return cleaned;
  }

  // ── Phase 3: Glossary translation ────────────────────────────────────────

  private pass3GrammarNoEn: string | null = null;

  async translateGlossary(
    jaData: Pass1Output,
  ): Promise<GlossaryTranslated> {
    if (!this.pass3System) {
      const full = await fs.readFile(path.join(DOCS_DIR, "glossary-translation-prompt-template.md"), "utf-8");
      const match = full.match(/`{4}[\r\n]+([\s\S]*?)`{4}/);
      this.pass3System = (match ? match[1]! : full).trim();
    }
    if (!this.pass3Grammar) {
      this.pass3Grammar = await fs.readFile(path.join(DOCS_DIR, "glossary-translation.gbnf"), "utf-8");
    }
    if (!this.pass3GrammarNoEn) {
      this.pass3GrammarNoEn = await fs.readFile(path.join(DOCS_DIR, "glossary-translation-no-en.gbnf"), "utf-8");
    }

    const isTw = this.locale === "zh-tw";
    const language = isTw ? "繁體中文" : "简体中文";
    const sys = this.pass3System
      .replace(/\{\{OUTPUT_LANGUAGE\}\}/g, language)
      .replace(/\{\{EX_REINA\}\}/g, isTw ? "田中麗奈" : "田中丽奈")
      .replace(/\{\{EX_SEIYUU\}\}/g, isTw ? "聲優" : "声优")
      .replace(/\{\{EX_WRITER\}\}/g, isTw ? "腳本作家" : "脚本作家")
      .replace(/\{\{EX_SAKURA\}\}/g, isTw ? "櫻" : "樱")
      .replace(/\{\{EX_JK_NOTE\}\}/g, isTw ? "清純系女高中生的表現" : "清纯系女高中生的表现");

    type TermInput = { ja: string; note?: string };
    const hasEnglish = (str: string) => /[A-Za-zａ-ｚＡ-Ｚ]/.test(str);

    const inputPermissive = { cv: [] as TermInput[], characters: [] as TermInput[], circles: [] as TermInput[], terms: [] as TermInput[] };
    const inputStrict = { cv: [] as TermInput[], characters: [] as TermInput[], circles: [] as TermInput[], terms: [] as TermInput[] };

    const partition = (str: string, item: TermInput, destPermissive: TermInput[], destStrict: TermInput[]) => {
      if (hasEnglish(str)) destPermissive.push(item);
      else destStrict.push(item);
    };

    jaData.ja.cv_list.forEach(({ name, note }) => partition(name, note ? { ja: name, note } : { ja: name }, inputPermissive.cv, inputStrict.cv));
    jaData.ja.character_list.forEach(({ name, note }) => partition(name, note ? { ja: name, note } : { ja: name }, inputPermissive.characters, inputStrict.characters));
    partition(jaData.ja.circle, { ja: jaData.ja.circle }, inputPermissive.circles, inputStrict.circles);
    jaData.ja.term_list.forEach(({ ja, note }) => partition(ja, note ? { ja, note } : { ja }, inputPermissive.terms, inputStrict.terms));

    const invokePhase3 = async (inputObj: typeof inputPermissive, grammar: string, label: string) => {
      const hasAny = inputObj.cv.length || inputObj.characters.length || inputObj.circles.length || inputObj.terms.length;
      if (!hasAny) return { cv: [], characters: [], circles: [], terms: [] };

      const prompt = buildChatPromptWithSystem(sys, JSON.stringify(inputObj, null, 2), true);
      const raw = await this.client.complete(prompt, { grammar, temperature: 0.7, seed: this.seed, label });
      return JSON.parse(extractJsonObject(raw)) as {
        cv: { ja: string; zh: string; note?: string }[];
        characters: { ja: string; zh: string; note?: string }[];
        circles: { ja: string; zh: string; note?: string }[];
        terms: { ja: string; zh: string; note?: string }[];
      };
    };

    const [parsedStrict, parsedPermissive] = await Promise.all([
      invokePhase3(inputStrict, this.pass3GrammarNoEn!, "metadata-phase-3-strict"),
      invokePhase3(inputPermissive, this.pass3Grammar!, "metadata-phase-3-permissive")
    ]);

    type GlossaryRow = { ja: string; zh: string; note?: string };
    const toEntry = (item: GlossaryRow): GlossaryTranslatedEntry =>
      item.note ? { zh: item.zh, note: item.note } : { zh: item.zh };

    const result: GlossaryTranslated = { cv: {}, characters: {}, circles: {}, terms: {} };
    
    const populate = (parsed: typeof parsedStrict) => {
      for (const item of parsed.cv ?? []) if (item.ja && item.zh) result.cv[item.ja] = toEntry(item);
      for (const item of parsed.characters ?? []) if (item.ja && item.zh) result.characters[item.ja] = toEntry(item);
      for (const item of parsed.circles ?? []) if (item.ja && item.zh) result.circles[item.ja] = toEntry(item);
      for (const item of parsed.terms ?? []) if (item.ja && item.zh) result.terms[item.ja] = toEntry(item);
    };

    populate(parsedStrict);
    populate(parsedPermissive);

    return result;
  }

  // ── Phase 4: Prose translation ───────────────────────────────────────────

  async translateProse(
    jaData: Pass1Output,
    glossary: GlossaryTranslated,
  ): Promise<Pass4bOutput> {
    if (!this.pass4System) {
      const full = await fs.readFile(path.join(DOCS_DIR, "metadata-translation-prompt-template.md"), "utf-8");
      const match = full.match(/`{4}[\r\n]+([\s\S]*?)`{4}/);
      this.pass4System = (match ? match[1]! : full).trim();
    }
    if (!this.pass4Grammar) {
      this.pass4Grammar = await fs.readFile(path.join(DOCS_DIR, "metadata-translation.gbnf"), "utf-8");
    }

    const language = this.locale === "zh-tw" ? "繁體中文" : "简体中文";
    const sys = this.pass4System.replace(/\{\{OUTPUT_LANGUAGE\}\}/g, language);

    // Build glossary subset for this work
    const subset: Record<string, string> = {};
    const addEntry = (ja: string, map: Record<string, GlossaryTranslatedEntry>) => {
      if (map[ja]) subset[ja] = map[ja].zh;
    };
    addEntry(jaData.ja.circle, glossary.circles);
    for (const cv of jaData.ja.cv_list) addEntry(cv.name, glossary.cv);
    for (const ch of jaData.ja.character_list) addEntry(ch.name, glossary.characters);
    for (const t of jaData.ja.term_list) addEntry(t.ja, glossary.terms);

    const proseInput = {
      title: jaData.ja.title,
      track_list: jaData.ja.track_list.map(t => ({ number: t.number, title: t.title })),
      summary: jaData.ja.summary,
    };

    const sysWithContext = sys
      .replace(/\{\{GLOSSARY_SUBSET_JSON\}\}/g, JSON.stringify(subset, null, 2))
      .replace(/\{\{PROSE_INPUT_JSON\}\}/g, JSON.stringify(proseInput, null, 2));

    const prompt = buildChatPromptWithSystem(sysWithContext, "", true);
    const raw = await this.client.complete(prompt, { grammar: this.pass4Grammar, temperature: 0.7, seed: this.seed, label: "metadata-phase-4" });
    return JSON.parse(extractJsonObject(raw)) as Pass4bOutput;
  }

  // ── Assembly ─────────────────────────────────────────────────────────────

  assembleFinal(
    jaData: Pass1Output,
    glossary: GlossaryTranslated,
    prose: Pass4bOutput,
  ): FinalMetadata {
    const lookupZh = (ja: string, map: Record<string, GlossaryTranslatedEntry>) => map[ja]?.zh ?? ja;

    return {
      asr: jaData.asr,
      translate: {
        title: prose.title,
        circle: glossary.circles[jaData.ja.circle]?.zh ?? jaData.ja.circle,
        cv_mapping: jaData.ja.cv_list.map(cv => ({
          ja: cv.name,
          zh: lookupZh(cv.name, glossary.cv),
        })),
        character_mapping: jaData.ja.character_list.map(ch => ({
          ja: ch.name,
          zh: lookupZh(ch.name, glossary.characters),
        })),
        track_list: jaData.ja.track_list.map(t => ({
          number: t.number,
          ja: t.title,
          zh: prose.track_list.find(p => p.number === t.number)?.zh ?? t.title,
        })),
        summary: prose.summary,
        term_mapping: jaData.ja.term_list.map(t => ({
          ja: t.ja,
          zh: lookupZh(t.ja, glossary.terms),
        })),
      },
    };
  }

  // ── Full pipeline ────────────────────────────────────────────────────────

  /**
   * Run the full metadata extraction + translation pipeline.
   * Returns FinalMetadata and the GlossaryLang needed for translation prompts.
   */
  async extract(metadataMd: string): Promise<{ metadata: FinalMetadata; glossary: GlossaryLang }> {
    console.log("[Metadata] Phase 1: Extracting Japanese entities...");
    const jaData = await this.extractJapanese(metadataMd);
    console.log(`[Metadata] Extracted: title="${jaData.ja.title}", ${jaData.ja.cv_list.length} CVs, ${jaData.ja.term_list.length} terms`);

    console.log("[Metadata] Phase 2: Cleaning ASR prompt...");
    const cleanedPrompt = await this.cleanAsrPrompt(jaData.asr.prompt, jaData.ja.summary);
    jaData.asr.prompt = cleanedPrompt;

    console.log("[Metadata] Phase 3: Translating glossary...");
    const glossary = await this.translateGlossary(jaData);

    console.log("[Metadata] Phase 4: Translating prose...");
    const prose = await this.translateProse(jaData, glossary);

    const metadata = this.assembleFinal(jaData, glossary, prose);
    console.log(`[Metadata] Done: "${metadata.translate.title}"`);

    // Build the GlossaryLang for translation prompts
    const glossaryLang: GlossaryLang = {
      cvs: metadata.translate.cv_mapping,
      characters: metadata.translate.character_mapping,
      terms: metadata.translate.term_mapping,
      summary: metadata.translate.summary,
    };

    return { metadata, glossary: glossaryLang };
  }
}
