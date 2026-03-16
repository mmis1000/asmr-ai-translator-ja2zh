import type { Segment, GlossaryEntry, GlossaryLang } from "../util/types.js";

// ── Prompt templates (copied verbatim from training pipeline) ────────────────
// These must match EXACTLY what the model was trained on.

// ── Base mode ────────────────────────────────────────────────────────────────

export function buildPromptZhTw(
  trackName: string,
  summary: string,
  glossaryJson: string,
  transcriptionJson: string,
): string {
  return `將以下日語ASMR逐字稿翻譯成繁體中文。

音軌：${trackName}
場景說明：${summary}

術語表（請嚴格使用zh欄位的譯名）：
${glossaryJson}

翻譯前請靜默修正下列Whisper識別錯誤：
- 重複片語（連續3次以上且無變化）：僅保留一次
- 錯字／同音異字：依上下文修正
- 字幕版權行（字幕：／翻訳：／QQ／LINE水印）：text設為null
- 錯誤專有名詞：依術語表修正

翻譯規則：
- 呻吟與氣息聲（あ、ん、はあ）→ 自然對應（啊、嗯、哈、呼）
- 擬聲詞：日語形式翻譯（パンパン→啪啪）；中文形式保留原樣
- 保留角色語氣與口吻
- text欄位只輸出譯文，不加注釋或括號說明

輸入：逐字稿JSON陣列 — {"id": <n>, "text": "<日文>", "start": <ms>, "end": <ms>}

輸出：將連續構成同一句話的片段合併，JSON陣列格式：
{"ids": [<n>, ...], "text": "<繁體中文>", "start": <最早ms>, "end": <最晚ms>}

字幕版權行：{"ids": [<n>], "text": null, "start": <ms>, "end": <ms>}
每個輸入id必須恰好出現在一個輸出項中。

逐字稿：
${transcriptionJson}`;
}

export function buildPromptZhCn(
  trackName: string,
  summary: string,
  glossaryJson: string,
  transcriptionJson: string,
): string {
  return `将以下日语ASMR逐字稿翻译成简体中文。

音轨：${trackName}
场景说明：${summary}

术语表（请严格使用zh栏位的译名）：
${glossaryJson}

翻译前请静默修正以下Whisper识别错误：
- 重复片语（连续3次以上且无变化）：仅保留一次
- 错字／同音异字：依上下文修正
- 字幕版权行（字幕：／翻訳：／QQ／LINE水印）：text设为null
- 错误专有名词：依术语表修正

翻译规则：
- 呻吟与气息声（あ、ん、はあ）→ 自然对应（啊、嗯、哈、呼）
- 拟声词：日语形式翻译（パンパン→啪啪）；中文形式保留原样
- 保留角色语气与口吻
- text字段只输出译文，不加注释或括号说明

输入：逐字稿JSON数组 — {"id": <n>, "text": "<日文>", "start": <ms>, "end": <ms>}

输出：将连续构成同一句话的片段合并，JSON数组格式：
{"ids": [<n>, ...], "text": "<简体中文>", "start": <最早ms>, "end": <最晚ms>}

字幕版权行：{"ids": [<n>], "text": null, "start": <ms>, "end": <ms>}
每个输入id必须恰好出现在一个输出项中。

逐字稿：
${transcriptionJson}`;
}

// ── Echo mode ────────────────────────────────────────────────────────────────

export function buildPromptEchoZhTw(
  trackName: string,
  summary: string,
  glossaryJson: string,
  transcriptionJson: string,
): string {
  return `將以下日語ASMR逐字稿翻譯成繁體中文。

音軌：${trackName}
場景說明：${summary}

術語表（請嚴格使用zh欄位的譯名）：
${glossaryJson}

翻譯前請靜默修正下列Whisper識別錯誤：
- 重複片語（連續3次以上且無變化）：僅保留一次
- 錯字／同音異字：依上下文修正
- 字幕版權行（字幕：／翻訳：／QQ／LINE水印）：text設為null
- 錯誤專有名詞：依術語表修正

翻譯規則：
- 呻吟與氣息聲（あ、ん、はあ）→ 自然對應（啊、嗯、哈、呼）
- 擬聲詞：日語形式翻譯（パンパン→啪啪）；中文形式保留原樣
- 保留角色語氣與口吻
- text欄位只輸出譯文，不加注釋或括號說明

輸入：逐字稿JSON陣列 — {"id": <n>, "text": "<日文>", "start": <ms>, "end": <ms>}

輸出：將連續構成同一句話的片段合併，JSON陣列格式：
{"ids": [<n>, ...], "input": "<合併後的原始日文，以空格連接>", "text": "<繁體中文>", "start": <最早ms>, "end": <最晚ms>}

字幕版權行：{"ids": [<n>], "input": "<原始日文>", "text": null, "start": <ms>, "end": <ms>}
每個輸入id必須恰好出現在一個輸出項中。
input欄位為ids所對應的原始日文片段以空格連接，text為其繁體中文翻譯。

逐字稿：
${transcriptionJson}`;
}

export function buildPromptEchoZhCn(
  trackName: string,
  summary: string,
  glossaryJson: string,
  transcriptionJson: string,
): string {
  return `将以下日语ASMR逐字稿翻译成简体中文。

音轨：${trackName}
场景说明：${summary}

术语表（请严格使用zh栏位的译名）：
${glossaryJson}

翻译前请静默修正以下Whisper识别错误：
- 重复片语（连续3次以上且无变化）：仅保留一次
- 错字／同音异字：依上下文修正
- 字幕版权行（字幕：／翻訳：／QQ／LINE水印）：text设为null
- 错误专有名词：依术语表修正

翻译规则：
- 呻吟与气息声（あ、ん、はあ）→ 自然对应（啊、嗯、哈、呼）
- 拟声词：日语形式翻译（パンパン→啪啪）；中文形式保留原样
- 保留角色语气与口吻
- text字段只输出译文，不加注释或括号说明

输入：逐字稿JSON数组 — {"id": <n>, "text": "<日文>", "start": <ms>, "end": <ms>}

输出：将连续构成同一句话的片段合并，JSON数组格式：
{"ids": [<n>, ...], "input": "<合并后的原始日文，以空格连接>", "text": "<简体中文>", "start": <最早ms>, "end": <最晚ms>}

字幕版权行：{"ids": [<n>], "input": "<原始日文>", "text": null, "start": <ms>, "end": <ms>}
每个输入id必须恰好出现在一个输出项中。
input字段为ids所对应的原始日文片段以空格连接，text为其简体中文翻译。

逐字稿：
${transcriptionJson}`;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

export function formatTranscriptionJson(segments: Segment[]): string {
  if (segments.length === 0) return "[]";
  const maxIdLen = String(segments[segments.length - 1]!.id).length;
  const lines = segments.map(
    (s) =>
      `  {"id": ${String(s.id).padStart(maxIdLen)}, "text": ${JSON.stringify(s.text)}, "start": ${s.start}, "end": ${s.end}}`,
  );
  return "[\n" + lines.join(",\n") + "\n]";
}

export function filterGlossary(g: GlossaryLang, japaneseText: string): GlossaryLang {
  const filter = (entries: GlossaryEntry[]) =>
    entries.filter((e) => e.ja && japaneseText.includes(e.ja));
  return {
    ...g,
    cvs: filter(g.cvs),
    characters: filter(g.characters),
    terms: filter(g.terms),
  };
}

export function buildGlossaryJson(g: GlossaryLang): string {
  return JSON.stringify(
    { cvs: g.cvs, characters: g.characters, terms: g.terms },
    null,
    2,
  );
}

/** Select the correct prompt builder based on locale and mode. */
export function getPromptBuilder(
  locale: "zh-tw" | "zh-cn",
  mode: "base" | "echo",
): typeof buildPromptZhTw {
  if (locale === "zh-tw") return mode === "echo" ? buildPromptEchoZhTw : buildPromptZhTw;
  return mode === "echo" ? buildPromptEchoZhCn : buildPromptZhCn;
}
