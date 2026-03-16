import type { TranslatorConfig } from "../config.js";

const IM_S = "<|im_start|>";
const IM_E = "<|im_end|>";
const STOP = ["<|im_end|>", "<|endoftext|>"];

/** Default per-request timeout (5 minutes). */
const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

export function buildChatPrompt(user: string): string {
  return `${IM_S}user\n${user}${IM_E}\n${IM_S}assistant\n<think></think>\n`;
}

export function buildChatPromptWithSystem(system: string, user: string, disableThinking = true): string {
  let prompt = `${IM_S}system\n${system}${IM_E}\n${IM_S}user\n${user}${IM_E}\n${IM_S}assistant\n`;
  if (disableThinking) prompt += `<think></think>\n`;
  return prompt;
}

/** Extract JSON object from LLM response (strips <think> block, finds { to }). */
export function extractJsonObject(raw: string): string {
  const afterThink = raw.includes("</think>")
    ? raw.slice(raw.lastIndexOf("</think>") + "</think>".length)
    : raw;
  let out = afterThink.trim();
  const first = out.indexOf("{");
  const last = out.lastIndexOf("}");
  if (first !== -1 && last !== -1) out = out.slice(first, last + 1);
  return out;
}

/** Extract JSON array from LLM response (strips <think> block, finds [ to ]). */
export function extractJsonArray(raw: string): string {
  const afterThink = raw.includes("</think>")
    ? raw.slice(raw.lastIndexOf("</think>") + "</think>".length)
    : raw;
  let out = afterThink.trim();
  const first = out.indexOf("[");
  const last = out.lastIndexOf("]");
  if (first !== -1 && last !== -1) out = out.slice(first, last + 1);
  return out;
}

export class LlmClient {
  constructor(
    private baseUrl: string,
    private config: Pick<TranslatorConfig, "temperature" | "topP" | "topK" | "minP">,
  ) {}

  /** Check if the server is responsive. */
  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Send a prompt to llama-server /completion and return the raw content.
   * Supports GBNF grammar, per-request timeout, and AbortSignal.
   */
  async complete(
    prompt: string,
    options?: {
      grammar?: string | undefined;
      temperature?: number | undefined;
      nPredict?: number | undefined;
      timeoutMs?: number | undefined;
    },
  ): Promise<string> {
    const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      return await this._doComplete(prompt, options, controller.signal);
    } finally {
      clearTimeout(timeout);
    }
  }

  private async _doComplete(
    prompt: string,
    options: {
      grammar?: string | undefined;
      temperature?: number | undefined;
      nPredict?: number | undefined;
    } | undefined,
    signal: AbortSignal,
  ): Promise<string> {
    const body = {
      prompt,
      stream: true,
      grammar: options?.grammar,
      temperature: options?.temperature ?? this.config.temperature,
      top_p: this.config.topP,
      top_k: this.config.topK,
      min_p: this.config.minP,
      n_predict: options?.nPredict ?? 8192,
      stop: STOP,
    };

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/completion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
    } catch (err: any) {
      if (err.name === "AbortError") {
        throw new Error("LLM request timed out");
      }
      const cause = err.cause
        ? ` (Cause: ${err.cause.message || err.cause.code || String(err.cause)})`
        : "";
      throw new Error(`fetch request to llama-server failed: ${err.message}${cause}`);
    }

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`llama-server responded with ${res.status}: ${text}`);
    }
    if (!res.body) throw new Error("No response body from llama-server");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let rawContent = "";
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const dataStr = line.slice(6).trim();
          if (!dataStr || dataStr === "[DONE]") continue;
          try {
            const data = JSON.parse(dataStr);
            if (data.content) rawContent += data.content;
          } catch { /* ignore malformed SSE */ }
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        throw new Error(`LLM request timed out (received ${rawContent.length} chars before timeout)`);
      }
      throw err;
    }

    if (!rawContent) {
      throw new Error("LLM returned empty response");
    }

    return rawContent;
  }
}
