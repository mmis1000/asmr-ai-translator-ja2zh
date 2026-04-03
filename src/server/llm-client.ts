import fs from "fs/promises";
import path from "path";
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
    private config: Pick<TranslatorConfig, "temperature" | "topP" | "topK" | "minP" | "repeatPenalty" | "debugLog" | "outputDir">,
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
      /** Fixed RNG seed for reproducible outputs. Omit for random (default). */
      seed?: number | undefined;
      label?: string | undefined;
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
      seed?: number | undefined;
      label?: string | undefined;
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
      repeat_penalty: this.config.repeatPenalty,
      n_predict: options?.nPredict ?? 8192,
      stop: STOP,
      ...(options?.seed !== undefined ? { seed: options.seed } : {}),
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

    if (this.config.debugLog && this.config.outputDir) {
      try {
        const debugDir = path.join(this.config.outputDir, "debug_logs");
        await fs.mkdir(debugDir, { recursive: true });
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const rng = Math.random().toString(36).substring(2, 8);
        const prefix = options?.label ? `${options.label}_` : "";
        const filename = `llm_log_${prefix}${timestamp}_${rng}.md`;
        // ensure path component correctness if label has strange chars but trackName might have slashes?
        // Actually trackName could contain slashes like "dir/file.mp3". We should sanitize it.
        const safeFilename = filename.replace(/[\/\\]/g, "-");
        
        const debugContent = `# PROMPT\n\n\`\`\`\n${prompt}\n\`\`\`\n\n# RESPONSE\n\n\`\`\`\n${rawContent}\n\`\`\`\n`;
        await fs.writeFile(path.join(debugDir, safeFilename), debugContent, "utf-8");
      } catch (e) {
        console.error("Failed to write debug log:", e);
      }
    }

    return rawContent;
  }
}
