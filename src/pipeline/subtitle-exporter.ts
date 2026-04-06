import type { TranslationEntry } from "../util/types.js";

/** Silence gap threshold in milliseconds — emit an empty LRC line if gap exceeds this. */
const SILENCE_THRESHOLD_MS = 2000;

/**
 * Collapse runs of 4+ identical characters down to 3 + ellipsis.
 * Prevents LLM-echoed moaning loops (e.g. 79× 嗯) from bloating LRC lines.
 * Example: 嗯嗯嗯嗯嗯嗯 → 嗯嗯嗯…
 */
function collapseRuns(text: string): string {
  return text.replace(/(.)\1{3,}/gu, "$1$1$1\u2026");
}

/**
 * Merge consecutive TranslationEntry items that share the same start timestamp.
 * Same-start entries arise when the ASR produced duplicate-start segments that
 * slipped through cleaning; without merging they emit multiple LRC lines at the
 * same timestamp. Texts are joined with ～; end time is extended to the latest.
 */
function mergeSameStart(entries: TranslationEntry[]): TranslationEntry[] {
  if (entries.length === 0) return entries;

  const result: TranslationEntry[] = [];
  let current: TranslationEntry = { ...entries[0]!, ids: [...entries[0]!.ids] };

  for (let i = 1; i < entries.length; i++) {
    const entry = entries[i]!;
    if (entry.start === current.start) {
      if (entry.text != null) {
        current.text = current.text != null
          ? current.text + "\uff5e" + entry.text
          : entry.text;
      }
      current.end = Math.max(current.end, entry.end);
      current.ids = current.ids.concat(entry.ids);
    } else {
      result.push(current);
      current = { ...entry, ids: [...entry.ids] };
    }
  }
  result.push(current);
  return result;
}

function msToLrc(ms: number): string {
  const totalSec = ms / 1000;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec - min * 60;
  return `[${String(min).padStart(2, "0")}:${sec.toFixed(2).padStart(5, "0")}]`;
}

function msToVtt(ms: number): string {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const frac = ms % 1000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(frac).padStart(3, "0")}`;
}

/** Convert translation entries to LRC subtitle format. */
export function toLrc(entries: TranslationEntry[]): string {
  const lines: string[] = [];
  const valid = mergeSameStart(entries.filter(e => e.text != null));

  for (let i = 0; i < valid.length; i++) {
    const entry = valid[i]!;
    lines.push(`${msToLrc(entry.start)}${collapseRuns(entry.text!)}`);

    // Emit empty line at end timestamp if there's a long silence before next entry
    const next = valid[i + 1];
    if (next && (next.start - entry.end) > SILENCE_THRESHOLD_MS) {
      lines.push(`${msToLrc(entry.end)}`);
    }
  }

  return lines.join("\n") + "\n";
}

/** Convert translation entries to WebVTT subtitle format. */
export function toVtt(entries: TranslationEntry[]): string {
  const lines: string[] = ["WEBVTT", ""];
  const valid = mergeSameStart(entries.filter(e => e.text != null));

  for (let i = 0; i < valid.length; i++) {
    const entry = valid[i]!;
    lines.push(String(i + 1));
    lines.push(`${msToVtt(entry.start)} --> ${msToVtt(entry.end)}`);
    lines.push(collapseRuns(entry.text!));
    lines.push("");
  }

  return lines.join("\n");
}
