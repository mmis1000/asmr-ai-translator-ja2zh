import type { TranslationEntry } from "../util/types.js";

/** Silence gap threshold in milliseconds — emit an empty LRC line if gap exceeds this. */
const SILENCE_THRESHOLD_MS = 2000;

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
  const valid = entries.filter(e => e.text != null);

  for (let i = 0; i < valid.length; i++) {
    const entry = valid[i]!;
    lines.push(`${msToLrc(entry.start)}${entry.text}`);

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
  const valid = entries.filter(e => e.text != null);

  for (let i = 0; i < valid.length; i++) {
    const entry = valid[i]!;
    lines.push(String(i + 1));
    lines.push(`${msToVtt(entry.start)} --> ${msToVtt(entry.end)}`);
    lines.push(entry.text!);
    lines.push("");
  }

  return lines.join("\n");
}
