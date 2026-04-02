import type { Segment } from "../util/types.js";

/**
 * Escape a string to be safely embedded as a literal inside a GBNF string rule.
 * GBNF string literals use \" for a literal quote and \\ for a literal backslash.
 */
function makeGbnfStringLiteral(text: string): string {
  let escaped = "";
  for (const ch of text) {
    if (ch === '"') escaped += '\\"';
    else if (ch === "\\") escaped += "\\\\";
    else if (ch === "\n" || ch === "\r") escaped += " ";
    else escaped += ch;
  }
  return '"' + escaped + '"';
}

/**
 * Generate a per-window GBNF grammar that hardcodes valid segment IDs and
 * timestamps. This structurally prevents the model from:
 *   - Emitting IDs not present in the window
 *   - Generating more output entries than input segments
 *   - Producing wrong or duplicate timestamps
 *
 * Each state sN covers segment index N-1. From each state the model may:
 *   - Translate segment N alone (→ s(N+1))
 *   - Merge segments N and N+1 into one entry (→ s(N+2))
 *   - Merge segments N, N+1, N+2 into one entry (→ s(N+3))
 *
 * Timestamps are embedded as literals; the model only generates the text field.
 */
export function generateTranslationGrammar(
  segments: Segment[],
  mode: "base" | "echo",
): string {
  const n = segments.length;
  if (n === 0) return 'root ::= "[]"';

  let gbnf = 'root ::= "[" ws s1 ws "]"\n\n';
  gbnf += 'text-value ::= "null" | any-string\n';
  gbnf +=
    'any-string ::= "\\"" ([^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""\n';
  gbnf += "ws ::= [ \\t\\n\\r]*\n\n";

  for (let i = 0; i < n; i++) {
    const stateNum = i + 1;
    const alternatives: string[] = [];

    for (let mergeLen = 1; mergeLen <= 3; mergeLen++) {
      if (i + mergeLen > n) break;

      const mergedSegs = segments.slice(i, i + mergeLen);
      const ids = mergedSegs.map((s) => s.id);
      const start = mergedSegs[0]!.start;
      const end = mergedSegs[mergedSegs.length - 1]!.end;
      const idsJson = "[" + ids.join(", ") + "]";

      // Build the hardcoded prefix up to (but not including) the text value.
      // Echo mode: include the "input" field as a free text-value (not a
      // hardcoded literal). Embedding the Japanese input text as a GBNF
      // literal causes grammar parse failures in llama.cpp for long/complex
      // Unicode strings; the echo model is trained to output the correct
      // Echo mode
      let entry: string;
      if (mode === "base") {
        const prefix = `{"ids": ${idsJson}, "text": `;
        const suffix = `, "start": ${start}, "end": ${end}}`;
        entry = `(${makeGbnfStringLiteral(prefix)} text-value ${makeGbnfStringLiteral(suffix)})`;
      } else {
        const inputText = mergedSegs.map((s) => s.text).join("");
        const prefix = `{"ids": ${idsJson}, "input": ${JSON.stringify(inputText)}, "text": `;
        const suffix = `, "start": ${start}, "end": ${end}}`;
        entry = `(${makeGbnfStringLiteral(prefix)} text-value ${makeGbnfStringLiteral(suffix)})`;
      }

      const hasNext = i + mergeLen < n;
      if (hasNext) {
        alternatives.push(`(${entry} "," ws s${i + mergeLen + 1})`);
      } else {
        alternatives.push(entry);
      }
    }

    gbnf += `s${stateNum} ::= (\n  ${alternatives.join(" |\n  ")}\n)\n\n`;
  }

  return gbnf;
}
