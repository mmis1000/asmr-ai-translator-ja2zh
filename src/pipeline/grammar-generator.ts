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

    for (let mergeLen = 1; mergeLen <= 5; mergeLen++) {
      if (i + mergeLen > n) break;

      const mergedSegs = segments.slice(i, i + mergeLen);
      const ids = mergedSegs.map((s) => s.id);
      const start = mergedSegs[0]!.start;
      const end = mergedSegs[mergedSegs.length - 1]!.end;
      
      const gStr = (s: string) => makeGbnfStringLiteral(s);
      const wsToken = (tok: string) => `${gStr(tok)} ws`;

      const gIds = ids.map(id => gStr(id.toString())).join(` ${wsToken(",")} `);
      const idsGbnf = `${wsToken("[")} ${gIds} ws ${gStr("]")}`;

      const keyIds = wsToken('"ids"');
      const keyInput = wsToken('"input"');
      const keyText = wsToken('"text"');
      const keyStart = wsToken('"start"');
      const keyEnd = wsToken('"end"');
      const colon = wsToken(":");
      const comma = wsToken(",");

      let entry: string;
      if (mode === "base") {
        entry = `( ${gStr("{")} ws ${keyIds} ${colon} ${idsGbnf} ws ` +
                `${comma} ${keyText} ${colon} text-value ws ` +
                `${comma} ${keyStart} ${colon} ${gStr(start.toString())} ws ` +
                `${comma} ${keyEnd} ${colon} ${gStr(end.toString())} ws ${gStr("}")} )`;
      } else {
        const inputText = mergedSegs.map((s) => s.text).join(" ");
        const inputLit = gStr(JSON.stringify(inputText));
        entry = `( ${gStr("{")} ws ${keyIds} ${colon} ${idsGbnf} ws ` +
                `${comma} ${keyInput} ${colon} ${inputLit} ws ` +
                `${comma} ${keyText} ${colon} text-value ws ` +
                `${comma} ${keyStart} ${colon} ${gStr(start.toString())} ws ` +
                `${comma} ${keyEnd} ${colon} ${gStr(end.toString())} ws ${gStr("}")} )`;
      }

      const hasNext = i + mergeLen < n;
      if (hasNext) {
        alternatives.push(`(${entry} ws ${gStr(",")} ws s${i + mergeLen + 1})`);
      } else {
        alternatives.push(entry);
      }
    }

    gbnf += `s${stateNum} ::= (\n  ${alternatives.join(" |\n  ")}\n)\n\n`;
  }

  return gbnf;
}
