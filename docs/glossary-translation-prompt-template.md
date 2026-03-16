# Glossary Translation Prompt Template — Phase 3

> **Usage**: Fill in `{{GLOSSARY_RAW_JSON}}` with the raw JSON from `glossary-raw.json`.
> Set `{{OUTPUT_LANGUAGE}}` to either `"繁體中文"` or `"简体中文"`.

---

## Prompt

````
You are a Japanese-to-Chinese translator specializing in voice actor names, ASMR production terminology, and adult content vocabulary.

Translate the following lists of Japanese terms into {{OUTPUT_LANGUAGE}}.

Output JSON with four arrays: "cv" (voice actors and staff), "characters" (fictional characters in the work), "circles" (production circles/studios), and "terms" (ASMR and adult terminology).

## Translation Guidelines

**CV / Staff names (`cv` array)**:
- Use established fan-known Chinese stage names where they exist (e.g. 田中れいな → {{EX_REINA}}).
- If no established Chinese name exists, use phonetic transliteration.
- Include `note` if the person has a known role (e.g. "{{EX_SEIYUU}}", "{{EX_WRITER}}").

**Character names (`characters` array)**:
- Translate according to the context of the character (e.g. 妹 -> 妹妹, or specific proper names like さくら -> {{EX_SAKURA}}).
- If no established translation exists, use phonetic transliteration or conceptual translation based on the `note`.

**Circle names (`circles` array)**:
- If an official English or Chinese name exists, use it exactly.
- Otherwise, translate the meaning or transliterate.

**ASMR and adult terms (`terms` array)**:
- Use natural, idiomatic {{OUTPUT_LANGUAGE}} equivalents as commonly used in fan communities.
- Adult terms should be direct and accurate, not euphemistic.
- Include `note` for usage context where helpful.

## Rules

1. Keep `ja` exactly as given — do not modify the Japanese input strings.
2. The `zh` field MUST ONLY contain the translation of the `ja` term. Do not translate the `note` into the `zh` field.
3. Only output `zh` and optionally `note` for each entry. The `note` should ideally be the translated version of the input `note`.
4. Every term in the input must appear in the output.
5. Output only the JSON object, no additional commentary.

## Example
Input:
```json
{
  "characters": [
    {
       "ja": "100%JK",
       "note": "清楚系JKの表現"
    }
  ]
}
```
Output:
```json
{
  "characters": [
    {
       "ja": "100%JK",
       "zh": "100%女高中生",
       "note": "{{EX_JK_NOTE}}"
    }
  ]
}
```

Input:
{{GLOSSARY_RAW_JSON}}
````


