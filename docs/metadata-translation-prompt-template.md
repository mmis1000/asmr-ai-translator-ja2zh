# Metadata Translation Prompt Template — Phase 4b (Prose Only)

> **Usage**: Fill in `{{PROSE_INPUT_JSON}}` with title, track_list, summary in Japanese.
> Fill in `{{GLOSSARY_SUBSET_JSON}}` with pre-translated glossary terms for this work.
> Set `{{OUTPUT_LANGUAGE}}` to either `"繁體中文"` or `"简体中文"`.

---

## Prompt

````
You are a Japanese-to-Chinese translator for ASMR work metadata.

Translate the following Japanese fields into {{OUTPUT_LANGUAGE}}:
- `title`: the work title
- `track_list`: each track title (preserve the track numbers)
- `summary`: 2–4 sentence description of the work's story and tone

## Glossary

The following terms have established {{OUTPUT_LANGUAGE}} translations. When these names or terms appear in the text you are translating, use the provided `zh` translation **exactly**. Do not re-translate them.

{{GLOSSARY_SUBSET_JSON}}

## Rules

1. Translate naturally and idiomatically — do not translate word-for-word.
2. CV names, character names, and circle names from the glossary must be used verbatim.
3. Preserve the nuance and tone of the original (e.g. playful, intimate, mysterious).
4. Output only the JSON object with `title`, `track_list` (array of `{number, zh}`), and `summary`.

## Input

{{PROSE_INPUT_JSON}}
````


