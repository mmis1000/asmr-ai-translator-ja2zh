import { parse } from "node-html-parser";

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

export interface DlsiteMetadata {
  id: string;
  title?: string | undefined;
  circle?: string | undefined;
  va?: string | undefined;
  description?: string | undefined;
  /** Formatted markdown for LLM metadata extraction. */
  metadataMd: string;
}

/**
 * Parse a DLSite ID from a user-supplied string.
 * Accepts: "RJ123456", "RJ01234567", or a full DLSite URL.
 */
export function parseDlsiteId(input: string): string {
  // Full URL
  const urlMatch = input.match(/product_id\/(RJ\d+)/i) ?? input.match(/(RJ\d+)/i);
  if (urlMatch?.[1]) return urlMatch[1].toUpperCase();
  throw new Error(`Could not parse DLSite ID from: ${input}`);
}

/**
 * Scrape DLSite product page for metadata.
 * Only fetches metadata — does NOT download audio.
 */
export async function fetchDlsiteMetadata(id: string): Promise<DlsiteMetadata> {
  const url = `https://www.dlsite.com/maniax/work/=/product_id/${id}`;
  console.log(`[DLSite] Fetching metadata from ${url}`);

  const res = await fetch(url, {
    headers: { "User-Agent": USER_AGENT },
  });

  if (!res.ok) {
    throw new Error(`DLSite responded with ${res.status} for ${id}`);
  }

  const html = await res.text();
  const root = parse(html);

  const title = root.querySelector("#work_name")?.textContent?.trim();
  const circle = root.querySelector(".maker_name")?.textContent?.trim();
  const va = root
    .querySelectorAll("th")
    .find((th) => th.textContent.trim() === "声優")
    ?.nextElementSibling?.textContent?.trim();
  const description = root
    .querySelector(".work_parts_container")
    ?.textContent?.trim();

  // Build formatted markdown for LLM extraction
  let metadataMd = "";
  if (title) metadataMd += `# Title\n\n${title}\n\n`;
  if (circle) metadataMd += `# Circle\n\n${circle}\n\n`;
  if (va) metadataMd += `# VA\n\n${va}\n\n`;
  if (description) metadataMd += `# Description\n\n${description}\n\n`;

  console.log(`[DLSite] ${id}: title="${title}", circle="${circle}", va="${va}"`);

  return { id, title, circle, va, description, metadataMd };
}
