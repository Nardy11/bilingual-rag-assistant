// Text processing for the browser, ported rule for rule from rag/normalize.py, rag/ingest.py and
// rag/chunking.py. Uploaded files go through exactly the same cleaning, sentence splitting,
// chunking and BM25 tokenisation as the documents indexed in Python.

// ---- normalisation (rag/normalize.py) --------------------------------------------------------
const TASHKEEL = /[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]/g;
const TATWEEL = /ـ/g;
const INVISIBLE = /[​-‏‪-‮⁦-⁩﻿]/g;
const ARABIC_DIGITS = /[٠-٩۰-۹]/g;
const TOKEN = /[0-9]+(?:[.,][0-9]+)?|[a-z]+|[ء-ي]+/g;
const ARABIC_CHAR = /[؀-ۿ]/;

let STOPWORDS = new Set();
let AR_PREFIXES = [];

// The stopword lists and Arabic prefixes come from index.json, written by Python, so both sides
// always use the same lists.
export function configure({ stopwords_en, stopwords_ar, ar_prefixes }) {
  STOPWORDS = new Set([...stopwords_en, ...stopwords_ar]);
  AR_PREFIXES = ar_prefixes;
}

function digitToAscii(d) {
  const code = d.charCodeAt(0);
  return String(code >= 0x06f0 ? code - 0x06f0 : code - 0x0660);
}

export function normalizeForMatch(text) {
  return text
    .normalize("NFKC")
    .replace(INVISIBLE, "")
    .replace(TASHKEEL, "")
    .replace(TATWEEL, "")
    .replace(/[آأإٱ]/g, "ا") // alef forms -> bare alef
    .replace(/ى/g, "ي") // alef maqsura -> yaa
    .replace(/ة/g, "ه") // taa marbuta -> haa
    .replace(/[ؤئ]/g, "ء") // hamza on waw / yaa -> hamza
    .replace(ARABIC_DIGITS, digitToAscii)
    .toLowerCase();
}

export function tokenize(text, removeStopwords = true) {
  const out = [];
  for (let tok of normalizeForMatch(text).match(TOKEN) || []) {
    if (removeStopwords && STOPWORDS.has(tok)) continue;
    if (ARABIC_CHAR.test(tok[0])) {
      const p = AR_PREFIXES.find((pre) => tok.startsWith(pre) && tok.length - pre.length >= 3);
      if (p) tok = tok.slice(p.length);
    } else if (tok.length > 3 && tok.endsWith("s") && !tok.endsWith("ss")) {
      tok = tok.slice(0, -1);
    }
    out.push(tok);
  }
  return out;
}

export function detectLanguage(text) {
  const ar = (text.match(/[؀-ۿ]/g) || []).length;
  const en = (text.match(/[A-Za-z]/g) || []).length;
  return ar > en ? "ar" : "en";
}

export function cleanText(text) {
  return text
    .normalize("NFKC")
    .replace(INVISIBLE, "")
    .replace(TATWEEL, "")
    .replace(/[ \t ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// ---- ingest (rag/ingest.py) --------------------------------------------------------------------
// PDF text comes with hard line wraps: a newline followed by a lowercase letter, digit or bracket,
// and not preceded by sentence punctuation, is a wrap, not a new paragraph.
export function unwrapLines(text) {
  return text.replace(/(?<![.!?:؟])\n(?=[a-z0-9(])/g, " ");
}

// ---- chunking (rag/chunking.py) ---------------------------------------------------------------
const SENT_END = /(?<=[.!?؟。])\s+|\n+/;
const EDGE = /^[ >\-*\t]+|[ >\-*\t]+$/g;

export function splitSentences(text) {
  return text
    .split(SENT_END)
    .map((s) => s.replace(/^#+\s*/, "").trim().replace(EDGE, ""))
    .filter(Boolean);
}

export function simpleTokenCount(text) {
  return (text.match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) || []).length;
}

function splitLong(sentence, maxTokens, count) {
  const out = [];
  let cur = [];
  for (const w of sentence.split(/\s+/).filter(Boolean)) {
    if (cur.length && count([...cur, w].join(" ")) > maxTokens) {
      out.push(cur.join(" "));
      cur = [];
    }
    cur.push(w);
  }
  if (cur.length) out.push(cur.join(" "));
  return out;
}

// Sentences are packed into a chunk until it reaches maxTokens; the next chunk starts with the
// last sentences of the previous one (up to overlapTokens), so a fact on a boundary is never lost.
export function chunkPages(pages, { maxTokens = 160, overlapTokens = 40, count = simpleTokenCount } = {}) {
  const chunks = [];
  for (const page of pages) {
    const sents = [];
    for (const s of splitSentences(page.text)) {
      const pieces = count(s) <= maxTokens ? [s] : splitLong(s, maxTokens, count);
      for (const p of pieces) sents.push([p, count(p)]);
    }
    let n = 0;
    let i = 0;
    while (i < sents.length) {
      const cur = [];
      let size = 0;
      let j = i;
      while (j < sents.length && (!cur.length || size + sents[j][1] <= maxTokens)) {
        cur.push(sents[j][0]);
        size += sents[j][1];
        j += 1;
      }
      const text = cur.join("\n");
      chunks.push({ id: `${page.doc}#p${page.page}-c${n}`, doc: page.doc, page: page.page, text, lang: detectLanguage(text) });
      n += 1;
      if (j >= sents.length) break;
      let back = 0;
      let k = j;
      while (k - 1 > i && back + sents[k - 1][1] <= overlapTokens) {
        k -= 1;
        back += sents[k][1];
      }
      i = k;
    }
  }
  return chunks;
}

// rag/generate.py is_answer_candidate: headings (fewer than 4 words) and questions (FAQ titles)
// are never quoted as answers.
export function isAnswerCandidate(sentence) {
  const s = sentence.trim();
  return tokenize(s, false).length >= 4 && !/[?؟]$/.test(s);
}
