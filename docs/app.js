// Browser port of rag/: normalise -> (translate) -> BM25 + dense -> reciprocal rank fusion -> (re-rank) -> extractive answer.
// Document tokens, chunk vectors and sentence vectors come precomputed from Python (index.json);
// only the query is embedded here, with the ONNX export of the same model.
import { pipeline, env, AutoTokenizer, AutoModelForSequenceClassification } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

env.allowLocalModels = false;

const $ = (s) => document.querySelector(s);
const RRF_K = 60, POOL = 30, TOP_K = 5, TOP_CHUNKS = 3, MAX_SENTENCES = 2, RERANK_POOL = 10;
const NOT_FOUND = { en: "Not found in the documents.", ar: "لم أجد الإجابة في المستندات." };
const NOT_SURE = { // rag/render.py
  en: "Not sure this answers it. The closest passage in the documents says:",
  ar: "لست متأكداً أن هذا يجيب عن سؤالك. أقرب ما وجدته في المستندات:",
};

// ---- normalisation: mirrors rag/normalize.py (tests pin the Python side) ----------------------
const TASHKEEL = /[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]/g;
const INVISIBLE = /[​-‏‪-‮⁦-⁩﻿]/g;
const DIGITS = { "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
  "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9" };
const TOKEN = /[0-9]+(?:[.,][0-9]+)?|[a-z]+|[ء-ي]+/g;
let STOP = new Set(), PREFIXES = [];

function normalizeForMatch(t) {
  return t.normalize("NFKC").replace(INVISIBLE, "").replace(TASHKEEL, "").replace(/ـ/g, "")
    .replace(/[آأإٱ]/g, "ا").replace(/ى/g, "ي").replace(/ة/g, "ه")
    .replace(/[ؤئ]/g, "ء").replace(/[٠-٩۰-۹]/g, (d) => DIGITS[d]).toLowerCase();
}
function tokenize(t, removeStop = true) {
  const out = [];
  for (let tok of normalizeForMatch(t).match(TOKEN) || []) {
    if (removeStop && STOP.has(tok)) continue;
    if (/^[؀-ۿ]/.test(tok)) {
      const p = PREFIXES.find((p) => tok.startsWith(p) && tok.length - p.length >= 3);
      if (p) tok = tok.slice(p.length);
    } else if (tok.length > 3 && tok.endsWith("s") && !tok.endsWith("ss")) tok = tok.slice(0, -1);
    out.push(tok);
  }
  return out;
}
const lang = (t) => ((t.match(/[؀-ۿ]/g) || []).length > (t.match(/[A-Za-z]/g) || []).length ? "ar" : "en");

// ---- BM25 Okapi, same formula and defaults as rank_bm25 (k1=1.5, b=0.75, epsilon=0.25) ---------
function buildBM25(docs) {
  const N = docs.length, df = new Map(), tfs = docs.map((d) => {
    const tf = new Map();
    for (const t of d) tf.set(t, (tf.get(t) || 0) + 1);
    for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    return tf;
  });
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / N;
  const idf = new Map(); let sum = 0; const neg = [];
  for (const [t, n] of df) { const v = Math.log(N - n + 0.5) - Math.log(n + 0.5); idf.set(t, v); sum += v; if (v < 0) neg.push(t); }
  const eps = 0.25 * (sum / df.size);
  for (const t of neg) idf.set(t, eps);
  return (q) => docs.map((d, i) => q.reduce((s, t) => {
    const f = tfs[i].get(t) || 0;
    return s + (idf.get(t) || 0) * (f * 2.5) / (f + 1.5 * (1 - 0.75 + 0.75 * d.length / avgdl));
  }, 0));
}

// ---- data ----------------------------------------------------------------------------------
const f32 = (b64) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
const dot = (v, M, i, d) => { let s = 0; for (let j = 0; j < d; j++) s += v[j] * M[i * d + j]; return s; };
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let IDX, CV, SV, bm25, embed;
// Cross-language extras (rag/translate.py + rag/rerank.py), loaded in the background after the
// embedder: until they are ready, questions use the fast path (hybrid search + cosine bands).
let rerankTok = null, rerankModel = null;
const MT = {};
let extrasReady = false;

function progress(label) {
  const seen = {};
  return (p) => {
    if (p.status === "progress" && p.total) {
      seen[p.file] = [p.loaded, p.total];
      const [l, t] = Object.values(seen).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      $("#status").textContent = `${label}… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB (cached after the first visit)`;
    }
  };
}

async function load() {
  IDX = await (await fetch("index.json")).json();
  STOP = new Set([...IDX.stopwords_en, ...IDX.stopwords_ar]);
  PREFIXES = IDX.ar_prefixes;
  CV = f32(IDX.chunk_vectors); SV = f32(IDX.sentence_vectors);
  bm25 = buildBM25(IDX.chunks.map((c) => c.tokens));
  embed = await pipeline("feature-extraction", IDX.model, {
    dtype: "q8", progress_callback: progress("Loading the multilingual embedding model"),
  });
  $("#status").textContent = ready("fast mode; loading cross-language search in the background");
  $("#go").disabled = false;
  loadExtras().catch((err) => { $("#status").textContent = ready(`fast mode (cross-language search unavailable: ${err.message})`); });
}

function ready(mode) {
  return `Ready (${mode}). ${IDX.chunks.length} passages from ${new Set(IDX.chunks.map((c) => c.doc)).size} documents indexed.`;
}

async function loadExtras() {
  const r = IDX.reranker, label = "Loading cross-language search (re-ranker + translation)";
  rerankTok = await AutoTokenizer.from_pretrained(r.model);
  rerankModel = await AutoModelForSequenceClassification.from_pretrained(r.model, {
    model_file_name: r.file, dtype: "fp32", progress_callback: progress(label),
  });
  for (const [l, id] of Object.entries(IDX.translators)) {
    MT[l] = await pipeline("translation", id, { dtype: "q8", progress_callback: progress(label) });
  }
  extrasReady = true;
  $("#status").textContent = ready("full mode: translation + re-ranking");
}

// mirrors rag/translate.py plausible(): small MT models sometimes loop, and a looping output is worse than none
function plausible(question, t) {
  const words = t.split(/\s+/).filter(Boolean);
  if (!words.length || t.length > 3 * question.length + 20) return false;
  return words.length < 6 || new Set(words).size / words.length >= 0.5;
}

async function translate(question) {
  if (!extrasReady) return "";
  const out = await MT[lang(question)](question.slice(0, 300), { max_new_tokens: 64, no_repeat_ngram_size: 3 });
  const t = (out[0]?.translation_text || "").trim();
  return t && plausible(question, t) && t.toLowerCase() !== question.trim().toLowerCase() ? t : "";
}

const byDesc = (arr) => [...arr.keys()].sort((a, b) => arr[b] - arr[a]);

// rag/retrieval.py hybrid_search: dense + BM25 ranks (pool 30) fused by RRF, top k
async function hybrid(query, k) {
  const d = IDX.dim, n = IDX.chunks.length;
  const qv = (await embed("query: " + query, { pooling: "mean", normalize: true })).data;
  const cos = IDX.chunks.map((_, i) => dot(qv, CV, i, d));
  const qtok = tokenize(query);
  const bm = qtok.length ? bm25(qtok) : new Array(n).fill(0);
  const denseRank = byDesc(cos).slice(0, POOL), bmRank = byDesc(bm).slice(0, POOL).filter((i) => bm[i] > 0);
  const fused = new Map();
  for (const r of [denseRank, bmRank]) r.forEach((i, j) => fused.set(i, (fused.get(i) || 0) + 1 / (RRF_K + j + 1)));
  const hits = [...fused.keys()].sort((a, b) => fused.get(b) - fused.get(a)).slice(0, k)
    .map((i) => ({ i, score: fused.get(i), cos: cos[i], dRank: denseRank.indexOf(i) + 1, bRank: bmRank.indexOf(i) + 1 }));
  return { qv, qtok, hits };
}

// rag/retrieval.py fuse_hits: RRF across the question and its translation, keeping the best cosine
function fuseHits(lists, k) {
  const score = new Map(), best = new Map();
  for (const hits of lists) hits.forEach((h, j) => {
    score.set(h.i, (score.get(h.i) || 0) + 1 / (RRF_K + j + 1));
    if (!best.has(h.i) || h.cos > best.get(h.i).cos) best.set(h.i, h);
  });
  return [...score.keys()].sort((a, b) => score.get(b) - score.get(a)).slice(0, k)
    .map((i) => ({ ...best.get(i), score: score.get(i) }));
}

// rag/rerank.py: cross-encoder score against every wording, keep the best
async function rerank(queries, hits) {
  const texts = hits.map((h) => IDX.chunks[h.i].text);
  const q = queries.flatMap((x) => texts.map(() => x)), p = queries.flatMap(() => texts);
  const logits = [];
  for (let s = 0; s < q.length; s += 10) {
    const inp = rerankTok(q.slice(s, s + 10), { text_pair: p.slice(s, s + 10), padding: true, truncation: true, max_length: 512 });
    logits.push(...(await rerankModel(inp)).logits.data);
  }
  return hits.map((h, i) => ({ ...h, rr: Math.max(...queries.map((_, j) => logits[j * hits.length + i])) }))
    .sort((a, b) => b.rr - a.rr);
}

async function ask(question) {
  const t0 = performance.now(), d = IDX.dim, full = extrasReady, timing = {};
  let t = performance.now();
  const lap = (name) => { const now = performance.now(); timing[name] = now - t; t = now; };
  const translated = full ? await translate(question) : "";
  if (full) lap("translate");
  const queries = translated ? [question, translated] : [question];
  const n = full ? RERANK_POOL : TOP_K;
  const runs = [];
  for (const q of queries) runs.push(await hybrid(q, n));
  let hits = runs.length === 1 ? runs[0].hits : fuseHits(runs.map((r) => r.hits), n);
  lap("search");
  if (full) { hits = await rerank(queries, hits); lap("re-rank"); }
  hits = hits.slice(0, TOP_K);

  const kind = full ? "rerank" : "cosine";
  const [floor, sure] = full ? [IDX.min_score, IDX.sure_score] : [IDX.min_confidence, IDX.sure_confidence];
  const confidence = full ? (hits.length ? hits[0].rr : floor - 1) : Math.max(...hits.map((h) => h.cos));
  const ql = lang(question);
  let answer = null;
  if (hits.length && confidence >= floor) {
    // same as rag/generate.py ExtractiveGenerator: candidates by chunk rank, then sentence order;
    // repeated sentences (chunk overlap) keep the higher-ranked copy; each score part is the
    // better of the question and its translation
    const top = hits.slice(0, TOP_CHUNKS).map((h) => h.i), qsets = runs.map((r) => new Set(r.qtok));
    const cands = [], seen = new Set();
    top.forEach((ci, k) => IDX.sentences.forEach((s, si) => {
      if (s.chunk !== ci || seen.has(s.text)) return;
      seen.add(s.text);
      const st = new Set(tokenize(s.text));
      const lex = Math.max(...qsets.map((qs) => [...qs].filter((t) => st.has(t)).length / Math.max(qs.size, 1)));
      const dense = Math.max(...runs.map((r) => dot(r.qv, SV, si, d)));
      cands.push({ si, order: cands.length, n: k + 1, text: s.text, score: 0.75 * dense + 0.25 * lex });
    }));
    if (cands.length) {
      const max = Math.max(...cands.map((c) => c.score));
      const best = [...cands].sort((a, b) => b.score - a.score).slice(0, MAX_SENTENCES).filter((c) => c.score >= max - 0.08);
      best.sort((a, b) => a.order - b.order);
      answer = best;
    }
  }
  render({ question, ql, hits, answer, confidence, kind, floor, sure, translated, timing, ms: performance.now() - t0 });
}

function render({ question, ql, hits, answer, confidence, kind, floor, sure, translated, timing, ms }) {
  const box = $("#answer");
  const isSure = answer && confidence >= sure;
  box.className = "answer" + (answer ? (isSure ? "" : " unsure") : " notfound");
  const usedText = new Set((answer || []).map((a) => a.text));
  let body;
  if (answer) {
    const ad = lang(answer.map((a) => a.text).join(" "));
    box.dir = ad === "ar" ? "rtl" : "ltr"; box.lang = ad;
    body = answer.map((a) => `${esc(a.text)}<sup class="cite"><a href="#src-${a.n}">${a.n}</a></sup>`).join(" ");
    if (!isSure) body = `<span class="note" dir="${ql === "ar" ? "rtl" : "ltr"}" lang="${ql}">${esc(NOT_SURE[ql])}</span>` + body;
  } else {
    box.dir = ql === "ar" ? "rtl" : "ltr"; box.lang = ql;
    body = esc(NOT_FOUND[ql]);
  }
  const label = ql === "ar" ? "الإجابة (مقتطفة من المستندات)" : "Answer (quoted from the documents)";
  const via = translated ? ` · also searched as “${esc(translated)}”` : "";
  const scoreName = kind === "rerank" ? "re-ranker score" : "cosine (fast mode)";
  box.innerHTML = `<span class="label">${label}</span>${body}<span class="meta" dir="ltr">${scoreName} ${confidence.toFixed(3)} ` +
    `(sure ≥ ${sure}, not found &lt; ${floor})${via} · ${ms.toFixed(0)} ms` +
    ` (${Object.entries(timing).map(([k, v]) => `${k} ${v.toFixed(0)}`).join(" · ")})</span>`;
  $("#sources").innerHTML = hits.map((h, k) => {
    const c = IDX.chunks[h.i];
    let text = esc(c.text);
    for (const s of usedText) if (c.text.includes(s)) text = text.replace(esc(s), `<mark>${esc(s)}</mark>`);
    const rr = h.rr !== undefined ? `re-ranker ${h.rr.toFixed(2)} · ` : "";
    return `<li class="src" id="src-${k + 1}"><header dir="ltr"><span class="n">[${k + 1}]</span><b>${esc(c.doc)}</b><span>page ${c.page}</span><code>${esc(c.id)}</code>` +
      `<span>${rr}cosine ${h.cos.toFixed(3)} · dense #${h.dRank || "–"} · BM25 #${h.bRank || "–"}</span></header>` +
      `<p dir="${c.lang === "ar" ? "rtl" : "ltr"}" lang="${c.lang}">${text}</p></li>`;
  }).join("");
  $("#result").hidden = false;
  document.body.dataset.done = "1";
}

$("#ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (!q || !embed) return;
  const u = new URL(location); u.searchParams.set("q", q); history.replaceState(null, "", u);
  ask(q).catch((err) => { $("#status").textContent = "Error: " + err.message; $("#status").className = "status error"; });
});
document.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
  $("#q").value = b.textContent; $("#ask").requestSubmit();
}));

load().then(() => {
  const q = new URLSearchParams(location.search).get("q");
  if (q) { $("#q").value = q; $("#ask").requestSubmit(); }
}).catch((err) => { $("#status").textContent = "Could not load the model: " + err.message; $("#status").className = "status error"; });
