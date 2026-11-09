// Browser port of rag/: normalise -> BM25 + dense -> reciprocal rank fusion -> extractive answer.
// Document tokens, chunk vectors and sentence vectors come precomputed from Python (index.json);
// only the query is embedded here, with the ONNX export of the same model.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

env.allowLocalModels = false;

const $ = (s) => document.querySelector(s);
const RRF_K = 60, POOL = 30, TOP_K = 5, TOP_CHUNKS = 3, MAX_SENTENCES = 2;
const NOT_FOUND = { en: "Not found in the documents.", ar: "لم أجد الإجابة في المستندات." };

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

async function load() {
  const status = $("#status");
  IDX = await (await fetch("index.json")).json();
  STOP = new Set([...IDX.stopwords_en, ...IDX.stopwords_ar]);
  PREFIXES = IDX.ar_prefixes;
  CV = f32(IDX.chunk_vectors); SV = f32(IDX.sentence_vectors);
  bm25 = buildBM25(IDX.chunks.map((c) => c.tokens));
  const seen = {};
  embed = await pipeline("feature-extraction", IDX.model, {
    dtype: "q8",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        seen[p.file] = [p.loaded, p.total];
        const [l, t] = Object.values(seen).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
        status.textContent = `Loading the multilingual embedding model… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB (cached after the first visit)`;
      }
    },
  });
  status.textContent = `Ready. ${IDX.chunks.length} passages from ${new Set(IDX.chunks.map((c) => c.doc)).size} documents indexed.`;
  $("#go").disabled = false;
}

async function ask(question) {
  const t0 = performance.now(), d = IDX.dim, n = IDX.chunks.length;
  const out = await embed("query: " + question, { pooling: "mean", normalize: true });
  const q = out.data;
  const cos = IDX.chunks.map((_, i) => dot(q, CV, i, d));
  const qtok = tokenize(question);
  const bm = qtok.length ? bm25(qtok) : new Array(n).fill(0);
  const byDesc = (arr) => [...arr.keys()].sort((a, b) => arr[b] - arr[a]);
  const denseRank = byDesc(cos).slice(0, POOL), bmRank = byDesc(bm).slice(0, POOL).filter((i) => bm[i] > 0);
  const fused = new Map();
  for (const r of [denseRank, bmRank]) r.forEach((i, k) => fused.set(i, (fused.get(i) || 0) + 1 / (RRF_K + k + 1)));
  const hits = [...fused.keys()].sort((a, b) => fused.get(b) - fused.get(a)).slice(0, TOP_K)
    .map((i) => ({ i, rrf: fused.get(i), cos: cos[i], bm: bm[i], dRank: denseRank.indexOf(i) + 1, bRank: bmRank.indexOf(i) + 1 }));
  const confidence = Math.max(...hits.map((h) => h.cos));
  const ql = lang(question);
  let answer = null;
  if (confidence >= IDX.min_confidence) {
    const top = hits.slice(0, TOP_CHUNKS).map((h) => h.i), qset = new Set(qtok);
    // same candidate order as rag/generate.py: by chunk rank, then sentence order; repeated
    // sentences (chunk overlap) keep the higher-ranked copy
    const cands = [], seen = new Set();
    top.forEach((ci, n) => IDX.sentences.forEach((s, si) => {
      if (s.chunk !== ci || seen.has(s.text)) return;
      seen.add(s.text);
      const st = new Set(tokenize(s.text));
      const lex = [...qset].filter((t) => st.has(t)).length / Math.max(qset.size, 1);
      cands.push({ si, order: cands.length, n: n + 1, text: s.text, score: 0.75 * dot(q, SV, si, d) + 0.25 * lex });
    }));
    if (cands.length) {
      const max = Math.max(...cands.map((c) => c.score));
      let best = [...cands].sort((a, b) => b.score - a.score).slice(0, MAX_SENTENCES).filter((c) => c.score >= max - 0.08);
      best.sort((a, b) => a.order - b.order);
      answer = best;
    }
  }
  render(question, ql, hits, answer, confidence, performance.now() - t0);
}

function render(question, ql, hits, answer, confidence, ms) {
  const box = $("#answer");
  box.className = "answer" + (answer ? "" : " notfound");
  const usedText = new Set((answer || []).map((a) => a.text));
  let body;
  if (answer) {
    const ad = lang(answer.map((a) => a.text).join(" "));
    box.dir = ad === "ar" ? "rtl" : "ltr"; box.lang = ad;
    body = answer.map((a) => `${esc(a.text)}<sup class="cite"><a href="#src-${a.n}">${a.n}</a></sup>`).join(" ");
  } else {
    box.dir = ql === "ar" ? "rtl" : "ltr"; box.lang = ql;
    body = esc(NOT_FOUND[ql]);
  }
  const label = ql === "ar" ? "الإجابة (مقتطفة من المستندات)" : "Answer (quoted from the documents)";
  box.innerHTML = `<span class="label">${label}</span>${body}<span class="meta" dir="ltr">confidence ${confidence.toFixed(3)} (threshold ${IDX.min_confidence}) · ${ms.toFixed(0)} ms</span>`;
  $("#sources").innerHTML = hits.map((h, k) => {
    const c = IDX.chunks[h.i];
    let text = esc(c.text);
    for (const s of usedText) if (c.text.includes(s)) text = text.replace(esc(s), `<mark>${esc(s)}</mark>`);
    return `<li class="src" id="src-${k + 1}"><header dir="ltr"><span class="n">[${k + 1}]</span><b>${esc(c.doc)}</b><span>page ${c.page}</span><code>${esc(c.id)}</code>` +
      `<span>cosine ${h.cos.toFixed(3)} · dense #${h.dRank || "–"} · BM25 #${h.bRank || "–"}</span></header>` +
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
