// The RAG engine that runs in your browser: the same steps as rag/pipeline.py.
//
//   question -> (translate) -> embed -> dense search + BM25 -> reciprocal rank fusion
//            -> (cross-encoder re-rank) -> confidence band -> answer (quoted sentences or an LLM)
//
// The sample documents were chunked and embedded in Python (scripts/export_web_index.py writes
// index.json); files you upload are chunked and embedded here, with the same rules and the same
// model, so both live in one vector space and one BM25 index.
import { pipeline, env, AutoTokenizer, AutoModelForSequenceClassification } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
import * as T from "./text.js";
import * as LLM from "./llm.js";
import { readFile } from "./ingest.js";

env.allowLocalModels = false;

export const RRF_K = 60; // reciprocal rank fusion constant (rag/retrieval.py)
export const POOL = 30; // candidates each retriever contributes before fusion
export const TOP_K = 5; // passages shown and handed to the generator
export const RERANK_POOL = 10; // fused candidates the cross-encoder re-scores (rag/pipeline.py)
export const TOP_CHUNKS = 3; // passages the extractive answer may quote from
export const MAX_SENTENCES = 2;
export const NOT_SURE = {
  en: "Not sure this answers it. The closest passage in the documents says:",
  ar: "لست متأكداً أن هذا يجيب عن سؤالك. أقرب ما وجدته في المستندات:",
};

// ---- small vector helpers ------------------------------------------------------------------
function f32(b64) {
  const s = atob(b64);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
// Vectors are unit length, so the dot product is the cosine similarity.
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
const ranked = (scores) => [...scores.keys()].sort((a, b) => scores[b] - scores[a]);

// ---- BM25 Okapi: same formula and defaults as rank_bm25 (k1 = 1.5, b = 0.75, epsilon = 0.25) ----
function buildBM25(docs) {
  const N = docs.length;
  const df = new Map();
  const tfs = docs.map((d) => {
    const tf = new Map();
    for (const t of d) tf.set(t, (tf.get(t) || 0) + 1);
    for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    return tf;
  });
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / Math.max(N, 1);
  const idf = new Map();
  const negative = [];
  let sum = 0;
  for (const [t, n] of df) {
    const v = Math.log(N - n + 0.5) - Math.log(n + 0.5);
    idf.set(t, v);
    sum += v;
    if (v < 0) negative.push(t);
  }
  const eps = 0.25 * (sum / Math.max(df.size, 1));
  for (const t of negative) idf.set(t, eps);
  return (query) =>
    docs.map((d, i) =>
      query.reduce((s, t) => {
        const f = tfs[i].get(t) || 0;
        return s + ((idf.get(t) || 0) * f * 2.5) / (f + 1.5 * (1 - 0.75 + (0.75 * d.length) / avgdl));
      }, 0),
    );
}

// ---- confidence as a percentage ----------------------------------------------------------------
// The raw signal is a cross-encoder logit (full mode) or a cosine (fast mode); neither is a
// probability. For people, the three bands are mapped onto one scale: below the floor -> under
// 40 %, between floor and "sure" -> 40 to 75 %, above "sure" -> 75 % and up. It says how strongly
// the best passage matches the question, not whether the answer is right.
export function confidencePercent(score, floor, sure, kind) {
  const span = kind === "rerank" ? 6 : 0.08;
  let p;
  if (score < floor) p = 40 - (40 * (floor - score)) / span;
  else if (score < sure) p = 40 + (35 * (score - floor)) / (sure - floor);
  else p = 75 + (24 * (score - sure)) / span;
  return Math.round(Math.min(99, Math.max(1, p)));
}

export class Engine {
  constructor() {
    this.config = null;
    this.docs = new Map(); // name -> { name, source: "sample" | "upload", lang, pages, chunks, enabled }
    this.chunks = []; // { id, doc, page, lang, text, tokens, vec }
    this.byId = new Map();
    this.sentences = new Map(); // chunk id -> [{ text, vec }]
    this.active = []; // the chunks of enabled documents, in index order
    this.bm25 = null;
    this.embedder = null;
    this.extras = null; // { tok, model, mt: { ar, en } } once full mode has loaded
  }

  get mode() {
    return this.extras ? "full" : "fast";
  }

  // -- loading ----------------------------------------------------------------------------------
  async loadIndex(url = "index.json") {
    const cfg = await (await fetch(url)).json();
    this.config = cfg;
    T.configure(cfg);
    const d = cfg.dim;
    const cv = f32(cfg.chunk_vectors);
    const sv = f32(cfg.sentence_vectors);
    const chunks = cfg.chunks.map((c, i) => ({ ...c, vec: cv.subarray(i * d, (i + 1) * d) }));
    const sentences = cfg.sentences.map((s, i) => ({ chunkId: cfg.chunks[s.chunk].id, text: s.text, vec: sv.subarray(i * d, (i + 1) * d) }));
    this._add(chunks, sentences, "sample");
  }

  async loadEmbedder(onProgress) {
    this.embedder = await pipeline("feature-extraction", this.config.model, { dtype: "q8", progress_callback: onProgress });
  }

  async loadExtras(onProgress) {
    const r = this.config.reranker;
    const tok = await AutoTokenizer.from_pretrained(r.model);
    const model = await AutoModelForSequenceClassification.from_pretrained(r.model, {
      model_file_name: r.file,
      dtype: "fp32",
      progress_callback: onProgress,
    });
    const mt = {};
    for (const [lang, id] of Object.entries(this.config.translators)) {
      mt[lang] = await pipeline("translation", id, { dtype: "q8", progress_callback: onProgress });
    }
    this.extras = { tok, model, mt };
  }

  // -- the index ----------------------------------------------------------------------------------
  _add(chunks, sentences, source) {
    for (const c of chunks) {
      if (!c.tokens) c.tokens = T.tokenize(c.text);
      this.chunks.push(c);
      this.byId.set(c.id, c);
      const doc = this.docs.get(c.doc) || { name: c.doc, source, pages: new Set(), chunks: 0, enabled: true, chars: 0, langs: {} };
      doc.pages.add(c.page);
      doc.chunks += 1;
      doc.langs[c.lang] = (doc.langs[c.lang] || 0) + 1;
      doc.lang = Object.entries(doc.langs).sort((a, b) => b[1] - a[1])[0][0];
      this.docs.set(c.doc, doc);
    }
    for (const s of sentences) {
      if (!this.sentences.has(s.chunkId)) this.sentences.set(s.chunkId, []);
      this.sentences.get(s.chunkId).push(s);
    }
    this.reindex();
  }

  // BM25 statistics (document frequencies, average length) cover the enabled documents only.
  reindex() {
    this.active = this.chunks.filter((c) => this.docs.get(c.doc)?.enabled);
    this.bm25 = buildBM25(this.active.map((c) => c.tokens));
  }

  setEnabled(name, enabled) {
    const doc = this.docs.get(name);
    if (doc) {
      doc.enabled = enabled;
      this.reindex();
    }
  }

  removeDocument(name) {
    this.chunks = this.chunks.filter((c) => {
      if (c.doc !== name) return true;
      this.byId.delete(c.id);
      this.sentences.delete(c.id);
      return false;
    });
    this.docs.delete(name);
    this.reindex();
  }

  chunksOf(name) {
    return this.chunks.filter((c) => c.doc === name);
  }

  // -- embedding ------------------------------------------------------------------------------------
  // E5 was trained with "query: " and "passage: " prefixes; leaving them out costs accuracy.
  async embed(texts, prefix, onBatch = () => {}) {
    const d = this.config.dim;
    const out = [];
    for (let s = 0; s < texts.length; s += 16) {
      const batch = texts.slice(s, s + 16).map((t) => prefix + t);
      const res = await this.embedder(batch, { pooling: "mean", normalize: true });
      for (let i = 0; i < batch.length; i++) out.push(res.data.slice(i * d, (i + 1) * d));
      onBatch(Math.min(s + 16, texts.length), texts.length);
    }
    return out;
  }

  countTokens(text) {
    return this.embedder.tokenizer.encode(text, { add_special_tokens: false }).length;
  }

  // -- adding your own file: read -> chunk -> tokenise -> embed -> index ------------------------------
  async addFile(file, step = () => {}) {
    const t0 = performance.now();
    step("read", { status: "running" });
    const pages = await readFile(file, (n, total) => step("read", { status: "running", detail: `page ${n} of ${total}` }));
    if (!pages.length) throw new Error(`${file.name}: no text found. Scanned PDFs need OCR first.`);
    const chars = pages.reduce((s, p) => s + p.text.length, 0);
    step("read", { status: "done", detail: `${pages.length} page${pages.length > 1 ? "s" : ""}, ${chars.toLocaleString()} characters` });

    step("chunk", { status: "running" });
    const chunks = T.chunkPages(pages, { count: (t) => this.countTokens(t) });
    const sentences = [];
    for (const c of chunks) {
      for (const s of T.splitSentences(c.text)) if (T.isAnswerCandidate(s)) sentences.push({ chunkId: c.id, text: s });
    }
    step("chunk", { status: "done", detail: `${chunks.length} passages of up to 160 tokens, ${sentences.length} quotable sentences`, chunks });

    step("tokens", { status: "running" });
    for (const c of chunks) c.tokens = T.tokenize(c.text);
    const vocab = new Set(chunks.flatMap((c) => c.tokens));
    step("tokens", { status: "done", detail: `${vocab.size.toLocaleString()} distinct BM25 terms`, sample: chunks[0]?.tokens.slice(0, 12) });

    step("embed", { status: "running" });
    const total = chunks.length + sentences.length;
    const cv = await this.embed(chunks.map((c) => c.text), "passage: ", (n) => step("embed", { status: "running", detail: `${n} of ${total} texts` }));
    const sv = await this.embed(sentences.map((s) => s.text), "passage: ", (n) =>
      step("embed", { status: "running", detail: `${chunks.length + n} of ${total} texts` }),
    );
    chunks.forEach((c, i) => (c.vec = cv[i]));
    sentences.forEach((s, i) => (s.vec = sv[i]));
    step("embed", { status: "done", detail: `${total} vectors of ${this.config.dim} numbers`, vector: cv[0] });

    step("index", { status: "running" });
    if (this.docs.has(file.name)) this.removeDocument(file.name);
    this._add(chunks, sentences, "upload");
    step("index", { status: "done", detail: `searchable now, ${this.active.length} passages in the index` });
    return { name: file.name, pages, chunks, sentences, ms: performance.now() - t0, bytes: file.size };
  }

  // Uploaded documents are kept in IndexedDB so they survive a reload without re-embedding.
  restore(record) {
    const chunks = record.chunks.map((c, i) => ({ ...c, vec: record.chunkVecs[i] }));
    const sentences = record.sentences.map((s, i) => ({ ...s, vec: record.sentVecs[i] }));
    this._add(chunks, sentences, "upload");
  }

  // -- retrieval ------------------------------------------------------------------------------------
  async translate(question) {
    if (!this.extras) return "";
    const out = await this.extras.mt[T.detectLanguage(question)](question.slice(0, 300), { max_new_tokens: 64, no_repeat_ngram_size: 3 });
    const t = (out[0]?.translation_text || "").trim();
    return t && plausible(question, t) && t.toLowerCase() !== question.trim().toLowerCase() ? t : "";
  }

  // rag/retrieval.py hybrid_search: rank by cosine and by BM25 separately, fuse the two rank lists.
  hybrid(query, qv, k) {
    const cos = this.active.map((c) => dot(qv, c.vec));
    const tokens = T.tokenize(query);
    const bm = tokens.length ? this.bm25(tokens) : cos.map(() => 0);
    const denseRank = ranked(cos).slice(0, POOL);
    const bmRank = ranked(bm).slice(0, POOL).filter((i) => bm[i] > 0);
    const fused = new Map();
    for (const list of [denseRank, bmRank]) list.forEach((i, r) => fused.set(i, (fused.get(i) || 0) + 1 / (RRF_K + r + 1)));
    const hits = [...fused.keys()]
      .sort((a, b) => fused.get(b) - fused.get(a))
      .slice(0, k)
      .map((i) => ({ chunk: this.active[i], score: fused.get(i), cos: cos[i], dRank: denseRank.indexOf(i) + 1, bRank: bmRank.indexOf(i) + 1, bm: bm[i] }));
    return {
      tokens,
      hits,
      dense: denseRank.slice(0, 5).map((i) => ({ chunk: this.active[i], score: cos[i] })),
      bm25: bmRank.slice(0, 5).map((i) => ({ chunk: this.active[i], score: bm[i] })),
    };
  }

  // rag/retrieval.py fuse_hits: fuse the question's and the translation's lists, keep the best cosine.
  fuseHits(lists, k) {
    const score = new Map();
    const best = new Map();
    for (const hits of lists) {
      hits.forEach((h, r) => {
        const id = h.chunk.id;
        score.set(id, (score.get(id) || 0) + 1 / (RRF_K + r + 1));
        if (!best.has(id) || h.cos > best.get(id).cos) best.set(id, h);
      });
    }
    return [...score.keys()]
      .sort((a, b) => score.get(b) - score.get(a))
      .slice(0, k)
      .map((id) => ({ ...best.get(id), score: score.get(id) }));
  }

  // rag/rerank.py: a cross-encoder reads question and passage together; each passage keeps its
  // best score over the wordings (question, translation).
  async rerank(queries, hits) {
    const { tok, model } = this.extras;
    const texts = hits.map((h) => h.chunk.text);
    const q = queries.flatMap((x) => texts.map(() => x));
    const p = queries.flatMap(() => texts);
    const logits = [];
    for (let s = 0; s < q.length; s += 10) {
      const inputs = tok(q.slice(s, s + 10), { text_pair: p.slice(s, s + 10), padding: true, truncation: true, max_length: 512 });
      logits.push(...(await model(inputs)).logits.data);
    }
    return hits
      .map((h, i) => ({ ...h, before: i + 1, rr: Math.max(...queries.map((_, j) => logits[j * hits.length + i])) }))
      .sort((a, b) => b.rr - a.rr);
  }

  // rag/generate.py ExtractiveGenerator: score every sentence of the top passages,
  // 0.75 x cosine + 0.25 x share of the question's words, and quote the best one or two.
  extract(hits, runs) {
    const qsets = runs.map((r) => new Set(r.tokens));
    const cands = [];
    const seen = new Set();
    hits.slice(0, TOP_CHUNKS).forEach((h, k) => {
      for (const s of this.sentences.get(h.chunk.id) || []) {
        if (seen.has(s.text)) continue;
        seen.add(s.text);
        const st = new Set(T.tokenize(s.text));
        const lex = Math.max(...qsets.map((qs) => [...qs].filter((t) => st.has(t)).length / Math.max(qs.size, 1)));
        const dense = Math.max(...runs.map((r) => dot(r.qv, s.vec)));
        cands.push({ order: cands.length, n: k + 1, text: s.text, dense, lex, score: 0.75 * dense + 0.25 * lex });
      }
    });
    if (!cands.length) return { parts: [], cands };
    const max = Math.max(...cands.map((c) => c.score));
    const best = [...cands]
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_SENTENCES)
      .filter((c) => c.score >= max - 0.08)
      .sort((a, b) => a.order - b.order);
    return { parts: best, cands: [...cands].sort((a, b) => b.score - a.score).slice(0, 6) };
  }

  // -- one question, start to finish -------------------------------------------------------------------
  // stage(name, data) is called as each step finishes, so the page can draw the trace live.
  async ask(question, { llm = null, stage = () => {}, signal } = {}) {
    if (!this.active.length) throw new Error("No documents are switched on. Turn one on in Documents.");
    const t0 = performance.now();
    const timing = {};
    let t = t0;
    const lap = (name) => {
      const now = performance.now();
      timing[name] = now - t;
      t = now;
      return timing[name];
    };
    const full = Boolean(this.extras);
    const lang = T.detectLanguage(question);

    stage("understand", { lang, normalized: T.normalizeForMatch(question), tokens: T.tokenize(question) });

    let translated = "";
    if (full) {
      translated = await this.translate(question);
      stage("translate", { text: translated, ms: lap("translate") });
    } else {
      stage("translate", { skipped: "fast mode: the translation model is still loading" });
    }
    const queries = translated ? [question, translated] : [question];

    const qvs = await this.embed(queries, "query: ");
    stage("embed", { vector: qvs[0], dims: this.config.dim, count: qvs.length, ms: lap("embed") });

    const n = full ? RERANK_POOL : TOP_K;
    const runs = queries.map((q, i) => ({ query: q, qv: qvs[i], ...this.hybrid(q, qvs[i], n) }));
    let hits = runs.length === 1 ? runs[0].hits : this.fuseHits(runs.map((r) => r.hits), n);
    stage("search", { runs, fused: hits.slice(0, n), ms: lap("search") });

    if (full) {
      hits = await this.rerank(queries, hits);
      stage("rerank", { hits, ms: lap("rerank") });
    } else {
      stage("rerank", { skipped: "fast mode: the re-ranker is still loading" });
    }
    hits = hits.slice(0, TOP_K);

    const kind = full ? "rerank" : "cosine";
    const [floor, sure] = full ? [this.config.min_score, this.config.sure_score] : [this.config.min_confidence, this.config.sure_confidence];
    const confidence = full ? (hits.length ? hits[0].rr : floor - 1) : Math.max(...hits.map((h) => h.cos), 0);
    let band = !hits.length || confidence < floor ? "none" : confidence >= sure ? "sure" : "unsure";
    const percent = confidencePercent(confidence, floor, sure, kind);
    stage("decide", { kind, confidence, floor, sure, band, percent });

    let answer = { type: "none", text: LLM.NOT_FOUND[lang], used: [] };
    let extraction = null;
    if (band !== "none") {
      if (llm && LLM.isConfigured(llm)) {
        try {
          const g = await LLM.generate(llm, question, hits, lang, signal);
          const notFound = Object.values(LLM.NOT_FOUND).some((nf) => g.text.startsWith(nf));
          answer = { type: "llm", text: g.text, used: g.used, model: llm.model, provider: llm.provider, messages: g.messages, notFound };
          if (notFound) band = "none";
        } catch (err) {
          if (err.name === "AbortError") throw err;
          extraction = this.extract(hits, runs);
          answer = { type: "extractive", parts: extraction.parts, used: [...new Set(extraction.parts.map((p) => p.n))], llmError: err.message };
        }
      } else {
        extraction = this.extract(hits, runs);
        answer = { type: "extractive", parts: extraction.parts, used: [...new Set(extraction.parts.map((p) => p.n))] };
      }
    }
    stage("answer", { answer, candidates: extraction?.cands || [], ms: lap("answer") });

    return {
      id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
      at: Date.now(),
      question,
      lang,
      mode: this.mode,
      translated,
      kind,
      confidence,
      floor,
      sure,
      band,
      percent,
      answer,
      hits: hits.map((h) => ({ id: h.chunk.id, cos: h.cos, rr: h.rr, dRank: h.dRank, bRank: h.bRank })),
      timing,
      ms: performance.now() - t0,
    };
  }
}

// rag/translate.py plausible(): small translation models sometimes loop, and a looping output is
// worse than no translation at all.
function plausible(question, t) {
  const words = t.split(/\s+/).filter(Boolean);
  if (!words.length || t.length > 3 * question.length + 20) return false;
  return words.length < 6 || new Set(words).size / words.length >= 0.5;
}
