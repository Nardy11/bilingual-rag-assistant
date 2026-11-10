// The page: wires the engine (engine.js) to the four views (Ask, Documents, History, Code tour).
import { Engine, NOT_SURE, TOP_K, RERANK_POOL } from "./engine.js";
import { PROVIDERS, isConfigured, NOT_FOUND } from "./llm.js";
import { history, llmSettings, prefs, uploads } from "./store.js";
import { detectLanguage } from "./text.js";
import { openTour } from "./tour.js";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const dirOf = (t) => (detectLanguage(t) === "ar" ? "rtl" : "ltr");
const shortDoc = (name) => name.replace(/\.(md|markdown|txt|pdf)$/i, "");
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

const engine = new Engine();
let embedderReady;
let current = null; // the result on screen in Ask
let controller = null;
let lastVector = null; // the question vector drawn in the trace

// ======================================================================================
// Loading the models
// ======================================================================================
function progress(label) {
  const seen = {};
  return (p) => {
    if (p.status === "progress" && p.total) {
      seen[p.file] = [p.loaded, p.total];
      const [l, t] = Object.values(seen).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      setStatus(`${label}… ${(l / 1e6).toFixed(0)} of ${(t / 1e6).toFixed(0)} MB (saved by your browser after the first visit)`);
    }
  };
}
function setStatus(text, state) {
  const el = $("#status");
  el.textContent = text;
  el.className = "loadline" + (state ? ` ${state}` : "");
}
function setMode(state, text) {
  const m = $("#mode");
  m.dataset.state = state;
  $("#mode-text").textContent = text;
  m.title = {
    loading: "Loading the embedding model",
    fast: "Fast mode: search by meaning and words, scored by cosine. Translation and re-ranking are loading.",
    full: "Full mode: translation and cross-encoder re-ranking are on.",
    error: "A model failed to load",
  }[state];
}

async function boot() {
  setMode("loading", "Loading model");
  await engine.loadIndex("index.json");
  for (const rec of await uploads.all()) engine.restore(rec);
  if (location.hash.startsWith("#documents")) route(); // a link straight to a document needs the index first
  embedderReady = engine.loadEmbedder(progress("Loading the multilingual embedding model (about 120 MB)"));
  await embedderReady;
  $("#go").disabled = false;
  setMode("fast", "Fast mode");
  setStatus("Ready in fast mode. Loading translation and re-ranking in the background for better cross-language answers…");
  engine
    .loadExtras(progress("Loading translation and re-ranking (about 340 MB)"))
    .then(() => {
      setMode("full", "Full mode");
      setStatus("", "done");
    })
    .catch((err) => setStatus(`Fast mode only: translation and re-ranking could not load (${err.message}).`, "error"));
}

// ======================================================================================
// Router: #ask, #documents/<doc>/<chunk>, #history, #code/<file>
// ======================================================================================
function route() {
  const [view, ...rest] = (location.hash.slice(1) || "ask").split("/");
  const name = ["ask", "documents", "history", "code"].includes(view) ? view : "ask";
  $$(".view").forEach((v) => (v.hidden = v.dataset.view !== name));
  $$(".tabs a").forEach((a) => (a.dataset.view === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (name === "documents") {
    renderDocs();
    if (rest[0]) openDoc(decodeURIComponent(rest[0]), rest[1] ? decodeURIComponent(rest.slice(1).join("/")) : null);
  }
  if (name === "history") renderHistory();
  if (name === "code") openTour(rest.map(decodeURIComponent).join("/"));
}
window.addEventListener("hashchange", route);

// ======================================================================================
// The trace: one row per pipeline step
// ======================================================================================
const STAGES = [
  { id: "understand", title: "Read the question", side: "q", file: "rag/normalize.py",
    what: "Detect the language, unify Arabic letter forms, keep the words that keyword search will match." },
  { id: "translate", title: "Translate it", side: "q", file: "rag/translate.py",
    what: "A second wording in the other language, so an Arabic question can match English text." },
  { id: "embed", title: "Turn it into a vector", side: "q", file: "rag/embeddings.py",
    what: "multilingual-e5 maps the question to 384 numbers. Similar meanings land close together." },
  { id: "search", title: "Search two ways", side: "qd", file: "rag/retrieval.py",
    what: "Nearest vectors find the meaning, BM25 finds the exact words. Reciprocal rank fusion merges both lists." },
  { id: "rerank", title: "Re-rank the top 10", side: "qd", file: "rag/rerank.py",
    what: "A cross-encoder reads question and passage together, which is slower but far more precise." },
  { id: "decide", title: "Decide how sure", side: "d", file: "rag/pipeline.py",
    what: "The best score lands in one of three bands: sure, not sure, or not found." },
  { id: "answer", title: "Write the answer", side: "d", file: "rag/generate.py",
    what: "Quote the best sentences, or let an LLM write from the top five passages. Either way with [n] citations." },
];

function resetTrace(idle = false) {
  $("#trace-total").textContent = "";
  $("#rail").innerHTML = STAGES.map(
    (s) => `<li class="step" data-stage="${s.id}" data-side="${s.side}" data-state="${idle ? "idle" : "pending"}">
      <span class="node" aria-hidden="true"></span>
      <div class="step-head"><span class="step-title">${s.title}</span><span class="step-ms"></span></div>
      <p class="step-what">${s.what} <a class="step-learn" href="#code/${s.file}">Code</a></p>
      <div class="step-body"></div>
    </li>`,
  ).join("");
}

function setStep(id, state, html = "", ms) {
  const li = $(`.step[data-stage="${id}"]`);
  if (!li) return;
  li.dataset.state = state;
  if (ms !== undefined) $(".step-ms", li).textContent = fmtMs(ms);
  $(".step-body", li).innerHTML = html;
  const next = li.nextElementSibling;
  if (state !== "running" && next && next.dataset.state === "pending") next.dataset.state = "running";
}

const chunkLabel = (c) => `${shortDoc(c.doc)} · p${c.page}-c${c.id.split("-c").pop()}`;

function miniList(items, val) {
  return `<ol class="mini">${items
    .map((it, i) => `<li><span class="r">${i + 1}</span><span class="who" title="${esc(it.chunk.id)}">${esc(chunkLabel(it.chunk))}</span><span class="val">${val(it)}</span></li>`)
    .join("")}</ol>`;
}

function drawBarcode(canvas, vec) {
  const css = getComputedStyle(document.documentElement);
  const q = css.getPropertyValue("--q").trim();
  const d = css.getPropertyValue("--d").trim();
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 34;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  const max = Math.max(...vec.map(Math.abs)) || 1;
  const bw = w / vec.length;
  vec.forEach((v, i) => {
    const bh = (Math.abs(v) / max) * (h / 2 - 2);
    g.fillStyle = v >= 0 ? q : d;
    g.fillRect(i * bw, v >= 0 ? h / 2 - bh : h / 2, Math.max(bw - 0.3, 0.6), bh);
  });
}

function stageHtml(name, data) {
  switch (name) {
    case "understand":
      return `<dl class="kv"><dt>Language</dt><dd>${data.lang === "ar" ? "Arabic" : "English"}</dd>
        <dt>Normalised</dt><dd dir="${dirOf(data.normalized)}">${esc(data.normalized)}</dd>
        <dt>BM25 words</dt><dd><span class="tokens">${data.tokens.map((t) => `<span class="tok">${esc(t)}</span>`).join("") || '<span class="muted">none, only stopwords</span>'}</span></dd></dl>`;
    case "translate":
      if (data.skipped) return `<span class="muted">Skipped. ${esc(data.skipped)}.</span>`;
      return data.text
        ? `Also searching as <b dir="${dirOf(data.text)}">“${esc(data.text)}”</b>`
        : `<span class="muted">No useful translation (it was empty, identical or looping), so only the original wording is searched.</span>`;
    case "embed":
      return `<canvas class="barcode" aria-label="The 384 numbers of the question vector, blue above zero, violet below"></canvas>
        <p class="help">Input: <code>query: ${esc(current?.question || "")}</code>. First values: <code>${[...data.vector.slice(0, 5)].map((v) => v.toFixed(3)).join(", ")}…</code>${data.count > 1 ? ". The translation got its own vector too." : ""}</p>`;
    case "search": {
      const runs = data.runs
        .map((r, i) => `${data.runs.length > 1 ? `<p class="sub-h">${i === 0 ? "Question" : "Translation"}: <span dir="${dirOf(r.query)}">${esc(r.query)}</span></p>` : ""}
          <div class="cols"><div><h4>By meaning (cosine)</h4>${miniList(r.dense, (x) => x.score.toFixed(3))}</div>
          <div><h4>By words (BM25)</h4>${r.bm25.length ? miniList(r.bm25, (x) => x.score.toFixed(2)) : '<p class="help">No passage shares a word with the question.</p>'}</div></div>`)
        .join("");
      return `${runs}<h4 class="sub-h">Fused by reciprocal rank fusion, 1 / (60 + rank)</h4>${miniList(data.fused.slice(0, 6), (x) => `${(x.score * 1000).toFixed(1)}e-3`)}`;
    }
    case "rerank": {
      if (data.skipped) return `<span class="muted">Skipped. ${esc(data.skipped)}, so the order above stands and confidence uses the cosine.</span>`;
      const max = Math.max(...data.hits.map((h) => h.rr));
      const min = Math.min(...data.hits.map((h) => h.rr));
      return miniList(data.hits.slice(0, 6), (h) => {
        const moved = h.before - data.hits.indexOf(h) - 1;
        const mv = moved > 0 ? `<span class="move up">▲${moved}</span>` : moved < 0 ? `<span class="move down">▼${-moved}</span>` : "";
        return `${mv}<span class="bar-s" style="width:${4 + (40 * (h.rr - min)) / (max - min || 1)}px"></span>${h.rr.toFixed(2)}`;
      });
    }
    case "decide": {
      const lo = data.kind === "rerank" ? data.floor - 4 : data.floor - 0.06;
      const hi = data.kind === "rerank" ? data.sure + 6 : data.sure + 0.08;
      const pos = (v) => Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100));
      const word = { sure: "Sure", unsure: "Not sure", none: "Not found" }[data.band];
      const digits = data.kind === "rerank" ? 2 : 3;
      return `<div class="scale" style="--f:${pos(data.floor)}%;--s:${pos(data.sure)}%">
          <span class="scale-line"></span>
          <span class="scale-tick" style="left:${pos(data.floor)}%"><span>${data.floor}</span></span>
          <span class="scale-tick" style="left:${pos(data.sure)}%"><span>${data.sure}</span></span>
          <span class="scale-dot" style="left:${pos(data.confidence)}%"></span></div>
        <p class="help"><b>${word}</b>: top ${data.kind === "rerank" ? "re-ranker score" : "cosine"} ${data.confidence.toFixed(digits)}, shown as ${data.percent}% match confidence. Below ${data.floor} the answer is “not found”, from ${data.sure} it is “sure”.</p>`;
    }
    case "answer": {
      const a = data.answer;
      if (a.type === "none") return `<span class="muted">${a.notFound ? "The LLM read the passages and said they do not contain the answer." : "Not answered: the best passage scored below the floor."}</span>`;
      if (a.type === "llm") {
        const prompt = a.messages.map((m) => `--- ${m.role} ---\n${m.content}`).join("\n\n");
        return `Written by <b>${esc(a.model)}</b> from the top ${TOP_K} passages.
          <details class="prompt"><summary>See the exact prompt sent</summary><pre>${esc(prompt)}</pre></details>`;
      }
      const picks = new Set(a.parts.map((p) => p.text));
      return `${a.llmError ? `<p class="help">LLM call failed, quoted instead: ${esc(a.llmError)}</p>` : ""}
        <p class="help">Each sentence of the top 3 passages is scored 0.75 × cosine + 0.25 × shared words. The best one or two are quoted.</p>
        ${data.candidates
          .map((c) => `<div class="cand ${picks.has(c.text) ? "pick" : ""}"><span dir="${dirOf(c.text)}">${picks.has(c.text) ? "<b>" : ""}${esc(c.text.length > 110 ? c.text.slice(0, 110) + "…" : c.text)}${picks.has(c.text) ? "</b>" : ""}</span>
            <span class="math">.75×${c.dense.toFixed(2)} + .25×${c.lex.toFixed(2)} = ${c.score.toFixed(3)}</span></div>`)
          .join("")}`;
    }
  }
  return "";
}

// ======================================================================================
// Ask
// ======================================================================================
function llmConfig() {
  const cfg = llmSettings.get();
  return isConfigured(cfg) ? cfg : null;
}

async function ask(question) {
  if (controller) controller.abort();
  controller = new AbortController();
  const useLlm = $('input[name="gen"]:checked').value === "llm";
  current = { question };
  resetTrace();
  $(".step").dataset.state = "running";
  $("#welcome").hidden = true;
  $("#result").hidden = true;
  $("#go").disabled = true;
  try {
    const result = await engine.ask(question, {
      llm: useLlm ? llmConfig() : null,
      signal: controller.signal,
      stage: (name, data) => {
        const skipped = Boolean(data.skipped);
        setStep(name, skipped ? "skipped" : "done", stageHtml(name, data), skipped ? undefined : data.ms);
        if (name === "embed") {
          lastVector = [...data.vector];
          drawBarcode($('.step[data-stage="embed"] canvas'), lastVector);
        }
        if (name === "answer" && data.answer.type === "llm") $('.step[data-stage="answer"] .step-ms').textContent = fmtMs(data.ms);
      },
    });
    $("#trace-total").textContent = `total ${fmtMs(result.ms)} · ${result.mode} mode`;
    current = result;
    const { messages, ...rest } = result.answer;
    history.add({ ...result, answer: rest });
    renderResult(result);
    document.body.dataset.done = "1";
  } catch (err) {
    if (err.name === "AbortError") return;
    $(".step[data-state='running']")?.setAttribute("data-state", "pending");
    setStatus(`Something went wrong: ${err.message}`, "error");
  } finally {
    $("#go").disabled = false;
  }
}

function citeChip(n) {
  return `<button type="button" class="cite" data-cite="${n}" aria-label="Source ${n}">${n}</button>`;
}

function answerHtml(r) {
  const a = r.answer;
  if (a.type === "none") return esc(a.text || NOT_FOUND[r.lang]);
  if (a.type === "llm") return esc(a.text).replace(/\[(\d+)\]/g, (m, n) => (Number(n) <= r.hits.length ? citeChip(n) : m)).replace(/\n/g, "<br>");
  return a.parts.map((p) => `${esc(p.text)}${citeChip(p.n)}`).join(" ");
}

// a quoted answer in fast mode whose top passage is in the other language than the question
function crossFast(r) {
  const top = engine.byId.get(r.hits[0]?.id);
  return r.mode === "fast" && r.answer.type === "extractive" && top && top.lang !== r.lang;
}

function ring(percent, band) {
  return `<span class="ring" data-band="${band}" style="--p:${percent}" aria-hidden="true"><b>${percent}</b></span>`;
}

function renderResult(r, fromHistory = false) {
  current = r;
  const a = r.answer;
  const box = $("#answer");
  box.dataset.band = r.band;
  const textDir = a.type === "none" ? (r.lang === "ar" ? "rtl" : "ltr") : dirOf(a.type === "llm" ? a.text : a.parts.map((p) => p.text).join(" "));
  const gen = a.type === "llm" ? `<i class="ph ph-sparkle"></i>Written by ${esc(a.model)}` : a.type === "extractive" ? `<i class="ph ph-quotes"></i>Quoted from the documents` : `<i class="ph ph-magnifying-glass"></i>No answer`;
  const word = { sure: "Sure", unsure: "Not sure", none: "Not found" }[r.band];
  const entry = history.all().find((e) => e.id === r.id);
  const llmOff = !llmConfig();
  box.innerHTML = `
    <div class="answer-top">
      <span class="gen-tag">${gen}</span>
      <span class="conf" title="Match confidence: how strongly the best passage matches the question, mapped from the ${r.kind === "rerank" ? "re-ranker score" : "cosine"} ${r.confidence.toFixed(r.kind === "rerank" ? 2 : 3)}. It is not a measured accuracy; mark answers right or wrong for that.">
        ${ring(r.percent, r.band)}<span class="conf-text"><b>${word}</b><small>${r.percent}% match</small></span></span>
    </div>
    ${r.band === "unsure" && a.type === "extractive" ? `<span class="note" dir="${r.lang === "ar" ? "rtl" : "ltr"}">${esc(NOT_SURE[r.lang])}</span>` : ""}
    <p class="answer-text" dir="${textDir}">${answerHtml(r)}</p>
    ${crossFast(r) ? `<p class="warn"><i class="ph ph-translate"></i>Fast mode compares sentences across languages poorly, so the quote may be off. ${engine.extras ? "Full mode is ready now: ask again." : "Full mode (translation + re-ranking) is still loading."} An LLM key also fixes this.</p>` : ""}
    ${a.llmError ? `<p class="warn"><i class="ph ph-warning"></i>${esc(a.llmError)} The answer above is quoted instead.</p>` : ""}
    <div class="answer-foot">
      <span class="verdict" data-id="${r.id}">Was this right?
        <button type="button" data-v="right" aria-pressed="${entry?.verdict === "right"}"><i class="ph ph-check"></i>Right</button>
        <button type="button" data-v="wrong" aria-pressed="${entry?.verdict === "wrong"}"><i class="ph ph-x"></i>Wrong</button></span>
      ${a.type === "extractive" && llmOff ? `<p class="hint">Quoting can pick the wrong sentence. <button type="button" data-open-llm>Add an LLM key</button> to get a written answer.</p>` : `<span class="hint">${fmtMs(r.ms)}${r.translated ? ` · also searched as “${esc(r.translated)}”` : ""}</span>`}
    </div>`;

  const quoted = new Set(a.type === "extractive" ? a.parts.map((p) => p.text) : []);
  const used = new Set(a.used || []);
  $("#sources-note").textContent = r.mode === "full" ? `top ${TOP_K} after re-ranking ${RERANK_POOL}` : `top ${TOP_K} by fused rank`;
  $("#sources").innerHTML = r.hits
    .map((h, k) => {
      const c = engine.byId.get(h.id);
      if (!c) return `<li class="src" id="src-${k + 1}"><div class="src-head"><span class="src-n">[${k + 1}]</span><span class="src-meta">${esc(h.id)}, document removed</span></div></li>`;
      let text = esc(c.text);
      for (const s of quoted) if (c.text.includes(s)) text = text.replace(esc(s), `<mark>${esc(s)}</mark>`);
      const scores = [h.rr !== undefined ? `re-rank ${h.rr.toFixed(2)}` : "", `cos ${h.cos.toFixed(3)}`, `dense #${h.dRank || "-"}`, `bm25 #${h.bRank || "-"}`].filter(Boolean);
      return `<li class="src ${used.has(k + 1) ? "used" : ""}" id="src-${k + 1}">
        <div class="src-head"><span class="src-n">[${k + 1}]</span><span class="src-doc">${esc(c.doc)}</span><span class="src-meta">page ${c.page}</span>
          <span class="src-scores">${scores.map((s) => `<span>${s}</span>`).join("")}</span></div>
        <p class="src-body" dir="${c.lang === "ar" ? "rtl" : "ltr"}" lang="${c.lang}">${text}</p>
        <div class="src-foot"><a class="linkish" href="#documents/${encodeURIComponent(c.doc)}/${encodeURIComponent(c.id)}"><i class="ph ph-arrow-square-out"></i>Open in the document</a></div>
      </li>`;
    })
    .join("");
  $("#result").hidden = false;
  $("#welcome").hidden = true;
  // a passage box shows about 13 lines: scroll each one so its first quoted sentence is in view
  $$("#sources mark").forEach((m) => {
    const body = m.closest(".src-body");
    if (body && !body.dataset.scrolled) {
      body.scrollTop = Math.max(0, m.offsetTop - 24);
      body.dataset.scrolled = "1";
    }
  });
  if (fromHistory) {
    resetTrace(true);
    $("#trace-total").innerHTML = `<button type="button" class="btn small" id="reask"><i class="ph ph-arrow-clockwise"></i>Ask again to see the steps</button>`;
    $("#reask").onclick = () => ask(r.question);
  }
}

// citations, verdicts and the LLM hint inside the answer
document.addEventListener("click", (e) => {
  const cite = e.target.closest("[data-cite]");
  if (cite) {
    const el = $(`#src-${cite.dataset.cite}`);
    el?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    el?.classList.remove("flash");
    void el?.offsetWidth;
    el?.classList.add("flash");
  }
  const v = e.target.closest(".verdict button");
  if (v) {
    const id = v.parentElement.dataset.id;
    const on = v.getAttribute("aria-pressed") !== "true";
    history.mark(id, on ? v.dataset.v : null);
    $$("button", v.parentElement).forEach((b) => b.setAttribute("aria-pressed", String(on && b === v)));
    if (!$("#view-history").hidden) renderHistory();
  }
  if (e.target.closest("[data-open-llm]")) openLlm();
});

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (!q || $("#go").disabled) return;
  const u = new URL(location);
  u.searchParams.set("q", q);
  window.history.replaceState(null, "", u);
  ask(q);
});
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#ask-form").requestSubmit();
  }
});
$$("#examples .chip").forEach((b) =>
  b.addEventListener("click", () => {
    $("#q").value = b.textContent;
    $("#ask-form").requestSubmit();
  }),
);

// ======================================================================================
// LLM settings
// ======================================================================================
function syncLlmUi() {
  const cfg = llmConfig();
  $("#llm-label").textContent = cfg ? PROVIDERS[cfg.provider].label.split(" (")[0] : "Add LLM key";
  $("#llm-open").classList.toggle("on", Boolean(cfg));
  $("#gen-llm-label").title = cfg ? `${PROVIDERS[cfg.provider].label}: ${cfg.model}` : "Add a key first";
  if (!cfg) $('input[name="gen"][value="extractive"]').checked = true;
}

function openLlm() {
  const cfg = llmSettings.get() || { provider: "gemini", model: PROVIDERS.gemini.model };
  const form = $("#llm-form");
  form.provider.value = cfg.provider;
  $("#llm-model").value = cfg.model || PROVIDERS[cfg.provider].model;
  $("#llm-key").value = cfg.key || "";
  $("#llm-base").value = cfg.base || "";
  $("#llm-remember").checked = Boolean(cfg.remember);
  $("#llm-error").hidden = true;
  syncProvider(false);
  $("#llm-dialog").showModal();
}

function syncProvider(resetModel = true) {
  const p = $("#llm-form").provider.value;
  $("#base-field").hidden = p !== "compatible";
  $("#llm-key").placeholder = PROVIDERS[p].keyHint;
  if (resetModel) $("#llm-model").value = PROVIDERS[p].model;
}

$$('input[name="provider"]').forEach((r) => r.addEventListener("change", () => syncProvider()));
$("#llm-open").addEventListener("click", openLlm);
$("#llm-form").addEventListener("submit", (e) => {
  const cfg = {
    provider: e.target.provider.value,
    model: $("#llm-model").value.trim(),
    key: $("#llm-key").value.trim(),
    base: $("#llm-base").value.trim(),
    remember: $("#llm-remember").checked,
  };
  if (!isConfigured(cfg)) {
    e.preventDefault();
    $("#llm-error").textContent = cfg.provider === "compatible" ? "Enter the base URL and the model name." : "Enter the API key and the model name.";
    $("#llm-error").hidden = false;
    return;
  }
  llmSettings.set(cfg);
  $('input[name="gen"][value="llm"]').checked = true;
  syncLlmUi();
});
$("#llm-remove").addEventListener("click", () => {
  llmSettings.clear();
  syncLlmUi();
  $("#llm-dialog").close();
});
$$("dialog [data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));
$('input[name="gen"][value="llm"]').addEventListener("change", () => {
  if (!llmConfig()) openLlm();
});
$("#llm-dialog").addEventListener("close", syncLlmUi);

// ======================================================================================
// Documents
// ======================================================================================
let openName = null;
let onlyCited = false;

function citations() {
  // chunk id -> [{ entry, quoted sentence texts }]
  const map = new Map();
  for (const e of history.all()) {
    for (const n of e.answer.used || []) {
      const id = e.hits[n - 1]?.id;
      if (!id) continue;
      const quotes = e.answer.type === "extractive" ? e.answer.parts.filter((p) => p.n === n).map((p) => p.text) : [];
      if (!map.has(id)) map.set(id, []);
      map.get(id).push({ entry: e, quotes });
    }
  }
  return map;
}

function docIcon(name) {
  return /\.pdf$/i.test(name) ? "ph-file-pdf" : /\.(md|markdown)$/i.test(name) ? "ph-file-code" : "ph-file-text";
}

function renderDocs() {
  const cites = citations();
  const list = [...engine.docs.values()].sort((a, b) => (a.source === b.source ? a.name.localeCompare(b.name) : a.source === "upload" ? -1 : 1));
  $("#doclist").innerHTML = list
    .map((d) => {
      const n = engine.chunksOf(d.name).reduce((s, c) => s + (cites.get(c.id)?.length || 0), 0);
      return `<li class="doc ${d.enabled ? "" : "off"}" data-doc="${esc(d.name)}" ${d.name === openName ? 'aria-current="true"' : ""} tabindex="0">
        <span class="doc-ic"><i class="ph ${docIcon(d.name)}"></i></span>
        <span><span class="doc-name">${esc(d.name)}</span>
          <span class="doc-meta"><span class="tag ${d.source === "upload" ? "mine" : ""}">${d.source === "upload" ? "Yours" : "Sample"}</span>${d.lang === "ar" ? "Arabic" : "English"} · ${d.pages.size} page${d.pages.size > 1 ? "s" : ""} · ${d.chunks} passages${n ? ` · cited ${n}×` : ""}</span></span>
        <label class="switch" title="${d.enabled ? "Searched" : "Not searched"}"><input type="checkbox" ${d.enabled ? "checked" : ""} aria-label="Search ${esc(d.name)}"><span></span></label>
      </li>`;
    })
    .join("");
}

$("#doclist").addEventListener("click", (e) => {
  const li = e.target.closest(".doc");
  if (!li) return;
  if (e.target.closest(".switch")) return;
  location.hash = `#documents/${encodeURIComponent(li.dataset.doc)}`;
});
$("#doclist").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.classList.contains("doc")) location.hash = `#documents/${encodeURIComponent(e.target.dataset.doc)}`;
});
$("#doclist").addEventListener("change", (e) => {
  const li = e.target.closest(".doc");
  engine.setEnabled(li.dataset.doc, e.target.checked);
  renderDocs();
});

function openDoc(name, chunkId) {
  const doc = engine.docs.get(name);
  if (!doc) return;
  openName = name;
  $$(".doc").forEach((li) => (li.dataset.doc === name ? li.setAttribute("aria-current", "true") : li.removeAttribute("aria-current")));
  const cites = citations();
  const chunks = engine.chunksOf(name);
  const cited = chunks.filter((c) => cites.has(c.id)).length;
  let page = null;
  const body = chunks
    .filter((c) => !onlyCited || cites.has(c.id) || c.id === chunkId)
    .map((c) => {
      const head = doc.pages.size > 1 && c.page !== page ? `<p class="pageh">PAGE ${c.page}</p>` : "";
      page = c.page;
      const cs = cites.get(c.id) || [];
      let text = esc(c.text);
      for (const s of new Set(cs.flatMap((x) => x.quotes))) text = text.replace(esc(s), `<mark>${esc(s)}</mark>`);
      const qs = [...new Map(cs.map((x) => [x.entry.question, x.entry])).values()].slice(0, 4);
      return `${head}<article class="chunk ${cs.length ? "cited" : ""}" id="chunk-${CSS.escape(c.id)}" data-chunk="${esc(c.id)}">
        <div class="chunk-head"><span class="chunk-id">${esc(c.id)}</span>${cs.length ? `<span class="chunk-cited">cited in ${cs.length} answer${cs.length > 1 ? "s" : ""}</span>` : ""}</div>
        <p class="chunk-text" dir="${c.lang === "ar" ? "rtl" : "ltr"}" lang="${c.lang}">${text}</p>
        ${qs.length ? `<ul class="chunk-qs">${qs.map((e) => `<li><a href="#ask" data-replay="${e.id}" dir="${dirOf(e.question)}">“${esc(e.question)}”</a></li>`).join("")}</ul>` : ""}
      </article>`;
    })
    .join("");
  $("#reader").innerHTML = `
    <div class="reader-head">
      <div><h2 class="reader-title">${esc(name)}</h2>
        <p class="reader-sub">${doc.source === "upload" ? "Your file" : "Sample document"} · ${doc.chunks} passages of up to 160 tokens, overlapping by about 40 · ${cited} cited</p></div>
      <div class="reader-tools">
        <label class="check" style="margin:0"><input type="checkbox" id="only-cited" ${onlyCited ? "checked" : ""}><span>Only cited passages</span></label>
        ${doc.source === "upload" ? `<button type="button" class="btn small ghost danger" id="doc-remove"><i class="ph ph-trash"></i>Remove</button>` : ""}
      </div>
    </div>${body || `<p class="muted">No passage of this document has been cited yet.</p>`}`;
  $("#only-cited").onchange = (e) => {
    onlyCited = e.target.checked;
    openDoc(name);
  };
  const rm = $("#doc-remove");
  if (rm)
    rm.onclick = async () => {
      if (!confirm(`Remove ${name} from this device? Its passages leave the index.`)) return;
      engine.removeDocument(name);
      await uploads.remove(name);
      openName = null;
      $("#reader").innerHTML = `<div class="reader-empty"><i class="ph ph-check-circle"></i><p>${esc(name)} was removed.</p></div>`;
      renderDocs();
    };
  if (chunkId) {
    const el = document.getElementById(`chunk-${CSS.escape(chunkId)}`) || document.querySelector(`[data-chunk="${CSS.escape(chunkId)}"]`);
    if (el) {
      el.scrollIntoView({ block: "center" });
      el.classList.add("flash");
    }
  }
}

document.addEventListener("click", (e) => {
  const r = e.target.closest("[data-replay]");
  if (!r) return;
  const entry = history.all().find((x) => x.id === r.dataset.replay);
  if (entry) {
    e.preventDefault();
    location.hash = "#ask";
    $("#q").value = entry.question;
    renderResult(entry, true);
    window.scrollTo({ top: 0 });
  }
});

// uploads
const ING_STEPS = [
  ["read", "Read", "Extract the text page by page and clean it."],
  ["chunk", "Split into passages", "Sentences packed into passages of up to 160 tokens, overlapping by 40."],
  ["tokens", "Index the words", "Normalised words for BM25 keyword search."],
  ["embed", "Embed", "Each passage and sentence becomes a vector with multilingual-e5."],
  ["index", "Add to the index", "From now on, questions search this file too."],
];

async function ingest(files) {
  const box = $("#ingest");
  box.hidden = false;
  for (const file of files) {
    box.innerHTML = `<li class="ingest-file"><span>${esc(file.name)}</span><span class="muted">${file.size < 1024 ? `${file.size} bytes` : `${Math.round(file.size / 1024)} KB`}</span></li>${ING_STEPS.map(
      ([id, title, what]) => `<li class="ingest-step" data-step="${id}" data-state="pending"><i class="ph ph-circle"></i><div><b>${title}</b><span>${what}</span></div></li>`,
    ).join("")}`;
    const step = (id, info) => {
      const li = $(`[data-step="${id}"]`, box);
      li.dataset.state = info.status;
      $("i", li).className = `ph ${info.status === "done" ? "ph-check-circle" : info.status === "running" ? "ph-circle-notch" : "ph-circle"}`;
      if (info.detail) $("span", li).textContent = info.detail;
    };
    try {
      if (!engine.embedder) {
        step("read", { status: "running", detail: "Waiting for the embedding model to finish loading…" });
        await embedderReady;
      }
      const res = await engine.addFile(file, step);
      await uploads.put({
        name: res.name,
        bytes: res.bytes,
        addedAt: Date.now(),
        chunks: res.chunks.map(({ vec, ...c }) => c),
        chunkVecs: res.chunks.map((c) => c.vec),
        sentences: res.sentences.map(({ vec, ...s }) => s),
        sentVecs: res.sentences.map((s) => s.vec),
      });
      box.insertAdjacentHTML("beforeend", `<li class="ingest-step"><i class="ph ph-sparkle"></i><div><b>Done in ${fmtMs(res.ms)}</b><span><a href="#ask" data-ask-about>Ask about it</a> or read its passages below.</span></div></li>`);
      renderDocs();
      location.hash = `#documents/${encodeURIComponent(res.name)}`;
    } catch (err) {
      const running = $('[data-state="running"]', box) || $('[data-step="read"]', box);
      running.dataset.state = "error";
      $("i", running).className = "ph ph-warning-circle";
      box.insertAdjacentHTML("beforeend", `<li class="ingest-err">${esc(err.message)}</li>`);
    }
  }
}

$("#file").addEventListener("change", (e) => {
  ingest([...e.target.files]);
  e.target.value = "";
});
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  ingest([...e.dataTransfer.files]);
});
document.addEventListener("click", (e) => {
  if (e.target.closest("[data-ask-about]")) setTimeout(() => $("#q").focus(), 50);
});

// ======================================================================================
// History
// ======================================================================================
function ago(t) {
  const s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t).toLocaleDateString();
}

function renderHistory() {
  const all = history.all();
  const marked = all.filter((e) => e.verdict);
  const right = marked.filter((e) => e.verdict === "right").length;
  const answered = all.filter((e) => e.band !== "none");
  const avg = all.length ? Math.round(all.reduce((s, e) => s + e.percent, 0) / all.length) : 0;
  const llm = all.filter((e) => e.answer.type === "llm").length;
  $("#stats").innerHTML = `
    <div class="stat"><div class="stat-v">${marked.length ? `${Math.round((100 * right) / marked.length)}%` : "No marks yet"}</div>
      <div class="stat-l">${marked.length ? `accuracy: ${right} of ${marked.length} answers you marked were right` : "Mark answers right or wrong to measure accuracy"}</div></div>
    <div class="stat"><div class="stat-v">${all.length}</div><div class="stat-l">questions, ${answered.length} answered</div></div>
    <div class="stat"><div class="stat-v">${all.length ? `${avg}%` : "-"}</div><div class="stat-l">average match confidence</div></div>
    <div class="stat"><div class="stat-v">${llm}</div><div class="stat-l">written by an LLM, ${all.length - llm} quoted</div></div>`;
  if (!all.length) {
    $("#hist").innerHTML = `<li class="empty"><i class="ph ph-chat-circle-dots"></i>No questions yet. <a href="#ask">Ask one</a> and it will show up here with its answer and confidence.</li>`;
    return;
  }
  $("#hist").innerHTML = all
    .map((e) => {
      const a = e.answer;
      const text = a.type === "extractive" ? a.parts.map((p) => p.text).join(" ") : a.text || "";
      const gen = a.type === "llm" ? esc(a.model) : a.type === "extractive" ? "Quoted" : "Not found";
      return `<li class="hitem">
        ${ring(e.percent, e.band)}
        <div><p class="hitem-q" dir="${dirOf(e.question)}">${esc(e.question)}</p>
          <p class="hitem-a" dir="${dirOf(text || e.question)}">${esc(text)}</p>
          <div class="hitem-meta"><span class="tag">${gen}</span><span class="tag">${e.mode} mode</span><span>${{ sure: "Sure", unsure: "Not sure", none: "Not found" }[e.band]}</span><span>· ${ago(e.at)}</span>${e.verdict ? `<span class="tag ${e.verdict === "right" ? "mine" : ""}">marked ${e.verdict}</span>` : ""}</div></div>
        <div class="hitem-side">
          <span class="verdict" data-id="${e.id}"><button type="button" data-v="right" aria-pressed="${e.verdict === "right"}" aria-label="Mark right"><i class="ph ph-check"></i></button><button type="button" data-v="wrong" aria-pressed="${e.verdict === "wrong"}" aria-label="Mark wrong"><i class="ph ph-x"></i></button></span>
          <span><button type="button" class="btn small ghost" data-replay="${e.id}">Open</button><button type="button" class="btn small ghost" data-del="${e.id}" aria-label="Delete"><i class="ph ph-trash"></i></button></span>
        </div></li>`;
    })
    .join("");
}

$("#hist").addEventListener("click", (e) => {
  const del = e.target.closest("[data-del]");
  if (del) {
    history.remove(del.dataset.del);
    renderHistory();
  }
});
$("#hist-clear").addEventListener("click", () => {
  if (!history.all().length || !confirm("Clear all questions and marks from this device?")) return;
  history.clear();
  renderHistory();
});
$("#hist-export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(history.all(), null, 1)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "rag-history.json" });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ======================================================================================
// Theme
// ======================================================================================
function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}
function syncThemeIcon() {
  $("#theme i").className = `ph ${isDark() ? "ph-sun" : "ph-moon-stars"}`;
}
$("#theme").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  prefs.set({ theme: next });
  syncThemeIcon();
  // the vector barcode is drawn on a canvas with the theme's colours: redraw it
  const c = $('.step[data-stage="embed"] canvas');
  if (c && lastVector) drawBarcode(c, lastVector);
});

// ======================================================================================
// Start
// ======================================================================================
syncThemeIcon();
syncLlmUi();
resetTrace(true);
route();
boot()
  .then(() => {
    const q = new URLSearchParams(location.search).get("q");
    if (q) {
      $("#q").value = q;
      $("#ask-form").requestSubmit();
    }
  })
  .catch((err) => {
    setMode("error", "Model failed");
    setStatus(`Could not load the model: ${err.message}. Check your connection and reload.`, "error");
  });
