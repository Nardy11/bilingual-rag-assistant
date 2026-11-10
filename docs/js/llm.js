// The generation step with a large language model (the browser side of rag/generate.py).
// Your key is sent only to the provider you pick, straight from this tab; there is no server in
// between. The prompt is the same one the Python service uses.

export const NOT_FOUND = { en: "Not found in the documents.", ar: "لم أجد الإجابة في المستندات." };

export const SYSTEM_PROMPT =
  "You answer questions using ONLY the numbered sources provided. " +
  "Answer in the same language as the question (Arabic or English), in at most three sentences. " +
  "After each claim, cite the source number in square brackets, e.g. [1] or [2]. " +
  "If the sources do not contain the answer, reply exactly: {not_found}";

export const PROVIDERS = {
  gemini: { label: "Google Gemini", model: "gemini-2.5-flash", keyHint: "AIza…", base: "" },
  openai: { label: "OpenAI", model: "gpt-4o-mini", keyHint: "sk-…", base: "https://api.openai.com/v1" },
  compatible: { label: "OpenAI-compatible (Groq, OpenRouter, Ollama…)", model: "", keyHint: "optional", base: "" },
};

export function formatSources(hits) {
  return hits.map((h, i) => `[${i + 1}] (${h.chunk.doc}, page ${h.chunk.page})\n${h.chunk.text}`).join("\n\n");
}

export function buildMessages(question, hits, lang) {
  return [
    { role: "system", content: SYSTEM_PROMPT.replace("{not_found}", NOT_FOUND[lang]) },
    { role: "user", content: `Sources:\n${formatSources(hits)}\n\nQuestion: ${question}` },
  ];
}

// "[2]" markers that point at a real source, as 1-based numbers, sorted and unique.
export function citedSources(text, n) {
  const used = new Set();
  for (const m of text.matchAll(/\[(\d+)\]/g)) {
    const i = Number(m[1]);
    if (i >= 1 && i <= n) used.add(i);
  }
  return [...used].sort((a, b) => a - b);
}

async function failure(r) {
  let detail = "";
  try {
    const body = await r.json();
    detail = body.error?.message || body.message || "";
  } catch {
    /* not JSON */
  }
  if (r.status === 401 || r.status === 403) return new Error(`The provider refused the key (${r.status}). Check it in LLM settings.`);
  if (r.status === 429) return new Error("The provider says the key is over its rate limit or quota (429). Try again in a minute.");
  return new Error(`The provider answered ${r.status}${detail ? `: ${detail}` : ""}`);
}

async function callGemini(cfg, messages, signal) {
  const [system, user] = messages;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(cfg.model)}:generateContent`;
  const r = await fetch(url, {
    method: "POST",
    signal,
    headers: { "content-type": "application/json", "x-goog-api-key": cfg.key },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system.content }] },
      contents: [{ role: "user", parts: [{ text: user.content }] }],
      generationConfig: { temperature: 0 },
    }),
  });
  if (!r.ok) throw await failure(r);
  const data = await r.json();
  return (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("").trim();
}

async function callOpenAI(cfg, messages, signal) {
  const base = (cfg.base || PROVIDERS.openai.base).replace(/\/+$/, "");
  const headers = { "content-type": "application/json" };
  if (cfg.key) headers.authorization = `Bearer ${cfg.key}`;
  const r = await fetch(`${base}/chat/completions`, {
    method: "POST",
    signal,
    headers,
    body: JSON.stringify({ model: cfg.model, temperature: 0, messages }),
  });
  if (!r.ok) throw await failure(r);
  const data = await r.json();
  return (data.choices?.[0]?.message?.content || "").trim();
}

export async function generate(cfg, question, hits, lang, signal) {
  const messages = buildMessages(question, hits, lang);
  const text = cfg.provider === "gemini" ? await callGemini(cfg, messages, signal) : await callOpenAI(cfg, messages, signal);
  if (!text) throw new Error("The model returned an empty answer.");
  return { text, used: citedSources(text, hits.length), messages };
}

export function isConfigured(cfg) {
  if (!cfg || !cfg.model) return false;
  return cfg.provider === "compatible" ? Boolean(cfg.base) : Boolean(cfg.key);
}
