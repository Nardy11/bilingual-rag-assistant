---
title: Bilingual RAG Assistant
emoji: 📚
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Arabic/English document QA with hybrid search and citations
---

# Bilingual RAG Assistant (Hugging Face Space)

Arabic/English question answering over documents, with hybrid retrieval (multilingual-e5-small + BM25,
reciprocal rank fusion), Arabic-English query translation, cross-encoder re-ranking, and answers that
cite their passages, flagged "not sure" when the match is weak. Source and evaluation:
https://github.com/Nardy11/bilingual-rag-assistant

Runs on the free CPU tier with the extractive answerer. To generate answers with an LLM, add a
`GEMINI_API_KEY` secret (or `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL`) in the Space settings.
