// The code tour: each source file next to its line-by-line notes.
// learn/manifest.json lists the files (written by scripts/build_code_tour.py, which also copies the
// Python sources under learn/src and checks that the notes cover every line of the current code).
const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let manifest = null;
const cache = new Map();

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function loadManifest() {
  if (!manifest) manifest = await json("learn/manifest.json");
  return manifest;
}

function renderNav(active) {
  $("#tour-nav").innerHTML = manifest.groups
    .map(
      (g) => `<div class="tour-group"><h3>${esc(g.title)}</h3>${g.files
        .map((f) => {
          const m = manifest.files[f];
          return `<a href="#code/${f}" ${f === active ? 'aria-current="page"' : ""}><span>${esc(f.split("/").pop())}</span><small>${m.lines}</small></a>`;
        })
        .join("")}</div>`,
    )
    .join("");
}

function renderPaths() {
  const path = (key, cls, title, text) => `<section class="path ${cls}"><h2>${title}</h2><p>${text}</p><ol>${manifest.paths[key]
    .map((s) => `<li><a href="#code/${s.file}"><b>${esc(s.file.split("/").pop())}</b><span>${esc(s.label)}</span></a></li>`)
    .join("")}</ol></section>`;
  return (
    path("document", "d", "The path a document takes", "From a file on disk to passages you can search: read, clean, split, embed, index.") +
    path("question", "q", "The path a question takes", "From the words you type to a cited answer: normalise, translate, embed, search, re-rank, decide, generate.")
  );
}

// highlight.js wraps tokens in spans that can cross line breaks (docstrings, template strings);
// to show one row per line, close the open spans at each newline and reopen them on the next line.
function splitLines(html) {
  const lines = [];
  const stack = [];
  let cur = "";
  const re = /(<span[^>]*>)|(<\/span>)|(\n)|([^<\n]+)/g;
  let m;
  while ((m = re.exec(html))) {
    if (m[1]) {
      stack.push(m[1]);
      cur += m[1];
    } else if (m[2]) {
      stack.pop();
      cur += m[2];
    } else if (m[3]) {
      lines.push(cur + "</span>".repeat(stack.length));
      cur = stack.join("");
    } else cur += m[4];
  }
  lines.push(cur + "</span>".repeat(stack.length));
  return lines;
}

function highlight(code, lang) {
  if (window.hljs) {
    try {
      return splitLines(window.hljs.highlight(code, { language: lang }).value);
    } catch {
      /* fall through to plain text */
    }
  }
  return code.split("\n").map(esc);
}

// notes use a small subset of Markdown: `code`, **bold**, and "L12: ..." lines
function noteHtml(note) {
  return note
    .split("\n")
    .filter((l) => l.trim())
    .map((line) => {
      let h = esc(line)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
      h = h.replace(/^(L\d+(?:\s*-\s*\d+)?):\s*/, '<span class="lref">$1</span>');
      return `<p>${h}</p>`;
    })
    .join("");
}

async function renderFile(path) {
  const meta = manifest.files[path];
  const main = $("#tour-main");
  main.innerHTML = `<div class="skeleton" style="width:30%"></div><div class="skeleton" style="width:60%"></div><div class="skeleton"></div><div class="skeleton" style="width:80%"></div>`;
  if (!cache.has(path)) {
    const [src, notes] = await Promise.all([fetch(meta.src).then((r) => r.text()), meta.notes ? json(meta.notes).catch(() => null) : null]);
    cache.set(path, { src: src.replace(/\r\n/g, "\n").replace(/\n$/, ""), notes });
  }
  const { src, notes } = cache.get(path);
  const lines = highlight(src, meta.lang);
  const blocks = notes?.blocks?.length ? notes.blocks : [{ from: 1, to: lines.length, note: "Notes for this file are being written." }];
  const all = manifest.order;
  const i = all.indexOf(path);
  const prev = all[i - 1];
  const next = all[i + 1];
  main.innerHTML = `
    <header class="file-head">
      <div class="file-path">${esc(path)}</div>
      <h1 class="file-title">${esc(notes?.title || meta.title || path)}</h1>
      ${notes?.summary ? `<p class="file-sum">${noteHtml(notes.summary).replace(/<\/?p>/g, "")}</p>` : ""}
      ${notes?.key_ideas?.length ? `<div class="ideas">${notes.key_ideas.map((k) => `<div class="idea"><b class="idea-t">${esc(k.term)}</b>${noteHtml(k.explain).replace(/<\/?p>/g, "")}</div>`).join("")}</div>` : ""}
    </header>
    <div class="tour-tools"><span class="muted">${lines.length} lines · ${blocks.length} notes</span>
      <a href="${esc(meta.github)}" target="_blank" rel="noopener">Open on GitHub</a></div>
    <div class="annot">${blocks
      .map(
        (b) => `<div class="arow" id="L${b.from}" tabindex="-1"><pre class="acode"><code>${lines
          .slice(b.from - 1, b.to)
          .map((l, j) => `<span class="ln" data-n="${b.from + j}">${l || " "}</span>`)
          .join("")}</code></pre><div class="anote">${noteHtml(b.note)}</div></div>`,
      )
      .join("")}</div>
    <nav class="tour-pager" aria-label="Next and previous file">
      ${prev ? `<a href="#code/${prev}"><small>Previous</small><b>${esc(prev)}</b></a>` : "<span></span>"}
      ${next ? `<a href="#code/${next}" style="text-align:right"><small>Next</small><b>${esc(next)}</b></a>` : ""}
    </nav>`;
  main.querySelector(".annot").addEventListener("click", (e) => {
    const row = e.target.closest(".arow");
    if (!row) return;
    main.querySelectorAll(".arow.focus").forEach((r) => r.classList.remove("focus"));
    row.classList.add("focus");
  });
  window.scrollTo({ top: 0 });
}

export async function openTour(path) {
  try {
    await loadManifest();
  } catch (err) {
    $("#tour-main").innerHTML = `<p class="muted">The code tour could not load (${esc(err.message)}).</p>`;
    return;
  }
  renderNav(path);
  if (path && manifest.files[path]) {
    await renderFile(path);
  } else {
    $("#tour-main").innerHTML = `<h1 id="code-h" class="title sm">Code tour</h1>
      <p class="lede">Every line of the project, explained next to the code. Follow the path a document takes, or the path a question takes, or open any file on the left.</p>
      <div class="paths">${renderPaths()}</div>`;
  }
}
