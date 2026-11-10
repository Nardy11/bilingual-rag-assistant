// Reading uploaded files into pages of clean text (the browser side of rag/ingest.py).
// PDFs are read with pdf.js, one page at a time; .txt and .md are a single page.
// Nothing is uploaded anywhere: the file is read by this tab and stays on your device.
import { cleanText, unwrapLines } from "./text.js";

const PDFJS = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168";
export const SUPPORTED = [".pdf", ".txt", ".md", ".markdown"];
export const MAX_FILE_BYTES = 10 * 1024 * 1024; // same limits as the Python API
export const MAX_PDF_PAGES = 200;

let pdfjs = null;
async function loadPdfJs() {
  if (!pdfjs) {
    pdfjs = await import(`${PDFJS}/pdf.min.mjs`);
    pdfjs.GlobalWorkerOptions.workerSrc = `${PDFJS}/pdf.worker.min.mjs`;
  }
  return pdfjs;
}

export function extensionOf(name) {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i).toLowerCase();
}

// pdf.js returns positioned text items; a new line starts when the item says so (hasEOL).
function pageText(content) {
  let out = "";
  for (const item of content.items) {
    out += item.str;
    if (item.hasEOL) out += "\n";
  }
  return out;
}

export async function readFile(file, onPage = () => {}) {
  const ext = extensionOf(file.name);
  if (!SUPPORTED.includes(ext)) throw new Error(`${file.name}: only PDF, TXT and Markdown files can be read.`);
  if (file.size > MAX_FILE_BYTES) throw new Error(`${file.name} is larger than 10 MB.`);
  if (ext !== ".pdf") {
    const text = cleanText(await file.text());
    return text ? [{ doc: file.name, page: 1, text }] : [];
  }
  const lib = await loadPdfJs();
  const pdf = await lib.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
  if (pdf.numPages > MAX_PDF_PAGES) throw new Error(`${file.name} has ${pdf.numPages} pages; the limit is ${MAX_PDF_PAGES}.`);
  const pages = [];
  for (let n = 1; n <= pdf.numPages; n++) {
    const page = await pdf.getPage(n);
    const text = unwrapLines(cleanText(pageText(await page.getTextContent())));
    if (text) pages.push({ doc: file.name, page: n, text });
    onPage(n, pdf.numPages);
  }
  return pages;
}
