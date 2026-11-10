// What this page remembers, all of it on your device only:
//   history      localStorage  your last 100 questions, answers and the right/wrong marks you gave
//   llm settings sessionStorage (or localStorage when you tick "remember on this device")
//   uploads      IndexedDB     your files' passages and vectors, so a reload does not re-embed them
// Every access is wrapped: in a private window storage can be missing, and the page still works.

const HISTORY = "rag.history.v1";
const LLM_KEY = "rag.llm.v1";
const PREFS = "rag.prefs.v1";

function read(storage, key, fallback) {
  try {
    const v = storage.getItem(key);
    return v ? JSON.parse(v) : fallback;
  } catch {
    return fallback;
  }
}
function write(storage, key, value) {
  try {
    if (value === null) storage.removeItem(key);
    else storage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: keep working without it */
  }
}

export const history = {
  all: () => read(localStorage, HISTORY, []),
  add(entry) {
    write(localStorage, HISTORY, [entry, ...history.all().filter((e) => e.id !== entry.id)].slice(0, 100));
  },
  mark(id, verdict) {
    write(localStorage, HISTORY, history.all().map((e) => (e.id === id ? { ...e, verdict } : e)));
  },
  remove(id) {
    write(localStorage, HISTORY, history.all().filter((e) => e.id !== id));
  },
  clear: () => write(localStorage, HISTORY, null),
};

export const llmSettings = {
  get() {
    return read(sessionStorage, LLM_KEY, null) || read(localStorage, LLM_KEY, null);
  },
  set(cfg) {
    write(sessionStorage, LLM_KEY, cfg);
    write(localStorage, LLM_KEY, cfg.remember ? cfg : null);
  },
  clear() {
    write(sessionStorage, LLM_KEY, null);
    write(localStorage, LLM_KEY, null);
  },
};

export const prefs = {
  get: () => read(localStorage, PREFS, {}),
  set: (patch) => write(localStorage, PREFS, { ...prefs.get(), ...patch }),
};

// ---- IndexedDB for uploaded documents ------------------------------------------------------------
function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("rag-uploads", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("docs", { keyPath: "name" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function tx(mode, fn) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const t = db.transaction("docs", mode);
    const result = fn(t.objectStore("docs"));
    t.oncomplete = () => resolve(result.result ?? result);
    t.onerror = () => reject(t.error);
  });
}

export const uploads = {
  async all() {
    try {
      return await tx("readonly", (s) => s.getAll());
    } catch {
      return [];
    }
  },
  async put(doc) {
    try {
      await tx("readwrite", (s) => s.put(doc));
    } catch {
      /* not persisted; the document still works until the tab closes */
    }
  },
  async remove(name) {
    try {
      await tx("readwrite", (s) => s.delete(name));
    } catch {
      /* nothing to remove */
    }
  },
};
