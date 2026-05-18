// flint_search v2 — static-page front end.
//
// Architecture: this page runs DuckDB-WASM entirely in the browser, attached
// to a 44 MB read-only .duckdb file fetched once and cached. For each query
// the page POSTs the text to a tiny Cloudflare Worker which holds the HF API
// token server-side and returns the BGE-large embedding; the embedding then
// drives a hybrid (semantic + BM25) re-ranking SQL inside the local DuckDB.
//
// Update WORKER_URL after `wrangler deploy` prints the deployment URL.

import * as duckdb from './lib/duckdb-wasm/duckdb-browser.bundled.mjs';

const WORKER_URL = 'https://flint-embed.flint-search.workers.dev/embed';
const DB_URL     = './data/flint_hf_lean_1024.duckdb';
const DB_ALIAS   = 'flint';

const QUERY_PROMPT = 'Represent this sentence for searching relevant passages: ';

// Semantic-only re-ranking. Phase-2 benchmark (see
// flint_duckdb_phase2_benchmark_results.md) showed semantic-only matches the
// full RRF (semantic + BM25) baseline at Jaccard@10 = 1.00 on the 30-query
// suite, so we lose nothing material by dropping BM25 here. The hybrid path
// can be re-added once the cross-catalog FTS quirks in DuckDB-WASM are
// resolved.
//
// The embedding is inlined as a SQL list literal rather than bound via
// prepared statement — DuckDB-WASM's parameter binding currently rejects
// FLOAT[1024] list values with "Invalid column type encountered for
// argument 0". Inlining costs ~25 KB of SQL per query but executes cleanly.
function buildRerankSql(embeddingArr) {
  const literal = '[' + embeddingArr.join(',') + ']::FLOAT[1024]';
  return `
SELECT id, uid, content, image_lookup,
       "From","Sent","To","Cc","Subject","Attachment",
       "thread_index", match_confidence,
       array_cosine_distance(embedding, ${literal}) AS distance
FROM ${DB_ALIAS}.documents
ORDER BY distance ASC
LIMIT 100
`;
}

// Archive.org repos that have a browseable "details" landing page.
// (Same list as utils.VALID_ARCHIVE.)
const VALID_ARCHIVE = new Set([
  'MSP004', 'MSP005', 'MSP008', 'MSP009', 'MSP011', 'MSP012',
  'Staff_1', 'Staff_10', 'Staff_11', 'Staff_12', 'Staff_13',
  'Staff_14', 'Staff_15', 'Staff_16', 'Staff_17',
]);

const S3_PREFIX = 'https://rjdgrlmrpwnzwmdwgizseanca.s3.us-east-2.amazonaws.com';
const METADATA_KEYS = ['From', 'Date', 'To', 'Cc', 'Subject', 'Attachment'];

// ----- DuckDB-WASM bootstrap ------------------------------------------------

let conn = null;        // AsyncDuckDBConnection
let dbReady = null;     // Promise that resolves once attach+FTS load completes

async function initDuckDB() {
  setStatus('Loading DuckDB engine…', 'info');
  const base = new URL('./lib/duckdb-wasm/', window.location.href).href;
  const MANUAL_BUNDLES = {
    mvp: {
      mainModule: base + 'duckdb-mvp.wasm',
      mainWorker: base + 'duckdb-browser-mvp.worker.js',
    },
    eh: {
      mainModule: base + 'duckdb-eh.wasm',
      mainWorker: base + 'duckdb-browser-eh.worker.js',
    },
  };
  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
  const workerBlob = new Blob(
    [`importScripts("${bundle.mainWorker}");`],
    { type: 'text/javascript' },
  );
  const workerUrl = URL.createObjectURL(workerBlob);
  const worker = new Worker(workerUrl);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule);
  URL.revokeObjectURL(workerUrl);

  setStatus('Downloading search database (44 MB, one-time)…', 'info');
  const absoluteDbUrl = new URL(DB_URL, window.location.href).href;
  await db.registerFileURL(
    'flint_hf_lean_1024.duckdb',
    absoluteDbUrl,
    duckdb.DuckDBDataProtocol.HTTP,
    false,
  );

  const c = await db.connect();
  await c.query(`ATTACH 'flint_hf_lean_1024.duckdb' AS ${DB_ALIAS} (READ_ONLY);`);

  setStatus('', null);
  conn = c;
  return c;
}

// ----- Embedding (calls the Worker) -----------------------------------------

class RateLimitError extends Error {
  constructor(retryAfter) {
    super(`Too many queries in a short time. Please wait ~${retryAfter}s before searching again.`);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

async function fetchEmbedding(userQuery) {
  const text = QUERY_PROMPT + userQuery;
  const res = await fetch(WORKER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: text }),
  });
  if (res.status === 429) {
    let retryAfter = 60;
    try {
      const j = await res.json();
      if (typeof j.retry_after === 'number') retryAfter = j.retry_after;
    } catch { /* ignore */ }
    throw new RateLimitError(retryAfter);
  }
  if (!res.ok) {
    let detail = '';
    try {
      const j = await res.json();
      detail = j.error || j.body || JSON.stringify(j);
    } catch {
      detail = await res.text();
    }
    throw new Error(`Embedding service returned ${res.status}: ${detail.slice(0, 300)}`);
  }
  const { embedding } = await res.json();
  if (!Array.isArray(embedding) || embedding.length !== 1024) {
    throw new Error(`Unexpected embedding shape: length=${embedding?.length}`);
  }
  return Float32Array.from(embedding);
}

// ----- Search ---------------------------------------------------------------

async function search(userQuery) {
  await dbReady;
  const embedding = await fetchEmbedding(userQuery);
  const sql = buildRerankSql(Array.from(embedding));
  const arrowTable = await conn.query(sql);
  return arrowTable.toArray().map((row) => row.toJSON());
}

// ----- Result rendering -----------------------------------------------------

function unwrapBraces(s) {
  if (s == null) return '';
  return String(s).replaceAll('{', '').replaceAll('}', '');
}

function escapeText(s) {
  // Strip markdown-syntactic chars that show up in OCR'd text.
  return String(s).replace(/[\\`*_{}#+$"]/g, '').replaceAll('\t', '');
}

function parsePgTextArray(s) {
  if (!s) return [];
  const inner = String(s).trim().replace(/^\{/, '').replace(/\}$/, '');
  if (!inner) return [];
  return inner.split(',')
    .map((part) => part.trim().replace(/^"/, '').replace(/"$/, ''))
    .filter(Boolean);
}

function formatImagePath(textName) {
  const stem = textName.split('.')[0];
  const directory = textName.split('_').slice(0, -1).join('_');
  return `${S3_PREFIX}/${directory}_jp2/${stem}.png`;
}

function archiveLinkForRepo(repo) {
  return VALID_ARCHIVE.has(repo)
    ? `https://archive.org/details/snyder_flint_emails/${repo}/`
    : `https://archive.org/download/snyder_flint_emails/${repo}.pdf`;
}

function renderResult(doc) {
  const el = document.createElement('article');
  el.className = 'result';

  // metadata block
  for (const key of METADATA_KEYS) {
    const dbKey = key === 'Date' ? 'Sent' : key;
    const raw = unwrapBraces(doc[dbKey]);
    if (!raw) continue;
    const parts = raw.split('","').map(escapeText).filter(Boolean);
    if (!parts.length) continue;
    const p = document.createElement('p');
    p.className = 'meta';
    const small = document.createElement('small');
    small.appendChild(document.createTextNode(`${key}: `));
    const b = document.createElement('b');
    b.textContent = parts.join(', ');
    small.appendChild(b);
    p.appendChild(small);
    el.appendChild(p);
  }

  // parsed-text expander
  const det = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = 'See the Parsed Text';
  det.appendChild(sum);
  const pre = document.createElement('pre');
  pre.textContent = doc.content || '';
  det.appendChild(pre);
  el.appendChild(det);

  // interpolated-row warning
  if (doc.match_confidence === 'interpolated') {
    const p = document.createElement('p');
    p.className = 'interpolated';
    p.textContent =
      "⚠️ Approximate page association — this email's body text could not be "
      + 'uniquely matched in the source OCR, so its page/thread was inferred from '
      + 'neighboring emails. See the About expander above for details.';
    el.appendChild(p);
  }

  // images
  const textNames = parsePgTextArray(doc.image_lookup);
  if (!textNames.length) return el;

  const imagePaths = textNames.map(formatImagePath);
  const tabsContainer = document.createElement('div');
  tabsContainer.className = 'image-tabs';

  if (imagePaths.length > 1) {
    const tabButtons = document.createElement('div');
    tabButtons.className = 'tab-buttons';
    const figure = document.createElement('figure');
    const img = document.createElement('img');
    const caption = document.createElement('figcaption');
    figure.appendChild(img);
    figure.appendChild(caption);
    imagePaths.forEach((path, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = path.split('/').pop();
      btn.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
      btn.addEventListener('click', () => {
        tabButtons.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', 'false'));
        btn.setAttribute('aria-selected', 'true');
        img.src = path;
        img.alt = path.split('/').pop();
        caption.textContent = path.split('/').pop();
      });
      tabButtons.appendChild(btn);
    });
    img.src = imagePaths[0];
    img.alt = imagePaths[0].split('/').pop();
    img.loading = 'lazy';
    caption.textContent = imagePaths[0].split('/').pop();
    tabsContainer.appendChild(tabButtons);
    tabsContainer.appendChild(figure);
  } else {
    const figure = document.createElement('figure');
    const img = document.createElement('img');
    img.src = imagePaths[0];
    img.alt = imagePaths[0].split('/').pop();
    img.loading = 'lazy';
    const caption = document.createElement('figcaption');
    caption.textContent = imagePaths[0].split('/').pop();
    figure.appendChild(img);
    figure.appendChild(caption);
    tabsContainer.appendChild(figure);
  }
  el.appendChild(tabsContainer);

  // archive.org link
  const match = imagePaths[0].match(/([^/]+)_jp2/);
  if (match) {
    const repo = match[1];
    const link = document.createElement('p');
    const a = document.createElement('a');
    a.href = archiveLinkForRepo(repo);
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = 'Click here to explore more on Archive.org';
    link.appendChild(a);
    el.appendChild(link);
  }

  return el;
}

// Dedup by uid (exact). Levenshtein-fuzzy dedup from the Streamlit version
// (thefuzz, 85% threshold) is skipped for v2; can be added back if duplicates
// leak through.
function dedupeByUid(rows) {
  const seen = new Set();
  const out = [];
  for (const r of rows) {
    const uid = r.uid || '';
    if (uid && seen.has(uid)) continue;
    seen.add(uid);
    out.push(r);
  }
  return out;
}

// ----- UI wiring ------------------------------------------------------------

const elements = {
  form: document.getElementById('search-form'),
  query: document.getElementById('query'),
  numResults: document.getElementById('num-results'),
  status: document.getElementById('status'),
  results: document.getElementById('results'),
};

function setStatus(text, level) {
  elements.status.textContent = text || '';
  if (level) {
    elements.status.setAttribute('data-level', level);
  } else {
    elements.status.removeAttribute('data-level');
  }
}

async function runSearch() {
  const q = elements.query.value.trim();
  elements.results.innerHTML = '';
  if (!q) return;
  const n = Math.max(1, Math.min(100, Number(elements.numResults.value) || 10));
  setStatus(`Searching for “${q}”…`, 'info');
  try {
    const rows = await search(q);
    const unique = dedupeByUid(rows).slice(0, n);
    if (!unique.length) {
      setStatus('No results.', 'info');
      return;
    }
    setStatus(`${unique.length} result${unique.length === 1 ? '' : 's'}.`, null);
    const frag = document.createDocumentFragment();
    unique.forEach((row) => frag.appendChild(renderResult(row)));
    elements.results.appendChild(frag);
  } catch (e) {
    console.error(e);
    if (e instanceof RateLimitError) {
      setStatus(e.message, 'warning');
    } else {
      setStatus(`Error: ${e.message}`, 'error');
    }
  }
}

elements.form.addEventListener('submit', (e) => {
  e.preventDefault();
  runSearch();
});

document.querySelectorAll('.suggestions button').forEach((btn) => {
  btn.addEventListener('click', () => {
    elements.query.value = btn.dataset.q;
    runSearch();
  });
});

// Kick off the DB bootstrap once, asynchronously. runSearch() awaits it.
dbReady = initDuckDB().catch((e) => {
  console.error(e);
  setStatus(`Failed to initialize search engine: ${e.message}`, 'error');
});
