#!/usr/bin/env node

// provider-example.js
//
// Standalone demonstration of the text-embedder provider queue convention.
//
// Run:  node provider-example.js
//
// Shows:
//   1. Port pre-check before spawning a new binary
//   2. FIFO embed queue with single-worker processing
//   3. Queue health monitoring (depth, oldest wait time, thresholds)
//   4. Per-request timeout with consecutive-timeout recovery
//   5. Multiple concurrent callers serialised through the queue

// ---------------------------------------------------------------------------
// Configuration (mirrors the real provider's constants)
// ---------------------------------------------------------------------------

const BATCH_SIZE = 128;               // TEXTEMBEDDER_BATCH_SIZE
const EMBED_TIMEOUT_MS = 120_000;     // 2 min per request
const QUEUE_MONITOR_INTERVAL_MS = 5_000;  // check every 5s
const QUEUE_WARN_DEPTH = 10;
const QUEUE_CRITICAL_DEPTH = 30;
const MAX_CONSECUTIVE_TIMEOUTS = 3;
const HEALTH_POLL_MS = 200;
const BINARY_START_TIMEOUT_MS = 10_000;

// Port where an instance might already be running
const DEFAULT_PORT = 8089;

// ---------------------------------------------------------------------------
// Simulated text-embedder binary
// ---------------------------------------------------------------------------

// Starts a simple HTTP server that behaves like the real text-embedder
// on the given port.  Returns the server so the caller can close it.
function startFakeEmbedder(port) {
  const http = require('http');

  const server = http.createServer((req, res) => {
    if (req.url === '/health' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', model: 'landmark-lattice-v1', dims: 768 }));
      return;
    }

    if (req.url === '/embed/batch' && req.method === 'POST') {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        let texts;
        try {
          texts = JSON.parse(body).texts || [];
        } catch { texts = []; }

        // Simulate embedding work: ~22ms per text, but parallelised
        // internally by the binary.  We approximate by sleeping ~3ms per text
        // (mimicking 8-worker parallelism on the real binary).
        const sleepMs = Math.max(5, Math.round(texts.length * 22 / 8));

        setTimeout(() => {
          const results = texts.map((_, i) => ({
            index: i,
            embedding: new Array(768).fill(0).map((_, d) => (i * 997 + d * 31) % 10001),
            dimensions: 768,
            token_count: texts[i].split(/\s+/).length,
          }));

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ results, model: 'landmark-lattice-v1', count: texts.length }));
        }, sleepMs);
      });
      return;
    }

    res.writeHead(404);
    res.end();
  });

  server.listen(port);
  return server;
}

// ---------------------------------------------------------------------------
// 1. Port pre-check ---------------------------------------------------------
// ---------------------------------------------------------------------------

// Before spawning a new binary, probe the port.  If something is already
// answering on /health, use that instance instead.
async function probePort(port) {
  const url = `http://localhost:${port}/health`;
  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(2000) });
    if (resp.ok) {
      const health = await resp.json();
      log(`[port-probe] Found existing instance on port ${port} ` +
          `(model=${health.model}, dims=${health.dims}) — reusing`);
      return `http://localhost:${port}`;
    }
  } catch {
    // Nothing listening — will spawn
  }
  return null;
}

// ---------------------------------------------------------------------------
// 2. Embed request queue ----------------------------------------------------
// ---------------------------------------------------------------------------

// Each request is an object pushed by embed() and processed by the worker:
//   { texts, resolve, reject, submittedAt, batchCount }

const embedQueue = [];
let queueWorkerRunning = false;
let queueMonitorTimer = null;
let consecutiveTimeouts = 0;
let binaryUrl = null;

// embed() — the public method.  Pushes a request onto the queue and returns
// a Promise.  The queue worker processes requests one at a time.
function embed(texts) {
  if (texts.length === 0) return Promise.resolve([]);

  startQueueMonitor();

  return new Promise((resolve, reject) => {
    embedQueue.push({
      texts,
      resolve,
      reject,
      submittedAt: Date.now(),
      batchCount: Math.ceil(texts.length / BATCH_SIZE),
    });
    // Kick off the worker (no-op if already running)
    processQueue();
  });
}

// ---------------------------------------------------------------------------
// 3. Queue worker -----------------------------------------------------------
// ---------------------------------------------------------------------------

// Processes requests one at a time.  Uses a guard flag (queueWorkerRunning)
// so only one worker loop runs at any time — this is the same pattern as the
// real provider's `processQueue()`.
async function processQueue() {
  if (queueWorkerRunning) return;
  queueWorkerRunning = true;

  while (embedQueue.length > 0) {
    const req = embedQueue.shift();
    const depthAfter = embedQueue.length;

    log(`[queue] processing request (${req.texts.length} texts, ` +
        `${req.batchCount} batch(es), ${depthAfter} still queued)`);

    try {
      const startTime = Date.now();
      const results = await doEmbed(req.texts);
      const duration = Date.now() - startTime;

      if (duration > 10_000) {
        log(`[queue] SLOW request: ${duration}ms for ${req.texts.length} texts`);
      }

      // Healthy completion resets the timeout counter
      consecutiveTimeouts = 0;
      req.resolve(results);
      log(`[queue] completed in ${duration}ms`);
    } catch (err) {
      const isTimeout = err instanceof Error && (
        err.name === 'TimeoutError' || err.message.includes('aborted')
      );

      if (isTimeout) {
        consecutiveTimeouts++;
        log(`[queue] TIMEOUT (${consecutiveTimeouts}/${MAX_CONSECUTIVE_TIMEOUTS})`);

        if (consecutiveTimeouts >= MAX_CONSECUTIVE_TIMEOUTS) {
          log('[queue] Consecutive timeouts — killing binary for recovery');
          killBinary();
          consecutiveTimeouts = 0;
        }
      }

      const msg = err instanceof Error ? err.message : String(err);
      req.reject(new Error(`embed failed: ${msg}`));
    }
  }

  queueWorkerRunning = false;
}

// ---------------------------------------------------------------------------
// 4. Actual HTTP call (doEmbed) ---------------------------------------------
// ---------------------------------------------------------------------------

// Sends the request to the binary in batches.  Each batch has a timeout via
// AbortController.  Same pattern as the real provider's doEmbed().
async function doEmbed(texts) {
  const baseUrl = await resolveBaseUrl();
  const results = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS);

    try {
      const resp = await fetch(`${baseUrl}/embed/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: batch }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const body = await resp.text().catch(() => '');
        throw new Error(`/embed/batch failed (${resp.status}): ${body}`);
      }

      const data = await resp.json();
      results.push(...data.results.map((r) => r.embedding));
    } finally {
      clearTimeout(timer);
    }
  }

  return results;
}

// ---------------------------------------------------------------------------
// 5. Binary lifecycle -------------------------------------------------------
// ---------------------------------------------------------------------------

async function resolveBaseUrl() {
  // In a real provider this also checks TEXTEMBEDDER_URL env var.
  // Here we just check if we already have a URL or need to start.

  if (binaryUrl) return binaryUrl;

  // Port pre-check — is something already running?
  const existing = await probePort(DEFAULT_PORT);
  if (existing) {
    binaryUrl = existing;
    return existing;
  }

  log('[binary] No existing instance found — would spawn new binary');
  log('[binary] (not spawning — no binary available in this demo)');
  throw new Error('No text-embedder instance available');
}

function killBinary() {
  binaryUrl = null;
  log('[binary] Binary reference cleared (will re-probe on next request)');
}

// ---------------------------------------------------------------------------
// 6. Queue monitor ----------------------------------------------------------
// ---------------------------------------------------------------------------

function startQueueMonitor() {
  if (queueMonitorTimer) return;

  queueMonitorTimer = setInterval(() => {
    const depth = embedQueue.length;
    if (depth === 0) {
      consecutiveTimeouts = 0;  // healthy — reset counter
      return;
    }

    const oldest = embedQueue[0];
    const elapsed = Date.now() - oldest.submittedAt;

    log(`[monitor] queue depth: ${depth}, oldest waiting: ${elapsed}ms, ` +
        `batches: ${oldest.batchCount}`);

    if (depth >= QUEUE_CRITICAL_DEPTH) {
      log(`[monitor] *** CRITICAL: queue depth ${depth}, oldest ${elapsed}ms`);
    } else if (depth >= QUEUE_WARN_DEPTH) {
      log(`[monitor] *** WARNING: queue depth ${depth}, oldest ${elapsed}ms`);
    }
  }, QUEUE_MONITOR_INTERVAL_MS);
}

function stopQueueMonitor() {
  if (queueMonitorTimer) {
    clearInterval(queueMonitorTimer);
    queueMonitorTimer = null;
  }
}

// ---------------------------------------------------------------------------
// 7. Logging helper ---------------------------------------------------------
// ---------------------------------------------------------------------------

function log(msg) {
  const ts = new Date().toISOString().slice(11, 23);
  console.log(`${ts}  ${msg}`);
}

// ---------------------------------------------------------------------------
// 8. Demo runner ------------------------------------------------------------
// ---------------------------------------------------------------------------

async function main() {
  log('╔══════════════════════════════════════════════════════════════╗');
  log('║  text-embedder provider queue — example                      ║');
  log('╚══════════════════════════════════════════════════════════════╝');
  log('');

  // Start a fake text-embedder on the default port so the port pre-check
  // finds it and the queue can process real HTTP requests.
  const fakePort = DEFAULT_PORT;
  let fakeServer = null;

  try {
    // Check if something is already on the port (e.g. a real text-embedder)
    const running = await probePort(fakePort);
    if (!running) {
      log('[demo] No existing instance — starting fake binary for demo');
      fakeServer = startFakeEmbedder(fakePort);
      // Wait for it to be ready
      await new Promise((r) => setTimeout(r, 100));
      const ok = await probePort(fakePort);
      if (!ok) throw new Error('Fake server failed to start');
    } else {
      log('[demo] Using existing instance on port ' + fakePort);
    }
  } catch (err) {
    log('[demo] ERROR: ' + err.message);
    log('[demo] Make sure nothing is on port ' + fakePort + ' or start a ' +
        'text-embedder manually with:');
    log('  ./embedder --addr=:' + fakePort);
    process.exit(1);
  }

  log('');
  log('─── Scenario: 3 concurrent callers, staggered texts ───');
  log('');

  // Three concurrent callers, each embedding different amounts of text.
  // Since the queue processes one at a time, they will complete in FIFO
  // order even though they were created near-simultaneously.

  const startAll = Date.now();

  const p1 = embed(['caller-1 text A', 'caller-1 text B']);
  const p2 = embed(['caller-2 text alpha', 'caller-2 text beta', 'caller-2 text gamma']);
  const p3 = embed(['caller-3 text x']);

  const all = await Promise.allSettled([p1, p2, p3]);

  log('');
  log('─── Results ───');
  log('');

  const labels = ['caller-1', 'caller-2', 'caller-3'];
  for (let i = 0; i < all.length; i++) {
    const r = all[i];
    if (r.status === 'fulfilled') {
      log(`${labels[i]}: OK — ${r.value.length} vectors returned`);
    } else {
      log(`${labels[i]}: FAILED — ${r.reason.message}`);
    }
  }

  log('');
  log(`Total wall clock: ${Date.now() - startAll}ms`);
  log('(If requests ran in parallel this would be ~the slowest single ' +
      'request; because they are queued it is the SUM of all three.)');
  log('');

  // ── Scenario 2: Queue depth warning ──
  log('─── Scenario: Queue depth warning (10+ queued) ───');
  log('');

  // Push 15 requests rapidly to trigger the QUEUE_WARN_DEPTH monitor
  const many = [];
  for (let i = 0; i < 15; i++) {
    many.push(embed([`burst-text-${i}`]));
  }
  await Promise.allSettled(many);
  log('All 15 burst requests completed');

  // ── Cleanup ──
  stopQueueMonitor();
  if (fakeServer) {
    fakeServer.close();
    log('[demo] Fake binary stopped');
  }

  log('');
  log('╔══════════════════════════════════════════════════════════════╗');
  log('║  Example complete.                                          ║');
  log('╚══════════════════════════════════════════════════════════════╝');
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
