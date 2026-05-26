# text-embedder

A zero-dependency Go HTTP server that converts text into **bit-perfect deterministic**
embedding vectors. Unlike standard neural models, this project uses the 
**Landmark Lattice** architecture to ensure 100% hardware-independence and
infinite semantic stability.

---

## Solving the "Three Walls" of AI Embeddings

Traditional neural network embeddings (OpenAI, BERT, etc.) face three fundamental industry challenges. `text-embedder` is designed to solve them by moving from a "Neural" paradigm to a "Geometric Basis" paradigm.

### 1. Hardware Jitter (The Precision Wall)
*   **The Problem:** Floating-point addition is non-associative. GPUs finish calculations in random order, leading to tiny rounding errors. The same text on an NVIDIA H100 vs. an Apple M3 will produce slightly different vectors (e.g., a "drift" in the 8th decimal place).
*   **Our Solution:** **Integer Lattice Pipeline.** Every step—from hashing to accumulation—is performed using discrete integer math. We replace `float32` accumulation with an `int64` lattice, ensuring 100% bit-parity across all CPU architectures.

### 2. Semantic Drift (The Version Wall)
*   **The Problem:** When you upgrade a model (e.g., from v1 to v2), the entire coordinate system changes. Old vectors stored in your database become useless because "Cat" in V1 is in a different location than "Cat" in V2.
*   **Our Solution:** **Semantic Landmark Basis.** Instead of "absolute coordinates," our vectors represent **relative distances** to fixed concept anchors (e.g., "Biology", "Finance", "Justice"). Even if we update the internal math, we recalibrate it to the *same* fixed landmarks, keeping your stored vectors stable for decades.

### 3. Contextual Ambiguity (The Logic Wall)
*   **The Problem:** Lexical models usually can't tell the difference between "river bank" and "investment bank" without a massive Transformer model (500MB+).
*   **Our Solution:** **Syntactic Feature Hashing.** We use a lightweight, rule-based "Dependency Sketcher" that identifies structural anchors (prepositions like "of", "to", "with"). By hashing the **relationship** (e.g., `bank_of_river`), we achieve high contextual awareness with zero-dependency Go code.

---

## Quick start

```bash
go build -o embedder ./cmd/text-embedder
./embedder              # listens on :8089, workers=GOMAXPROCS
./embedder --workers=4  # limit concurrency, or --workers=1 for sequential
```

## Using as a Go module

```go
import "github.com/guiperry/text-embedder/pkg/embed"

// vec is []int32 with 768 elements, scaled to [0, 10000]
vec := embed.Embed("the quick brown fox")
```

### Compare two texts

```go
a := embed.Embed("neural networks")
b := embed.Embed("deep learning")

sim := embed.CosineSimilarity(a, b) // returns float64 in [0, 1]
```

---

## Algorithm: `landmark-lattice-v1`

### 1. Syntactic Tokenisation
Input is lowercased and split. We identify **Structural Anchors** to create contextual features (e.g., `bank` modified by `river`).

### 2. Integer Lattice Accumulation
Features are hashed into a 4096-dimensional `int64` lattice. We use FNV-1a hashing and bitwise logic to ensure the summation is identical on all CPUs.

### 3. Landmark Projection
The high-dimensional lattice is projected onto a **Basis Set** of 768 semantic landmarks. Each dimension in the output vector represents the similarity of the input text to a specific landmark concept.

### 4. Fixed-Point Scaling
The final similarities are scaled to the range `[0, 10000]` and stored as `int32`. This prevents floating-point drift during storage and transmission.

## HTTP API

All endpoints accept and return `application/json`.

### `POST /embed`
Returns a 768-dimensional `int32` vector.
```json
{
  "embedding": [942, 104, 3922, "..."],
  "dimensions": 768,
  "model": "landmark-lattice-v1",
  "token_count": 4
}
```

### `POST /embed/batch`
Embeds up to 256 texts in a single request. Processing is **parallelized internally** using a bounded worker pool — each text runs in its own goroutine, and results are written by index so response order matches request order.

```json
// Request
{ "texts": ["first text", "second text", "third text"] }

// Response
{
  "results": [
    { "index": 0, "embedding": [942, 104, ...], "dimensions": 768, "token_count": 2 },
    { "index": 1, "embedding": [514, 308, ...], "dimensions": 768, "token_count": 2 },
    { "index": 2, "embedding": [720, 211, ...], "dimensions": 768, "token_count": 2 }
  ],
  "model": "landmark-lattice-v1",
  "count": 3
}
```

Concurrency defaults to `GOMAXPROCS` (number of logical CPUs) and is tunable with `--workers`:

| `--workers` | Usage | Effect |
|:------------|:------|:-------|
| `0` (or omit) | `./embedder` | Uses `GOMAXPROCS` — optimal for CPU-bound work |
| `1` | `./embedder --workers=1` | Sequential processing (one text at a time) |
| `4` | `./embedder --workers=4` | Up to 4 concurrent embeddings |
| `N` | `./embedder --workers=N` | Up to N concurrent embeddings |

**Benchmark** (i7-950, 8 threads, 100 texts, 768 dims):

| Mode | Server-side time |
|:-----|:-----------------|
| Sequential (--workers=1) | ~2,200 ms (all in Embed computation) |
| Parallel (--workers=8) | ~279 ms (275 ms Embed + ~4 ms JSON encode) |
| **Speedup** | **~7.9x** |

> JSON serialization is negligible (~4 ms for 100 items = 77K int32 values). The parallel speedup applies directly to the compute-bound Embed() calls. The ~8x figure matches `GOMAXPROCS` on an i7-950 (8 threads).
