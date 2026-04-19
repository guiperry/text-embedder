# text-embedder

A zero-dependency Go HTTP server that converts text into deterministic embedding
vectors. No ML models, no GPU, no internet access required — the same input always
produces the exact same vector.

## Quick start

```bash
go build -o embedder ./cmd/text-embedder
./embedder              # listens on :8089 by default
./embedder --addr :9000 # custom port
```

Or with `make`:

```bash
make run          # build + run on :8089
make run ADDR=:9000
make test         # run all tests
make bench        # run benchmarks
```

---

## Using as a Go module

Import the `embed` package directly into your own Go project — no HTTP server
needed.

```bash
go get github.com/guiperry/text-embedder
```

```go
import "github.com/guiperry/text-embedder/pkg/embed"
```

### Embed a single text

```go
vec := embed.Embed("machine learning is fascinating")
// vec is []float32 with embed.Dims (768) elements, L2-normalised to unit length
```

### Compare two texts

```go
a := embed.Embed("neural networks")
b := embed.Embed("deep learning")

sim := embed.CosineSimilarity(a, b) // float64 in [-1, 1]
```

`CosineSimilarity` expects both inputs to already be unit vectors (as returned by
`Embed`). A value of `1.0` means identical, `0.0` means unrelated, `-1.0` means
maximally opposite.

### Constants

```go
embed.Dims    // 768  — length of every embedding vector
embed.ModelID // "hash-ngram-v1" — stable algorithm identifier
```

Both values are safe to use in persistent storage schemas: the algorithm is
versioned and will not change under this identifier.

---

## HTTP API

All endpoints accept and return `application/json`.

---

### `GET /health`

Liveness check.

```bash
curl http://localhost:8089/health
```

```json
{
  "status": "ok",
  "model": "hash-ngram-v1",
  "dims": 768,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### `POST /embed`

Embed a single text string into a 768-dimensional unit vector.

```bash
curl -X POST http://localhost:8089/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "machine learning is fascinating"}'
```

```json
{
  "embedding": [0.042, -0.017, 0.093, "...768 values total..."],
  "dimensions": 768,
  "model": "hash-ngram-v1",
  "token_count": 4
}
```

`token_count` is a rough word count of the input (whitespace-delimited), useful
for logging and rate-limiting. It does not affect the embedding.

---

### `POST /embed/batch`

Embed up to 256 texts in one request. Results are returned in the same order
as the input; each item carries its original index.

```bash
curl -X POST http://localhost:8089/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "goodbye world", "machine learning"]}'
```

```json
{
  "results": [
    {"index": 0, "embedding": ["...768 values..."], "dimensions": 768, "token_count": 2},
    {"index": 1, "embedding": ["...768 values..."], "dimensions": 768, "token_count": 2},
    {"index": 2, "embedding": ["...768 values..."], "dimensions": 768, "token_count": 2}
  ],
  "model": "hash-ngram-v1",
  "count": 3
}
```

Sending more than 256 texts returns `400 Bad Request`.

---

### `POST /similarity`

Compute the cosine similarity between two texts in one round-trip.
Returns a value in `[-1, 1]` where `1` means identical and `0` means unrelated.

```bash
curl -X POST http://localhost:8089/similarity \
  -H "Content-Type: application/json" \
  -d '{"text_a": "neural networks", "text_b": "deep learning"}'
```

```json
{
  "similarity": 0.3421,
  "model": "hash-ngram-v1"
}
```

---

## Algorithm: `hash-ngram-v1`

The embedding is produced in four deterministic steps — no trained weights,
no external state, no randomness.

### 1. Tokenisation

The input is lowercased and split on any run of non-alphanumeric characters.
Pure-numeric tokens are discarded; only tokens containing at least one letter
are kept.

```
"The quick brown fox!" → ["the", "quick", "brown", "fox"]
```

### 2. Feature extraction

Three feature types are generated from the token list:

| Type | Example (from "quick") | Purpose |
|---|---|---|
| Word unigram | `quick` | Whole-word identity |
| Word bigram | `2:the_quick` | Local word order |
| Char trigram | `c:\x02qu`, `c:qui`, `c:uic`, `c:ick`, `c:ck\x03` | Subword morphology, typo tolerance |

`\x02` and `\x03` are BOS/EOS padding bytes that anchor trigrams at word edges.

### 3. Feature hashing (the hashing trick)

Each feature string is mapped to a *(dimension, sign)* pair:

```
dimension = FNV-1a-64(feature) mod 768
sign      = FNV-1a-64(reverse(feature)) mod 2  →  +1 or -1
```

The sign comes from an independent hash of the reversed bytes, which prevents
semantically opposite features from accidentally landing on the same dimension
with opposite signs and cancelling each other out.

Term frequency (how many times a feature appears) is accumulated at that
dimension:

```
vec[dimension] += sign × tf(feature)
```

### 4. L2 normalisation

The raw vector is divided by its Euclidean norm, producing a unit vector
(`‖v‖₂ = 1`). This makes cosine similarity equivalent to a dot product and
keeps all vectors on the same scale regardless of text length.

### Properties

- **Deterministic** — identical input → identical output, always.
- **Fast** — ~41 µs per embed on a single core; no model loading.
- **No dependencies** — uses only the Go standard library (`hash/fnv`, `math`).
- **Lexical similarity** — texts sharing words, word pairs, or character
  sequences will have higher cosine similarity. Not semantic (no synonym
  awareness), but useful for deduplication, clustering, and retrieval over
  domain-specific corpora.
- **Stable across versions** — the model ID `hash-ngram-v1` is a commitment:
  the same text will always produce the same vector for this version.

## Project layout

```
text-embedder/
├── cmd/
│   └── text-embedder/
│       ├── main.go                 # HTTP server, handlers, middleware
│       └── integration_test.go     # handler + live-server integration tests
├── pkg/
│   └── embed/
│       ├── embed.go                # tokeniser, feature extractor, hasher, normaliser
│       └── embed_test.go           # unit tests + benchmark
├── go.mod
└── Makefile
```

## Benchmark

```
BenchmarkEmbed-2   28803   41393 ns/op   18573 B/op   142 allocs/op
```

~41 µs per embedding on a 2.1 GHz Xeon core. For high-throughput use, run
multiple goroutines — `Embed` is stateless and safe for concurrent calls.
