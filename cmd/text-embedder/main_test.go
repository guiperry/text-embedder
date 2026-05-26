// text-embedder benchmark: simulates provider calling patterns.
//
// The SocratiCode provider sends batches of up to 128 texts to
// /embed/batch. The server processes them in parallel using a bounded
// worker pool (default GOMAXPROCS workers). These benchmarks test
// the end-to-end handler pipeline with the same batch sizes the
// provider uses.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/guiperry/text-embedder/pkg/embed"
)

// makeTexts generates n unique texts for embedding.
func makeTexts(n int) []string {
	texts := make([]string, n)
	for i := range texts {
		texts[i] = fmt.Sprintf("The quick brown fox jumps over the lazy dog near the river bank %d", i)
	}
	return texts
}

// ---- Provider-like batch sizes -----------------------------------------

const providerBatchSize = 128 // matches TEXTEMBEDDER_BATCH_SIZE

// BenchmarkBatchHandler_providerBatchSize benchmarks /embed/batch with the
// exact batch size the SocratiCode provider sends (128 texts).
func BenchmarkBatchHandler_providerBatchSize(b *testing.B) {
	texts := makeTexts(providerBatchSize)
	body, err := json.Marshal(batchRequest{Texts: texts})
	if err != nil {
		b.Fatal(err)
	}

	// Use the default worker count (GOMAXPROCS) like the provider does
	batchWorkers = runtime.GOMAXPROCS(0)

	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
		handleBatch(w, r)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

// BenchmarkBatchHandler_smallBatch benchmarks the same call but with a
// smaller batch (16 texts) — useful for low-latency queries.
func BenchmarkBatchHandler_smallBatch(b *testing.B) {
	texts := makeTexts(16)
	body, err := json.Marshal(batchRequest{Texts: texts})
	if err != nil {
		b.Fatal(err)
	}
	batchWorkers = runtime.GOMAXPROCS(0)

	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
		handleBatch(w, r)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

// BenchmarkBatchHandler_maxBatch benchmarks the maximum batch size (256).
func BenchmarkBatchHandler_maxBatch(b *testing.B) {
	texts := makeTexts(256)
	body, err := json.Marshal(batchRequest{Texts: texts})
	if err != nil {
		b.Fatal(err)
	}
	batchWorkers = runtime.GOMAXPROCS(0)

	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
		handleBatch(w, r)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

// ---- Worker count comparison -------------------------------------------

func BenchmarkBatchHandler_workers_1(b *testing.B)   { benchmarkWithWorkers(b, 1, 128) }
func BenchmarkBatchHandler_workers_2(b *testing.B)   { benchmarkWithWorkers(b, 2, 128) }
func BenchmarkBatchHandler_workers_4(b *testing.B)   { benchmarkWithWorkers(b, 4, 128) }
func BenchmarkBatchHandler_workers_8(b *testing.B)   { benchmarkWithWorkers(b, 8, 128) }

func benchmarkWithWorkers(b *testing.B, workers, textCount int) {
	texts := makeTexts(textCount)
	body, err := json.Marshal(batchRequest{Texts: texts})
	if err != nil {
		b.Fatal(err)
	}
	batchWorkers = workers

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
		handleBatch(w, r)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

// ---- Order preservation test -------------------------------------------

// TestBatchOrder verifies that /embed/batch returns results in the same
// order as the input texts, even though they're processed in parallel.
func TestBatchOrder(t *testing.T) {
	batchWorkers = runtime.GOMAXPROCS(0)
	n := 137 // odd prime, not a multiple of workers — stresses ordering

	texts := makeTexts(n)
	body, err := json.Marshal(batchRequest{Texts: texts})
	if err != nil {
		t.Fatal(err)
	}

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
	handleBatch(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp batchResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Count != n {
		t.Fatalf("expected count %d, got %d", n, resp.Count)
	}
	for i, item := range resp.Results {
		if item.Index != i {
			t.Fatalf("order violation at position %d: expected index %d, got %d", i, i, item.Index)
		}
	}
}

// ---- JSON serialization benchmark (own) ---------------------------------

// BenchmarkBatchJSON benchmarks the JSON encode/decode of a full batch
// response — the remaining sequential bottleneck after parallel Embed.
func BenchmarkBatchJSON(b *testing.B) {
	for _, n := range []int{16, 64, 128, 256} {
		items := make([]batchItem, n)
		emb := make([]int32, 768)
		for i := range emb {
			emb[i] = int32((i * 997) % 10001)
		}
		for i := range items {
			items[i] = batchItem{
				Index:      i,
				Embedding:  emb,
				Dimensions: 768,
				TokenCount: 10,
			}
		}
		resp := batchResponse{
			Results: items,
			Model:   embed.ModelID,
			Count:   n,
		}
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				json.Marshal(resp)
			}
		})
	}
}

// ---- Concurrent callers (provider-like load) ----------------------------

// BenchmarkBatchHandler_concurrent simulates the provider's embed() method
// being called concurrently by multiple callers (e.g., indexing + search).
// Each caller sends a batch of 128 texts. The Go server handles concurrent
// HTTP requests in separate goroutines, while the internal worker pool
// bounds total parallelism.
func BenchmarkBatchHandler_concurrent(b *testing.B) {
	batchWorkers = runtime.GOMAXPROCS(0)

	body128, _ := json.Marshal(batchRequest{Texts: makeTexts(128)})

	for _, callers := range []int{1, 2, 4} {
		b.Run(fmt.Sprintf("callers=%d", callers), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				var wg sync.WaitGroup
				for c := 0; c < callers; c++ {
					wg.Add(1)
					go func() {
						defer wg.Done()
						w := httptest.NewRecorder()
						r := httptest.NewRequest(
							http.MethodPost, "/embed/batch",
							bytes.NewReader(body128),
						)
						handleBatch(w, r)
						if w.Code != http.StatusOK {
							b.Error("non-200 status")
						}
					}()
				}
				wg.Wait()
			}
		})
	}
}

// ---- Determinism: same inputs => same outputs ---------------------------

// TestBatchDeterminism verifies that embedding the same batch twice
// produces identical results (the core property of the algorithm).
func TestBatchDeterminism(t *testing.T) {
	batchWorkers = runtime.GOMAXPROCS(0)

	texts := []string{
		"the river bank",
		"the investment bank",
		"machine learning is fascinating",
		"hello world",
	}
	body, err := json.Marshal(batchRequest{Texts: texts})
	if err != nil {
		t.Fatal(err)
	}

	// First run
	w1 := httptest.NewRecorder()
	r1 := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
	handleBatch(w1, r1)

	// Second run
	w2 := httptest.NewRecorder()
	r2 := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
	handleBatch(w2, r2)

	var resp1, resp2 batchResponse
	json.NewDecoder(w1.Body).Decode(&resp1)
	json.NewDecoder(w2.Body).Decode(&resp2)

	for i := range resp1.Results {
		r1 := resp1.Results[i]
		r2 := resp2.Results[i]
		if len(r1.Embedding) != len(r2.Embedding) {
			t.Fatalf("item %d: embedding length mismatch", i)
		}
		for d := range r1.Embedding {
			if r1.Embedding[d] != r2.Embedding[d] {
				t.Fatalf("item %d dim %d: %d != %d", i, d, r1.Embedding[d], r2.Embedding[d])
			}
		}
	}
}

// ---- Minimum viable handler (batch of 1) ---------------------------------

func BenchmarkBatchHandler_single(b *testing.B) {
	body, _ := json.Marshal(batchRequest{Texts: []string{"hello world"}})
	batchWorkers = runtime.GOMAXPROCS(0)

	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
		handleBatch(w, r)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

// ---- Error handling tests -----------------------------------------------

func TestBatchHandler_EmptyInput(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/embed/batch",
		strings.NewReader(`{"texts":[]}`))
	handleBatch(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty texts, got %d", w.Code)
	}
}

func TestBatchHandler_ExceedsLimit(t *testing.T) {
	texts := make([]string, 257)
	for i := range texts {
		texts[i] = "x"
	}
	body, _ := json.Marshal(batchRequest{Texts: texts})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
	handleBatch(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for >256 texts, got %d", w.Code)
	}
}
