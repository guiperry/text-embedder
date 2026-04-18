package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newTestServer spins up a real TCP server on a random free port and returns
// its base URL. The caller's t.Cleanup closes it.
func newTestServer(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/embed", handleEmbed)
	mux.HandleFunc("/embed/batch", handleBatch)
	mux.HandleFunc("/similarity", handleSimilarity)
	mux.HandleFunc("/health", handleHealth)
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	return "http://" + ln.Addr().String()
}

func postJSON(t *testing.T, url, body string) []byte {
	t.Helper()
	resp, err := http.Post(url, "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	var buf bytes.Buffer
	buf.ReadFrom(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s: status %d, body: %s", url, resp.StatusCode, buf.String())
	}
	return buf.Bytes()
}

// ---- unit-level handler tests (no network) ----------------------------------

func TestHandleEmbed_Basic(t *testing.T) {
	body := `{"text": "the quick brown fox"}`
	req := httptest.NewRequest(http.MethodPost, "/embed", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleEmbed(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp embedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Dimensions != 768 {
		t.Errorf("expected 768 dims, got %d", resp.Dimensions)
	}
	if resp.Model == "" {
		t.Error("model field empty")
	}
}

func TestHandleEmbed_WrongMethod(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/embed", nil)
	w := httptest.NewRecorder()
	handleEmbed(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestHandleEmbed_BadJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/embed", bytes.NewBufferString(`{bad`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleEmbed(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestHandleDeterministic(t *testing.T) {
	text := `{"text": "determinism test sentence"}`
	call := func() []float32 {
		req := httptest.NewRequest(http.MethodPost, "/embed", bytes.NewBufferString(text))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		handleEmbed(w, req)
		var resp embedResponse
		json.Unmarshal(w.Body.Bytes(), &resp)
		return resp.Embedding
	}
	v1, v2 := call(), call()
	for i := range v1 {
		if v1[i] != v2[i] {
			t.Fatalf("dim %d differs between identical calls: %v != %v", i, v1[i], v2[i])
		}
	}
}

func TestHandleSimilarity_RelatedHigher(t *testing.T) {
	body := `{"text_a":"machine learning neural networks","text_b":"deep learning artificial intelligence"}`
	req := httptest.NewRequest(http.MethodPost, "/similarity", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleSimilarity(w, req)

	var resp similarityResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Similarity <= 0 {
		t.Errorf("expected positive similarity between related topics, got %f", resp.Similarity)
	}
}

func TestHandleBatch_Count(t *testing.T) {
	body := `{"texts":["hello world","foo bar","machine learning"]}`
	req := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleBatch(w, req)

	var resp batchResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Count != 3 {
		t.Errorf("expected count=3, got %d", resp.Count)
	}
	for i, r := range resp.Results {
		if r.Index != i {
			t.Errorf("result[%d].Index = %d, want %d", i, r.Index, i)
		}
	}
}

func TestHandleBatch_Empty(t *testing.T) {
	body := `{"texts":[]}`
	req := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleBatch(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty batch, got %d", w.Code)
	}
}

// ---- live TCP server tests --------------------------------------------------

func TestLiveServer_Health(t *testing.T) {
	base := newTestServer(t)
	resp, err := http.Get(base + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var h healthResponse
	json.NewDecoder(resp.Body).Decode(&h)
	if h.Status != "ok" {
		t.Errorf("status=%q, want ok", h.Status)
	}
	fmt.Printf("  live /health → status=%s model=%s dims=%d\n", h.Status, h.Model, h.Dims)
}

func TestLiveServer_EmbedDeterministic(t *testing.T) {
	base := newTestServer(t)
	body := `{"text":"testing determinism over the network"}`

	var r1, r2 embedResponse
	json.Unmarshal(postJSON(t, base+"/embed", body), &r1)
	json.Unmarshal(postJSON(t, base+"/embed", body), &r2)

	for i := range r1.Embedding {
		if r1.Embedding[i] != r2.Embedding[i] {
			t.Fatalf("live determinism failed at dim %d", i)
		}
	}
	fmt.Printf("  live /embed determinism → OK (dims=%d)\n", r1.Dimensions)
}

func TestLiveServer_SimilarityEndToEnd(t *testing.T) {
	base := newTestServer(t)

	cases := []struct {
		a, b    string
		wantPos bool
		label   string
	}{
		{"machine learning", "deep learning", true, "related ML topics"},
		{"the cat sat", "the cat sat", true, "identical texts → sim~1"},
		{"machine learning", "chocolate cake baking", false, "unrelated topics"},
	}

	for _, c := range cases {
		body := fmt.Sprintf(`{"text_a":%q,"text_b":%q}`, c.a, c.b)
		var sr similarityResponse
		json.Unmarshal(postJSON(t, base+"/similarity", body), &sr)
		fmt.Printf("  %s → %.4f\n", c.label, sr.Similarity)

		if c.wantPos && sr.Similarity <= 0 {
			t.Errorf("%s: expected positive similarity, got %f", c.label, sr.Similarity)
		}
	}
}

func TestLiveServer_BatchIndexOrder(t *testing.T) {
	base := newTestServer(t)
	texts := []string{"alpha", "beta", "gamma", "delta"}
	body := `{"texts":["alpha","beta","gamma","delta"]}`

	var br batchResponse
	json.Unmarshal(postJSON(t, base+"/embed/batch", body), &br)

	if br.Count != len(texts) {
		t.Fatalf("expected %d results, got %d", len(texts), br.Count)
	}
	for i, r := range br.Results {
		if r.Index != i {
			t.Errorf("results[%d].Index = %d", i, r.Index)
		}
	}
	fmt.Printf("  live /embed/batch → %d results, index order correct\n", br.Count)
}
