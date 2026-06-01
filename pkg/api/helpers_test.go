package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestWriteJSON(t *testing.T) {
	w := httptest.NewRecorder()
	WriteJSON(w, http.StatusOK, map[string]string{"msg": "hello"})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("expected application/json, got %s", ct)
	}
	var m map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &m); err != nil {
		t.Fatal(err)
	}
	if m["msg"] != "hello" {
		t.Fatalf("expected msg=hello, got %s", m["msg"])
	}
}

func TestWriteError(t *testing.T) {
	w := httptest.NewRecorder()
	WriteError(w, http.StatusBadRequest, "bad stuff")
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
	var e ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &e); err != nil {
		t.Fatal(err)
	}
	if e.Error != "bad stuff" {
		t.Fatalf("expected error message, got %s", e.Error)
	}
	if e.Code != http.StatusBadRequest {
		t.Fatalf("expected code %d, got %d", http.StatusBadRequest, e.Code)
	}
}

func TestDecode_Valid(t *testing.T) {
	body := `{"text":"hello world"}`
	r := httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	var req EmbedRequest
	if !Decode(w, r, &req) {
		t.Fatal("Decode returned false for valid input")
	}
	if req.Text != "hello world" {
		t.Fatalf("expected text=hello world, got %s", req.Text)
	}
}

func TestDecode_InvalidJSON(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{bad`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	var req EmbedRequest
	if Decode(w, r, &req) {
		t.Fatal("Decode should return false for invalid JSON")
	}
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestDecode_UnknownFields(t *testing.T) {
	body := `{"text":"hi","extra":"field"}`
	r := httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	var req EmbedRequest
	if Decode(w, r, &req) {
		t.Fatal("Decode should reject unknown fields")
	}
}

func TestDecode_OversizedBody(t *testing.T) {
	big := make([]byte, MaxBodySize+1024)
	for i := range big {
		big[i] = 'a'
	}
	body := `{"text":"` + string(big) + `"}`
	r := httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	var req EmbedRequest
	if Decode(w, r, &req) {
		t.Fatal("Decode should reject oversized body")
	}
}

func TestSetCORSHeaders(t *testing.T) {
	w := httptest.NewRecorder()
	SetCORSHeaders(w)

	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Fatalf("expected *, got %s", origin)
	}
	if methods := w.Header().Get("Access-Control-Allow-Methods"); methods == "" {
		t.Fatal("expected CORS methods header")
	}
}

func TestResolveWorkers_FlagWins(t *testing.T) {
	n := ResolveWorkers(8)
	if n != 8 {
		t.Fatalf("expected 8, got %d", n)
	}
}

func TestResolveWorkers_EnvVar(t *testing.T) {
	os.Setenv("EMBED_BATCH_WORKERS", "6")
	defer os.Unsetenv("EMBED_BATCH_WORKERS")

	n := ResolveWorkers(0)
	if n != 6 {
		t.Fatalf("expected 6, got %d", n)
	}
}

func TestResolveWorkers_EnvVarFlagWins(t *testing.T) {
	os.Setenv("EMBED_BATCH_WORKERS", "6")
	defer os.Unsetenv("EMBED_BATCH_WORKERS")

	n := ResolveWorkers(4)
	if n != 4 {
		t.Fatalf("expected flag 4 to win over env var, got %d", n)
	}
}

func TestResolveWorkers_InvalidEnvVar(t *testing.T) {
	os.Setenv("EMBED_BATCH_WORKERS", "not-a-number")
	defer os.Unsetenv("EMBED_BATCH_WORKERS")

	n := ResolveWorkers(0)
	expected := runtime.GOMAXPROCS(0)
	if n != expected {
		t.Fatalf("expected GOMAXPROCS=%d, got %d", expected, n)
	}
}

func TestResolveWorkers_ZeroEnvVar(t *testing.T) {
	os.Setenv("EMBED_BATCH_WORKERS", "0")
	defer os.Unsetenv("EMBED_BATCH_WORKERS")

	n := ResolveWorkers(0)
	expected := runtime.GOMAXPROCS(0)
	if n != expected {
		t.Fatalf("expected GOMAXPROCS=%d, got %d", expected, n)
	}
}

func TestHandleEmbed_Success(t *testing.T) {
	body := `{"text":"hello world"}`
	r := httptest.NewRequest(http.MethodPost, "/embed", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	HandleEmbed(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp EmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Dimensions != 768 {
		t.Fatalf("expected 768 dims, got %d", resp.Dimensions)
	}
	if resp.Model == "" {
		t.Fatal("model should not be empty")
	}
	if len(resp.Embedding) != 768 {
		t.Fatalf("expected 768-dim embedding, got %d", len(resp.Embedding))
	}
}

func TestHandleEmbed_WrongMethod(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/embed", nil)
	w := httptest.NewRecorder()
	HandleEmbed(w, r)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleEmbed_BadJSON(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/embed", bytes.NewBufferString(`{bad`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleEmbed(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandleEmbed_Deterministic(t *testing.T) {
	r1 := httptest.NewRequest(http.MethodPost, "/embed",
		bytes.NewBufferString(`{"text":"test determinism"}`))
	r1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	HandleEmbed(w1, r1)

	r2 := httptest.NewRequest(http.MethodPost, "/embed",
		bytes.NewBufferString(`{"text":"test determinism"}`))
	r2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	HandleEmbed(w2, r2)

	var resp1, resp2 EmbedResponse
	json.Unmarshal(w1.Body.Bytes(), &resp1)
	json.Unmarshal(w2.Body.Bytes(), &resp2)

	for i := range resp1.Embedding {
		if resp1.Embedding[i] != resp2.Embedding[i] {
			t.Fatalf("dim %d differs: %d != %d", i, resp1.Embedding[i], resp2.Embedding[i])
		}
	}
}

func TestHandleEmbed_EmptyText(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/embed",
		bytes.NewBufferString(`{"text":""}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleEmbed(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for empty text, got %d", w.Code)
	}
	var resp EmbedResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.TokenCount != 0 {
		t.Fatalf("expected 0 tokens for empty text, got %d", resp.TokenCount)
	}
}

func TestHandleBatch_Success(t *testing.T) {
	body := `{"texts":["hello","world"]}`
	r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	HandleBatch(w, r, 4)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp BatchResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Count != 2 {
		t.Fatalf("expected count=2, got %d", resp.Count)
	}
	for i, item := range resp.Results {
		if item.Index != i {
			t.Fatalf("expected index=%d, got %d", i, item.Index)
		}
		if item.Dimensions != 768 {
			t.Fatalf("expected 768 dims, got %d", item.Dimensions)
		}
	}
}

func TestHandleBatch_Empty(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/embed/batch",
		strings.NewReader(`{"texts":[]}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleBatch(w, r, 4)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty texts, got %d", w.Code)
	}
}

func TestHandleBatch_ExceedsLimit(t *testing.T) {
	texts := make([]string, 257)
	for i := range texts {
		texts[i] = "x"
	}
	body, _ := json.Marshal(BatchRequest{Texts: texts})
	r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleBatch(w, r, 4)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for >256 texts, got %d", w.Code)
	}
}

func TestHandleBatch_WrongMethod(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/embed/batch", nil)
	w := httptest.NewRecorder()
	HandleBatch(w, r, 4)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleBatch_OrderPreserved(t *testing.T) {
	body := `{"texts":["alpha","beta","gamma","delta"]}`
	r := httptest.NewRequest(http.MethodPost, "/embed/batch", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleBatch(w, r, 2)

	var resp BatchResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	for i, item := range resp.Results {
		if item.Index != i {
			t.Fatalf("order violation at %d: index=%d", i, item.Index)
		}
	}
}

func TestHandleSimilarity_Success(t *testing.T) {
	body := `{"text_a":"machine learning","text_b":"deep learning"}`
	r := httptest.NewRequest(http.MethodPost, "/similarity", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleSimilarity(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp SimilarityResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Similarity <= 0 {
		t.Fatalf("expected positive similarity, got %f", resp.Similarity)
	}
}

func TestHandleSimilarity_Identical(t *testing.T) {
	body := `{"text_a":"same text","text_b":"same text"}`
	r := httptest.NewRequest(http.MethodPost, "/similarity", bytes.NewBufferString(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	HandleSimilarity(w, r)

	var resp SimilarityResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Similarity < 0.9999 {
		t.Fatalf("expected near-1 similarity for identical texts, got %f", resp.Similarity)
	}
}

func TestHandleSimilarity_WrongMethod(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/similarity", nil)
	w := httptest.NewRecorder()
	HandleSimilarity(w, r)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleHealth_Success(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/health", nil)
	HandleHealth(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp HealthResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Status != "ok" {
		t.Fatalf("expected status=ok, got %s", resp.Status)
	}
	if resp.Dims != 768 {
		t.Fatalf("expected dims=768, got %d", resp.Dims)
	}
	if resp.Model == "" {
		t.Fatal("model should not be empty")
	}
	if resp.Timestamp == "" {
		t.Fatal("timestamp should not be empty")
	}
}

func TestHealthResponse_ImplementsWriteJSON(t *testing.T) {
	w := httptest.NewRecorder()
	WriteJSON(w, http.StatusOK, HealthResponse{
		Status: "ok",
		Model:  "test",
		Dims:   768,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}
