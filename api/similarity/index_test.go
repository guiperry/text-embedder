package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	api "github.com/guiperry/text-embedder/internal/api"
)

func TestSimilarityHandler(t *testing.T) {
	body := `{"text_a":"machine learning","text_b":"deep learning"}`
	r := httptest.NewRequest(http.MethodPost, "/similarity", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Fatalf("expected CORS origin=*, got %s", origin)
	}
	var resp api.SimilarityResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Similarity <= 0 {
		t.Fatalf("expected positive similarity, got %f", resp.Similarity)
	}
}

func TestSimilarityHandler_OPTIONS(t *testing.T) {
	r := httptest.NewRequest(http.MethodOptions, "/similarity", nil)
	w := httptest.NewRecorder()
	Handler(w, r)
	if w.Code != http.StatusNoContent {
		t.Fatalf("expected 204 for OPTIONS, got %d", w.Code)
	}
}
