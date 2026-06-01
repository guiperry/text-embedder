package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	api "github.com/guiperry/text-embedder/internal/api"
)

func TestEmbedHandler(t *testing.T) {
	body := `{"text":"hello world"}`
	r := httptest.NewRequest(http.MethodPost, "/embed", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Fatalf("expected CORS origin=*, got %s", origin)
	}
	var resp api.EmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Embedding) != 768 {
		t.Fatalf("expected 768-dim embedding, got %d", len(resp.Embedding))
	}
}

func TestEmbedHandler_OPTIONS(t *testing.T) {
	r := httptest.NewRequest(http.MethodOptions, "/embed", nil)
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusNoContent {
		t.Fatalf("expected 204 for OPTIONS, got %d", w.Code)
	}
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Fatalf("expected CORS origin=*, got %s", origin)
	}
}

func TestEmbedHandler_WrongMethod(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/embed", nil)
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestEmbedHandler_BadJSON(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/embed", strings.NewReader(`{bad`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}
