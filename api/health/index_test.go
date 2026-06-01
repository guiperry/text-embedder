package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	api "github.com/guiperry/text-embedder/internal/api"
)

func TestHealthHandler(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Fatalf("expected CORS origin=*, got %s", origin)
	}
	var resp api.HealthResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Status != "ok" {
		t.Fatalf("expected status=ok, got %s", resp.Status)
	}
}

func TestHealthHandler_OPTIONS(t *testing.T) {
	r := httptest.NewRequest(http.MethodOptions, "/health", nil)
	w := httptest.NewRecorder()
	Handler(w, r)
	if w.Code != http.StatusNoContent {
		t.Fatalf("expected 204 for OPTIONS, got %d", w.Code)
	}
}
