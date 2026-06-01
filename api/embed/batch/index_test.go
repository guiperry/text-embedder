package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	api "github.com/guiperry/text-embedder/internal/api"
)

func TestBatchHandler(t *testing.T) {
	body := `{"texts":["hello","world"]}`
	r := httptest.NewRequest(http.MethodPost, "/embed/batch", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if origin := w.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Fatalf("expected CORS origin=*, got %s", origin)
	}
	var resp api.BatchResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Count != 2 {
		t.Fatalf("expected count=2, got %d", resp.Count)
	}
}

func TestBatchHandler_OPTIONS(t *testing.T) {
	r := httptest.NewRequest(http.MethodOptions, "/embed/batch", nil)
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusNoContent {
		t.Fatalf("expected 204 for OPTIONS, got %d", w.Code)
	}
}

func TestBatchHandler_WrongMethod(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/embed/batch", nil)
	w := httptest.NewRecorder()
	Handler(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}
