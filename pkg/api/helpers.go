// Package api provides shared request/response types, HTTP helpers, and
// handler functions used by both the local binary and Vercel serverless
// deployment. All handler logic lives here — the local binary and Vercel
// wrappers are thin registrations on top.
package api

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/guiperry/text-embedder/pkg/embed"
)

const (
	MaxBatchSize = 256
	MaxBodySize  = 1 << 20
)

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Embedding  []int32 `json:"embedding"`
	Dimensions int     `json:"dimensions"`
	Model      string  `json:"model"`
	TokenCount int     `json:"token_count"`
}

type BatchRequest struct {
	Texts []string `json:"texts"`
}

type BatchItem struct {
	Index      int     `json:"index"`
	Embedding  []int32 `json:"embedding"`
	Dimensions int     `json:"dimensions"`
	TokenCount int     `json:"token_count"`
}

type BatchResponse struct {
	Results []BatchItem `json:"results"`
	Model   string      `json:"model"`
	Count   int         `json:"count"`
}

type SimilarityRequest struct {
	TextA string `json:"text_a"`
	TextB string `json:"text_b"`
}

type SimilarityResponse struct {
	Similarity float64 `json:"similarity"`
	Model      string  `json:"model"`
}

type HealthResponse struct {
	Status    string `json:"status"`
	Model     string `json:"model"`
	Dims      int    `json:"dims"`
	Timestamp string `json:"timestamp"`
}

type ErrorResponse struct {
	Error string `json:"error"`
	Code  int    `json:"code"`
}

func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("json encode failed", "err", err)
	}
}

func WriteError(w http.ResponseWriter, status int, msg string) {
	WriteJSON(w, status, ErrorResponse{Error: msg, Code: status})
}

func Decode(w http.ResponseWriter, r *http.Request, dst any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, MaxBodySize)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

func SetCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func ResolveWorkers(flagVal int) int {
	if flagVal > 0 {
		return flagVal
	}
	if s := os.Getenv("EMBED_BATCH_WORKERS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return runtime.GOMAXPROCS(0)
}

func HandleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req EmbedRequest
	if !Decode(w, r, &req) {
		return
	}
	vec := embed.Embed(req.Text)
	WriteJSON(w, http.StatusOK, EmbedResponse{
		Embedding:  vec,
		Dimensions: embed.Dims,
		Model:      embed.ModelID,
		TokenCount: len(embed.Tokenize(req.Text)),
	})
}

func HandleBatch(w http.ResponseWriter, r *http.Request, workers int) {
	if r.Method != http.MethodPost {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req BatchRequest
	if !Decode(w, r, &req) {
		return
	}
	if len(req.Texts) == 0 {
		WriteError(w, http.StatusBadRequest, `"texts" array must not be empty`)
		return
	}
	if len(req.Texts) > MaxBatchSize {
		WriteError(w, http.StatusBadRequest, "batch size exceeds limit of 256")
		return
	}

	results := make([]BatchItem, len(req.Texts))
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)

	for i, text := range req.Texts {
		wg.Add(1)
		go func(idx int, txt string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			results[idx] = BatchItem{
				Index:      idx,
				Embedding:  embed.Embed(txt),
				Dimensions: embed.Dims,
				TokenCount: len(embed.Tokenize(txt)),
			}
		}(i, text)
	}

	wg.Wait()
	WriteJSON(w, http.StatusOK, BatchResponse{
		Results: results,
		Model:   embed.ModelID,
		Count:   len(results),
	})
}

func HandleSimilarity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req SimilarityRequest
	if !Decode(w, r, &req) {
		return
	}
	vA := embed.Embed(req.TextA)
	vB := embed.Embed(req.TextB)
	sim := embed.CosineSimilarity(vA, vB)
	WriteJSON(w, http.StatusOK, SimilarityResponse{
		Similarity: sim,
		Model:      embed.ModelID,
	})
}

func HandleHealth(w http.ResponseWriter, r *http.Request) {
	WriteJSON(w, http.StatusOK, HealthResponse{
		Status:    "ok",
		Model:     embed.ModelID,
		Dims:      embed.Dims,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	})
}
