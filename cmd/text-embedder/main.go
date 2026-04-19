// text-embedder: a local HTTP API server that converts text to deterministic
// embedding vectors using feature hashing (no external dependencies, no ML models).
//
// Endpoints:
//
//	POST /embed           – embed a single text
//	POST /embed/batch     – embed multiple texts
//	POST /similarity      – cosine similarity between two texts
//	GET  /health          – liveness check
//
// Usage:
//
//	go run . [--addr :8080]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/guiperry/text-embedder/pkg/embed"
)

// ---- request / response types -----------------------------------------------

type embedRequest struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
	Model      string    `json:"model"`
	TokenCount int       `json:"token_count"`
}

type batchRequest struct {
	Texts []string `json:"texts"`
}

type batchItem struct {
	Index      int       `json:"index"`
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
	TokenCount int       `json:"token_count"`
}

type batchResponse struct {
	Results []batchItem `json:"results"`
	Model   string      `json:"model"`
	Count   int         `json:"count"`
}

type similarityRequest struct {
	TextA string `json:"text_a"`
	TextB string `json:"text_b"`
}

type similarityResponse struct {
	Similarity float64 `json:"similarity"`
	Model      string  `json:"model"`
}

type healthResponse struct {
	Status    string `json:"status"`
	Model     string `json:"model"`
	Dims      int    `json:"dims"`
	Timestamp string `json:"timestamp"`
}

type errorResponse struct {
	Error string `json:"error"`
	Code  int    `json:"code"`
}

// ---- helpers ----------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("json encode failed", "err", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg, Code: status})
}

func decode(w http.ResponseWriter, r *http.Request, dst any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

// ---- handlers ---------------------------------------------------------------

func handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req embedRequest
	if !decode(w, r, &req) {
		return
	}
	vec := embed.Embed(req.Text)
	writeJSON(w, http.StatusOK, embedResponse{
		Embedding:  vec,
		Dimensions: embed.Dims,
		Model:      embed.ModelID,
		TokenCount: len(embed.Tokenize(req.Text)),
	})
}

func handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req batchRequest
	if !decode(w, r, &req) {
		return
	}
	if len(req.Texts) == 0 {
		writeError(w, http.StatusBadRequest, `"texts" array must not be empty`)
		return
	}
	if len(req.Texts) > 256 {
		writeError(w, http.StatusBadRequest, "batch size exceeds limit of 256")
		return
	}
	results := make([]batchItem, len(req.Texts))
	for i, text := range req.Texts {
		results[i] = batchItem{
			Index:      i,
			Embedding:  embed.Embed(text),
			Dimensions: embed.Dims,
			TokenCount: len(embed.Tokenize(text)),
		}
	}
	writeJSON(w, http.StatusOK, batchResponse{
		Results: results,
		Model:   embed.ModelID,
		Count:   len(results),
	})
}

func handleSimilarity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req similarityRequest
	if !decode(w, r, &req) {
		return
	}
	vA := embed.Embed(req.TextA)
	vB := embed.Embed(req.TextB)
	sim := embed.CosineSimilarity(vA, vB)
	writeJSON(w, http.StatusOK, similarityResponse{
		Similarity: sim,
		Model:      embed.ModelID,
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{
		Status:    "ok",
		Model:     embed.ModelID,
		Dims:      embed.Dims,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	})
}

// ---- logging middleware ------------------------------------------------------

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(status int) {
	rw.status = status
	rw.ResponseWriter.WriteHeader(status)
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.status,
			"duration", time.Since(start).String(),
			"remote", r.RemoteAddr,
		)
	})
}

// ---- main -------------------------------------------------------------------

func main() {
	addr := flag.String("addr", ":8089", "listen address")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	mux := http.NewServeMux()
	mux.HandleFunc("/embed", handleEmbed)
	mux.HandleFunc("/embed/batch", handleBatch)
	mux.HandleFunc("/similarity", handleSimilarity)
	mux.HandleFunc("/health", handleHealth)

	server := &http.Server{
		Addr:         *addr,
		Handler:      loggingMiddleware(mux),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		slog.Info("server starting",
			"addr", *addr,
			"model", embed.ModelID,
			"dims", embed.Dims,
		)
		fmt.Printf("\ntext-embedder running on http://localhost%s\n\n", *addr)
		fmt.Println("  POST /embed           – embed a single text")
		fmt.Println("  POST /embed/batch     – embed up to 256 texts")
		fmt.Println("  POST /similarity      – cosine similarity of two texts")
		fmt.Println("  GET  /health          – liveness check")
		fmt.Println()
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down…")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		slog.Error("shutdown error", "err", err)
	}
	slog.Info("server stopped")
}
