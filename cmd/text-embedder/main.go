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
//
// This binary is also Vercel-deployable via the api/ directory — see vercel.json.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	api "github.com/guiperry/text-embedder/pkg/api"
	"github.com/guiperry/text-embedder/pkg/embed"
)

func main() {
	addr := flag.String("addr", ":8089", "listen address")
	workers := flag.Int("workers", runtime.GOMAXPROCS(0), "max concurrent batch workers (0 = GOMAXPROCS)")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	mux := http.NewServeMux()
	mux.HandleFunc("/embed", api.HandleEmbed)
	mux.HandleFunc("/embed/batch", func(w http.ResponseWriter, r *http.Request) {
		api.HandleBatch(w, r, api.ResolveWorkers(*workers))
	})
	mux.HandleFunc("/similarity", api.HandleSimilarity)
	mux.HandleFunc("/health", api.HandleHealth)

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
			"batch_workers", api.ResolveWorkers(*workers),
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
