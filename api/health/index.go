package handler

import (
	"net/http"

	api "github.com/guiperry/text-embedder/internal/api"
)

func Handler(w http.ResponseWriter, r *http.Request) {
	api.SetCORSHeaders(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	api.HandleHealth(w, r)
}
