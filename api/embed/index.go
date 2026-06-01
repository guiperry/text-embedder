package handler

import (
	"net/http"

	api "github.com/guiperry/text-embedder/internal/api"
	"github.com/guiperry/text-embedder/pkg/embed"
)

func Handler(w http.ResponseWriter, r *http.Request) {
	api.SetCORSHeaders(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodPost {
		api.WriteError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	var req api.EmbedRequest
	if !api.Decode(w, r, &req) {
		return
	}
	vec := embed.Embed(req.Text)
	api.WriteJSON(w, http.StatusOK, api.EmbedResponse{
		Embedding:  vec,
		Dimensions: embed.Dims,
		Model:      embed.ModelID,
		TokenCount: len(embed.Tokenize(req.Text)),
	})
}
