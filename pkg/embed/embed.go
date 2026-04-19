// Package embed implements deterministic text embeddings using feature hashing.
//
// Algorithm:
//  1. Tokenize text into word unigrams, word bigrams, and character trigrams
//  2. Apply FNV-1a hash to map each feature to a dimension in [0, Dims)
//  3. Use a secondary hash to assign a sign (+1/-1), preventing cancellation
//  4. Weight dimensions by term frequency (TF) of each feature
//  5. L2-normalise the result to a unit vector
//
// The same input always produces the same output (no randomness, no state).
package embed

import (
	"hash/fnv"
	"math"
	"regexp"
	"strings"
	"unicode"
)

const (
	// Dims is the number of dimensions in the output embedding vector.
	// 768 matches popular sentence-transformer models for compatibility.
	Dims = 768

	// ModelID is a stable identifier for this embedding algorithm version.
	ModelID = "hash-ngram-v1"
)

// nonAlphanumRE matches runs of characters that are not letters or digits.
var nonAlphanumRE = regexp.MustCompile(`[^a-z0-9]+`)

// Tokenize returns the tokens that would be embedded for text.
// The count reflects exactly what the embedding algorithm operates on.
func Tokenize(text string) []string {
	return tokenise(text)
}

// Embed converts a text string into a deterministic unit-normalised vector.
// The same input always returns the exact same vector.
func Embed(text string) []float32 {
	features := extractFeatures(text)
	if len(features) == 0 {
		return make([]float32, Dims) // zero vector for empty input
	}

	// Count term frequencies
	tf := make(map[string]float64, len(features))
	for _, f := range features {
		tf[f]++
	}

	// Accumulate into fixed-size vector using the hashing trick
	vec := make([]float64, Dims)
	for feat, count := range tf {
		dim, sign := hashFeature(feat)
		vec[dim] += sign * count
	}

	// L2-normalise to unit length
	normalise(vec)

	// Convert to float32 for compact JSON output
	out := make([]float32, Dims)
	for i, v := range vec {
		out[i] = float32(v)
	}
	return out
}

// CosineSimilarity returns the cosine similarity in [-1, 1] between two unit
// vectors. Both inputs must already be L2-normalised (as returned by Embed).
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	// Clamp to [-1,1] to guard against floating-point drift
	if dot > 1 {
		dot = 1
	} else if dot < -1 {
		dot = -1
	}
	return dot
}

// ---------- internals --------------------------------------------------------

// extractFeatures returns all features for a text:
//   - word unigrams  (whole lowercase tokens)
//   - word bigrams   (adjacent token pairs, prefix "2:")
//   - char 3-grams   (sliding windows over each token, prefix "c:")
func extractFeatures(text string) []string {
	tokens := tokenise(text)
	if len(tokens) == 0 {
		return nil
	}

	features := make([]string, 0, len(tokens)*6)

	// Word unigrams
	for _, t := range tokens {
		features = append(features, t)
	}

	// Word bigrams
	for i := 0; i < len(tokens)-1; i++ {
		features = append(features, "2:"+tokens[i]+"_"+tokens[i+1])
	}

	// Character trigrams (subword coverage for morphology & typos)
	for _, t := range tokens {
		padded := "\x02" + t + "\x03" // BOS / EOS markers
		for i := 0; i <= len(padded)-3; i++ {
			features = append(features, "c:"+padded[i:i+3])
		}
	}

	return features
}

// tokenise lower-cases the input, splits on non-alphanumeric runs,
// removes pure-numeric tokens, and returns only non-empty tokens.
func tokenise(text string) []string {
	lower := strings.Map(func(r rune) rune {
		if unicode.IsUpper(r) {
			return unicode.ToLower(r)
		}
		return r
	}, text)

	raw := nonAlphanumRE.Split(lower, -1)
	tokens := raw[:0]
	for _, tok := range raw {
		if tok == "" {
			continue
		}
		// Keep tokens that have at least one letter (skip pure numbers)
		hasLetter := false
		for _, r := range tok {
			if unicode.IsLetter(r) {
				hasLetter = true
				break
			}
		}
		if hasLetter {
			tokens = append(tokens, tok)
		}
	}
	return tokens
}

// hashFeature returns (dimension, sign) for a feature string.
//
// Primary hash  → dimension index in [0, Dims)
// Secondary hash → sign in {-1, +1}   prevents opposite features cancelling
func hashFeature(feat string) (int, float64) {
	// Primary: FNV-1a 64-bit
	h1 := fnv.New64a()
	h1.Write([]byte(feat))
	dim := int(h1.Sum64() % uint64(Dims))

	// Secondary: FNV-1a over reversed bytes to get an independent bit
	b := []byte(feat)
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	h2 := fnv.New64a()
	h2.Write(b)
	sign := 1.0
	if h2.Sum64()%2 == 0 {
		sign = -1.0
	}

	return dim, sign
}

// normalise divides vec by its L2 norm in-place. If the norm is zero the
// vector is left unchanged (all-zero input → all-zero output).
func normalise(vec []float64) {
	var sumSq float64
	for _, v := range vec {
		sumSq += v * v
	}
	if sumSq == 0 {
		return
	}
	norm := math.Sqrt(sumSq)
	for i := range vec {
		vec[i] /= norm
	}
}
