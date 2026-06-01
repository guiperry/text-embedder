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
	"sync"
	"unicode"
)

const (
	// Dims is the number of dimensions in the output embedding vector.
	Dims = 768

	// ModelID is a stable identifier for this embedding algorithm version.
	ModelID = "landmark-lattice-v1"

	// LatticeSize is the internal high-dimensional space for feature hashing.
	// Large enough to minimize collisions before landmark projection.
	LatticeSize = 4096

	// FixedPointScale is the multiplier for converting float similarities 
	// to integers [0, 10000].
	FixedPointScale = 10000
)

// functionalWords are structural anchors (prepositions, conjunctions, etc.)
// used for syntactic feature hashing. This set covers the majority of
// English relational words to provide robust context linking.
var functionalWords = map[string]bool{
	// Core Prepositions
	"of": true, "to": true, "in": true, "for": true, "on": true, "with": true,
	"at": true, "by": true, "from": true, "up": true, "about": true, "into": true,
	"over": true, "after": true, "through": true, "during": true, "before": true,
	"between": true, "under": true, "against": true, "among": true, "throughout": true,
	"despite": true, "towards": true, "upon": true, "concerning": true, "around": true,
	"near": true, "behind": true, "beyond": true, "within": true, "without": true,
	"until": true, "since": true, "past": true, "off": true, "across": true,
	"along": true, "above": true, "below": true, "down": true, "except": true,

	// Conjunctions (Coordinating & Subordinating)
	"and": true, "but": true, "or": true, "if": true, "because": true, "as": true,
	"while": true, "although": true, "though": true, "whereas": true,
	"unless": true, "whether": true, "yet": true, "so": true,
	"nor": true,

	// Relational & Deterministic Anchors
	"that": true, "which": true, "who": true, "whom": true, "whose": true,
	"this": true, "those": true, "these": true, "each": true, "every": true,
	"any": true, "all": true, "some": true, "no": true, "none": true,
}

// sync.Once cache for landmark vectors — computed once per process lifetime.
// Catastrophic for cold-start latency on batch requests without this: a batch of
// 256 texts would recompute 256 × 768 = 196,608 landmark vectorizations.
var (
	landmarkOnce    sync.Once
	landmarkVectors [][]int64
)

func cachedLandmarkVectors() [][]int64 {
	landmarkOnce.Do(func() {
		landmarkVectors = getLandmarkVectors()
	})
	return landmarkVectors
}

// Embed converts a text string into a deterministic similarity profile.
// The output is a []int32 representing similarity to fixed semantic landmarks.
func Embed(text string) []int32 {
	lat := embedToLattice(text)
	landmarks := cachedLandmarkVectors()

	// Calculate similarity to each landmark
	// To keep it 100% deterministic and jitter-free, we use integer dot products
	// and fixed-point division.
	out := make([]int32, Dims)
	for i := 0; i < len(landmarks); i++ {
		sim := latticeSimilarity(lat, landmarks[i])
		v := int32(sim * FixedPointScale)
		if v < 0 {
			v = 0
		}
		out[i] = v
	}
	return out
}

// embedToLattice converts text into a high-dimensional integer vector.
// This is bit-perfect across all platforms.
func embedToLattice(text string) []int64 {
	features := extractFeatures(text)
	vec := make([]int64, LatticeSize)
	if len(features) == 0 {
		return vec
	}

	for _, f := range features {
		dim, sign := hashFeature(f)
		vec[dim] += int64(sign)
	}
	return vec
}

// latticeSimilarity calculates a deterministic similarity between two lattice vectors.
func latticeSimilarity(a, b []int64) float64 {
	var dot, normA, normB int64
	for i := 0; i < LatticeSize; i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	// We use float64 for the final ratio, but since the inputs are large integers,
	// the precision loss is negligible and the result is stable.
	return float64(dot) / (math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
}

// extractFeatures returns all features for a text:
//   - word unigrams
//   - structural bigrams (word + functional word)
//   - char trigrams
func extractFeatures(text string) []string {
	tokens := tokenise(text)
	if len(tokens) == 0 {
		return nil
	}

	features := make([]string, 0, len(tokens)*6)

	// Unigrams and Structural Bigrams
	for i, t := range tokens {
		features = append(features, t)

		// Syntactic/Structural: if current or next is functional, link them
		if i < len(tokens)-1 {
			next := tokens[i+1]
			if functionalWords[t] || functionalWords[next] {
				features = append(features, "s:"+t+"_"+next)
			}
		}
	}

	// Character trigrams
	for _, t := range tokens {
		if functionalWords[t] {
			continue // Don't sub-hash functional words
		}
		padded := "\x02" + t + "\x03"
		for i := 0; i <= len(padded)-3; i++ {
			features = append(features, "c:"+padded[i:i+3])
		}
	}

	return features
}

// hashFeature returns (dimension, sign) for a feature string.
func hashFeature(feat string) (int, int) {
	h1 := fnv.New64a()
	h1.Write([]byte(feat))
	val1 := h1.Sum64()
	dim := int(val1 % uint64(LatticeSize))

	// Secondary hash for sign
	h2 := fnv.New64a()
	h2.Write([]byte(feat))
	h2.Write([]byte("reverse")) // Simpler than actual reverse for sign
	sign := 1
	if h2.Sum64()%2 == 0 {
		sign = -1
	}

	return dim, sign
}

// nonAlphanumRE matches runs of characters that are not letters or digits.
var nonAlphanumRE = regexp.MustCompile(`[^a-z0-9]+`)

// Tokenize returns the tokens that would be embedded for text.
// The count reflects exactly what the embedding algorithm operates on.
func Tokenize(text string) []string {
	return tokenise(text)
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

// CosineSimilarity returns the similarity between two fixed-point vectors.
func CosineSimilarity(a, b []int32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		vA := float64(a[i])
		vB := float64(b[i])
		dot += vA * vB
		normA += vA * vA
		normB += vB * vB
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

