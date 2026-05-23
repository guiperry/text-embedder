package embed

import (
	"math"
	"testing"
)

func TestEmbed_Deterministic(t *testing.T) {
	text := "The quick brown fox jumps over the lazy dog"
	v1 := Embed(text)
	v2 := Embed(text)

	for i := range v1 {
		if v1[i] != v2[i] {
			t.Fatalf("dimension %d differs: %v vs %v", i, v1[i], v2[i])
		}
	}
}

func TestEmbed_OutputDimensions(t *testing.T) {
	v := Embed("hello world")
	if len(v) != Dims {
		t.Fatalf("expected %d dims, got %d", Dims, len(v))
	}
}

func TestEmbed_FixedPointRange(t *testing.T) {
	texts := []string{
		"machine learning is fascinating",
		"river bank",
		"investment bank",
	}
	for _, text := range texts {
		v := Embed(text)
		for _, x := range v {
			if x < 0 || x > FixedPointScale {
				t.Errorf("text %q: value %d out of range [0, %d]", text, x, FixedPointScale)
			}
		}
	}
}

func TestEmbed_StructuralAwareness(t *testing.T) {
	// "bank of river" and "river bank" should be similar but not identical to "bank of money"
	vRiver1 := Embed("bank of the river")
	vRiver2 := Embed("river bank")
	vMoney := Embed("bank of the money")

	simRiver := CosineSimilarity(vRiver1, vRiver2)
	simMixed := CosineSimilarity(vRiver1, vMoney)

	if simMixed >= simRiver {
		t.Errorf("expected river-context to be more similar: river-river=%f mixed=%f",
			simRiver, simMixed)
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	v := Embed("")
	for i, x := range v {
		if x != 0 {
			t.Errorf("dimension %d: expected 0 for empty input, got %v", i, x)
		}
	}
}

func TestCosineSimilarity_Identical(t *testing.T) {
	v := Embed("artificial intelligence")
	sim := CosineSimilarity(v, v)
	if math.Abs(sim-1.0) > 1e-5 {
		t.Errorf("identical vectors: expected sim=1, got %f", sim)
	}
}

func TestCosineSimilarity_Similar(t *testing.T) {
	v1 := Embed("machine learning model training")
	v2 := Embed("deep learning neural network training")
	v3 := Embed("chocolate cake recipe baking")

	simClose := CosineSimilarity(v1, v2)
	simFar := CosineSimilarity(v1, v3)

	if simClose <= simFar {
		t.Errorf("expected ML texts to be more similar than unrelated texts: close=%f far=%f",
			simClose, simFar)
	}
}

func TestCosineSimilarity_CaseInsensitive(t *testing.T) {
	v1 := Embed("Hello World")
	v2 := Embed("hello world")
	sim := CosineSimilarity(v1, v2)
	if math.Abs(sim-1.0) > 1e-5 {
		t.Errorf("case variants should be identical, got sim=%f", sim)
	}
}

func BenchmarkEmbed(b *testing.B) {
	text := "The quick brown fox jumps over the lazy dog. " +
		"Natural language processing enables computers to understand human text."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Embed(text)
	}
}
